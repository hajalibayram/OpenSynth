import logging
from calendar import monthrange
from collections.abc import Generator
from datetime import date
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import torch
from tqdm.auto import tqdm

from opensynth.data_modules.goiener_data_module import GoiEnerDataModule
from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.models.energydiff import PLDiffusion1D

DATE_COLUMNS = ["date", "DATE"]
DATETIME_COLUMNS = ["datetime", "DATETIME"]


logger = logging.getLogger(__name__)


def load_lcl_data_by_year(
    fname: Path | str | None = None,
    year: int = 2013,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Load LCL data for a specific year.

    Returns a DataFrame in wide format. The first column contains the timestamp.

    ArgsL
        fname (str or Path): Location of the `train.csv` data file.
        year (int): Year to load.

    Returns:
        pl.DataFrame with KWH/hh measurements.
    """
    fname = (
        Path(__file__).parents[0] / "../../../../data/raw/historical/train.csv"
        if fname is None
        else Path(fname)
    )
    if not fname.exists():
        raise ValueError(
            "LCL dataset not found, "
            "please download it or supply correct path to train.csv"
        )

    logger.info(f"Loading LCL data from {str(fname.resolve())}...")
    lcl = (
        pl.scan_csv(
            fname,
            schema={
                "LCLid": pl.String,
                "stdorTtoU": pl.String,
                "DateTime": pl.String,
                "KWH/hh (per half hour)": pl.String,
            },
        )
        .with_columns(
            pl.col("KWH/hh (per half hour)")
            .str.strip_chars()
            .cast(pl.Float32, strict=False)
            .fill_null(0)
            .alias("kWH"),
            pl.col("DateTime")
            .str.slice(0, 16)
            .str.to_datetime()
            .alias("datetime"),
        )
        .collect()
        .unique()
        .pivot(on="LCLid", index="datetime", values="kWH")
        .sort("datetime")
        .fill_null(0)
    )

    logger.info(f"Selecting year {year}")
    lcl = lcl.filter(pl.col("datetime").dt.year() == year)

    if fmt == "pandas":
        return lcl.to_pandas()

    return lcl

def load_goiener_data_by_year(
    fname: Path | str | None = None,
    year: int = 2018,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Load GoiEner data for a specific year.

    Returns a DataFrame in wide format. The first column contains the timestamp.

    ArgsL
        fname (str or Path): Location of the `train.csv` data file.
        year (int): Year to load.

    Returns:
        pl.DataFrame with KWH/hh measurements.
    """
    fname = (
        Path(__file__).parents[0] / "../../../../data/raw/goiener/historical/train.csv"
        if fname is None
        else Path(fname)
    )
    if not fname.exists():
        raise ValueError(
            "GoiEner dataset not found, "
            "please download it or supply correct path to train.csv"
        )

    logger.info(f"Loading GoiEner data from {str(fname.resolve())}...")
    goiener = (
        pl.scan_csv(
            fname,
            schema={
                "profile_id": pl.String,
                "timestamp": pl.String,
                "kWh": pl.String,
            },
        )
        .with_columns(
            pl.col("kWh")
            .str.strip_chars()
            .cast(pl.Float32, strict=False)
            .fill_null(0)
            .alias("kWH"),
            pl.col("timestamp")
            .str.slice(0, 16)
            .str.to_datetime()
            .alias("datetime"),
        )
        .collect()
        .unique()
        .pivot(on="profile_id", index="datetime", values="kWH")
        .sort("datetime")
        .fill_null(0)
    )

    logger.info(f"Selecting year {year}")
    goiener = goiener.filter(pl.col("datetime").dt.year() == year)

    if fmt == "pandas":
        return goiener.to_pandas()

    return goiener

import torch
import numpy as np
import polars as pl
from datetime import date, datetime, timedelta, time
from typing import Union
from opensynth.data_modules.goiener_data_module import GoiEnerDataModule
from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.models.energydiff import PLDiffusion1D
from opensynth.models.energydiff.calibrate import calibrate

def generate_synthetic_samples(
    model: PLDiffusion1D,
    dm: Union[LCLDataModule, GoiEnerDataModule],
    n_profiles: int,
    batch_size: int = 1000,
    step: int = 100,
    year: int = 2022,
    month: int | None = None,
) -> pl.DataFrame:
    """
    Generate `n_profiles` synthetic half-hour traces for each day in `year` (or in a specific `month`).
    Returns a Polars DataFrame with:
      - N = n_days * 48 rows,
      - 1 `datetime` column (half-hour stamps),
      - n_profiles columns named `profile_0`…`profile_{n_profiles-1}`.
    """
    # 1) Build the day-level calendar
    cal = (
        pl.date_range(date(year, 1, 1), date(year, 12, 31), "1d", eager=True)
          .to_frame("date")
          .with_columns([
              pl.col("date").dt.weekday().alias("dayofweek"),
              pl.col("date").dt.month().alias("month"),
          ])
    )
    if month is not None:
        cal = cal.filter(pl.col("month") == month)
    n_days = cal.height
    if n_days == 0:
        raise ValueError(f"No days found for year={year} month={month}")

    # 2) Determine total daily samples needed
    total_draws = n_profiles * n_days

    # 3) Sample the diffusion model in one go
    ema = getattr(model, "ema", None)
    net = ema.ema_model if (ema and hasattr(ema, "ema_model")) else model.diffusion_model
    raw = net.dpm_solver_sample(
        total_num_sample=total_draws,
        batch_size=batch_size,
        step=step,
        shape=(48, 1),
    ).squeeze(-1)  # → (total_draws, 48)

    # 4) Reconstruct & calibrate
    batch = next(iter(dm.train_dataloader()))
    train_kwh = dm.reconstruct_kwh(batch["kwh"])
    samples_kwh = dm.reconstruct_kwh(raw).clamp(min=0)
    calib = calibrate(train_kwh, samples_kwh).to("cpu").numpy()
    arr = calib.reshape(n_profiles, n_days, 48)  # (profiles, days, half-hours)

    # 5) Extract actual Python dates from `cal`
    date_list = cal["date"].to_list()  # list of datetime.date
    start_date = date_list[0]
    end_date = date_list[-1]

    start_dt = datetime.combine(start_date, time(0, 0))  # 00:00 of first day
    end_dt = datetime.combine(end_date, time(23, 30))  # 23:30 of last day

    timestamps = pl.datetime_range(
        start_dt,  # datetime
        end_dt,  # datetime
        "30m",  # interval string
        eager=True
    )

    # 7) Assemble the wide DataFrame
    data = {"datetime": timestamps}
    for i in range(n_profiles):
        data[f"profile_{i}"] = arr[i].reshape(n_days * 48)

    return pl.DataFrame(data)



def g3enerate_synthetic_samples(
    model: PLDiffusion1D,
    dm: LCLDataModule | GoiEnerDataModule,
    num_samples: int,
    batch_size: int = 1000,
    step: int = 100,
    year: int = 2022,
    month: int = None,
) -> Tuple[torch.Tensor, pl.DataFrame]:
    """
    Generate `num_samples` synthetic daily profiles (48 half-hours) plus matching dates.

    Returns:
        samples_kwh_calib (torch.Tensor): shape (num_samples, 48)
        metadata_df       (pl.DataFrame): columns [sample_id, datetime, month, dayofweek], one row per sample
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    # 1) SAMPLE FROM THE DIFFUSION MODEL
    ema = getattr(model, "ema", None)
    net = ema.ema_model if (ema and hasattr(ema, "ema_model")) else model.diffusion_model
    raw = net.dpm_solver_sample(
        total_num_sample=num_samples,
        batch_size=batch_size,
        step=step,
        shape=(48, 1)
    ).squeeze(-1)  # → (num_samples, 48)

    # 2) RECONSTRUCT & CALIBRATE
    batch = next(iter(dm.train_dataloader()))
    train_kwh = dm.reconstruct_kwh(batch["kwh"])
    samples_kwh = dm.reconstruct_kwh(raw).clamp(min=0)
    from opensynth.models.energydiff.calibrate import calibrate
    calib = calibrate(train_kwh, samples_kwh).to("cpu").numpy()
    samples_kwh_calib = torch.tensor(calib)  # (num_samples, 48)

    # 3) BUILD A YEARLY CALENDAR
    cal = (
        pl.date_range(date(year, 1, 1), date(year, 12, 31), "1d", eager=True)
        .to_frame("datetime")
        .with_columns([
            pl.col("datetime").dt.weekday().alias("dayofweek"),
            pl.col("datetime").dt.month().alias("month"),
        ])
    )
    if month is not None:
        cal = cal.filter(pl.col("month") == month)

    # 4) SAMPLE WEEKDAYS FROM THE TRAINING DISTRIBUTION
    dow = batch["features"]["dayofweek"]
    probs = torch.bincount(dow).float() / dow.numel()
    sampled_dow = torch.multinomial(probs, num_samples=num_samples, replacement=True).numpy()

    # 5) MAKE A “SAMPLES” FRAME AND JOIN TO ALL MATCHING DATES
    samples_df = pl.DataFrame({
        "sample_id": np.arange(num_samples),
        "sampled_dayofweek": sampled_dow
    })
    joined = samples_df.join(
        cal,
        left_on="sampled_dayofweek",
        right_on="dayofweek",
        how="left"
    )

    # 6) PICK EXACTLY ONE DATE PER SAMPLE
    # Try Polars grouping first
    if hasattr(joined, "groupby"):
        picked = (
            joined
            .groupby("sample_id", maintain_order=True)
            .agg([pl.col("datetime").sample(1)])
            .explode("datetime")
        )
    else:
        # Fallback: convert to pandas, sample, then back
        jp = joined.to_pandas()
        # each group.sample(1) picks one random date
        p = (
            jp.groupby("sample_id", sort=False)["datetime"]
            .apply(lambda s: s.sample(n=1).iloc[0])
            .reset_index()
        )
        picked = pl.from_pandas(p.rename(columns={"datetime": "datetime"}))

    # 7) EXTRACT month & dayofweek
    metadata_df = (
        picked
        .with_columns([
            pl.col("datetime").dt.month().alias("month"),
            pl.col("datetime").dt.weekday().alias("dayofweek"),
        ])
        .select(["sample_id", "datetime", "month", "dayofweek"])
        .sort("sample_id")
    )

    return samples_kwh_calib, metadata_df

def g2enerate_synthetic_samples(
    model: PLDiffusion1D,
    dm: LCLDataModule | GoiEnerDataModule,
    num_samples: int,
    batch_size: int = 1000,
    step: int = 100,
    year: int = 2022,
    month: int | None = None,
) -> tuple[torch.Tensor, pl.DataFrame]:
    """Generate samples using EnergyDiff model with temporal organization."""
    # Get EMA model if available
    ema_model = model.ema.ema_model if hasattr(model, 'ema') else model.diffusion_model

    # Generate base samples
    samples = ema_model.dpm_solver_sample(
        total_num_sample=num_samples,
        batch_size=batch_size,
        step=step,
        shape=(48, 1)
    )
    samples = samples.squeeze(-1)

    # Get real data for calibration
    train_data = next(iter(dm.train_dataloader()))
    train_kwh = dm.reconstruct_kwh(train_data['kwh'])

    # Reconstruct and calibrate samples
    samples_kwh = dm.reconstruct_kwh(samples)
    samples_kwh = torch.clip(samples_kwh, min=0)

    # Calibrate samples
    from opensynth.models.energydiff import calibrate
    samples_kwh_calib = torch.tensor(
        calibrate.calibrate(train_kwh, samples_kwh).to('cpu')
    )

    # Create temporal metadata matching training data patterns
    metadata_df = (
        pl.date_range(date(year, 1, 1), date(year, 12, 31), "1d", eager=True)
        .alias("datetime")
        .to_frame()
        .with_columns([
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.col("datetime").dt.month().alias("month"),
            pl.col("datetime").dt.weekday().alias("dayofweek"),
        ])
    )

    if month is not None:
        metadata_df = metadata_df.filter(pl.col("month") == month)

    # Get weekday distribution from training data
    weekday_dist = torch.bincount(train_data['features']['dayofweek'])
    weekday_probs = weekday_dist / weekday_dist.sum()

    # Sample weekdays according to training distribution
    weekdays = torch.multinomial(
        weekday_probs,
        num_samples=len(samples_kwh_calib),
        replacement=True
    )

    # Sort metadata by sampled weekdays
    metadata_df = metadata_df.with_columns(
        pl.Series("sampled_weekday", weekdays.numpy())
    ).filter(
        pl.col("dayofweek") == pl.col("sampled_weekday")
    ).drop("sampled_weekday").head(len(samples_kwh_calib))

    return samples_kwh_calib, metadata_df

def g1enerate_synthetic_samples(
        model: PLDiffusion1D,
        dm: LCLDataModule | GoiEnerDataModule,
        num_samples: int,
        batch_size: int = 1000,
        step: int = 100,
        year: int = 2022,
        month: int | None = None,
) -> tuple[torch.Tensor, pd.DataFrame]:
    """Generate samples using EnergyDiff model with temporal organization.

    Args:
        model: EnergyDiff model
        dm: Data module
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        step: Number of DPM-Solver steps
        year: Year to generate samples for
        month: Optional specific month (1-12)

    Returns:
        tuple: (samples, metadata_df) where samples is tensor of shape (num_samples, 48)
        and metadata_df contains temporal information
    """
    # Get EMA model if available
    ema_model = model.ema.ema_model if hasattr(model, 'ema') else model.diffusion_model

    # Generate base samples
    samples = ema_model.dpm_solver_sample(
        total_num_sample=num_samples,
        batch_size=batch_size,
        step=step,
        shape=(48, 1)
    )
    samples = samples.squeeze(-1)

    # Get real data for calibration
    train_data = next(iter(dm.train_dataloader()))
    train_kwh = dm.reconstruct_kwh(train_data['kwh'])

    # Reconstruct and calibrate samples
    samples_kwh = dm.reconstruct_kwh(samples)
    samples_kwh = torch.clip(samples_kwh, min=0)

    # Calibrate samples
    from opensynth.models.energydiff import calibrate
    samples_kwh_calib = torch.tensor(
        calibrate.calibrate(train_kwh, samples_kwh).to('cpu')
    )

    # Create temporal metadata matching training data patterns
    metadata_df = (
        pl.date_range(date(year, 1, 1), date(year, 12, 31), "1d", eager=True)
        .alias("datetime")
        .to_frame()
        .with_columns([
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.col("datetime").dt.month().alias("month"),
        ])
    )

    if month is not None:
        metadata_df = metadata_df.filter(pl.col("month") == month)

    # Sort samples by weekday pattern to match temporal structure
    weekday_patterns = train_data['features']['dayofweek'].unique()
    sorted_indices = []

    for weekday in weekday_patterns:
        mask = train_data['features']['dayofweek'] == weekday
        weekday_samples = samples_kwh_calib[mask]
        sorted_indices.extend(weekday_samples.indices)

    samples_kwh_calib = samples_kwh_calib[sorted_indices]

    # Take subset of metadata to match number of samples
    metadata_df = metadata_df.sample(n=len(samples_kwh_calib), shuffle=False)

    return samples_kwh_calib, metadata_df

def mm(
    model: PLDiffusion1D,
    dm: LCLDataModule | GoiEnerDataModule,
    n_samples: int,
    year: int = 2017,
    month: int | None = None,
) -> Generator[
    Tuple[date, float, float, np.typing.NDArray[np.float64]], None, None
]:
    """Generate Faraday samples for a specific month/year combination.

    Samples will be generated with a timestamp that fits the specified year and month.
    If month is not specified, it can be any month.

    Args:
        model (PLDiffusion1D): Model
        dm (LCLDataModule | GoiEnerDataModule): Data module.
        n_samples (int): Number of synthetic samples to generate.
        year (int, optional): Year to use for timestamps.
        month (int, optional): Month (1-based) to use. If generated samples do not
            match the specified month, they will be discarded until enough samples
            are specified that do match.

    Yields:
        Tuple with datetime, month, day_of_week, generated sample values
    """
    if n_samples < 2:
        raise ValueError("n_samples must be higher than 1")

    sample_df = (
        pl.date_range(date(year, 1, 1), date(year, 12, 31), "1d", eager=True)
        .alias("datetime")
        .to_frame()
        .with_columns(
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.col("datetime").dt.month().alias("month"),
        )
    )

    gmm_samples = model.sample_gmm(n_samples)
    gmm_samples_reconstructed = dm.reconstruct_kwh(gmm_samples["kwh"])
    gmm_samples_reconstructed = torch.clip(gmm_samples_reconstructed, min=0)
    for torch_month, dayofweek, values in zip(
        gmm_samples["features"]["month"],
        gmm_samples["features"]["dayofweek"],
        gmm_samples_reconstructed,
    ):
        g_month = torch_month.numpy()[0]
        try:
            if month is None or g_month == month:
                yield (
                    sample_df.filter(
                        pl.col("weekday") == dayofweek.numpy()[0] + 1,
                        pl.col("month") == g_month,
                    ).sample(1)["datetime"][0],
                    g_month,
                    dayofweek.numpy()[0],
                    values.detach().numpy(),
                )

        except Exception as e:
            print(e)
            continue


def generate_synthetic_sample_df(
    model: PLDiffusion1D,
    dm: LCLDataModule | GoiEnerDataModule,
    n_samples: int,
    year: int = 2022,
    month: int | None = None,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Generate DataFrame Faraday samples for a specific month/year combination.

    Samples will be generated with a timestamp that fits the specified year and month.
    If month is not specified, it can be any month.

    Args:
        model (FaradayModel): Model
        dm (LCLDataModule | GoiEnerDataModule): Data module.
        n_samples (int): Number of synthetic samples to generate.
        year (int, optional): Year to use for timestamps.
        month (int, optional): Month (1-based) to use. If generated samples do not
            match the specified month, they will be discarded until enough samples
            are specified that do match.
        fmt (str, optional): Either "pandas" or "polars", default is "pandas".

    Returns:
        pl.DataFrame in wide format with datetime as first columns.
    """
    df = pl.DataFrame(
        np.array(
            [
                (datetime, m, d, *values)
                for datetime, m, d, values in generate_synthetic_samples(
                    model, dm, n_samples, year=year, month=month
                )
            ]
        ).tolist(),
        schema={"date": pl.Date, "month": int, "dayofweek": int}
        | {
            d: float
            for d in [f"{i // 2:02d}{(i % 2) * 30:02d}" for i in range(48)]
        },
        orient="row",
    )

    if fmt == "pandas":
        return df.to_pandas()

    return df


def generate_full_synthetic_month(
    model: PLDiffusion1D,
    dm: LCLDataModule | GoiEnerDataModule,
    year: int,
    month: int,
    n_samples: int = 2,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Generate DataFrame Faraday samples for a specific month.

    Samples will be generated with a timestamp that fits the specified month.

    Args:
        model (FaradayModel): Model
        dm (LCLDataModule | GoiEnerDataModule): Data module.
        year (int, optional): Year to use for timestamps.
        month (int, optional): Month (1-based) to use. If generated samples do not
            match the specified month, they will be discarded until enough samples
            are specified that do match.
        n_samples (int): Number of synthetic samples to generate.
        fmt (str, optional): Either "pandas" or "polars", default is "pandas".

    Returns:
        pl.DataFrame in wide format with datetime as first columns.
    """
    batch_size = 1000
    df = generate_synthetic_sample_df(
        model, dm, batch_size, year=year, month=month, fmt="polars"
    )

    while (
        df.group_by("date").len().min()["len"][0] < n_samples
        or len(df["date"].unique()) < monthrange(year, month)[1]
    ):
        df = pl.concat(
            (
                df,
                generate_synthetic_sample_df(
                    model,
                    dm,
                    batch_size,
                    year=year,
                    month=month,
                    fmt="polars",
                ),
            )
        )
    df = pl.concat(
        [p.sample(n_samples).with_row_index() for p in df.partition_by("date")]
    )

    if fmt == "pandas":
        return df.to_pandas()

    return df


def generate_full_synthetic_year(
    model: PLDiffusion1D,
    dm: LCLDataModule | GoiEnerDataModule,
    year: int,
    n_samples: int = 2,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Generate DataFrame Faraday samples for a specific year.

    Samples will be generated with all timesteps for all months in the specified year.

    Args:
        model (FaradayModel): Model
        dm (LCLDataModule | GoiEnerDataModule): Data module.
        year (int, optional): Year to use for timestamps.
        n_samples (int): Number of synthetic samples to generate.
        fmt (str, optional): Either "pandas" or "polars", default is "pandas".

    Returns:
        pl.DataFrame in wide format with datetime as first columns.
    """
    df = pl.concat(
        [
            generate_full_synthetic_month(
                model, dm, year, month, n_samples=n_samples, fmt="polars"
            )
            for month in tqdm(range(1, 13))
        ]
    )
    df = (
        semiwide_to_wide(
            df.select(pl.exclude("month", "dayofweek")),
            date_col="date",
            datetime_name="datetime",
        )
        .with_columns(pl.col("index").cast(str))
        .transpose(
            column_names="index", include_header=True, header_name="datetime"
        )
        .with_columns(pl.col("datetime").str.to_datetime())
    )

    if fmt == "pandas":
        return df.to_pandas()

    return df


def infer_date_column(df: pl.DataFrame) -> str:
    """Return column name for a Date columns in input DataFrame.

    Returns the column name of a column in Date format, or a String column that
    matches a Date string. If the DataFrame contains only one matching column,
    this function will return that column name. If multiple columns match, it will
    return the column name that matches a canonical Date name, such as "DATUM".
    In all other cases the function will raise a ValueError().

    Args:
        df (pl.DataFrame): DataFrame.

    Returns:
        str: column name of a column in Date or Date-like format.

    Raises:
        ValueError: if no columns are in a Date-like format or multiple columns are
        in Date-like format and match a canonical name.

    """
    date_columns = df.select(pl.col(pl.Date)).columns
    date_columns = list(
        set(df.columns).intersection(DATE_COLUMNS).union(date_columns)
    )
    canonical_columns = set(DATE_COLUMNS).intersection(date_columns)

    match len(date_columns):
        case 0:
            raise ValueError(
                "No Date or Date-like columns found in DataFrame!"
            )
        case 1:
            return date_columns[0]
        case _ if len(canonical_columns) == 1:
            return list(canonical_columns)[0]
        case _:
            raise ValueError(
                "Multiple Date-like columns found with a matching canonical name!"
            )


def semiwide_to_long(
    df: pl.DataFrame,
    on: Optional[list[str]] = None,
    date_col: Optional[str] = None,
    datetime_name: Optional[str] = None,
    value_name: Optional[str] = None,
) -> pl.DataFrame:
    """Convert polars DataFrame from semi-wide to long format.

    The semi-wide format is based on a split between date (rows) and time
    (columns). Therefore, this function will only work on a DateFrame with
    a Date column that contains dates and at least one timestamp column, by
    default in "%HH%mm" format.

    Args:
        df (polars.DataFrame): DataFrame in semi-wide wide format, containing DateTime-
            compatible column names.
        on (list, optional): Columns to use as timepoints. By default, all columns that
            match the pattern '[0-9][0-9][0-9][0-9]' will be used.
        date_col (str, optional): Column that contains the Date values. By default,
            a column that is in Date format, or that is a Date-compatible string, will
            be used, if there is only one column in that format. If there are multiple
            Date-compatible, columns, but only one matches a canonical name such as
            DATUM, that column will be used. Otherwise, this method will fail, and the
            date_col needs to be explicitly specified.
        datetime_name (str, optional): Name for the DateTime column in the long
            DataFrame, "DATUM_TIJD" by default.
        value_name (str, optional): Name to give to the value column. Defaults to
            "value".

    Returns:
        polars.DataFrame in long format.

    """
    on = df.select(cs.matches(r"^\d\d\d\d$")).columns if on is None else on
    date_col = infer_date_column(df) if date_col is None else date_col
    datetime_name = (
        DATETIME_COLUMNS[0] if datetime_name is None else datetime_name
    )
    value_name = "value" if value_name is None else value_name

    if str(df.select(date_col).dtypes[0]) == "String":
        df = df.with_columns(pl.col(date_col).str.to_date().alias(date_col))

    long_df = (
        df.unpivot(
            index=df.select(pl.exclude(on)).columns,
            on=df.select(on).columns,
            value_name=value_name,
        )
        .with_columns(
            (
                pl.col(date_col).dt.strftime("%Y-%m-%d")
                + " "
                + pl.col("variable")
            )
            .str.to_datetime(time_unit="ns", time_zone="UTC")
            .alias(datetime_name),
        )
        .drop(date_col, "variable")
        .sort(datetime_name)
    )

    return long_df


def semiwide_to_wide(
    df: pl.DataFrame,
    on: Optional[list[str]] = None,
    date_col: Optional[str] = None,
    datetime_name: Optional[str] = None,
) -> pl.DataFrame:
    """Convert polars DataFrame from semi-wide to wide format.

    The semi-wide format is based on a split between date (rows) and time
    (columns). Therefore, this function will only work on a DateFrame with
    a Date column that contains dates and at least one timestamp column, by
    default in "%HH%mm" format.

    Args:
        df (polars.DataFrame): DataFrame in semi-wide wide format, containing
            DateTime-compatible column names.
        on (list, optional): Columns to use as timepoints. By default, all
            columns that match the pattern '[0-9][0-9][0-9][0-9]' will be used.
        date_col (str, optional): Column that contains the Date values. By default,
            a column that is in Date format, or that is a Date-compatible string,
            will be used, if there is only one column in that format. If there are
            multiple Date-compatible, columns, but only one matches a canonical name
            such as DATUM, that column will be used. Otherwise, this method will fail,
            and the date_col needs to be explicitly specified.
        datetime_name (str, optional): Name for the DateTime column in the long
            DataFrame, "datetime" by default.

    Returns:
        polars.DataFrame in wide format.

    """
    date_col = infer_date_column(df) if date_col is None else date_col
    datetime_name = (
        DATETIME_COLUMNS[0] if datetime_name is None else datetime_name
    )

    return (
        semiwide_to_long(
            df, on=on, date_col=date_col, datetime_name=datetime_name
        )
        .with_columns(
            pl.col(datetime_name)
            .dt.strftime("%Y-%m-%d %H:%M")
            .alias(datetime_name)
        )
        .pivot(on=datetime_name, values="value", aggregate_function="first")
    )
