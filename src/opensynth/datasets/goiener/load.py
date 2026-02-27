# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

def load_goiener_data_by_year(
    fname: Path | str | None = None,
    year: int = 2024,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Load GoiEner data for a specific year.

    Returns a DataFrame in wide format.
    The first column contains the timestamp.

    Args:
        fname (str or Path): Location of the `train.csv` data file.
        year (int): Year to load.

    Returns:
        pl.DataFrame with kWh measurements.
    """
    fname = (
        Path(__file__).parents[0] / "./data/raw/goiener/historical/train.csv"
        if fname is None
        else Path(fname)
    )
    if not fname.exists():
        raise FileNotFoundError(
            "GoiEner dataset not found, "
            "please download it or supply correct path to train.csv"
        )

    logger.info(f"Loading GoiEner data from {str(fname.resolve())}...")
    goiener = (
        pl.scan_csv(
            fname,
            schema={
                "id": pl.String,
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
        .pivot(on="id", index="datetime", values="kWH")
        .sort("datetime")
        .fill_null(0)
    )

    logger.info(f"Selecting year {year}")
    goiener = goiener.filter(pl.col("datetime").dt.year() == year)

    if fmt == "pandas":
        return goiener.to_pandas()

    return goiener



logger = logging.getLogger(__name__)
