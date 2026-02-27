# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import json

random.seed(42)

logger = logging.getLogger(__name__)

import pandas as pd
from typing import Optional, List

def fit_categorical_encoders(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> dict[str, dict]:
    encoders: dict[str, dict] = {}

    if not feature_cols:
        return encoders

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found")

        cat = pd.Categorical(df[col].astype(str))
        encoders[col] = {
            "type": "categorical_int_encoded",
            "categories": list(cat.categories),
        }

    return encoders


def apply_categorical_encoders(
    df: pd.DataFrame,
    encoders: dict[str, dict],
) -> pd.DataFrame:
    df_out = df.copy()

    for col, spec in encoders.items():
        if spec.get("type") != "categorical_int_encoded":
            continue

        categories = spec["categories"]
        mapping = {cat: i for i, cat in enumerate(categories)}

        # map unknowns to -1 (or raise if you prefer strict behavior)
        df_out[col] = df_out[col].astype(str).map(mapping).fillna(-1).astype(int)

    return df_out

def split_household_ids(
    df: pd.DataFrame,
    id_col: str,
    sample_fraction: float = 0.75,
) -> Tuple[List[str], List[str]]:
    """
    Split dataset into training vs holdout households.

    Args:
        df (pd.DataFrame): dataset
        id_col (str): Name of the household ID column
        sample_fraction (float): Fraction of household ids to include in
        training set

    Returns:
        Tuple[List[str], List[str]]: List of training and holdout household ids
    """
    logger.info("Splitting households into train and holdout households")
    unique_ids = df[id_col].unique().tolist()
    random.shuffle(unique_ids)
    sample_size = int(len(unique_ids) * sample_fraction)

    train_ids = unique_ids[:sample_size]
    holdout_ids = unique_ids[sample_size:]

    return train_ids, holdout_ids


def split_historical_future_periods(
    df: pd.DataFrame,
    datetime_col: str,
    historical_start: str,
    historical_end: str,
    future_start: str,
    future_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into:
     1) historical period
     2) future period
    These periods are used for TSTR evaluation.

    Args:
        df (pd.DataFrame): pd.DataFrame
        datetime_col (str): Name of the datetime column
        historical_start (str): Start date for historical period
        historical_end (str): End date for historical period
        future_start (str): Start date for future period
        future_end (str): End date for future period

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Historical and Future dataframe.
    """
    # Format the start and end dates to include time information
    historical_start = pd.Timestamp(historical_start, tz="UTC").replace(
        hour=00, minute=00, second=00
    )
    historical_end = pd.Timestamp(historical_end, tz="UTC").replace(
        hour=23, minute=59, second=59
    )
    future_start = pd.Timestamp(future_start, tz="UTC").replace(
        hour=00, minute=00, second=00
    )
    future_end = pd.Timestamp(future_end, tz="UTC").replace(
        hour=23, minute=59, second=59
    )

    historical_start_mask = df[datetime_col] >= historical_start
    historical_end_mask = df[datetime_col] <= historical_end
    tstr_historical_mask = historical_start_mask & historical_end_mask

    future_start_mask = df[datetime_col] >= future_start
    future_end_mask = df[datetime_col] <= future_end
    tstr_future_mask = future_start_mask & future_end_mask

    df_historical = df.loc[tstr_historical_mask]
    df_future = df.loc[tstr_future_mask]
    return df_historical, df_future


def split_data(
    data_dir: str,
    csv_filename: Path,
    sample_fraction: float = 0.75,
    id_col: str = "ID",
    kwh_col: str = "kwh",
    datetime_col: str = "DateTime",
    utc=True,
    datetime_format: Optional[str] = None,
    historical_start: str = "2012-01-01",
    historical_end: str = "2013-12-31",
    future_start: str = "2014-01-01",
    future_end: str = "2015-12-31",
    feature_cols: Optional[List[str]] = None,
) -> None:
    """
    Split the dataset 4 ways:
    1) Historical Train household data
    2) Historical Holdout household data
    3) Future Train household data
    4) Future Holdout household data

    Historical data is used for training generative models.
    Future data is used for Train-Synthetic-Test-Real (TSTR) evaluation.

    Args:
        data_dir (str): Directory to store the processed data.
        csv_filename (Path): Path to the CSV file containing the raw data.
        sample_fraction (float): Fraction of households to include in the
            training set. Defaults to 0.75.
        id_col (str): Name of the household ID column. Defaults to "id".
        kwh_col (str): Name of the kWh column.
        datetime_col (str): Name of the datetime column. Defaults to
            "DateTime".
        utc (bool): Whether to parse datetime as UTC. Defaults to False.
        datetime_format (str, optional): Format of the datetime column.
            Defaults to None.
        historical_start (str): Start date for historical data. Defaults to
            "2012-01-01".
        historical_end (str): End date for historical data. Defaults to
            "2013-12-31".
        future_start (str): Start date for future data. Defaults to
            "2014-01-01".
        future_end (str): End date for future data. Defaults to "2014-12-31".
    Returns:
        None
    """

    logger.info(f"👀 Reading data from: {csv_filename}")
    df = pd.read_csv(csv_filename)

    logger.info("🧹 Formatting data")
    df[datetime_col] = pd.to_datetime(
        df[datetime_col], utc=utc, format=datetime_format
    )
    df[kwh_col] = df[kwh_col].replace("Null", np.float64())
    df[kwh_col] = df[kwh_col].astype(float)
    df[id_col] = df[id_col].astype(str)

    if feature_cols:
        feature_encoders = fit_categorical_encoders(df, feature_cols=feature_cols)
        df = apply_categorical_encoders(df, feature_encoders)

        enc_path = Path(f"{data_dir}/goiener_feature_encoders.json")
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(enc_path, "w") as f:
            json.dump(feature_encoders, f, indent=2)

    logger.info("🖖 Spliting households into train and holdout")
    train_ids, holdout_ids = split_household_ids(
        df,
        id_col,
        sample_fraction=sample_fraction,
    )
    logger.info(f"Train len: {len(train_ids)}")
    logger.info(f"Holdout len: {len(holdout_ids)}")

    logger.info("📆 Splitting data into train and holdout period")
    df_history, df_future = split_historical_future_periods(
        df,
        datetime_col,
        historical_start,
        historical_end,
        future_start,
        future_end,
    )
    logger.info(f"History len: {len(df_history)}")
    logger.info(f"Future len: {len(df_future)}")

    df_historical_train = df_history[df_history[id_col].isin(train_ids)]
    df_future_train = df_future[df_future[id_col].isin(train_ids)]
    df_historical_holdout = df_history[df_history[id_col].isin(holdout_ids)]
    df_future_holdout = df_future[df_future[id_col].isin(holdout_ids)]

    logger.info("📦 Saving train and holdout data")
    historical_path = Path(f"{data_dir}/raw/goiener/historical")
    future_path = Path(f"{data_dir}/raw/goiener/future")
    os.makedirs(historical_path, exist_ok=True)
    os.makedirs(future_path, exist_ok=True)

    df_historical_train.to_csv(
        f"{historical_path}/train.csv",
        index=False,
    )
    df_historical_holdout.to_csv(
        f"{historical_path}/holdout.csv",
        index=False,
    )
    df_future_train.to_csv(f"{future_path}/train.csv", index=False)
    df_future_holdout.to_csv(f"{future_path}/holdout.csv", index=False)

    logger.info("👍 Done!")
