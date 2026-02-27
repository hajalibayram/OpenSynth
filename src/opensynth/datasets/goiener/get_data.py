# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Optional

from opensynth.datasets import datasets_utils
from opensynth.datasets.goiener import (
    preprocess_goiener,
    split_households,
)

logger = logging.getLogger(__name__)


def split_preprocess_data(
    split: bool,
    preprocess: bool,
    data_dir: str,
    csv_data_path: str,
    sample_fraction: float,
    time_resolution: str,
    feature_cols: list[str],
    id_col: str,
    kwh_col: str,
    datetime_col: str,
    utc: bool,
    datetime_format: Optional[str],
    historical_start: str,
    historical_end: str,
    future_start: str,
    future_end: str,
    drop_nulls: bool,
):
    """
    Read in, split and preprocess the dataset.

    Requirements:
    - Dataset needs to be stored in CSV format in {data_dir}/{csv_data_path}.
    - Dataset needs to have an ID column, a kWh column and a datetime column.
    - If time_resolution is "half_hourly", data needs to be at half-hourly
      resolution. If "hourly", data needs to be at hourly resolution.
    - Datetime column needs to be in a format that can be parsed by
        pandas.to_datetime, if datetime_format is specified, then this will be
        parsed into pandas.to_datetime.
    - kWh column needs to be in numeric format.
    - feature_cols needs to be a list of columns in the dataset to include as
      features. These can be categorical or numeric.
    - historical_start, historical_end, future_start and future_end define
        the historical and future periods for for TSTR evaluation. Needs to be
        in the format "YYYY-MM-DD".
    Args:
        data_dir (str): Directory to save the processed data.
        csv_data_path (str): Path to the CSV file containing the dataset.
        sample_fraction (float): Fraction of households to include in the
            training set.
        time_resolution (str): Time resolution of the data. Allowed values:
            "half_hourly", "hourly".
        feature_cols (List[str]): List of feature columns to include.
        id_col (str): Name of the household ID column.
        kwh_col (str): Name of the kWh column.
        datetime_col (str): Name of the datetime column.
        utc (bool): Whether to parse datetime as UTC.
        datetime_format (str, optional): Format of the datetime column.
        historical_start (str): Start date for historical data.
        historical_end (str): End date for historical data.
        future_start (str): Start date for future data.
        future_end (str): End date for future data.
        drop_nulls (bool): Whether to drop rows with NaN kwh values. If False,
            will replace NaN kwh values with 0.0
    """

    CSV_FILE_NAME = Path(f"{data_dir}/{csv_data_path}")
    logger.info(
        f"Reading data from {CSV_FILE_NAME}. Storing data in {data_dir}."
    )

    if split:
        # Split dataset into training/ holdout sets
        split_households.split_data(
            data_dir,
            CSV_FILE_NAME,
            sample_fraction=sample_fraction,
            id_col=id_col,
            kwh_col=kwh_col,
            datetime_col=datetime_col,
            utc=utc,
            datetime_format=datetime_format,
            historical_start=historical_start,
            historical_end=historical_end,
            future_start=future_start,
            future_end=future_end,
            feature_cols=feature_cols,
        )
    if preprocess:
        # Preprocess the data into daily load profiles
        preprocess_goiener.preprocess_data(
            data_dir,
            datetime_col=datetime_col,
            kwh_col=kwh_col,
            id_col=id_col,
            utc=utc,
            datetime_format=datetime_format,
            time_resolution=time_resolution,
            feature_cols=feature_cols,
            drop_nulls=drop_nulls,
        )


if __name__ == "__main__":
    # Whether to split and/or preprocess the data
    split = True
    preprocess = True

    # Data directory
    data_dir = "./data"

    # Fraction of households to include in training set
    sample_fraction = 0.75

    # Dataset location
    csv_data_path = "raw/GoiEner_kWh.csv"

    # Dataset parameters
    # Adjust these parameters if using a different dataset
    # Parameters defined here are for the GoiEner dataset
    time_resolution = "hourly"
    feature_cols = list["cnae", "postal_code", "p1_kw", "tarriff"]
    id_col = "id"
    kwh_col = "kWh"
    date_col = "timestamp"
    utc = True
    datetime_format = None
    historical_start = "2021-06-01"
    historical_end = "3025-12-31"
    future_start = "3314-01-01"
    future_end = "3314-12-31"
    drop_nulls = True

    split_preprocess_data(
        split,
        preprocess,
        data_dir,
        csv_data_path,
        sample_fraction,
        time_resolution,
        feature_cols,
        id_col,
        kwh_col,
        date_col,
        utc,
        datetime_format,
        historical_start,
        historical_end,
        future_start,
        future_end,
        drop_nulls,
    )
