README — GoiEner dataset (for OpenSynth)

Overview
--------
This README describes how OpenSynth expects, splits and preprocesses the GoiEner dataset.
It summarizes the dataset conventions used in the repository, the preprocessing pipeline, the CLI exposed in `app/app.py`, and how to load processed data for modelling and evaluation.

Key assumptions / expected raw CSV schema
---------------------------------------
- File location (default): `./data/raw/Goiener_kWh.csv` (the repo also contains per-split CSVs under `data/raw/goiener/` after splitting).
- Required columns (case-sensitive unless you change the CLI args):
  - `id` — household identifier
  - `timestamp` — datetime string (default parsed with pandas, timezone-aware parsing when `--utc` is set)
  - `kWh` — energy reading (numeric or string convertible to float)
- Default timezone parsing: UTC (CLI `--utc` default: True)
- Default time resolution: hourly (CLI default: `hourly`). The pipeline supports `hourly` and `half_hourly`.

High-level pipeline (what preprocess does)
------------------------------------------
When you run the split/preprocess routines the code will:
1. Split raw dataset into train / holdout households and historical / future time windows. This produces files under `data/raw/goiener/historical` and `data/raw/goiener/future`:
   - `train.csv`
   - `holdout.csv`
   If `feature_cols` is provided, a `goiener_feature_encoders.json` file will be written under the `data` folder containing categorical encodings.
2. Preprocess CSVs into daily load profiles for Faraday model training and evaluation. For each processed split the pipeline will:
   - Parse and normalise datetimes
   - Extract date features (month, day, dayofweek, hour, minute, month_end)
   - Compute a settlement period (simple hour->period mapping used internally)
   - Drop duplicate readings; drop or fill null kWh values (controlled by `--drop_nulls`)
   - Filter days that do not have a complete set of readings (24 for hourly, 48 for half-hourly)
   - Pack time-series into one row per daily profile (kWh measurements collected into arrays/lists)
   - Compute dataset mean and std and write `mean_std.csv`
   - Generate synthetic outliers (Gaussian + Gamma noise) and write an outliers CSV

Processed outputs (paths)
-------------------------
- Raw splits written by the split step (if used):
  - `data/raw/goiener/historical/train.csv`
  - `data/raw/goiener/historical/holdout.csv`
  - `data/raw/goiener/future/train.csv`
  - `data/raw/goiener/future/holdout.csv`
- Preprocessed outputs (one folder per split):
  - `data/processed/goiener/historical/train/data.csv` — the packed daily profiles
  - `data/processed/goiener/historical/train/outliers.csv` — generated outliers
  - `data/processed/goiener/historical/train/mean_std.csv` — mean & std used for normalization
  - Equivalent files are produced for `historical/holdout` (and the `future` folders if you run the pipeline on them).
- Encoders (if features are used):
  - `data/goiener_feature_encoders.json` — categorical encoders (written by the split step)

CLI (app/app.py) — quick reference
---------------------------------
The repository exposes CLI commands in `app/app.py`. Two relevant commands are:

- `preprocess_goiener_data` — split and/or preprocess the GoiEner dataset

Important flags (these show defaults used in the CLI):
- `--loc` (data_dir): default `./data`
- `--csv_path`: default `raw/Goiener_kWh.csv` (path relative to `--loc`)
- `--split`: whether to split households into train / holdout (default: False)
- `--preprocess`: whether to preprocess CSVs into daily profiles (default: False)
- `--sample_fraction`: fraction of households to include in training (default: 0.75)
- `--time_resolution`: `hourly` or `half_hourly` (default: `hourly`)
- `--feature_cols`: list of feature column names (default: `["postal_code"]` in the CLI)
- `--id_col`: household id column (default: `id`)
- `--kwh_col`: kWh column (default: `kWh`)
- `--datetime_col`: datetime column (default: `timestamp`)
- `--utc`: parse datetimes as UTC (default: True)
- `--datetime_format`: pass a format string for rigid parsing (default: None)
- `--historical_start`, `--historical_end`, `--future_start`, `--future_end`: date windows used by the split step (defaults are in the CLI; change them to suit your evaluation plan)
- `--drop_nulls`: drop rows with NaN kWh (default: True). If False, NaNs are replaced with 0.0.

Example: run both split and preprocess from the repository root

```bash
python app/app.py preprocess_goiener_data \
  --split --preprocess \
  --loc ./data \
  --csv_path raw/Goiener_kWh.csv \
  --time_resolution hourly \
  --feature_cols postal_code \
  --id_col id --kwh_col kWh --datetime_col timestamp --utc True
```

Programmatic usage (Python)
---------------------------
You can call the helper functions from Python if you prefer to run the steps from a script or notebook. Example parameter values to pass to `split_preprocess_data` (shown as a plain list):

- split: True
- preprocess: True
- data_dir: ./data
- csv_data_path: raw/Goiener_kWh.csv
- sample_fraction: 0.75
- time_resolution: hourly
- feature_cols: ["postal_code"]
- id_col: id
- kwh_col: kWh
- datetime_col: timestamp
- utc: True
- datetime_format: None
- historical_start: 2021-06-01
- historical_end: 2023-12-31
- future_start: 2024-01-01
- future_end: 2024-12-31
- drop_nulls: True

(You can pass these into `opensynth.datasets.goiener.get_data.split_preprocess_data(...)` from Python or use the CLI in `app/app.py`.)

Implementation notes / details
------------------------------
- Datetime parsing: the preprocessing code renames user-provided datetime/kWh/id columns to `DateTime`, `kwh`, `ID` internally and then uses pandas `to_datetime` with `utc=<value>` and optional `format` if passed.
- Missing / duplicated readings: pipeline drops duplicates per (ID, date, settlement_period), then either drops NaNs or fills with 0 depending on `--drop_nulls`.
- Filtering incomplete days: for `hourly` the pipeline requires exactly 24 readings per day; for `half_hourly` requires 48. Days with insufficient readings are dropped.
- Outliers: the pipeline synthesises outlier daily profiles using a Gaussian and Gamma noise generator (NoiseFactory) and writes them to `outliers.csv` for use in robustness/fidelity tests.
- Feature encoding: if `feature_cols` are provided, categorical columns are encoded in `split_households.fit_categorical_encoders` and saved to `data/goiener_feature_encoders.json` during the split step. The encoded features are used by later steps (packaging into arrays).

Related notebook
----------------
- `notebooks/faraday/faraday_goiener_tutorial.ipynb` — step-by-step tutorial demonstrating preprocessing, loading GoiEner processed data and training/evaluating the Faraday model on the GoiEner split. Open this notebook in Jupyter or JupyterLab to follow the worked example.
