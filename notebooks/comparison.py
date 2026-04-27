from pyarrow import Tensor

from opensynth.models.faraday.losses import mmd_loss
import os
import sys
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
#%%
LCL_PROCESSED_DATA_PATH = Path("../data/processed/historical/")
LCL_RAW_DATA_PATH = Path("../data/raw/historical/")

GOIENER_PROCESSED_DATA_PATH = Path("../data/processed/goiener/historical/")
GOIENER_RAW_DATA_PATH = Path("../data/raw/goiener/historical/")


TRAINED_FARADAY_PATH = Path("./faraday/")
TRAINED_ENERGYDIFF_PATH = Path("./energydiff/")

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.data_modules.goiener_data_module import GoiEnerDataModule
import pytorch_lightning as pl

import matplotlib.pyplot as plt
#%% md
## LCL
#%%
data_path = LCL_PROCESSED_DATA_PATH / "train/lcl_data.csv"
stats_path = LCL_PROCESSED_DATA_PATH / "train/mean_std.csv"
outlier_path = LCL_PROCESSED_DATA_PATH / "train/outliers.csv"

# Original training data with no outliers
dm_lcl = LCLDataModule(data_path=data_path, stats_path=stats_path, batch_size=200, n_samples=20000)
dm_lcl.setup()

gmm_lcl_data_module = LCLDataModule(data_path=data_path, stats_path=stats_path, batch_size=25000, n_samples=100000)
gmm_lcl_data_module.setup()
#%%
# Holdout data
holdout_lcl_path = LCL_PROCESSED_DATA_PATH / "holdout/lcl_data.csv"
dm_lcl_holdout = LCLDataModule(data_path=holdout_lcl_path, stats_path=stats_path, batch_size=200, n_samples=20000)
dm_lcl_holdout.setup()

# Outlier
dm_lcl_with_outliers = LCLDataModule(data_path=data_path, stats_path=stats_path, batch_size=5000, n_samples=20000, outlier_path=outlier_path)
dm_lcl_with_outliers.setup()
#%% md
## GoiEner
#%%
data_path = GOIENER_PROCESSED_DATA_PATH / "train/goiener_data.csv"
stats_path = GOIENER_PROCESSED_DATA_PATH / "train/mean_std.csv"
outlier_path = GOIENER_PROCESSED_DATA_PATH / "train/outliers.csv"

# Original training data with no outliers
dm_goiener = GoiEnerDataModule(data_path=data_path, stats_path=stats_path, batch_size=200, n_samples=20000)
dm_goiener.setup()

gmm_goiener_data_module = GoiEnerDataModule(data_path=data_path, stats_path=stats_path, batch_size=25000, n_samples=100000)
gmm_goiener_data_module.setup()

# Outlier
dm_goiener_with_outliers = GoiEnerDataModule(data_path=data_path, stats_path=stats_path, batch_size=5000, n_samples=20000, outlier_path=outlier_path)
dm_goiener_with_outliers.setup()
#%%
# Holdout data
holdout_goiener_path = GOIENER_PROCESSED_DATA_PATH / "holdout/goiener_data.csv"
dm_goiener_holdout = GoiEnerDataModule(data_path=holdout_goiener_path, stats_path=stats_path, batch_size=200, n_samples=20000)
dm_goiener_holdout.setup()
#%%

#%% md
# 🤖 Load Pretrained models
#%%
from opensynth.models.faraday.model import FaradayModel
from opensynth.models.energydiff.diffusion import PLDiffusion1D
import numpy as np
import torch
#%% md
## LCL Faraday
#%%
faraday_lcl_models = {}
for n_components in [10, 50, 150]:
    faraday_lcl_models[n_components] = torch.load(TRAINED_FARADAY_PATH / f"faraday_model_{n_components}.pt", weights_only=False)
#%% md
## LCL EnergyDiff
#%%
model_path = TRAINED_ENERGYDIFF_PATH / "lightning_logs" / "version_3" / "checkpoints/" / "epoch=99-step=1900.ckpt"
kwh_path = TRAINED_ENERGYDIFF_PATH / "lightning_logs" / "version_3" / "samples" / "dpm_kwh_calib.pt"
df_lcl_model = PLDiffusion1D.load_from_checkpoint(model_path)
df_lcl_kwh = torch.load(kwh_path)
#%%

#%% md
## GoiEner Faraday
#%%
faraday_goiener_models = {}
for n_components in [10, 50, 150]:
    faraday_goiener_models[n_components] = torch.load(TRAINED_FARADAY_PATH / f"goiener_faraday_model_{n_components}.pt", weights_only=False)
#%% md
## GoiEner EnergyDiff
#%%
model_path = TRAINED_ENERGYDIFF_PATH / "lightning_logs" / "version_2" / "checkpoints/" / "epoch=99-step=1900.ckpt"
kwh_path = TRAINED_ENERGYDIFF_PATH / "lightning_logs" / "version_2" / "samples" / "dpm_kwh_calib.pt"
df_goiener_model = PLDiffusion1D.load_from_checkpoint(model_path)
df_goiener_kwh = torch.load(kwh_path)
df_goiener_kwh = torch.clip(df_goiener_kwh, min=0) # Clip min 0 to get read of negative values
#%%


from opensynth.evaluation.fidelity.generate_data import load_lcl_data_by_year, generate_full_synthetic_year, load_goiener_data_by_year
YEAR_LCL = 2013
YEAR_GOIENER = 2018
from opensynth.evaluation.fidelity.generate_ediff_data import generate_synthetic_samples
import polars as pls
ediff_lcl_yearly = generate_synthetic_samples(
    df_lcl_model,
    dm_lcl,
    n_profiles=1,   # ← 100 synthetic profiles
    year=YEAR_LCL,        # or whichever year you like
    month=None,       # ← ensure full 365 days
    batch_size=100,   # adjust to taste (GPU/memory)
    step=100,
)