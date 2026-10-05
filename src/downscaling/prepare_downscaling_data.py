"""
Preparation of coarse-to-fine rainfall datasets for statistical downscaling.
Constructs coarse regional/district spatial rainfall representations from station networks.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def create_coarse_rainfall_representation(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct genuine coarse regional rainfall signals by computing district-level spatial means
    and regional variances (representing coarse ~1.0 degree satellite/GCM observations).
    """
    df_down = df.copy()
    
    # Compute coarse spatial mean rainfall per district per date
    coarse_district = df_down.groupby(["District", "Date"])["Rainfall (mm)"].agg(
        coarse_rainfall_mean="mean",
        coarse_rainfall_max="max",
        coarse_rainfall_std="std"
    ).reset_index()
    
    coarse_district["coarse_rainfall_std"] = coarse_district["coarse_rainfall_std"].fillna(0.0)
    
    # Merge back to station-level dataset
    df_merged = pd.merge(
        df_down,
        coarse_district,
        on=["District", "Date"],
        how="left"
    )
    
    return df_merged
