"""
Dataset combination module for Odisha flood prediction system.
Harmonizes district names, filters spurious duplicated summary records,
maintains cross-year temporal continuity, and exports the unified 2001-2024 dataset.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard 30 districts of Odisha
STANDARD_DISTRICTS = [
    "ANGUL", "BALANGIR", "BALASORE", "BARGARH", "BHADRAK", "BOUDH", "CUTTACK",
    "DEOGARH", "DHENKANAL", "GAJAPATI", "GANJAM", "JAGATSINGHPUR", "JAJPUR",
    "JHARSUGUDA", "KALAHANDI", "KANDHAMAL", "KENDRAPARA", "KEONJHAR", "KHORDHA",
    "KORAPUT", "MALKANGIRI", "MAYURBHANJ", "NAWARANGPUR", "NAYAGARH", "NUAPADA",
    "PURI", "RAYAGADA", "SAMBALPUR", "SUBARNAPUR", "SUNDARGARH"
]

# District spelling normalization mapping
DISTRICT_SYNONYMS = {
    "BARAGARH": "BARGARH",
    "BOLANGIR": "BALANGIR",
    "NAWARANGHPUR": "NAWARANGPUR",
    "MALKANAGIRI": "MALKANGIRI",
}

# Authentic blocks of Sundargarh district
SUNDARGARH_AUTHENTIC_BLOCKS = {
    "Balisankara", "Bargaon", "Bisra", "Bonai", "Gurundia", "Hemgir",
    "Koira", "Kuarmunda", "Kutra", "Lahunipara", "Lathikata",
    "Lephripara", "Nuagaon", "Rajgangpur", "Subdega", "Sundargarh",
    "Tangarpali"
}


def standardize_district_name(name: str) -> str:
    """Normalize district name using standard casing and synonyms."""
    if not isinstance(name, str):
        return str(name)
    cleaned = name.strip().upper()
    return DISTRICT_SYNONYMS.get(cleaned, cleaned)


def clean_single_year_df(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Clean and harmonize a single yearly DataFrame:
    1. Standardize district names.
    2. Filter spurious district-summary records appended under SUNDARGARH in 2019-2024.
    3. Deduplicate exact and location-date duplicates if any remain.
    4. Parse and format dates.
    """
    df = df.copy()
    
    # 1. District standardization
    df["District"] = df["District"].apply(standardize_district_name)
    df["Block/Station"] = df["Block/Station"].astype(str).str.strip()
    
    # 2. Filter spurious entries under Sundargarh for 2019-2024
    if year >= 2019:
        # Non-Sundargarh rows
        non_sundargarh = df[df["District"] != "SUNDARGARH"]
        
        # Genuine Sundargarh rows (only authentic blocks)
        sundargarh = df[df["District"] == "SUNDARGARH"]
        sundargarh_authentic = sundargarh[sundargarh["Block/Station"].isin(SUNDARGARH_AUTHENTIC_BLOCKS)]
        
        # For the 'Sundargarh' block itself, drop duplicates per date
        sundargarh_authentic = sundargarh_authentic.drop_duplicates(
            subset=["District", "Block/Station", "Date"], keep="first"
        )
        
        df = pd.concat([non_sundargarh, sundargarh_authentic], ignore_index=True)
    
    # 3. Drop exact duplicates if any
    df = df.drop_duplicates(subset=["District", "Block/Station", "Date"], keep="first")
    
    # 4. Parse Date
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    
    return df


def recompute_cross_year_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute cross-year boundary lag features and Flood_Next_Day across continuous time series.
    Ensures that boundary nulls created by yearly partitioning are seamlessly filled.
    """
    logger.info("Recomputing cross-year lags and targets across continuous 24-year time series...")
    
    df = df.copy()
    df["Date_Parsed"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["District", "Block/Station", "Date_Parsed"]).reset_index(drop=True)
    
    grouped = df.groupby(["District", "Block/Station"])
    
    # Fill Flood_Next_Day: shift(-1) of Flood_Occurred within each station
    df["Flood_Next_Day"] = grouped["Flood_Occurred"].shift(-1)
    
    # Lag 1d
    df["Rainfall_Lag_1d"] = grouped["Rainfall (mm)"].shift(1)
    # Lag 2d
    df["Rainfall_Lag_2d"] = grouped["Rainfall (mm)"].shift(2)
    # Lag 3d
    df["Rainfall_Lag_3d"] = grouped["Rainfall (mm)"].shift(3)
    # Lag 7d
    df["Rainfall_Lag_7d"] = grouped["Rainfall (mm)"].shift(7)
    
    # Previous rolling sums (excluding current day, matching lag definition)
    # Rolling over shifted rainfall
    shifted_rf = grouped["Rainfall (mm)"].shift(1)
    
    # Note: The original features are:
    # Rainfall_Prev_3d_Sum = sum of previous 3 days (lag 1, lag 2, lag 3)
    # Let's compute rolling metrics per group cleanly
    df["Rainfall_Prev_3d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
    )
    df["Rainfall_Prev_7d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
    )
    df["Rainfall_Prev_15d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=15, min_periods=1).sum()
    )
    df["Rainfall_Prev_30d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=1).sum()
    )
    
    df["Rainfall_Prev_3d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).max()
    )
    df["Rainfall_Prev_7d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).max()
    )
    df["Rainfall_Prev_15d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=15, min_periods=1).max()
    )
    df["Rainfall_Prev_30d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=1).max()
    )
    
    # Rainy days (Rainfall >= 2.5 mm is standard IMD threshold for rainy day)
    rainy_flag = (df["Rainfall (mm)"] >= 2.5).astype(float)
    df["_rainy_flag"] = rainy_flag
    
    df["Rainy_Days_Prev_3d"] = df.groupby(["District", "Block/Station"])["_rainy_flag"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
    )
    df["Rainy_Days_Prev_7d"] = df.groupby(["District", "Block/Station"])["_rainy_flag"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
    )
    df["Rainy_Days_Prev_15d"] = df.groupby(["District", "Block/Station"])["_rainy_flag"].transform(
        lambda s: s.shift(1).rolling(window=15, min_periods=1).sum()
    )
    df["Rainy_Days_Prev_30d"] = df.groupby(["District", "Block/Station"])["_rainy_flag"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=1).sum()
    )
    
    df.drop(columns=["_rainy_flag", "Date_Parsed"], inplace=True)
    
    # Fill remaining initial boundary nulls (very first few days of 2001) with 0.0
    lag_cols = [
        "Rainfall_Lag_1d", "Rainfall_Lag_2d", "Rainfall_Lag_3d", "Rainfall_Lag_7d",
        "Rainfall_Prev_3d_Sum", "Rainfall_Prev_7d_Sum", "Rainfall_Prev_15d_Sum", "Rainfall_Prev_30d_Sum",
        "Rainfall_Prev_3d_Max", "Rainfall_Prev_7d_Max", "Rainfall_Prev_15d_Max", "Rainfall_Prev_30d_Max",
        "Rainy_Days_Prev_3d", "Rainy_Days_Prev_7d", "Rainy_Days_Prev_15d", "Rainy_Days_Prev_30d",
        "Consecutive_Rainy_Days_Before"
    ]
    for c in lag_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
            
    return df


def combine_all_years(
    source_dir: str = "Rain_new_updated",
    output_path: str = "data/combined/Odisha_Flood_2001_2024.csv",
    recompute_continuity: bool = True
) -> pd.DataFrame:
    """
    Combine all 24 yearly CSV files into a unified dataset.
    """
    logger.info(f"Starting combination of 24-year datasets from {source_dir}...")
    yearly_dfs = []
    
    for year in range(2001, 2025):
        file_path = os.path.join(source_dir, f"Odisha_Flood_{year}_Corrected_Preprocessed.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required file: {file_path}")
        
        logger.info(f"Loading and standardizing year {year} ({file_path})...")
        df_year = pd.read_csv(file_path)
        cleaned_df = clean_single_year_df(df_year, year)
        yearly_dfs.append(cleaned_df)
    
    logger.info("Concatenating all 24 cleaned yearly DataFrames...")
    combined_df = pd.concat(yearly_dfs, ignore_index=True)
    
    # Sort chronologically by District, Block, Date
    combined_df = combined_df.sort_values(by=["District", "Block/Station", "Date"]).reset_index(drop=True)
    
    if recompute_continuity:
        combined_df = recompute_cross_year_features(combined_df)
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Writing combined dataset to {output_path} ({len(combined_df):,} rows)...")
    combined_df.to_csv(output_path, index=False)
    logger.info("Combined dataset successfully created and saved.")
    
    return combined_df
