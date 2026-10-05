"""
Data loading utilities for Odisha flood prediction system.
Handles loading individual yearly CSV files, combined historical datasets, operational 2025 data, and predictions.
"""

import os
import glob
import pandas as pd
import yaml
from typing import Dict, List, Optional, Union


def load_config(config_path: str = "config.yaml") -> dict:
    """Load system configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_source_file_path(year: int, source_dir: str = "data/raw/historical") -> str:
    """Get the expected path for a given year's dataset."""
    # Check standardized path data/raw/historical/<year>.csv first
    p1 = os.path.join(source_dir, f"{year}.csv")
    if os.path.exists(p1):
        return p1
    # Fallback to Rain_new_updated
    p2 = os.path.join("Rain_new_updated", f"Odisha_Flood_{year}_Corrected_Preprocessed.csv")
    if os.path.exists(p2):
        return p2
    return p1


def get_all_source_files(source_dir: str = "data/raw/historical") -> Dict[int, str]:
    """Discover all 2001-2024 source CSV files."""
    files_map = {}
    for year in range(2001, 2025):
        path = get_source_file_path(year, source_dir)
        if os.path.exists(path):
            files_map[year] = path
    return files_map


def load_year_data(
    year: int, 
    source_dir: str = "data/raw/historical",
    parse_dates: bool = True
) -> pd.DataFrame:
    """Load a specific yearly CSV file."""
    path = get_source_file_path(year, source_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file for year {year} not found at {path}")
    
    date_cols = ["Date"] if parse_dates else False
    df = pd.read_csv(path, parse_dates=date_cols)
    return df


def load_combined_data(
    combined_path: str = "data/processed/historical_2001_2024.csv",
    parse_dates: bool = True
) -> pd.DataFrame:
    """Load the full combined 2001-2024 processed dataset."""
    if not os.path.exists(combined_path):
        # Fallback to legacy path if needed
        fallback = "data/combined/Odisha_Flood_2001_2024.csv"
        if os.path.exists(fallback):
            combined_path = fallback
        else:
            raise FileNotFoundError(
                f"Combined dataset not found at {combined_path}. "
                "Please run scripts/organize_data_structure.py first."
            )
    date_cols = ["Date"] if parse_dates else False
    df = pd.read_csv(combined_path, parse_dates=date_cols)
    return df


def load_operational_2025_data(
    ops_path: str = "data/processed/operational_2025.csv",
    parse_dates: bool = True
) -> pd.DataFrame:
    """Load the processed operational 2025 dataset with all engineered features."""
    if not os.path.exists(ops_path):
        raise FileNotFoundError(f"Operational 2025 dataset not found at {ops_path}")
    date_cols = ["Date"] if parse_dates else False
    return pd.read_csv(ops_path, parse_dates=date_cols)


def load_2025_predictions(
    pred_path: str = "data/predictions/2025_flood_risk_predictions.csv",
    parse_dates: bool = True
) -> pd.DataFrame:
    """Load the generated 2025 flood risk predictions dataset."""
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"2025 predictions not found at {pred_path}")
    date_cols = ["Date"] if parse_dates else False
    return pd.read_csv(pred_path, parse_dates=date_cols)
