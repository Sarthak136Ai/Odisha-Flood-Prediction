"""
Unit tests for standardized data architecture, raw/processed loading, and schema validation.
"""

import os
import pytest
import pandas as pd
from src.data.load_data import (
    load_combined_data,
    load_operational_2025_data,
    load_2025_predictions,
    load_config
)
from src.data.validate_data import EXPECTED_COLUMNS
from src.data.combine_data import STANDARD_DISTRICTS


def test_standardized_directory_structure():
    assert os.path.isdir("data/raw/historical"), "Missing data/raw/historical/"
    assert os.path.isdir("data/raw/operational"), "Missing data/raw/operational/"
    assert os.path.isdir("data/processed"), "Missing data/processed/"
    assert os.path.isdir("data/predictions"), "Missing data/predictions/"
    
    # Check all 24 historical raw files
    for yr in range(2001, 2025):
        raw_file = f"data/raw/historical/{yr}.csv"
        assert os.path.exists(raw_file), f"Raw historical file {raw_file} missing"
        
    # Check operational 2025 raw file
    assert os.path.exists("data/raw/operational/2025.csv"), "Missing data/raw/operational/2025.csv"
    
    # Check processed files
    assert os.path.exists("data/processed/historical_2001_2024.csv"), "Missing data/processed/historical_2001_2024.csv"
    assert os.path.exists("data/processed/operational_2025.csv"), "Missing data/processed/operational_2025.csv"
    
    # Check predictions
    assert os.path.exists("data/predictions/2025_flood_risk_predictions.csv"), "Missing data/predictions/2025_flood_risk_predictions.csv"


def test_combined_data_schema():
    df = load_combined_data(parse_dates=False)
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Expected column {col} missing in combined dataset"


def test_no_location_date_duplicates():
    df = load_combined_data(parse_dates=False)
    dups = df.duplicated(subset=["District", "Block/Station", "Date"]).sum()
    assert dups == 0, f"Found {dups} duplicate location-date records in combined dataset"


def test_standard_districts():
    df = load_combined_data(parse_dates=False)
    unique_districts = df["District"].unique()
    assert len(unique_districts) == 30, f"Expected 30 districts, found {len(unique_districts)}"
    for d in unique_districts:
        assert d in STANDARD_DISTRICTS, f"Non-standard district found: {d}"


def test_rainfall_bounds():
    df = load_combined_data(parse_dates=False)
    assert (df["Rainfall (mm)"] < 0).sum() == 0, "Negative rainfall values found"
    assert df["Rainfall (mm)"].max() <= 1000.0, f"Unrealistic rainfall value: {df['Rainfall (mm)'].max()}"


def test_operational_2025_dataset_integrity():
    df_25 = load_operational_2025_data(parse_dates=False)
    assert len(df_25) == 114610, f"Expected 114,610 rows in 2025 operational dataset, got {len(df_25)}"
    assert "Rainfall_Prev_7d_Sum" in df_25.columns
    assert "Rainfall_Lag_1d" in df_25.columns
    
    df_preds = load_2025_predictions(parse_dates=False)
    assert len(df_preds) == 114610, f"Expected 114,610 predictions, got {len(df_preds)}"
    assert "Predicted_Flood_Probability" in df_preds.columns
    assert "Predicted_Risk_Level" in df_preds.columns
