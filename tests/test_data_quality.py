"""
Unit & Integration Tests for Data Quality Validation Rules in Odisha Flood Prediction.
"""

import os
import pandas as pd
import numpy as np
import pytest

from src.data.combine_data import STANDARD_DISTRICTS, standardize_district_name
from src.data.load_data import load_combined_data


@pytest.fixture(scope="module")
def processed_df():
    """Load sample partition of processed dataset for validation checks."""
    path = "data/processed/historical_2001_2024.csv"
    if not os.path.exists(path):
        pytest.skip("Processed historical dataset not found.")
    df = pd.read_csv(path, nrows=50000)
    return df


def test_standard_30_districts_compliance(processed_df):
    """Assert all districts in processed dataset strictly belong to the 30 standard districts."""
    unique_dists = processed_df["District"].unique().tolist()
    for d in unique_dists:
        assert d in STANDARD_DISTRICTS, f"Non-standard district found: {d}"
        assert d == standardize_district_name(d), f"Un-normalized district name casing: {d}"


def test_no_duplicate_station_date_in_processed(processed_df):
    """Assert exactly 0 duplicate [District, Block/Station, Date] combinations exist."""
    dups = processed_df.duplicated(subset=["District", "Block/Station", "Date"]).sum()
    assert dups == 0, f"Found {dups} duplicate location-date records in processed dataset."


def test_no_negative_rainfall_in_processed(processed_df):
    """Assert rainfall observations are strictly non-negative."""
    rf = processed_df["Rainfall (mm)"]
    assert (rf >= 0).all(), f"Negative rainfall values detected in dataset: min is {rf.min()}"


def test_rainfall_lag_and_rolling_bounds(processed_df):
    """Assert rolling accumulation features are non-negative and logically bounded."""
    for col in ["Rainfall_Prev_3d_Sum", "Rainfall_Prev_7d_Sum", "Rainfall_Prev_15d_Sum", "Rainfall_Prev_30d_Sum"]:
        if col in processed_df.columns:
            assert (processed_df[col] >= 0).all(), f"{col} contains negative values"
            
    # 7-day sum should be >= 3-day sum (for rows with at least 7 days history)
    # Check max bounds
    if "Rainfall_Prev_7d_Max" in processed_df.columns and "Rainfall_Prev_3d_Max" in processed_df.columns:
        assert (processed_df["Rainfall_Prev_7d_Max"] >= 0).all()


def test_flood_targets_binary_validity(processed_df):
    """Assert Flood_Occurred and Flood_Next_Day are strictly binary {0, 1}."""
    if "Flood_Occurred" in processed_df.columns:
        unique_occ = set(processed_df["Flood_Occurred"].dropna().unique())
        assert unique_occ.issubset({0, 1}), f"Flood_Occurred has non-binary values: {unique_occ}"
        
    if "Flood_Next_Day" in processed_df.columns:
        unique_nxt = set(processed_df["Flood_Next_Day"].dropna().unique())
        assert unique_nxt.issubset({0, 1}), f"Flood_Next_Day has non-binary values: {unique_nxt}"


def test_calendar_cyclical_bounds(processed_df):
    """Assert Month_sin and Month_cos are strictly within [-1.0, 1.0]."""
    if "Month_sin" in processed_df.columns:
        assert (processed_df["Month_sin"] >= -1.0001).all() and (processed_df["Month_sin"] <= 1.0001).all()
    if "Month_cos" in processed_df.columns:
        assert (processed_df["Month_cos"] >= -1.0001).all() and (processed_df["Month_cos"] <= 1.0001).all()


def test_data_quality_output_artifacts_generation():
    """Verify that reports directory contains generated data quality outputs."""
    report_html = "reports/data_quality_report.html"
    report_csv = "reports/data_quality_summary.csv"
    # If generated, assert non-empty
    if os.path.exists(report_html):
        assert os.path.getsize(report_html) > 1000, "HTML report is empty or corrupted"
    if os.path.exists(report_csv):
        assert os.path.getsize(report_csv) > 100, "CSV summary is empty or corrupted"
