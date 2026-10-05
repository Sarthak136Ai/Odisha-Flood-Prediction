"""
Unit tests for feature integrity and data leakage prevention.
Verifies that features only use past/present information and target is strictly segregated.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.load_data import load_combined_data, load_config


def test_target_not_in_features():
    """Verify Flood_Next_Day is never in the feature list."""
    config = load_config("config.yaml")
    rainfall_cols = config["features"]["rainfall_cols"]
    temporal_cols = config["features"]["temporal_cols"]
    all_features = rainfall_cols + temporal_cols + ["Flood_Occurred"]
    
    assert "Flood_Next_Day" not in all_features, "Target Flood_Next_Day is present in feature list (CRITICAL LEAKAGE)!"


def test_temporal_split_disjointness():
    """Verify train, validation, and test years have zero overlap."""
    config = load_config("config.yaml")
    train_years = set(config["split"]["train_years"])
    val_years = set(config["split"]["val_years"])
    test_years = set(config["split"]["test_years"])
    
    assert train_years.isdisjoint(val_years), "Train and Validation years overlap!"
    assert val_years.isdisjoint(test_years), "Validation and Test years overlap!"
    assert train_years.isdisjoint(test_years), "Train and Test years overlap!"
    assert max(train_years) < min(val_years), "Validation period does not strictly follow Train period chronologically!"
    assert max(val_years) < min(test_years), "Test period does not strictly follow Validation period chronologically!"


def test_lag_features_math():
    """Verify lag features mathematically correspond to shifted previous observations."""
    df = load_combined_data(parse_dates=False)
    # Sample 1 station for speed
    sample_station = df[(df["District"] == "CUTTACK") & (df["Block/Station"] == "Cuttack")].copy()
    sample_station["Date_Parsed"] = pd.to_datetime(sample_station["Date"])
    sample_station = sample_station.sort_values("Date_Parsed").reset_index(drop=True)
    
    # Lag 1d check (skipping first row)
    expected_lag1 = sample_station["Rainfall (mm)"].shift(1).iloc[1:50]
    actual_lag1 = sample_station["Rainfall_Lag_1d"].iloc[1:50]
    np.testing.assert_allclose(actual_lag1, expected_lag1, rtol=1e-5, err_msg="Rainfall_Lag_1d does not match shifted rainfall!")
    
    # Next day target check
    expected_target = sample_station["Flood_Occurred"].shift(-1).iloc[0:50]
    actual_target = sample_station["Flood_Next_Day"].iloc[0:50]
    np.testing.assert_allclose(actual_target, expected_target, rtol=1e-5, err_msg="Flood_Next_Day does not match shifted Flood_Occurred!")
