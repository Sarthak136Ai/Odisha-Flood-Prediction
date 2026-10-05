"""
Unit tests for trained model loading, inference, and early warning risk calculation.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.models.predict import FloodPredictor


def test_model_artifact_exists():
    assert os.path.exists("models/flood_prediction/best_model.pkl"), "Best model artifact missing"
    assert os.path.exists("models/flood_prediction/model_metadata.json"), "Model metadata missing"


def test_flood_predictor_inference():
    predictor = FloodPredictor()
    
    # Mock sample features for a heavy rainfall day
    sample_heavy_rain = {
        "Year": 2024,
        "Month_Number": 8,
        "Day": 15,
        "Day_of_Year": 228,
        "Day_of_Week": 3,
        "Month_sin": -0.866,
        "Month_cos": -0.5,
        "Rainfall (mm)": 180.0,
        "Rainfall_Missing": 0,
        "Rainfall_Lag_1d": 120.0,
        "Rainfall_Lag_2d": 95.0,
        "Rainfall_Lag_3d": 60.0,
        "Rainfall_Lag_7d": 40.0,
        "Rainfall_Prev_3d_Sum": 275.0,
        "Rainfall_Prev_7d_Sum": 410.0,
        "Rainfall_Prev_15d_Sum": 650.0,
        "Rainfall_Prev_30d_Sum": 920.0,
        "Rainfall_Prev_3d_Max": 120.0,
        "Rainfall_Prev_7d_Max": 120.0,
        "Rainfall_Prev_15d_Max": 120.0,
        "Rainfall_Prev_30d_Max": 120.0,
        "Rainy_Days_Prev_3d": 3.0,
        "Rainy_Days_Prev_7d": 6.0,
        "Rainy_Days_Prev_15d": 12.0,
        "Rainy_Days_Prev_30d": 22.0,
        "Consecutive_Rainy_Days_Before": 7.0,
        "Flood_Occurred": 1
    }
    
    res_heavy = predictor.predict_single(sample_heavy_rain)
    assert "flood_probability" in res_heavy
    assert "predicted_flood_next_day" in res_heavy
    assert "risk_level" in res_heavy
    assert res_heavy["flood_probability"] > 0.5, "Expected high probability for extreme multi-day rainfall"
    
    # Mock sample features for a dry day
    sample_dry = {
        "Year": 2024,
        "Month_Number": 2,
        "Day": 10,
        "Day_of_Year": 41,
        "Day_of_Week": 5,
        "Month_sin": 0.866,
        "Month_cos": 0.5,
        "Rainfall (mm)": 0.0,
        "Rainfall_Missing": 0,
        "Rainfall_Lag_1d": 0.0,
        "Rainfall_Lag_2d": 0.0,
        "Rainfall_Lag_3d": 0.0,
        "Rainfall_Lag_7d": 0.0,
        "Rainfall_Prev_3d_Sum": 0.0,
        "Rainfall_Prev_7d_Sum": 0.0,
        "Rainfall_Prev_15d_Sum": 0.0,
        "Rainfall_Prev_30d_Sum": 5.0,
        "Rainfall_Prev_3d_Max": 0.0,
        "Rainfall_Prev_7d_Max": 0.0,
        "Rainfall_Prev_15d_Max": 0.0,
        "Rainfall_Prev_30d_Max": 5.0,
        "Rainy_Days_Prev_3d": 0.0,
        "Rainy_Days_Prev_7d": 0.0,
        "Rainy_Days_Prev_15d": 0.0,
        "Rainy_Days_Prev_30d": 1.0,
        "Consecutive_Rainy_Days_Before": 0.0,
        "Flood_Occurred": 0
    }
    
    res_dry = predictor.predict_single(sample_dry)
    assert res_dry["flood_probability"] < 0.3, "Expected low probability for dry weather"
    assert res_dry["risk_level"] == "Low"
