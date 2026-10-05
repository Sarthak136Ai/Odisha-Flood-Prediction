"""
Unit tests for 2025 unseen operational inference engine and distribution shift analysis.
"""

import pandas as pd
import numpy as np
from src.inference.unseen_2025_pipeline import Unseen2025InferenceEngine, analyze_rainfall_distribution_shift


def test_analyze_rainfall_distribution_shift():
    historical = pd.Series(np.random.gamma(shape=2, scale=10, size=500))
    current = pd.Series(np.random.gamma(shape=2, scale=10, size=200))
    
    res = analyze_rainfall_distribution_shift(historical, current, "Cuttack")
    assert "historical_mean_mm" in res
    assert "current_mean_mm" in res
    assert "ks_statistic" in res
    assert "p_value" in res
    assert 0.0 <= res["ks_statistic"] <= 1.0


def test_unseen_2025_inference_pipeline():
    engine = Unseen2025InferenceEngine()
    
    mock_2025 = pd.DataFrame({
        "District": ["CUTTACK", "PURI"],
        "Block/Station": ["Banki", "Kanas"],
        "Date": ["2025-08-15", "2025-08-15"],
        "Year": [2025, 2025],
        "Month_Number": [8, 8],
        "Day": [15, 15],
        "Day_of_Year": [227, 227],
        "Day_of_Week": [4, 4],
        "Month_sin": [-0.866, -0.866],
        "Month_cos": [-0.5, -0.5],
        "Rainfall (mm)": [140.0, 10.0],
        "Rainfall_Missing": [0, 0],
        "Rainfall_Lag_1d": [80.0, 5.0],
        "Rainfall_Lag_2d": [60.0, 0.0],
        "Rainfall_Lag_3d": [50.0, 0.0],
        "Rainfall_Lag_7d": [30.0, 0.0],
        "Rainfall_Prev_3d_Sum": [190.0, 5.0],
        "Rainfall_Prev_7d_Sum": [320.0, 10.0],
        "Rainfall_Prev_15d_Sum": [450.0, 20.0],
        "Rainfall_Prev_30d_Sum": [600.0, 40.0],
        "Rainfall_Prev_3d_Max": [80.0, 5.0],
        "Rainfall_Prev_7d_Max": [80.0, 5.0],
        "Rainfall_Prev_15d_Max": [80.0, 5.0],
        "Rainfall_Prev_30d_Max": [80.0, 10.0],
        "Rainy_Days_Prev_3d": [3, 1],
        "Rainy_Days_Prev_7d": [6, 2],
        "Rainy_Days_Prev_15d": [12, 3],
        "Rainy_Days_Prev_30d": [20, 5],
        "Consecutive_Rainy_Days_Before": [5, 0],
        "Flood_Occurred": [1, 0]
    })
    
    preds_df = engine.predict_2025_dataset(mock_2025)
    assert "Predicted_Flood_Probability" in preds_df.columns
    assert "Predicted_Risk_Level" in preds_df.columns
    assert "Predicted_Flood_Next_Day" in preds_df.columns
    assert preds_df.iloc[0]["Predicted_Flood_Probability"] > preds_df.iloc[1]["Predicted_Flood_Probability"]
