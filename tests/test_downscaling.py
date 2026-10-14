"""
Unit tests for the rainfall downscaling module (src/downscaling).
Tests data preparation, model training, evaluation metrics, non-negativity clipping,
rolling feature regeneration, and prevention of lookahead leakage.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.downscaling.prepare_downscaling_data import create_coarse_rainfall_representation
from src.downscaling.evaluate_downscaling import (
    evaluate_downscaling_quality,
    compare_downscaling_benchmarks,
    evaluate_intensity_tiers,
    compare_flood_prediction_impact
)
from src.downscaling.apply_downscaling import generate_downscaled_flood_features
from src.downscaling.train_downscaling import train_downscaling_model


@pytest.fixture
def sample_telemetry_df():
    """Create synthetic multi-station telemetry dataframe."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    records = []
    
    for d in dates:
        for block in ["Block_A", "Block_B", "Block_C"]:
            rf = float(np.random.exponential(scale=10.0))
            records.append({
                "District": "KHORDA",
                "Block/Station": block,
                "Date": d.strftime("%Y-%m-%d"),
                "Year": d.year,
                "Month_sin": np.sin(2 * np.pi * d.month / 12),
                "Month_cos": np.cos(2 * np.pi * d.month / 12),
                "Day_of_Year": d.dayofyear,
                "Rainfall (mm)": rf,
                "Flood_Occurred": 1 if rf > 25.0 else 0,
                "Flood_Next_Day": 0
            })
            
    return pd.DataFrame(records)


def test_create_coarse_rainfall_representation(sample_telemetry_df):
    """Test computation of coarse spatial means, maxima, and dispersion."""
    df_coarse = create_coarse_rainfall_representation(sample_telemetry_df)
    
    assert len(df_coarse) == len(sample_telemetry_df)
    assert "coarse_rainfall_mean" in df_coarse.columns
    assert "coarse_rainfall_std" in df_coarse.columns
    assert "coarse_rainfall_max" in df_coarse.columns
    
    # Check that for any date, coarse_rainfall_mean is equal across all stations in the district
    date_sample = df_coarse[df_coarse["Date"] == "2020-01-01"]
    means = date_sample["coarse_rainfall_mean"].unique()
    assert len(means) == 1
    
    # Mean of coarse baseline equals mean of station observations
    assert np.isclose(date_sample["coarse_rainfall_mean"].iloc[0], date_sample["Rainfall (mm)"].mean())
    assert not df_coarse["coarse_rainfall_std"].isna().any()


def test_evaluate_downscaling_quality_perfect_agreement():
    """Test metric calculations on identical series."""
    y_true = np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    metrics = evaluate_downscaling_quality(y_true, y_true)
    
    assert metrics["mae_mm"] == 0.0
    assert metrics["rmse_mm"] == 0.0
    assert metrics["r2_score"] == 1.0
    assert metrics["bias_mm"] == 0.0
    assert metrics["pearson_correlation"] == 1.0


def test_evaluate_downscaling_quality_known_bias():
    """Test calculation with known constant offset."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 22.0, 32.0, 42.0])  # +2 mm bias
    
    metrics = evaluate_downscaling_quality(y_true, y_pred)
    assert np.isclose(metrics["mae_mm"], 2.0)
    assert np.isclose(metrics["rmse_mm"], 2.0)
    assert np.isclose(metrics["bias_mm"], 2.0)
    assert np.isclose(metrics["pearson_correlation"], 1.0)


def test_evaluate_downscaling_zero_variance():
    """Test robustness when arrays have zero variance."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    
    metrics = evaluate_downscaling_quality(y_true, y_pred)
    assert metrics["mae_mm"] == 0.0
    assert metrics["pearson_correlation"] == 0.0


def test_compare_downscaling_benchmarks():
    """Test comparative benchmark summary table."""
    y_true = np.array([0.0, 5.0, 15.0, 30.0, 75.0])
    y_coarse = np.array([2.0, 8.0, 12.0, 25.0, 60.0])
    y_down = np.array([1.0, 6.0, 14.0, 28.0, 70.0])
    
    df_comp = compare_downscaling_benchmarks(y_true, y_coarse, y_down, split_name="Test (2022-2024)")
    assert len(df_comp) == 2
    assert "Split" in df_comp.columns
    assert "Series" in df_comp.columns
    assert "MAE (mm)" in df_comp.columns
    assert "RMSE (mm)" in df_comp.columns
    assert "R2" in df_comp.columns
    assert "Bias (mm)" in df_comp.columns
    assert "Pearson_r" in df_comp.columns


def test_evaluate_intensity_tiers():
    """Test performance stratification across IMD rainfall intensity categories."""
    y_true = np.array([1.0, 10.0, 70.0, 150.0, 250.0])
    y_coarse = np.array([1.5, 12.0, 65.0, 130.0, 200.0])
    y_down = np.array([1.2, 11.0, 68.0, 140.0, 220.0])
    
    df_tiers = evaluate_intensity_tiers(y_true, y_coarse, y_down)
    assert len(df_tiers) == 5
    assert "Rainfall_Intensity_Tier" in df_tiers.columns
    assert "Coarse_MAE_mm" in df_tiers.columns
    assert "Downscaled_MAE_mm" in df_tiers.columns
    assert "Coarse_Bias_mm" in df_tiers.columns
    assert "Downscaled_Bias_mm" in df_tiers.columns


def test_generate_downscaled_flood_features_non_negativity_and_no_leakage(sample_telemetry_df):
    """Test downscaling application: non-negativity constraint and lookahead leakage prevention."""
    df_coarse = create_coarse_rainfall_representation(sample_telemetry_df)
    
    # Train dummy model
    features = ["coarse_rainfall_mean", "coarse_rainfall_std", "Month_sin", "Month_cos", "Day_of_Year"]
    model = HistGradientBoostingRegressor(max_iter=10, random_state=42)
    model.fit(df_coarse[features], df_coarse["Rainfall (mm)"])
    
    df_downscaled = generate_downscaled_flood_features(df_coarse, model, downscaling_features=features)
    
    # 1. Non-negativity
    assert (df_downscaled["Rainfall (mm)"] >= 0.0).all()
    assert (df_downscaled["Downscaled_Rainfall (mm)"] >= 0.0).all()
    
    # 2. Recomputed features exist
    expected_cols = [
        "Rainfall_Lag_1d", "Rainfall_Lag_2d", "Rainfall_Lag_3d", "Rainfall_Lag_7d",
        "Rainfall_Prev_3d_Sum", "Rainfall_Prev_7d_Sum", "Rainfall_Prev_15d_Sum", "Rainfall_Prev_30d_Sum",
        "Rainfall_Prev_3d_Max", "Rainfall_Prev_7d_Max", "Rainfall_Prev_15d_Max", "Rainfall_Prev_30d_Max",
        "Rainy_Days_Prev_3d", "Rainy_Days_Prev_7d", "Rainy_Days_Prev_15d", "Rainy_Days_Prev_30d",
        "Consecutive_Rainy_Days_Before"
    ]
    for col in expected_cols:
        assert col in df_downscaled.columns, f"Missing feature: {col}"
        
    # 3. No leakage check: Lag_1d on day 2 must equal day 1 rainfall
    st_df = df_downscaled[df_downscaled["Block/Station"] == "Block_A"].sort_values("Date").reset_index(drop=True)
    assert st_df.loc[1, "Rainfall_Lag_1d"] == st_df.loc[0, "Rainfall (mm)"]
    assert st_df.loc[0, "Rainfall_Lag_1d"] == 0.0  # Initial boundary filled with 0.0


def test_train_downscaling_model(tmp_path, sample_telemetry_df):
    """Test training function and return metrics."""
    df_coarse = create_coarse_rainfall_representation(sample_telemetry_df)
    save_path = str(tmp_path / "test_downscaling_model.pkl")
    
    model, metrics = train_downscaling_model(
        train_df=df_coarse,
        features=["coarse_rainfall_mean", "coarse_rainfall_std", "Month_sin", "Month_cos", "Day_of_Year"],
        target="Rainfall (mm)",
        save_path=save_path
    )
    
    assert isinstance(model, HistGradientBoostingRegressor)
    assert "train_mae" in metrics
    assert "train_rmse" in metrics
    assert "train_r2" in metrics
    assert "train_bias" in metrics
    assert "train_pearson_r" in metrics
