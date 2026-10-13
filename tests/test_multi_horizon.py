"""
Unit tests for Multi-Horizon Flood Target Generation and Evaluation.

Validates:
1. Target naming conventions (Flood_T+1, Flood_T+2, Flood_T+3).
2. Consecutive calendar day shifting logic.
3. Calendar gap detection and NaN masking (zero temporal misalignment).
4. Station and district boundary isolation (no cross-station leakage).
5. Backward compatibility and equivalence of Flood_Next_Day with Flood_T+1.
6. Target alignment mathematical verification function.
7. MultiHorizonEvaluationEngine execution and degradation metric calculations on synthetic data.
8. Performance degradation plot generation.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.multi_horizon import (
    HORIZON_NAME_MAP,
    get_horizon_target_name,
    generate_horizon_target,
    generate_multi_horizon_targets,
    verify_target_alignment
)
from src.evaluation.multi_horizon_evaluation import (
    MultiHorizonEvaluationEngine,
    plot_multi_horizon_degradation
)


def test_get_horizon_target_name():
    """Verify target column name resolution."""
    assert get_horizon_target_name(1) == "Flood_T+1"
    assert get_horizon_target_name(2) == "Flood_T+2"
    assert get_horizon_target_name(3) == "Flood_T+3"
    assert get_horizon_target_name(5) == "Flood_T+5"


def test_single_horizon_target_generation_consecutive():
    """Verify consecutive day target generation for horizons 1, 2, and 3."""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    df = pd.DataFrame({
        "District": ["CUTTACK"] * 6,
        "Block/Station": ["Cuttack_HQ"] * 6,
        "Date": dates.strftime("%Y-%m-%d"),
        "Flood_Occurred": [0, 0, 1, 0, 1, 0]
    })

    # Horizon 1 (T+1)
    df1 = generate_horizon_target(df, horizon=1, align_calendar_days=True)
    expected_t1 = [0.0, 1.0, 0.0, 1.0, 0.0, np.nan]
    np.testing.assert_array_equal(df1["Flood_T+1"].values[:5], expected_t1[:5])
    assert np.isnan(df1["Flood_T+1"].values[5])

    # Horizon 2 (T+2)
    df2 = generate_horizon_target(df, horizon=2, align_calendar_days=True)
    expected_t2 = [1.0, 0.0, 1.0, 0.0, np.nan, np.nan]
    np.testing.assert_array_equal(df2["Flood_T+2"].values[:4], expected_t2[:4])
    assert np.isnan(df2["Flood_T+2"].values[4])
    assert np.isnan(df2["Flood_T+2"].values[5])

    # Horizon 3 (T+3)
    df3 = generate_horizon_target(df, horizon=3, align_calendar_days=True)
    expected_t3 = [0.0, 1.0, 0.0, np.nan, np.nan, np.nan]
    np.testing.assert_array_equal(df3["Flood_T+3"].values[:3], expected_t3[:3])
    assert np.isnan(df3["Flood_T+3"].values[3])
    assert np.isnan(df3["Flood_T+3"].values[4])
    assert np.isnan(df3["Flood_T+3"].values[5])


def test_calendar_day_gap_masking():
    """Verify that discontinuous dates are safely masked as NaN rather than shifted across gaps."""
    # Day 1, Day 2, Day 4 (Day 3 missing!)
    dates = ["2020-01-01", "2020-01-02", "2020-01-04"]
    df = pd.DataFrame({
        "District": ["BALASORE"] * 3,
        "Block/Station": ["Basta"] * 3,
        "Date": dates,
        "Flood_Occurred": [0, 1, 1]
    })

    df_aligned = generate_horizon_target(df, horizon=1, align_calendar_days=True)
    # Day 1 -> Day 2 is 1 day ahead (valid)
    assert df_aligned["Flood_T+1"].iloc[0] == 1.0
    # Day 2 -> Day 4 is 2 days ahead, NOT 1 day ahead! Must be masked to NaN
    assert np.isnan(df_aligned["Flood_T+1"].iloc[1])
    # Day 3 (last record) must be NaN
    assert np.isnan(df_aligned["Flood_T+1"].iloc[2])

    # For horizon 2 (T+2):
    df_aligned2 = generate_horizon_target(df, horizon=2, align_calendar_days=True)
    # Day 1 -> Day 4 is 3 days ahead, NOT 2 days ahead! Must be masked to NaN
    assert np.isnan(df_aligned2["Flood_T+2"].iloc[0])


def test_cross_station_isolation():
    """Verify that target shifts never cross station or district boundaries."""
    df = pd.DataFrame({
        "District": ["CUTTACK", "CUTTACK", "PURI", "PURI"],
        "Block/Station": ["Station_A", "Station_A", "Station_B", "Station_B"],
        "Date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
        "Flood_Occurred": [0, 1, 1, 0]
    })

    df_res = generate_multi_horizon_targets(df, horizons=[1, 2])
    # Station_A last row must be NaN for both T+1 and T+2
    assert df_res.loc[df_res["Block/Station"] == "Station_A", "Flood_T+1"].iloc[1] is np.nan or np.isnan(df_res.loc[df_res["Block/Station"] == "Station_A", "Flood_T+1"].iloc[1])
    assert np.isnan(df_res.loc[df_res["Block/Station"] == "Station_A", "Flood_T+2"].iloc[0])

    # Station_B first row for T+1 must be 0 (its own next day), never leaking Station_A
    assert df_res.loc[df_res["Block/Station"] == "Station_B", "Flood_T+1"].iloc[0] == 0.0


def test_backward_compatibility_flood_next_day():
    """Verify Flood_Next_Day is preserved and identical to Flood_T+1."""
    dates = pd.date_range("2020-05-01", periods=5, freq="D")
    df = pd.DataFrame({
        "District": ["GANJAM"] * 5,
        "Block/Station": ["Chhatrapur"] * 5,
        "Date": dates.strftime("%Y-%m-%d"),
        "Flood_Occurred": [0, 1, 0, 0, 1]
    })

    df_multi = generate_multi_horizon_targets(df, horizons=[1, 2, 3], preserve_next_day=True)
    assert "Flood_Next_Day" in df_multi.columns
    assert "Flood_T+1" in df_multi.columns

    # Check numerical equivalence for all valid records
    valid_mask = df_multi["Flood_T+1"].notna()
    np.testing.assert_array_equal(
        df_multi.loc[valid_mask, "Flood_Next_Day"].values,
        df_multi.loc[valid_mask, "Flood_T+1"].values
    )


def test_verify_target_alignment():
    """Verify mathematical audit tool detects alignment and corruption."""
    dates = pd.date_range("2020-07-01", periods=10, freq="D")
    df = pd.DataFrame({
        "District": ["BHADRAK"] * 10,
        "Block/Station": ["Bhadrak_Town"] * 10,
        "Date": dates.strftime("%Y-%m-%d"),
        "Flood_Occurred": [0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    })

    df_targets = generate_multi_horizon_targets(df, horizons=[1, 2])
    audit1 = verify_target_alignment(df_targets, horizon=1)
    assert audit1["is_perfectly_aligned"] is True
    assert audit1["value_mismatches"] == 0
    assert audit1["valid_target_rows"] == 9

    audit2 = verify_target_alignment(df_targets, horizon=2)
    assert audit2["is_perfectly_aligned"] is True
    assert audit2["valid_target_rows"] == 8

    # Intentional corruption check
    df_corrupted = df_targets.copy()
    df_corrupted.loc[0, "Flood_T+1"] = 1 - df_corrupted.loc[0, "Flood_T+1"]
    corrupt_audit = verify_target_alignment(df_corrupted, horizon=1)
    assert corrupt_audit["is_perfectly_aligned"] is False
    assert corrupt_audit["value_mismatches"] > 0


def test_multi_horizon_evaluation_engine_synthetic(tmp_path):
    """End-to-end verification of MultiHorizonEvaluationEngine and degradation plotting."""
    np.random.seed(42)
    n_days = 60
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    def make_partition(n):
        rf = np.random.exponential(scale=10.0, size=n)
        flood = (rf > 18.0).astype(int)
        df_p = pd.DataFrame({
            "District": ["CUTTACK"] * n,
            "Block/Station": ["HQ"] * n,
            "Date": dates[:n].strftime("%Y-%m-%d"),
            "Rainfall (mm)": rf,
            "Month_sin": np.sin(np.linspace(0, 3, n)),
            "Flood_Occurred": flood
        })
        return df_p

    train_df = make_partition(40)
    val_df = make_partition(30)
    test_df = make_partition(30)

    engine = MultiHorizonEvaluationEngine(horizons=[1, 2], optimization_metric="f1")
    model_factories = {
        "Logistic Regression": lambda spw: LogisticRegression(max_iter=100)
    }

    results = engine.run_multi_horizon_evaluation(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=["Rainfall (mm)", "Month_sin"],
        model_factories=model_factories
    )

    assert len(results) == 2
    assert "Horizon_Label" in results.columns
    assert "Test_ROC_AUC" in results.columns
    assert "Test_PR_AUC" in results.columns
    assert "Test_F1" in results.columns
    assert "Test_F2" in results.columns
    assert "Delta_Test_PR_AUC" in results.columns
    assert "Delta_Test_ROC_AUC" in results.columns

    # Test degradation plotting
    png_path = str(tmp_path / "degradation.png")
    svg_path = str(tmp_path / "degradation.svg")
    fig = plot_multi_horizon_degradation(results, save_path_png=png_path, save_path_svg=svg_path)
    assert os.path.exists(png_path)
    assert os.path.exists(svg_path)
    assert os.path.getsize(png_path) > 1000
