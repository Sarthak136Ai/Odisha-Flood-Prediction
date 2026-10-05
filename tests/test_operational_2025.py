"""
Unit tests for 2025 Unseen Operational Pipeline, Validation, Warm-up Handling, and Retrospective Evaluation.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.inference.unseen_2025_pipeline import (
    validate_2025_raw_dataset,
    engineer_2025_features,
    Operational2025Pipeline,
    Retrospective2025EvaluationEngine
)


def test_validate_2025_valid_data():
    sample_df = pd.DataFrame({
        "District": ["CUTTACK", "PURI"],
        "Block/Station": ["Cuttack", "Puri"],
        "Date": ["2025-08-01", "2025-08-01"],
        "Rainfall (mm)": [45.0, 12.5]
    })
    report = validate_2025_raw_dataset(sample_df)
    assert report["is_valid"] is True
    assert "✓ Valid operational dataset" in report["validation_status"]
    assert report["row_count"] == 2
    assert report["districts_count"] == 2


def test_validate_2025_missing_columns():
    invalid_df = pd.DataFrame({
        "District": ["CUTTACK"],
        "Rainfall (mm)": [45.0]
    })
    report = validate_2025_raw_dataset(invalid_df)
    assert report["is_valid"] is False
    assert "Missing required columns" in report["validation_status"]


def test_validate_2025_negative_rainfall():
    invalid_df = pd.DataFrame({
        "District": ["CUTTACK"],
        "Block/Station": ["Cuttack"],
        "Date": ["2025-08-01"],
        "Rainfall (mm)": [-5.0]
    })
    report = validate_2025_raw_dataset(invalid_df)
    assert report["is_valid"] is False
    assert report["negative_rainfall_count"] == 1


def test_engineer_2025_features_with_warmup():
    warmup_df = pd.DataFrame({
        "District": ["CUTTACK"] * 30,
        "Block/Station": ["Cuttack"] * 30,
        "Date": pd.date_range("2024-12-02", periods=30).strftime("%Y-%m-%d"),
        "Rainfall (mm)": [10.0] * 30
    })
    
    ops_df = pd.DataFrame({
        "District": ["CUTTACK"] * 5,
        "Block/Station": ["Cuttack"] * 5,
        "Date": pd.date_range("2025-01-01", periods=5).strftime("%Y-%m-%d"),
        "Rainfall (mm)": [25.0] * 5
    })
    
    df_feat, info = engineer_2025_features(ops_df, warmup_df=warmup_df)
    assert len(df_feat) == 5
    assert info["warmup_applied"] is True
    assert "Rainfall_Prev_30d_Sum" in df_feat.columns
    # With 30 preceding days of 10mm each, Rainfall_Prev_30d_Sum on 2025-01-01 should be 300mm
    assert df_feat["Rainfall_Prev_30d_Sum"].iloc[0] == 300.0
    assert df_feat["Flood_Occurred"].iloc[0] == 0


def test_operational_pipeline_predictions_output():
    pipeline = Operational2025Pipeline()
    sample_df = pd.DataFrame({
        "District": ["CUTTACK", "CUTTACK"],
        "Block/Station": ["Cuttack", "Cuttack"],
        "Date": ["2025-08-15", "2025-08-16"],
        "Rainfall (mm)": [120.0, 85.0]
    })
    
    df_preds, summary = pipeline.run_pipeline(sample_df, output_csv_path=None)
    assert len(df_preds) == 2
    assert "Predicted_Probability" in df_preds.columns
    assert "Risk_Level" in df_preds.columns
    assert "Top_Risk_Driver_1" in df_preds.columns
    assert "Flood_Occurred" not in df_preds.columns


def test_retrospective_evaluation():
    engine = Retrospective2025EvaluationEngine()
    
    df_preds = pd.DataFrame({
        "District": ["CUTTACK", "PURI", "BALASORE", "GANJAM"],
        "Block/Station": ["Cuttack", "Puri", "Balasore", "Ganjam"],
        "Date": ["2025-08-15", "2025-08-15", "2025-08-15", "2025-08-15"],
        "Predicted_Probability": [0.95, 0.10, 0.88, 0.05],
        "Risk_Level": ["HIGH", "LOW", "HIGH", "LOW"]
    })
    
    df_gt = pd.DataFrame({
        "District": ["CUTTACK", "PURI", "BALASORE", "GANJAM"],
        "Block/Station": ["Cuttack", "Puri", "Balasore", "Ganjam"],
        "Date": ["2025-08-15", "2025-08-15", "2025-08-15", "2025-08-15"],
        "Flood_Occurred": [1, 0, 1, 0]
    })
    
    metrics, df_matched = engine.evaluate(df_preds, df_gt)
    assert metrics["matched_observations"] == 4
    assert metrics["true_flood_days"] == 2
    assert "retrospective_f1" in metrics
    assert "retrospective_accuracy" in metrics
    assert metrics["retrospective_accuracy"] == 1.0
