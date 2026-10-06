"""
Unit and integration tests for the Temporal Walk-Forward Validation Engine.
Verifies chronological ordering, absence of future leakage, strict preprocessor isolation,
expanding training window monotonicity, and metric schema validity.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.evaluation.temporal_validation import (
    TemporalWalkForwardSplitter,
    TemporalWalkForwardValidator
)
from src.data.load_data import load_config
from src.models.logistic_regression import create_logistic_regression_pipeline


@pytest.fixture
def synthetic_temporal_data():
    """Create synthetic multi-year spatio-temporal dataset for testing."""
    np.random.seed(42)
    records = []
    districts = ["Cuttack", "Puri", "Khurda", "Balasore", "Ganjam"]
    
    for year in range(2001, 2025):
        for district in districts:
            for day_of_year in range(1, 366):
                rainfall = float(np.random.exponential(scale=4.0) if np.random.rand() > 0.7 else 0.0)
                # target correlated with rainfall but with lag
                flood_prob = 1.0 / (1.0 + np.exp(-(rainfall - 50.0) / 10.0))
                flood_occurred = int(np.random.rand() < flood_prob * 0.5)
                flood_next_day = int(np.random.rand() < flood_prob * 0.6)
                
                records.append({
                    "District": district,
                    "Block/Station": f"{district}_Block_1",
                    "Date": f"{year}-01-01",
                    "Year": year,
                    "Month_Number": 6,
                    "Day": 15,
                    "Day_of_Year": day_of_year,
                    "Day_of_Week": 2,
                    "Month_sin": 0.5,
                    "Month_cos": 0.866,
                    "Rainfall (mm)": rainfall,
                    "Rainfall_Missing": 0,
                    "Rainfall_Lag_1d": rainfall * 0.8,
                    "Rainfall_Lag_2d": rainfall * 0.6,
                    "Rainfall_Lag_3d": rainfall * 0.4,
                    "Rainfall_Lag_7d": rainfall * 0.2,
                    "Rainfall_Prev_3d_Sum": rainfall * 1.5,
                    "Rainfall_Prev_7d_Sum": rainfall * 2.5,
                    "Rainfall_Prev_15d_Sum": rainfall * 4.0,
                    "Rainfall_Prev_30d_Sum": rainfall * 6.0,
                    "Rainfall_Prev_3d_Max": rainfall,
                    "Rainfall_Prev_7d_Max": rainfall,
                    "Rainfall_Prev_15d_Max": rainfall,
                    "Rainfall_Prev_30d_Max": rainfall,
                    "Rainy_Days_Prev_3d": 1,
                    "Rainy_Days_Prev_7d": 2,
                    "Rainy_Days_Prev_15d": 4,
                    "Rainy_Days_Prev_30d": 7,
                    "Consecutive_Rainy_Days_Before": 1,
                    "Flood_Occurred": flood_occurred,
                    "Flood_Next_Day": flood_next_day
                })
    return pd.DataFrame(records)


def test_temporal_split_chronological_ordering():
    """Test that max(train_year) < min(val_year) for every generated fold."""
    splitter = TemporalWalkForwardSplitter(
        start_year=2001,
        initial_train_end_year=2016,
        val_step_years=1,
        n_folds=5,
        test_years=[2022, 2023, 2024]
    )
    folds = splitter.generate_fold_definitions()
    
    assert len(folds) == 5
    for fold in folds:
        train_years = fold["train_years"]
        val_years = fold["val_years"]
        assert max(train_years) < min(val_years), (
            f"Temporal causality violation: max(train)={max(train_years)} >= min(val)={min(val_years)}"
        )
        assert min(train_years) == 2001


def test_zero_train_validation_index_overlap(synthetic_temporal_data):
    """Verify that there is zero index overlap between train and val splits."""
    splitter = TemporalWalkForwardSplitter(n_folds=5)
    
    for fold_id, train_df, val_df, meta in splitter.split(synthetic_temporal_data):
        train_idx = set(train_df.index)
        val_idx = set(val_df.index)
        assert len(train_idx.intersection(val_idx)) == 0, (
            f"Fold {fold_id} has overlapping indices between train and validation!"
        )


def test_expanding_window_growth(synthetic_temporal_data):
    """Verify that training set strictly expands chronologically across folds."""
    splitter = TemporalWalkForwardSplitter(n_folds=5)
    train_sizes = []
    
    for fold_id, train_df, val_df, meta in splitter.split(synthetic_temporal_data):
        train_sizes.append(len(train_df))
        
    for i in range(len(train_sizes) - 1):
        assert train_sizes[i] < train_sizes[i + 1], (
            f"Training set size did not expand: Fold {i+1} ({train_sizes[i]}) >= Fold {i+2} ({train_sizes[i+1]})"
        )


def test_held_out_test_years_isolation(synthetic_temporal_data):
    """Verify test years (2022–2024) are completely excluded from walk-forward folds."""
    test_years = [2022, 2023, 2024]
    splitter = TemporalWalkForwardSplitter(test_years=test_years, n_folds=5)
    
    for fold_id, train_df, val_df, meta in splitter.split(synthetic_temporal_data):
        assert not any(y in test_years for y in train_df["Year"].unique()), (
            f"Fold {fold_id} train split contains held-out test years!"
        )
        assert not any(y in test_years for y in val_df["Year"].unique()), (
            f"Fold {fold_id} validation split contains held-out test years!"
        )


def test_scaler_preprocessor_training_isolation(synthetic_temporal_data):
    """Verify that feature scaling is fit strictly on training fold data."""
    splitter = TemporalWalkForwardSplitter(n_folds=2)
    feature_cols = ["Rainfall (mm)", "Rainfall_Lag_1d", "Rainfall_Prev_3d_Sum"]
    
    for fold_id, train_df, val_df, meta in splitter.split(synthetic_temporal_data):
        # Create pipeline
        pipeline = create_logistic_regression_pipeline()
        X_train = train_df[feature_cols]
        y_train = train_df["Flood_Next_Day"].values
        
        pipeline.fit(X_train, y_train)
        scaler: StandardScaler = pipeline.named_steps["scaler"]
        
        # Verify scaler mean matches training set mean, NOT combined or val mean
        expected_train_mean = X_train.mean().values
        np.testing.assert_allclose(
            scaler.mean_, expected_train_mean, rtol=1e-4,
            err_msg="StandardScaler mean differs from fold training mean (possible leakage)!"
        )


def test_walk_forward_evaluation_metrics_validity(synthetic_temporal_data):
    """Verify validation engine computes all required metrics within valid statistical bounds."""
    validator = TemporalWalkForwardValidator(
        config_path="config.yaml",
        random_state=42
    )
    # Test on lightweight subset of models (Logistic Regression & Decision Tree)
    results_df, summary_df, summary_dict = validator.run_validation(
        synthetic_temporal_data,
        models_to_run=["Logistic Regression", "Decision Tree"]
    )
    
    assert not results_df.empty
    assert not summary_df.empty
    
    # Check fold count
    assert len(results_df) == 10  # 2 models * 5 folds
    assert len(summary_df) == 2   # 2 models
    
    # Check metric bounds
    for col in ["ROC_AUC", "PR_AUC", "F1", "Precision", "Recall", "Accuracy", "Specificity", "Brier_Score"]:
        assert (results_df[col] >= 0.0).all() and (results_df[col] <= 1.0).all(), (
            f"Metric {col} violates [0, 1] probability bound!"
        )
        
    # Check summary mean and std columns exist
    for col in ["ROC_AUC_Mean", "PR_AUC_Mean", "F1_Mean", "Brier_Score_Mean"]:
        assert col in summary_df.columns
        assert (summary_df[col] >= 0.0).all() and (summary_df[col] <= 1.0).all()
