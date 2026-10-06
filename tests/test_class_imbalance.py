"""
Automated Unit and Integration Tests for Class Imbalance Analysis.

Verifies:
1. Target class counts, proportions, and positive rate consistency (~2.6% flood rate).
2. Split partition integrity: Train + Val + Test sums match total dataset records.
3. Year-wise distribution completeness (all 24 years from 2001 to 2024 present).
4. Confusion matrix invariants: TP + FP + TN + FN == N.
5. F2 / Disaster-averse threshold behavior under extreme imbalance.
6. Metric consistency: PR-AUC baseline corresponds to positive prevalence.
7. Absence of synthetic oversampling (no fabricated data injected into telemetry).
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.evaluation.class_imbalance import ClassImbalanceAnalyzer
from src.data.load_data import load_combined_data, load_config


@pytest.fixture
def sample_imbalanced_data():
    """Create a controlled imbalanced dataset with ~2.6% prevalence."""
    np.random.seed(42)
    n = 10000
    years = np.random.choice(range(2001, 2025), size=n)
    districts = np.random.choice(["Kendrapara", "Cuttack", "Puri", "Sambalpur", "Khordha"], size=n)
    y = np.random.binomial(1, 0.026, size=n)
    
    # Model probabilities
    proba = np.where(y == 1,
                     np.random.beta(4, 2, size=n),
                     np.random.beta(1, 20, size=n))
                     
    df = pd.DataFrame({
        "Year": years,
        "District": districts,
        "Flood_Next_Day": y
    })
    return df, proba


def test_target_class_distribution_integrity(sample_imbalanced_data):
    """Test that target distribution analysis returns valid proportions and counts."""
    df, _ = sample_imbalanced_data
    analyzer = ClassImbalanceAnalyzer(random_state=42)
    
    train_years = list(range(2001, 2019))
    val_years = [2019, 2020, 2021]
    test_years = [2022, 2023, 2024]
    
    res = analyzer.analyze_dataset_distributions(
        df=df,
        target_col="Flood_Next_Day",
        train_years=train_years,
        val_years=val_years,
        test_years=test_years
    )
    
    splits_df = res["splits_summary"]
    yearwise_df = res["yearwise_summary"]
    district_df = res["district_summary"]
    
    # Verify overall row count
    overall_row = splits_df[splits_df["Split"].str.contains("Complete")].iloc[0]
    assert overall_row["Total_Records"] == len(df)
    assert overall_row["Negative_Count_0"] + overall_row["Positive_Count_1"] == len(df)
    assert 0.0 < overall_row["Positive_Pct_1"] < 10.0
    
    # Verify split partition sum
    train_row = splits_df[splits_df["Split"].str.contains("Training")].iloc[0]
    val_row = splits_df[splits_df["Split"].str.contains("Validation")].iloc[0]
    test_row = splits_df[splits_df["Split"].str.contains("Test")].iloc[0]
    
    total_split_records = train_row["Total_Records"] + val_row["Total_Records"] + test_row["Total_Records"]
    assert total_split_records == len(df)
    
    # Verify year-wise completeness
    assert len(yearwise_df) == 24
    assert set(yearwise_df["Year"]) == set(range(2001, 2025))


def test_confusion_matrix_invariants(sample_imbalanced_data):
    """Test confusion matrix components satisfy conservation of total samples."""
    df, proba = sample_imbalanced_data
    y_true = df["Flood_Next_Day"].values
    analyzer = ClassImbalanceAnalyzer(random_state=42)
    
    thresholds = {
        "Standard Baseline": 0.50,
        "Low Cutoff": 0.20,
        "High Cutoff": 0.80
    }
    
    eval_df = analyzer.evaluate_threshold_performance_matrix(
        y_true=y_true,
        y_proba=proba,
        threshold_map=thresholds,
        model_name="TestModel"
    )
    
    for _, row in eval_df.iterrows():
        total_cm = row["True_Positives_TP"] + row["False_Positives_FP"] + row["True_Negatives_TN"] + row["False_Negatives_FN"]
        assert total_cm == len(y_true), "Confusion matrix total must equal sample count"
        assert row["True_Positives_TP"] + row["False_Negatives_FN"] == int(np.sum(y_true == 1))
        assert row["True_Negatives_TN"] + row["False_Positives_FP"] == int(np.sum(y_true == 0))


def test_f2_disaster_averse_behavior(sample_imbalanced_data):
    """Verify that F2 score places greater weight on recall than precision."""
    analyzer = ClassImbalanceAnalyzer(random_state=42)
    
    # High precision, low recall scenario
    f1_high_prec = 2 * (0.90 * 0.20) / (0.90 + 0.20)
    f2_high_prec = analyzer.calculate_f_beta(0.90, 0.20, beta=2.0)
    
    # Low precision, high recall scenario
    f1_high_rec = 2 * (0.20 * 0.90) / (0.20 + 0.90)
    f2_high_rec = analyzer.calculate_f_beta(0.20, 0.90, beta=2.0)
    
    # F1 is symmetric: f1_high_prec == f1_high_rec
    assert abs(f1_high_prec - f1_high_rec) < 1e-6
    # F2 favors high recall: f2_high_rec > f2_high_prec
    assert f2_high_rec > f2_high_prec


def test_pr_auc_baseline_prevalence(sample_imbalanced_data):
    """Test that PR-AUC calculation is bounded and evaluates discriminative power."""
    df, proba = sample_imbalanced_data
    y_true = df["Flood_Next_Day"].values
    analyzer = ClassImbalanceAnalyzer(random_state=42)
    
    eval_df = analyzer.evaluate_threshold_performance_matrix(
        y_true=y_true,
        y_proba=proba,
        threshold_map={"Default": 0.50},
        model_name="TestModel"
    )
    
    pr_auc = eval_df.iloc[0]["PR_AUC"]
    prevalence = np.mean(y_true)
    assert 0.0 <= pr_auc <= 1.0
    # Model with good separation should exceed random prevalence baseline
    assert pr_auc > prevalence


def test_raw_dataset_no_synthetic_injection():
    """Verify historical dataset maintains pure empirical integer/float values without SMOTE interpolation."""
    config = load_config("config.yaml")
    dataset_path = config["paths"]["combined_data_path"]
    df = load_combined_data(dataset_path, parse_dates=False)
    
    # Flood_Next_Day must be strictly binary integers (0 or 1)
    unique_targets = df["Flood_Next_Day"].dropna().unique()
    assert set(unique_targets).issubset({0, 1, 0.0, 1.0}), "Target must only contain 0 or 1"
    
    # Check overall dataset length is exactly the historical 2,752,252
    assert len(df) == 2752252, f"Expected 2,752,252 rows, found {len(df)}"
