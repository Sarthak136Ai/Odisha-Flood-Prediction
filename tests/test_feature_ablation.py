"""
Unit tests for Feature Ablation Study in Odisha Flood Prediction.

Validates:
1. Feature group definitions, integrity, and monotonic containment hierarchies.
2. F2 score computation and recall weighting.
3. Strict validation-only threshold optimization without test leakage.
4. Metric evaluation outputs (ROC-AUC, PR-AUC, Precision, Recall, F1, F2, Brier Score).
5. End-to-end execution of FeatureAblationStudy on synthetic data.
6. Multi-panel publication plot generation.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.feature_ablation import (
    FEATURE_ABLATION_GROUPS,
    CURRENT_RAINFALL_COLS,
    LAG_RAINFALL_COLS,
    ROLLING_RAINFALL_COLS,
    TEMPORAL_SEASONAL_COLS,
    INTENSITY_COLS,
    compute_f2_score,
    find_validation_optimal_threshold,
    evaluate_predictions,
    FeatureAblationStudy,
    plot_feature_ablation_comparison
)


def test_feature_groups_integrity():
    """Verify group definitions, member uniqueness, and expected column counts."""
    expected_groups = [
        "Experiment_A", "Experiment_B", "Experiment_C",
        "Experiment_D", "Experiment_E", "Experiment_F"
    ]
    for g in expected_groups:
        assert g in FEATURE_ABLATION_GROUPS, f"Missing group {g} in registry"
        feats = FEATURE_ABLATION_GROUPS[g]["features"]
        assert len(feats) == len(set(feats)), f"Duplicate features in {g}"

    assert len(FEATURE_ABLATION_GROUPS["Experiment_A"]["features"]) == 2
    assert len(FEATURE_ABLATION_GROUPS["Experiment_B"]["features"]) == 6
    assert len(FEATURE_ABLATION_GROUPS["Experiment_C"]["features"]) == 19
    assert len(FEATURE_ABLATION_GROUPS["Experiment_D"]["features"]) == 5
    assert len(FEATURE_ABLATION_GROUPS["Experiment_E"]["features"]) == 22
    assert len(FEATURE_ABLATION_GROUPS["Experiment_F"]["features"]) == 29

    # Hierarchy verification
    exp_e_set = set(FEATURE_ABLATION_GROUPS["Experiment_E"]["features"])
    assert set(FEATURE_ABLATION_GROUPS["Experiment_A"]["features"]).issubset(exp_e_set)
    assert set(FEATURE_ABLATION_GROUPS["Experiment_B"]["features"]).issubset(exp_e_set)
    assert set(FEATURE_ABLATION_GROUPS["Experiment_C"]["features"]).issubset(exp_e_set)
    assert set(FEATURE_ABLATION_GROUPS["Experiment_D"]["features"]).issubset(exp_e_set)

    exp_f_set = set(FEATURE_ABLATION_GROUPS["Experiment_F"]["features"])
    assert exp_e_set.issubset(exp_f_set)
    assert set(INTENSITY_COLS).issubset(exp_f_set)


def test_f2_score_properties():
    """Verify F2 metric calculation, boundary values, and recall-weighting preference."""
    # Zero division
    assert compute_f2_score(0.0, 0.0) == 0.0

    # Perfect prediction
    assert pytest.approx(compute_f2_score(1.0, 1.0)) == 1.0

    # Equal precision/recall
    f2_sym = compute_f2_score(0.5, 0.5)
    assert pytest.approx(f2_sym) == 0.5

    # Recall priority: high recall, moderate precision vs high precision, moderate recall
    # In flood disaster warning, high recall is favored:
    # F2 = 5 * (P * R) / (4*P + R)
    f2_high_rec = compute_f2_score(precision=0.4, recall=0.8)  # 5 * 0.32 / (1.6 + 0.8) = 1.6 / 2.4 = 0.667
    f2_high_prec = compute_f2_score(precision=0.8, recall=0.4) # 5 * 0.32 / (3.2 + 0.4) = 1.6 / 3.6 = 0.444
    assert f2_high_rec > f2_high_prec, "F2 must strictly favor Recall over Precision"


def test_validation_only_threshold_tuning():
    """Verify threshold optimization returns valid cutoff in (0, 1)."""
    np.random.seed(42)
    y_true = np.array([0] * 900 + [1] * 100)
    y_proba = np.where(y_true == 1, np.random.uniform(0.4, 0.9, size=1000), np.random.uniform(0.01, 0.4, size=1000))

    t_f1, best_f1 = find_validation_optimal_threshold(y_true, y_proba, metric="f1")
    assert 0.0 < t_f1 < 1.0
    assert best_f1 > 0.5

    t_f2, best_f2 = find_validation_optimal_threshold(y_true, y_proba, metric="f2")
    assert 0.0 < t_f2 < 1.0
    # Recall-oriented F2 threshold should typically be lower or equal to F1 threshold to capture more positive cases
    assert t_f2 <= t_f1 + 0.05


def test_evaluate_predictions_metrics():
    """Verify all 7 requested metrics are present and bounded."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.15, 0.05, 0.6, 0.25])
    threshold = 0.5

    metrics = evaluate_predictions(y_true, y_proba, threshold=threshold)
    required_keys = [
        "threshold", "roc_auc", "pr_auc", "precision", "recall",
        "f1", "f2", "brier_score", "tp", "fp", "tn", "fn"
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric {key}"

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["f2"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["tp"] == 4
    assert metrics["fn"] == 0


def test_feature_ablation_study_synthetic_run():
    """End-to-end verification of FeatureAblationStudy on synthetic data."""
    np.random.seed(42)
    n_train, n_val, n_test = 300, 100, 100

    def generate_synthetic_df(n):
        df = pd.DataFrame()
        # All columns required by groups A to F
        all_cols = FEATURE_ABLATION_GROUPS["Experiment_F"]["features"]
        for c in all_cols:
            if c == "Rainfall_Missing":
                df[c] = 0.0
            elif "Indicator" in c or c in INTENSITY_COLS:
                df[c] = np.random.choice([0.0, 1.0], size=n)
            else:
                df[c] = np.random.exponential(scale=10.0, size=n)
        # Synthetic target correlated with rainfall
        flood_prob = 1 / (1 + np.exp(-(df["Rainfall (mm)"] - 15) / 5))
        df["Flood_Next_Day"] = (np.random.uniform(size=n) < flood_prob).astype(int)
        return df

    train_df = generate_synthetic_df(n_train)
    val_df = generate_synthetic_df(n_val)
    test_df = generate_synthetic_df(n_test)

    # Use a 2-group subset for fast testing
    test_groups = {
        "Experiment_A": FEATURE_ABLATION_GROUPS["Experiment_A"],
        "Experiment_D": FEATURE_ABLATION_GROUPS["Experiment_D"]
    }

    study = FeatureAblationStudy(groups=test_groups, optimization_metric="f1")
    model_factories = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=100)
    }

    results = study.run_ablation(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col="Flood_Next_Day",
        model_factory_dict=model_factories
    )

    assert len(results) == 2
    assert "Val_ROC_AUC" in results.columns
    assert "Val_PR_AUC" in results.columns
    assert "Val_F1" in results.columns
    assert "Val_F2" in results.columns
    assert "Val_Brier_Score" in results.columns
    assert "Test_ROC_AUC" in results.columns
    assert "Test_PR_AUC" in results.columns
    assert "Test_F1" in results.columns
    assert "Test_F2" in results.columns
    assert "Test_Brier_Score" in results.columns


def test_plot_generation(tmp_path):
    """Verify plot generation creates valid png and svg files."""
    results_data = [
        {
            "Model": "Logistic Regression",
            "Experiment_ID": f"Experiment_{c}",
            "Group_Name": f"Group {c}",
            "Val_PR_AUC": 0.2 + i * 0.05,
            "Test_PR_AUC": 0.22 + i * 0.04,
            "Val_ROC_AUC": 0.65 + i * 0.04,
            "Test_ROC_AUC": 0.68 + i * 0.03,
            "Test_F1": 0.18 + i * 0.03,
            "Test_F2": 0.25 + i * 0.03,
            "Test_Brier_Score": 0.05 - i * 0.005
        }
        for i, c in enumerate(["A", "B", "C", "D", "E", "F"])
    ]
    df = pd.DataFrame(results_data)

    png_path = str(tmp_path / "test_ablation.png")
    svg_path = str(tmp_path / "test_ablation.svg")

    fig = plot_feature_ablation_comparison(df, save_path_png=png_path, save_path_svg=svg_path)
    assert os.path.exists(png_path)
    assert os.path.exists(svg_path)
    assert os.path.getsize(png_path) > 1000
    assert os.path.getsize(svg_path) > 1000
