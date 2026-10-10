"""
Automated Unit Tests for Confusion Matrix Evaluation & Calculations.

Verifies:
1. Exact calculation of TN, FP, FN, TP and sample conservation (TN + FP + FN + TP == N).
2. Calculation of Accuracy, Precision, Recall, F1, F2.
3. Calculation of operational error rates: False Negative Rate (FNR = Miss Rate) and False Positive Rate (FPR).
4. Boundary condition: FNR + Recall == 1.0 and FPR + Specificity == 1.0.
5. Zero-division robustness (edge cases where TP+FP == 0 or TP+FN == 0).
6. Generation of PNG and SVG figure artifacts.
"""

import os
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.confusion_matrix import (
    calculate_confusion_matrix_metrics,
    plot_styled_confusion_matrix,
    plot_multi_model_confusion_matrix_grid
)


@pytest.fixture
def controlled_cm_data():
    """Create controlled binary labels and predictions with known confusion matrix."""
    # 70 TN, 10 FP, 5 FN, 15 TP -> Total = 100
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.array([0] * 70 + [1] * 10 + [0] * 5 + [1] * 15)
    return y_true, y_pred


def test_confusion_matrix_basic_metrics(controlled_cm_data):
    """Test exact mathematical correctness of confusion matrix metrics."""
    y_true, y_pred = controlled_cm_data
    
    metrics = calculate_confusion_matrix_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_name="TestModel",
        threshold=0.5
    )
    
    assert metrics["total_samples"] == 100
    assert metrics["true_negatives_tn"] == 70
    assert metrics["false_positives_fp"] == 10
    assert metrics["false_negatives_fn"] == 5
    assert metrics["true_positives_tp"] == 15
    
    # Check metric formulas
    assert abs(metrics["accuracy"] - 0.85) < 1e-5           # (70+15)/100 = 0.85
    assert abs(metrics["precision"] - (15 / 25)) < 1e-5     # 15/25 = 0.60
    assert abs(metrics["recall_sensitivity"] - (15 / 20)) < 1e-5 # 15/20 = 0.75
    assert abs(metrics["specificity"] - (70 / 80)) < 1e-5   # 70/80 = 0.875
    
    # Check F1
    expected_f1 = 2 * (0.60 * 0.75) / (0.60 + 0.75)       # 0.90 / 1.35 = 0.66667
    assert abs(metrics["f1_score"] - expected_f1) < 1e-4
    
    # Check F2 (beta=2)
    expected_f2 = 5 * (0.60 * 0.75) / ((4 * 0.60) + 0.75) # 2.25 / 3.15 = 0.71429
    assert abs(metrics["f2_score"] - expected_f2) < 1e-4
    
    # Check Error Rates
    expected_fnr = 5 / 20                                  # 0.25 (25% missed floods)
    expected_fpr = 10 / 80                                 # 0.125 (12.5% false alarms)
    assert abs(metrics["false_negative_rate_fnr"] - expected_fnr) < 1e-5
    assert abs(metrics["false_positive_rate_fpr"] - expected_fpr) < 1e-5
    
    # Check Complementarity Invariants
    assert abs(metrics["false_negative_rate_fnr"] + metrics["recall_sensitivity"] - 1.0) < 1e-5
    assert abs(metrics["false_positive_rate_fpr"] + metrics["specificity"] - 1.0) < 1e-5


def test_confusion_matrix_zero_division_safety():
    """Verify metrics calculation never crashes on extreme all-zero or all-one edge cases."""
    # Case 1: Model predicts all 0s
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred_all_0 = np.array([0, 0, 0, 0, 0])
    m1 = calculate_confusion_matrix_metrics(y_true, y_pred_all_0)
    assert m1["true_positives_tp"] == 0
    assert m1["precision"] == 0.0
    assert m1["recall_sensitivity"] == 0.0
    assert m1["f1_score"] == 0.0
    assert m1["false_negative_rate_fnr"] == 1.0  # 100% missed floods
    
    # Case 2: Model predicts all 1s
    y_pred_all_1 = np.array([1, 1, 1, 1, 1])
    m2 = calculate_confusion_matrix_metrics(y_true, y_pred_all_1)
    assert m2["true_negatives_tn"] == 0
    assert m2["specificity"] == 0.0
    assert m2["false_positive_rate_fpr"] == 1.0  # 100% false alarms


def test_styled_confusion_matrix_plot_generation(tmp_path, controlled_cm_data):
    """Test generating PNG and SVG confusion matrix figure files."""
    y_true, y_pred = controlled_cm_data
    png_file = str(tmp_path / "test_cm.png")
    svg_file = str(tmp_path / "test_cm.svg")
    
    fig = plot_styled_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        model_name="Test Model",
        threshold=0.5,
        save_path_png=png_file,
        save_path_svg=svg_file
    )
    
    assert fig is not None
    assert os.path.exists(png_file)
    assert os.path.exists(svg_file)
    assert os.path.getsize(png_file) > 0
    assert os.path.getsize(svg_file) > 0
    plt.close(fig)


def test_multi_model_confusion_matrix_grid(tmp_path, controlled_cm_data):
    """Test generating multi-model confusion matrix grid figure."""
    y_true, y_pred = controlled_cm_data
    grid_png = str(tmp_path / "test_grid.png")
    grid_svg = str(tmp_path / "test_grid.svg")
    
    models_dict = {
        "Model A": (y_true, y_pred, 0.5),
        "Model B": (y_true, (y_pred == 0).astype(int), 0.8)
    }
    
    fig = plot_multi_model_confusion_matrix_grid(
        models_predictions=models_dict,
        save_path_png=grid_png,
        save_path_svg=grid_svg
    )
    
    assert fig is not None
    assert os.path.exists(grid_png)
    assert os.path.exists(grid_svg)
    assert os.path.getsize(grid_png) > 0
    assert os.path.getsize(grid_svg) > 0
    plt.close(fig)
