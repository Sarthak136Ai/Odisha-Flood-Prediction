"""
Unit tests for probability calibration and reliability metric calculations.
"""

import numpy as np
from src.evaluation.calibration import evaluate_calibration


def test_evaluate_calibration_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.15, 0.3, 0.7, 0.8, 0.85, 0.9, 0.65, 0.95])
    
    cal_stats = evaluate_calibration(y_true, y_proba, n_bins=5)
    
    assert "brier_score" in cal_stats
    assert "expected_calibration_error" in cal_stats
    assert 0.0 <= cal_stats["brier_score"] <= 1.0
    assert 0.0 <= cal_stats["expected_calibration_error"] <= 1.0
    assert len(cal_stats["prob_true"]) > 0
    assert len(cal_stats["prob_pred"]) > 0
