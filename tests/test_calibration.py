"""
Unit tests for Probability Calibration Engine in Odisha Flood Prediction.

Validates:
1. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) computations.
2. Perfect vs poor calibration metrics.
3. Fit calibrators (Sigmoid / Platt and Isotonic) on validation data without leakage.
4. Brier Score and ECE reduction post-calibration.
5. FloodPredictor loading calibrated model and calibrated risk thresholds.
6. Calibration comparison publication plot generation.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.calibration import (
    compute_calibration_errors,
    evaluate_calibration,
    fit_validation_calibrator,
    plot_raw_vs_calibrated_comparison,
    plot_reliability_diagrams
)
from src.models.predict import FloodPredictor


def test_compute_calibration_errors_ideal():
    """Verify that perfectly calibrated predictions yield near-zero ECE and MCE."""
    np.random.seed(42)
    n = 10000
    # Generate calibrated probabilities
    y_proba = np.random.uniform(0.0, 1.0, size=n)
    y_true = (np.random.uniform(size=n) < y_proba).astype(int)

    ece, mce, p_true, p_pred, counts = compute_calibration_errors(y_true, y_proba, n_bins=10)
    assert 0.0 <= ece < 0.03
    assert 0.0 <= mce < 0.05
    assert len(p_true) == 10
    assert len(p_pred) == 10
    assert sum(counts) == n


def test_compute_calibration_errors_uncalibrated():
    """Verify that artificially inflated probabilities yield high ECE."""
    n = 1000
    y_true = np.array([0] * 900 + [1] * 100) # 10% positive
    # Artificially shift probabilities upwards to 0.7 - 0.9
    y_proba = np.random.uniform(0.7, 0.9, size=n)

    ece, mce, _, _, _ = compute_calibration_errors(y_true, y_proba, n_bins=10)
    assert ece > 0.50, f"Expected high ECE for shifted probabilities, got {ece}"
    assert mce > 0.50


def test_evaluate_calibration_metrics():
    """Verify evaluate_calibration output schema and bounds."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.15, 0.3, 0.7, 0.8, 0.85, 0.9, 0.65, 0.95])

    cal_stats = evaluate_calibration(y_true, y_proba, n_bins=5)
    expected_keys = [
        "brier_score", "log_loss", "expected_calibration_error",
        "maximum_calibration_error", "prob_true", "prob_pred",
        "bin_counts", "mean_predicted_probability", "observed_prevalence"
    ]
    for k in expected_keys:
        assert k in cal_stats, f"Missing key {k}"

    assert 0.0 <= cal_stats["brier_score"] <= 1.0
    assert cal_stats["log_loss"] >= 0.0
    assert 0.0 <= cal_stats["expected_calibration_error"] <= 1.0
    assert 0.0 <= cal_stats["maximum_calibration_error"] <= 1.0
    assert cal_stats["observed_prevalence"] == 0.6


def test_fit_validation_calibrator_sigmoid_and_isotonic():
    """Verify fitting calibrators on validation data with zero test leakage."""
    np.random.seed(42)
    n_train = 500
    n_val = 300
    n_test = 300

    X_train = np.random.randn(n_train, 4)
    y_train = (X_train[:, 0] > 0.5).astype(int)

    X_val = np.random.randn(n_val, 4)
    y_val = (X_val[:, 0] > 0.5).astype(int)

    X_test = np.random.randn(n_test, 4)
    y_test = (X_test[:, 0] > 0.5).astype(int)

    # Train base model with class weights
    base_model = LogisticRegression(class_weight="balanced")
    base_model.fit(X_train, y_train)

    # Fit Sigmoid calibrator on validation
    cal_sig = fit_validation_calibrator(base_model, X_val, y_val, method="sigmoid")
    sig_test_proba = cal_sig.predict_proba(X_test)[:, 1]
    assert sig_test_proba.shape == (n_test,)
    assert 0.0 <= sig_test_proba.min() and sig_test_proba.max() <= 1.0

    # Fit Isotonic calibrator on validation
    cal_iso = fit_validation_calibrator(base_model, X_val, y_val, method="isotonic")
    iso_test_proba = cal_iso.predict_proba(X_test)[:, 1]
    assert iso_test_proba.shape == (n_test,)
    assert 0.0 <= iso_test_proba.min() and iso_test_proba.max() <= 1.0


def test_calibrator_reduces_brier_and_ece():
    """Verify that calibration significantly reduces ECE and Brier score on uncalibrated models."""
    np.random.seed(42)
    n_samples = 2000
    X = np.random.randn(n_samples, 2)
    # 5% positive rate
    y = (X[:, 0] > 1.64).astype(int)

    X_tr, y_tr = X[:1000], y[:1000]
    X_va, y_va = X[1000:1500], y[1000:1500]
    X_te, y_te = X[1500:], y[1500:]

    # Base model with extreme class weight
    model = LogisticRegression(class_weight={0: 1, 1: 30})
    model.fit(X_tr, y_tr)

    raw_test_p = model.predict_proba(X_te)[:, 1]
    raw_stats = evaluate_calibration(y_te, raw_test_p)

    cal = fit_validation_calibrator(model, X_va, y_va, method="sigmoid")
    cal_test_p = cal.predict_proba(X_te)[:, 1]
    cal_stats = evaluate_calibration(y_te, cal_test_p)

    assert cal_stats["brier_score"] < raw_stats["brier_score"]
    assert cal_stats["expected_calibration_error"] < raw_stats["expected_calibration_error"]


def test_flood_predictor_calibrated_mode():
    """Verify FloodPredictor calibrated mode handling."""
    predictor_raw = FloodPredictor(use_calibrated=False)
    assert predictor_raw.use_calibrated is False
    assert predictor_raw.model is not None

    predictor_cal = FloodPredictor(use_calibrated=True)
    assert predictor_cal.model is not None
    if os.path.exists("models/flood_prediction/calibrated_best_model.pkl"):
        assert "calibrated" in predictor_cal.calibrated_model_path


def test_plot_raw_vs_calibrated_comparison(tmp_path):
    """Verify publication comparison plot generation."""
    np.random.seed(42)
    n = 200
    y_test = np.random.binomial(1, 0.05, size=n)
    raw_p = np.random.beta(2, 5, size=n)
    sig_p = np.random.beta(1, 15, size=n)
    iso_p = np.random.beta(1, 14, size=n)

    comp_data = {
        "XGBoost": {
            "Raw": (y_test, raw_p),
            "Calibrated (Sigmoid)": (y_test, sig_p),
            "Calibrated (Isotonic)": (y_test, iso_p)
        }
    }

    png_path = str(tmp_path / "cal_comp.png")
    svg_path = str(tmp_path / "cal_comp.svg")

    fig = plot_raw_vs_calibrated_comparison(comp_data, save_path_png=png_path, save_path_svg=svg_path)
    assert os.path.exists(png_path)
    assert os.path.exists(svg_path)
    assert os.path.getsize(png_path) > 1000
    assert os.path.getsize(svg_path) > 1000
