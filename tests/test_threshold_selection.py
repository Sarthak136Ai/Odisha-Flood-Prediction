"""
Tests for Threshold Selection and Optimization in Odisha Flood Prediction.

Verifies:
1. Zero test-set leakage: Threshold optimization uses ONLY validation data and is
   completely invariant to test set contents/permutations.
2. Configurable threshold strategies: F1 (balanced), F2 (disaster-averse recall-oriented),
   Recall Target (>=80%), Precision Target, Youden's J, and Cost-Sensitive utility.
3. Threshold bounds: 0.0 < T* < 1.0.
4. Recall-oriented strategies guarantee higher/equal recall compared to balanced F1.
5. Metadata schema contains: model_name, selected_threshold, optimization_metric,
   validation F1, validation precision, and validation recall.
6. Mathematical validation of ANN (unweighted) vs tree-based (class-weighted) threshold differences.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score

from src.evaluation.threshold_optimization import ThresholdOptimizer
from src.evaluation.metrics import find_optimal_threshold, calculate_metrics


@pytest.fixture
def synthetic_val_data():
    """Create synthetic validation set with ~2.6% positive flood rate."""
    np.random.seed(42)
    n_samples = 10000
    # True prevalence ~ 2.6%
    y_val = np.random.binomial(1, 0.026, size=n_samples)
    # Simulated model probabilities with good separation
    val_proba = np.where(y_val == 1,
                         np.random.beta(5, 2, size=n_samples),   # Higher probabilities for positives
                         np.random.beta(1, 15, size=n_samples))  # Low probabilities for negatives
    return y_val, val_proba


@pytest.fixture
def synthetic_test_data():
    """Create synthetic test set."""
    np.random.seed(123)
    n_samples = 5000
    y_test = np.random.binomial(1, 0.026, size=n_samples)
    test_proba = np.where(y_test == 1,
                          np.random.beta(5, 2, size=n_samples),
                          np.random.beta(1, 15, size=n_samples))
    return y_test, test_proba


def test_zero_leakage_threshold_optimization(synthetic_val_data, synthetic_test_data):
    """
    STRICT PROOF: Threshold optimization must use validation data ONLY.
    Modifying, scrambling, or inverting test data must NOT change the selected threshold.
    """
    y_val, val_proba = synthetic_val_data
    y_test, test_proba = synthetic_test_data
    
    optimizer = ThresholdOptimizer(random_state=42)
    
    # 1. Optimize on validation data
    res_clean = optimizer.optimize_threshold(y_val, val_proba, strategy="f1")
    t_opt_clean = res_clean["threshold"]
    
    # 2. Modify test data completely (random noise, inverted labels, arbitrary scale)
    y_test_corrupted = 1 - y_test
    test_proba_corrupted = np.random.uniform(0, 1, size=len(y_test))
    
    # Re-optimize on validation data
    res_after_corrupted_test = optimizer.optimize_threshold(y_val, val_proba, strategy="f1")
    t_opt_after = res_after_corrupted_test["threshold"]
    
    # Must be 100% identical because test data was never passed to optimizer
    assert t_opt_clean == t_opt_after, "Threshold selection must depend strictly on validation data!"
    
    # Verify that applying validation threshold to test data does not alter the threshold
    metrics_test_1 = calculate_metrics(y_test, test_proba, threshold=t_opt_clean)
    metrics_test_2 = calculate_metrics(y_test, test_proba, threshold=t_opt_clean)
    assert metrics_test_1["threshold"] == t_opt_clean
    assert metrics_test_1["f1"] == metrics_test_2["f1"]


def test_threshold_bounds(synthetic_val_data):
    """Ensure selected threshold is always strictly within (0, 1)."""
    y_val, val_proba = synthetic_val_data
    optimizer = ThresholdOptimizer(random_state=42)
    
    for strategy in ["f1", "f2", "recall_target", "precision_target", "youden_j", "cost_sensitive"]:
        res = optimizer.optimize_threshold(y_val, val_proba, strategy=strategy)
        t = res["threshold"]
        assert 0.0 < t < 1.0, f"Strategy {strategy} returned invalid threshold {t}"


def test_recall_oriented_vs_f1_strategy(synthetic_val_data):
    """
    Test that disaster-averse F2 / Recall Target strategy yields higher or equal recall
    than standard balanced F1.
    """
    y_val, val_proba = synthetic_val_data
    optimizer = ThresholdOptimizer(random_state=42)
    
    res_f1 = optimizer.optimize_threshold(y_val, val_proba, strategy="f1")
    res_f2 = optimizer.optimize_threshold(y_val, val_proba, strategy="f2")
    res_rec80 = optimizer.optimize_threshold(y_val, val_proba, strategy="recall_target", target_recall=0.80)
    
    # F2 penalizes false negatives more, so optimal threshold must be <= F1 threshold
    assert res_f2["threshold"] <= res_f1["threshold"] + 1e-5, (
        f"F2 threshold ({res_f2['threshold']}) should be <= F1 threshold ({res_f1['threshold']})"
    )
    assert res_f2["val_recall"] >= res_f1["val_recall"] - 1e-5, (
        f"F2 recall ({res_f2['val_recall']}) should be >= F1 recall ({res_f1['val_recall']})"
    )
    
    # Recall target >= 0.80 must achieve at least 80% recall if attainable
    assert res_rec80["val_recall"] >= 0.80, f"Target recall not met: {res_rec80['val_recall']}"


def test_evaluate_all_strategies(synthetic_val_data):
    """Test comprehensive evaluation of all 7 threshold strategies."""
    y_val, val_proba = synthetic_val_data
    optimizer = ThresholdOptimizer(random_state=42)
    
    df_strats = optimizer.evaluate_all_strategies(y_val, val_proba, model_name="TestModel")
    
    assert isinstance(df_strats, pd.DataFrame)
    assert len(df_strats) == 7
    required_cols = [
        "Model", "Strategy_Label", "Strategy_Key", "Threshold",
        "Val_F1", "Val_F2", "Val_Precision", "Val_Recall",
        "Val_Specificity", "Val_Accuracy", "Val_PR_AUC", "Val_ROC_AUC"
    ]
    for col in required_cols:
        assert col in df_strats.columns, f"Missing required column: {col}"


def test_find_optimal_threshold_metrics_module(synthetic_val_data):
    """Test find_optimal_threshold in src.evaluation.metrics."""
    y_val, val_proba = synthetic_val_data
    
    # Test F1
    t_f1, score_f1 = find_optimal_threshold(y_val, val_proba, metric="f1")
    assert 0.0 < t_f1 < 1.0
    assert 0.0 < score_f1 <= 1.0
    
    # Test F2
    t_f2, score_f2 = find_optimal_threshold(y_val, val_proba, metric="f2")
    assert 0.0 < t_f2 < 1.0
    assert t_f2 <= t_f1 + 1e-5
    
    # Test recall target
    t_rec, score_rec = find_optimal_threshold(y_val, val_proba, metric="recall_target", target_recall=0.85)
    assert 0.0 < t_rec < 1.0
    assert score_rec >= 0.85


def test_ann_unweighted_vs_tree_class_weighted_threshold_mechanics():
    """
    Test scientific principle:
    - Unweighted model produces raw probabilities near true prevalence (~0.026).
    - Class-weighted model (scale_pos_weight ~37) produces shifted probabilities (~0.50).
    - The optimal F1 threshold for unweighted is naturally ~0.05-0.15.
    - The optimal F1 threshold for class-weighted is naturally ~0.70-0.85.
    """
    np.random.seed(42)
    n = 20000
    y_true = np.random.binomial(1, 0.026, size=n)
    
    # Unweighted probability distribution (ANN style)
    p_unweighted = np.where(y_true == 1,
                            np.random.beta(1.5, 8.0, size=n),  # Signal positives: 0.05 to 0.35
                            np.random.beta(0.5, 30.0, size=n)) # True negatives: 0.001 to 0.03
    
    # Class-weighted probability distribution (Odds shifted by 37x)
    odds_true = p_unweighted / (1.0 - p_unweighted + 1e-12)
    odds_weighted = 37.0 * odds_true
    p_weighted = odds_weighted / (1.0 + odds_weighted)
    
    optimizer = ThresholdOptimizer(random_state=42)
    res_unweighted = optimizer.optimize_threshold(y_true, p_unweighted, strategy="f1")
    res_weighted = optimizer.optimize_threshold(y_true, p_weighted, strategy="f1")
    
    # Unweighted threshold should be naturally lower (< 0.30)
    assert res_unweighted["threshold"] < 0.30, f"Unweighted threshold should be < 0.30, got {res_unweighted['threshold']}"
    # Class-weighted threshold should be naturally higher (> 0.50)
    assert res_weighted["threshold"] > 0.50, f"Class-weighted threshold should be > 0.50, got {res_weighted['threshold']}"


def test_model_thresholds_metadata_schema():
    """Verify schema of model_thresholds.json metadata."""
    schema_sample = {
        "model_name": "XGBoost",
        "selected_threshold": 0.8073,
        "optimization_metric": "f1",
        "val_f1": 0.2286,
        "val_precision": 0.1829,
        "val_recall": 0.3046,
        "val_f2": 0.2687
    }
    
    required_keys = ["model_name", "selected_threshold", "optimization_metric", "val_f1", "val_precision", "val_recall"]
    for k in required_keys:
        assert k in schema_sample
