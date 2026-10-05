"""
Evaluation metrics module for flood prediction models.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score,
and determines optimal classification thresholds for early warning sensitivity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve
)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find optimal classification probability threshold on validation set.
    metric: 'f1' or 'recall_at_precision'
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid div by zero in f1 calculation
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    if metric == "f1":
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        return best_threshold, best_f1
    
    # Default fallback: 0.5
    return 0.5, 0.0


def calculate_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for binary classification.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # AUC calculations
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.0
        
    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = 0.0
        
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        f"{prefix}threshold": round(float(threshold), 4),
        f"{prefix}accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        f"{prefix}precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 5),
        f"{prefix}recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 5),
        f"{prefix}specificity": round(float(specificity), 5),
        f"{prefix}f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 5),
        f"{prefix}roc_auc": round(roc_auc, 5),
        f"{prefix}pr_auc": round(pr_auc, 5),
        f"{prefix}brier_score": round(float(brier_score_loss(y_true, y_proba)), 5),
        f"{prefix}tp": int(tp),
        f"{prefix}fp": int(fp),
        f"{prefix}tn": int(tn),
        f"{prefix}fn": int(fn),
    }
    return metrics
