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
    metric: str = "f1",
    beta: float = 2.0,
    target_recall: float = 0.80,
    target_precision: float = 0.25
) -> Tuple[float, float]:
    """
    Find optimal classification probability threshold on validation set.
    Supports: 'f1', 'f2', 'f_beta', 'recall_at_precision' / 'recall_target', 'precision_target', 'youden_j'.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    prec_t = precisions[:-1]
    rec_t = recalls[:-1]
    
    if metric == "f1":
        f1_scores = 2 * (prec_t * rec_t) / (prec_t + rec_t + 1e-10)
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx]), float(f1_scores[best_idx])
        
    elif metric in ["f2", "f_beta"]:
        b = 2.0 if metric == "f2" else beta
        beta_sq = b ** 2
        denom = (beta_sq * prec_t) + rec_t + 1e-10
        fb_scores = (1 + beta_sq) * (prec_t * rec_t) / denom
        best_idx = int(np.argmax(fb_scores))
        return float(thresholds[best_idx]), float(fb_scores[best_idx])
        
    elif metric in ["recall_at_precision", "recall_target"]:
        valid_indices = np.where(rec_t >= target_recall)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(prec_t[valid_indices])]
            return float(thresholds[best_idx]), float(rec_t[best_idx])
        best_idx = int(np.argmax(rec_t))
        return float(thresholds[best_idx]), float(rec_t[best_idx])
        
    elif metric == "precision_target":
        valid_indices = np.where(prec_t >= target_precision)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(rec_t[valid_indices])]
            return float(thresholds[best_idx]), float(prec_t[best_idx])
        best_idx = int(np.argmax(prec_t))
        return float(thresholds[best_idx]), float(prec_t[best_idx])
        
    elif metric == "youden_j":
        from sklearn.metrics import roc_curve
        fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        return float(roc_thresh[best_idx]), float(j_scores[best_idx])
    
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
