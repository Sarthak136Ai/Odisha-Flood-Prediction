"""
Comprehensive Confusion Matrix Evaluation and Visualization Engine for Odisha Flood Prediction.

Provides:
1. Complete Confusion Matrix decomposition: TN, FP, FN, TP, Total.
2. Core Classification & Operational Metrics:
   - Accuracy, Precision, Recall (Sensitivity / TPR), Specificity (TNR)
   - F1 Score, F2 Score (Disaster-Averse, Recall-Weighted)
   - False Negative Rate (FNR = Miss Rate = FN / (TP + FN))
   - False Positive Rate (FPR = Fall-Out = FP / (TN + FP))
   - Negative Predictive Value (NPV), Positive Predictive Value (PPV)
3. Disaster Risk Reduction Focus:
   - EXPLICIT HIGHLIGHTING of False Negatives (FN) representing unwarned catastrophic flood events.
4. Multi-format artifact generation:
   - Individual Confusion Matrix plots (PNG & SVG)
   - Multi-model comparative confusion matrix grid (PNG & SVG)
   - Formatted CSV and JSON evaluation summaries.
5. Dynamic model discovery and evaluation on the untouched 2022–2024 test set.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.5,
    split_name: str = "Test Set (2022–2024)"
) -> Dict[str, Any]:
    """
    Calculate full confusion matrix metrics with emphasis on False Negatives (missed floods).
    
    Parameters:
    -----------
    y_true: Ground truth binary labels (0 = No Flood, 1 = Flood).
    y_pred: Predicted binary labels (0 = No Flood, 1 = Flood).
    model_name: Identifier name of the model.
    threshold: Decision threshold used to generate y_pred.
    split_name: Dataset split label.
    
    Returns:
    --------
    Dictionary containing all confusion matrix components and operational rates.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = int(tn + fp + fn + tp)
    actual_positives = int(tp + fn)
    actual_negatives = int(tn + fp)
    predicted_positives = int(tp + fp)
    predicted_negatives = int(tn + fn)
    
    # Primary Metrics
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    
    # F-Beta Scores
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    # F2: beta=2.0 weights recall 2x over precision
    f2_denom = (4 * precision) + recall
    f2 = float(5 * precision * recall / f2_denom) if f2_denom > 0 else 0.0
    
    # Critical Operational Disaster Error Rates
    # FNR = Miss Rate (False Negatives / Total Floods) -> Extreme life & property damage risk
    fnr = float(fn / actual_positives) if actual_positives > 0 else 0.0
    # FPR = Fall-out (False Positives / Total Non-Floods) -> False alarm fatigue & resource deployment cost
    fpr = float(fp / actual_negatives) if actual_negatives > 0 else 0.0
    
    prevalence = float(actual_positives / total) if total > 0 else 0.0
    
    return {
        "model_name": model_name,
        "split": split_name,
        "threshold": round(float(threshold), 4),
        "total_samples": total,
        "true_negatives_tn": int(tn),
        "false_positives_fp": int(fp),
        "false_negatives_fn": int(fn),
        "true_positives_tp": int(tp),
        "actual_floods": actual_positives,
        "actual_non_floods": actual_negatives,
        "predicted_floods": predicted_positives,
        "predicted_non_floods": predicted_negatives,
        "prevalence": round(prevalence, 5),
        "accuracy": round(accuracy, 5),
        "precision": round(precision, 5),
        "recall_sensitivity": round(recall, 5),
        "specificity": round(specificity, 5),
        "f1_score": round(f1, 5),
        "f2_score": round(f2, 5),
        "false_negative_rate_fnr": round(fnr, 5),
        "false_positive_rate_fpr": round(fpr, 5),
        "negative_predictive_value_npv": round(npv, 5),
        "missed_floods_percentage": round(fnr * 100.0, 3),
        "false_alarms_percentage": round(fpr * 100.0, 3)
    }


def plot_styled_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.5,
    save_path_png: Optional[str] = None,
    save_path_svg: Optional[str] = None,
    labels: list = ["No Flood (0)", "Flood (1)"]
) -> plt.Figure:
    """
    Generate an individual styled confusion matrix with distinct highlighting
    for False Negatives (Missed Floods) and False Positives (False Alarms).
    """
    metrics = calculate_confusion_matrix_metrics(y_true, y_pred, model_name=model_name, threshold=threshold)
    tn = metrics["true_negatives_tn"]
    fp = metrics["false_positives_fp"]
    fn = metrics["false_negatives_fn"]
    tp = metrics["true_positives_tp"]
    total = metrics["total_samples"]
    
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    
    for spine in ax.spines.values():
        spine.set_color("#334155")
        
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    
    # Custom colored cell annotations
    # TN: Slate/Navy, FP: Amber/Orange, FN: Crimson Red (Danger), TP: Emerald Green (Success)
    cell_colors = [
        ["#1e3a8a", "#9a3412"],  # TN, FP
        ["#7f1d1d", "#064e3b"]   # FN, TP
    ]
    
    for r in range(2):
        for c in range(2):
            val = cm_matrix[r][c]
            pct = (val / total) * 100
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=cell_colors[r][c], edgecolor="#475569", lw=1.5)
            ax.add_patch(rect)
            
            # Label typography
            if r == 0 and c == 0:
                header = "TRUE NEGATIVE (TN)"
                desc = "Correct Dry/Normal"
                sub_color = "#93c5fd"
            elif r == 0 and c == 1:
                header = "FALSE POSITIVE (FP)"
                desc = "False Alarm (Precautionary Cost)"
                sub_color = "#fdba74"
            elif r == 1 and c == 0:
                header = "⚠️ FALSE NEGATIVE (FN) ⚠️"
                desc = "MISSED FLOOD (CATASTROPHIC)"
                sub_color = "#fca5a5"
            else:
                header = "TRUE POSITIVE (TP)"
                desc = "Caught Flood (Life Saved)"
                sub_color = "#86efac"
                
            ax.text(c, r - 0.18, header, ha="center", va="center", color="#ffffff", fontsize=10.5, fontweight="bold")
            ax.text(c, r + 0.02, f"{val:,}", ha="center", va="center", color="#ffffff", fontsize=16, fontweight="heavy")
            ax.text(c, r + 0.18, f"({pct:.2f}% of test records)\n{desc}", ha="center", va="center", color=sub_color, fontsize=8.5, style="italic")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred: {labels[0]}", f"Pred: {labels[1]}"], color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_yticklabels([f"True: {labels[0]}", f"True: {labels[1]}"], color="#f8fafc", fontsize=11, fontweight="bold")
    
    ax.set_xlabel("Predicted Emergency Decision", color="#94a3b8", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Ground Truth Inundation", color="#94a3b8", fontsize=12, labelpad=10)
    
    # Title & Subtitle banner
    title_text = f"{model_name} — Confusion Matrix (Untouched Test Set 2022–2024)"
    subtitle_text = (
        f"Threshold T* = {threshold:.4f} | Recall: {metrics['recall_sensitivity']*100:.1f}% | "
        f"Precision: {metrics['precision']*100:.1f}% | F1: {metrics['f1_score']:.4f} | "
        f"Miss Rate (FNR): {metrics['false_negative_rate_fnr']*100:.2f}%"
    )
    plt.title(f"{title_text}\n{subtitle_text}", color="#ffffff", fontsize=12, fontweight="bold", pad=16)
    
    plt.tight_layout()
    
    if save_path_png:
        os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PNG confusion matrix plot to {save_path_png}")
        
    if save_path_svg:
        os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)
        plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
        logger.info(f"Saved SVG confusion matrix plot to {save_path_svg}")
        
    return fig


def plot_multi_model_confusion_matrix_grid(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path_png: Optional[str] = "reports/plots/confusion_matrices_all_models.png",
    save_path_svg: Optional[str] = "reports/plots/confusion_matrices_all_models.svg",
    labels: list = ["No Flood", "Flood"]
) -> plt.Figure:
    """
    Generate a combined multi-panel comparison grid of confusion matrices across all models.
    
    Parameters:
    -----------
    models_predictions: Dict mapping model_name -> (y_true, y_pred, threshold)
    """
    n_models = len(models_predictions)
    if n_models == 0:
        return None
        
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 5.2 * rows), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    
    if n_models == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
        
    for idx, (model_name, (y_true, y_pred, threshold)) in enumerate(models_predictions.items()):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
        metrics = calculate_confusion_matrix_metrics(y_true, y_pred, model_name=model_name, threshold=threshold)
        tn, fp, fn, tp = metrics["true_negatives_tn"], metrics["false_positives_fp"], metrics["false_negatives_fn"], metrics["true_positives_tp"]
        total = metrics["total_samples"]
        
        cm_matrix = np.array([[tn, fp], [fn, tp]])
        cell_colors = [["#1e3a8a", "#9a3412"], ["#7f1d1d", "#064e3b"]]
        
        for i in range(2):
            for j in range(2):
                val = cm_matrix[i][j]
                pct = (val / total) * 100
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cell_colors[i][j], edgecolor="#475569", lw=1.2)
                ax.add_patch(rect)
                
                tag = "TN" if (i == 0 and j == 0) else ("FP" if (i == 0 and j == 1) else ("FN ⚠️" if (i == 1 and j == 0) else "TP"))
                sub = "Missed" if (i == 1 and j == 0) else ("Alarm" if (i == 0 and j == 1) else ("Caught" if (i == 1 and j == 1) else "Dry"))
                
                ax.text(j, i - 0.14, f"{tag}: {val:,}", ha="center", va="center", color="#ffffff", fontsize=10, fontweight="bold")
                ax.text(j, i + 0.14, f"{pct:.1f}% ({sub})", ha="center", va="center", color="#cbd5e1", fontsize=8.5)
                
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], color="#cbd5e1", fontsize=9.5)
        ax.set_yticklabels(["True 0", "True 1"], color="#cbd5e1", fontsize=9.5)
        
        # Subplot Title
        ax.set_title(
            f"{model_name} (T={threshold:.3f})\nRecall={metrics['recall_sensitivity']*100:.1f}% | Prec={metrics['precision']*100:.1f}% | FNR={metrics['false_negative_rate_fnr']*100:.1f}%",
            color="#f8fafc", fontsize=10, fontweight="bold", pad=8
        )
        
    # Hide unused subplots
    for idx in range(n_models, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")
        
    plt.suptitle("Test Set (2022–2024) Confusion Matrices Comparison Across All Major Models",
                 color="#ffffff", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path_png:
        os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PNG multi-model confusion matrix grid to {save_path_png}")
        
    if save_path_svg:
        os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)
        plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
        logger.info(f"Saved SVG multi-model confusion matrix grid to {save_path_svg}")
        
    return fig


# Backward compatibility alias
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = "results/plots/confusion_matrix.png",
    labels: list = ["No Flood (0)", "Flood (1)"]
) -> plt.Figure:
    """Backward-compatible wrapper for plotting single confusion matrix."""
    return plot_styled_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        model_name=title,
        threshold=0.5,
        save_path_png=save_path,
        labels=labels
    )
