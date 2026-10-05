"""
Probability Calibration and Reliability Analysis Module.
Computes calibration curves, Brier scores, and plots reliability diagrams.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Dict, Tuple, Optional, Any


def evaluate_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Calculate probability calibration statistics including Brier score and calibration curve bins.
    """
    brier = float(brier_score_loss(y_true, y_proba))
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    
    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(y_proba, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    
    ece = 0.0
    total_samples = len(y_true)
    for b in range(n_bins):
        mask = bin_assignments == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_proba[mask])
            bin_weight = np.sum(mask) / total_samples
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return {
        "brier_score": brier,
        "expected_calibration_error": float(ece),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist()
    }


def plot_reliability_diagrams(
    model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Reliability Diagram (Probability Calibration)",
    save_path: Optional[str] = "results/plots/calibration_curves.png",
    n_bins: int = 10
) -> plt.Figure:
    """
    Plot calibration curves / reliability diagrams for multiple models.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [3, 1]})
    
    # Reference perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated (Ideal)", alpha=0.7)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for i, (name, (y_true, y_proba)) in enumerate(model_predictions.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        brier = brier_score_loss(y_true, y_proba)
        color = colors[i % len(colors)]
        ax1.plot(prob_pred, prob_true, "s-", color=color, label=f"{name} (Brier: {brier:.4f})", linewidth=2, markersize=5)
        ax2.hist(y_proba, range=(0, 1), bins=n_bins, label=name, histtype="step", lw=1.5, color=color)
        
    ax1.set_ylabel("Empirical Flood Frequency (Observed)", fontsize=11, fontweight="bold")
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    ax2.set_xlabel("Mean Predicted Flood Probability", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    return fig
