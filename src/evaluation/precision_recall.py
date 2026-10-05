"""
Precision-Recall curve comparison plotting module.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from typing import Dict, Tuple, Optional


def plot_precision_recall_curves(
    models_probas: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Precision-Recall (PR) Curves",
    save_path: Optional[str] = "results/plots/precision_recall_curve.png"
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models.
    models_probas: Dict[model_name, (y_true, y_proba)]
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for idx, (name, (y_true, y_proba)) in enumerate(models_probas.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        color = colors[idx % len(colors)]
        ax.plot(
            recall, precision,
            label=f"{name} (PR-AUC = {pr_auc:.4f})",
            color=color,
            lw=2.2
        )
        
    # Baseline prevalence line
    first_y = list(models_probas.values())[0][0]
    baseline_prevalence = float(np.mean(first_y))
    ax.axhline(
        y=baseline_prevalence,
        color="grey",
        linestyle="--",
        label=f"Baseline Prevalence ({baseline_prevalence:.3f})"
    )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12, labelpad=8)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        
    return fig
