"""
ROC curve comparison plotting module.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import Dict, Tuple, Optional


def plot_roc_curves(
    models_probas: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Receiver Operating Characteristic (ROC) Curves",
    save_path: Optional[str] = "results/plots/roc_curve.png"
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    models_probas: Dict[model_name, (y_true, y_proba)]
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for idx, (name, (y_true, y_proba)) in enumerate(models_probas.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        ax.plot(
            fpr, tpr,
            label=f"{name} (AUC = {roc_auc:.4f})",
            color=color,
            lw=2.2
        )
        
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=8)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        
    return fig
