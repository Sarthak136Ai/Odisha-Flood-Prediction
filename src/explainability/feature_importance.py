"""
Feature importance extraction and visualization module.
Calculates standardized coefficients for linear models and Gini/gain importance for tree models.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional


def extract_feature_importance(
    model,
    feature_names: List[str]
) -> pd.DataFrame:
    """Extract normalized feature importance from model object or pipeline."""
    # Check if pipeline
    clf = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    
    if hasattr(clf, "coef_"):
        # Linear model coefficients
        coefs = clf.coef_[0]
        # Use absolute magnitude for ranking, with signed value for directional effect
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(coefs),
            "Signed_Coefficient": coefs,
            "Type": "Linear Coefficient"
        })
    elif hasattr(clf, "feature_importances_"):
        # Tree model importance
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": clf.feature_importances_,
            "Signed_Coefficient": clf.feature_importances_,
            "Type": "Tree Gini / Gain"
        })
    else:
        raise ValueError("Model does not support coefficient or feature_importances_ extraction.")
        
    df_imp = df_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df_imp


def plot_feature_importance(
    df_imp: pd.DataFrame,
    title: str = "Feature Importance - Flood Prediction Model",
    top_n: int = 15,
    save_path: Optional[str] = "results/plots/feature_importance.png"
) -> plt.Figure:
    """Plot and save a styled bar chart of top N features."""
    top_df = df_imp.head(top_n).sort_values(by="Importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Color based on sign if linear, or gradient if tree
    colors = ["#1f77b4" if sc >= 0 else "#d62728" for sc in top_df["Signed_Coefficient"]]
    
    bars = ax.barh(top_df["Feature"], top_df["Importance"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance / Absolute Standardized Effect", fontsize=12, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    
    # Legend for positive vs negative effect
    if "Linear" in top_df["Type"].iloc[0]:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#1f77b4", label="Positive Driver (+ Risk)"),
            Patch(facecolor="#d62728", label="Negative Driver (- Risk)")
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True)
        
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        
    return fig
