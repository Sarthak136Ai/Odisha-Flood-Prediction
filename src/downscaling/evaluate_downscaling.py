"""
Evaluation and comparative benchmarking module for rainfall downscaling and flood forecasting impact.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple, Optional

from src.evaluation.metrics import calculate_metrics


def evaluate_downscaling_quality(
    y_true_rf: np.ndarray,
    y_downscaled_rf: np.ndarray
) -> Dict[str, float]:
    """Calculate meteorological downscaling accuracy metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true_rf, y_downscaled_rf)))
    mae = float(mean_absolute_error(y_true_rf, y_downscaled_rf))
    r2 = float(r2_score(y_true_rf, y_downscaled_rf))
    corr, _ = pearsonr(y_true_rf, y_downscaled_rf)
    bias = float(np.mean(y_downscaled_rf - y_true_rf))
    
    return {
        "rmse_mm": round(rmse, 4),
        "mae_mm": round(mae, 4),
        "r2_score": round(r2, 4),
        "pearson_correlation": round(float(corr), 4),
        "bias_mm": round(bias, 4)
    }


def compare_flood_prediction_impact(
    flood_model,
    X_test_orig: pd.DataFrame,
    X_test_downscaled: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Directly compare flood prediction skill between Original vs Downscaled rainfall inputs.
    """
    # 1. Baseline: Original Rainfall
    proba_orig = flood_model.predict_proba(X_test_orig)[:, 1]
    metrics_orig = calculate_metrics(y_test, proba_orig, threshold=threshold, prefix="")
    metrics_orig["Rainfall_Source"] = "Original Observed Rainfall"
    
    # 2. Downscaled Rainfall
    proba_down = flood_model.predict_proba(X_test_downscaled)[:, 1]
    metrics_down = calculate_metrics(y_test, proba_down, threshold=threshold, prefix="")
    metrics_down["Rainfall_Source"] = "Downscaled Rainfall"
    
    df_comp = pd.DataFrame([metrics_orig, metrics_down])
    return df_comp


def plot_downscaling_comparison(
    y_true_rf: np.ndarray,
    y_downscaled_rf: np.ndarray,
    sample_dates: pd.Series,
    save_path: Optional[str] = "results/downscaling/rainfall_comparison.png"
) -> plt.Figure:
    """Plot scatter and time-series comparison between original and downscaled rainfall."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Subplot 1: Density Scatter
    sample_indices = np.random.choice(len(y_true_rf), size=min(5000, len(y_true_rf)), replace=False)
    y_t_samp = y_true_rf[sample_indices]
    y_d_samp = y_downscaled_rf[sample_indices]
    
    axes[0].scatter(y_t_samp, y_d_samp, alpha=0.3, color="#1f77b4", edgecolors="none", s=15)
    max_val = max(float(np.max(y_t_samp)), float(np.max(y_d_samp)))
    axes[0].plot([0, max_val], [0, max_val], "r--", label="1:1 Perfect Agreement")
    axes[0].set_title("Station Rainfall: Observed vs Downscaled", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Observed Station Rainfall (mm)", fontsize=11)
    axes[0].set_ylabel("Downscaled Station Rainfall (mm)", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: 60-day Time-series snippet
    axes[1].plot(range(60), y_t_samp[:60], label="Observed", color="#1f77b4", lw=2)
    axes[1].plot(range(60), y_d_samp[:60], label="Downscaled", color="#ff7f0e", lw=1.8, linestyle="--")
    axes[1].set_title("Sample 60-Day Temporal Profile", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Time Step (Days)", fontsize=11)
    axes[1].set_ylabel("Rainfall (mm)", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        
    return fig
