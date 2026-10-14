"""
Scientific evaluation and comparative benchmarking module for rainfall downscaling.
Computes MAE, RMSE, R², Bias, and Pearson correlation comparing:
  - Observed station rainfall (ground truth)
  - Original coarse regional rainfall (district spatial mean)
  - Statistical downscaled rainfall (model estimate)
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.evaluation.metrics import calculate_metrics


def evaluate_downscaling_quality(
    y_true_rf: np.ndarray,
    y_pred_rf: np.ndarray
) -> Dict[str, float]:
    """
    Calculate scientific meteorological downscaling accuracy metrics.
    
    Parameters
    ----------
    y_true_rf : np.ndarray
        Observed station rain gauge measurements (ground truth in mm).
    y_pred_rf : np.ndarray
        Predicted/downscaled or coarse estimated rainfall (in mm).
        
    Returns
    -------
    Dict[str, float]
        Dictionary with MAE, RMSE, R², Bias, and Pearson correlation.
    """
    y_true = np.asarray(y_true_rf, dtype=float)
    y_pred = np.asarray(y_pred_rf, dtype=float)
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # Handle constant arrays safely for Pearson correlation
    if np.std(y_true) < 1e-9 or np.std(y_pred) < 1e-9:
        corr = 0.0
    else:
        corr_val, _ = pearsonr(y_true, y_pred)
        corr = float(corr_val) if not np.isnan(corr_val) else 0.0
        
    bias = float(np.mean(y_pred - y_true))
    
    return {
        "mae_mm": round(mae, 4),
        "rmse_mm": round(rmse, 4),
        "r2_score": round(r2, 4),
        "bias_mm": round(bias, 4),
        "pearson_correlation": round(corr, 4)
    }


def compare_downscaling_benchmarks(
    y_observed: np.ndarray,
    y_coarse: np.ndarray,
    y_downscaled: np.ndarray,
    split_name: str = "Test"
) -> pd.DataFrame:
    """
    Directly compare Observed vs Coarse vs Downscaled rainfall across all 5 scientific metrics:
    MAE, RMSE, R², Bias, and Pearson correlation.
    
    Parameters
    ----------
    y_observed : np.ndarray
        Observed block station ground truth rainfall (mm).
    y_coarse : np.ndarray
        Original coarse district spatial mean rainfall (mm).
    y_downscaled : np.ndarray
        Statistical downscaling model predicted rainfall (mm).
    split_name : str
        Evaluation split identifier (e.g. 'Train (2001-2018)', 'Val (2019-2021)', 'Test (2022-2024)').
        
    Returns
    -------
    pd.DataFrame
        Structured comparative summary table.
    """
    m_coarse = evaluate_downscaling_quality(y_observed, y_coarse)
    m_down = evaluate_downscaling_quality(y_observed, y_downscaled)
    
    records = [
        {
            "Split": split_name,
            "Series": "Original Coarse Regional Rainfall (District Mean)",
            "MAE (mm)": m_coarse["mae_mm"],
            "RMSE (mm)": m_coarse["rmse_mm"],
            "R2": m_coarse["r2_score"],
            "Bias (mm)": m_coarse["bias_mm"],
            "Pearson_r": m_coarse["pearson_correlation"],
            "Samples": len(y_observed)
        },
        {
            "Split": split_name,
            "Series": "Statistical Downscaled Rainfall (Model)",
            "MAE (mm)": m_down["mae_mm"],
            "RMSE (mm)": m_down["rmse_mm"],
            "R2": m_down["r2_score"],
            "Bias (mm)": m_down["bias_mm"],
            "Pearson_r": m_down["pearson_correlation"],
            "Samples": len(y_observed)
        }
    ]
    return pd.DataFrame(records)


def evaluate_intensity_tiers(
    y_observed: np.ndarray,
    y_coarse: np.ndarray,
    y_downscaled: np.ndarray
) -> pd.DataFrame:
    """
    Calculate MAE, RMSE, and Bias broken down by standard IMD meteorological rainfall intensity categories.
    """
    y_obs = np.asarray(y_observed, dtype=float)
    y_crs = np.asarray(y_coarse, dtype=float)
    y_dwn = np.asarray(y_downscaled, dtype=float)
    
    tiers = [
        ("Dry / Trace (< 2.5 mm)", y_obs < 2.5),
        ("Moderate (2.5 - 64.4 mm)", (y_obs >= 2.5) & (y_obs < 64.5)),
        ("Heavy (64.5 - 115.5 mm)", (y_obs >= 64.5) & (y_obs < 115.6)),
        ("Very Heavy (115.6 - 204.4 mm)", (y_obs >= 115.6) & (y_obs < 204.5)),
        ("Extremely Heavy (>= 204.5 mm)", y_obs >= 204.5)
    ]
    
    rows = []
    total_n = len(y_obs)
    for name, mask in tiers:
        n = int(np.sum(mask))
        if n == 0:
            continue
        pct = (n / total_n) * 100.0
        
        m_c = evaluate_downscaling_quality(y_obs[mask], y_crs[mask])
        m_d = evaluate_downscaling_quality(y_obs[mask], y_dwn[mask])
        
        rows.append({
            "Rainfall_Intensity_Tier": name,
            "Sample_Count": n,
            "Sample_Pct": round(pct, 2),
            "Coarse_MAE_mm": m_c["mae_mm"],
            "Coarse_Bias_mm": m_c["bias_mm"],
            "Downscaled_MAE_mm": m_d["mae_mm"],
            "Downscaled_Bias_mm": m_d["bias_mm"]
        })
        
    return pd.DataFrame(rows)


def compare_flood_prediction_impact(
    flood_model,
    X_test_orig: pd.DataFrame,
    X_test_downscaled: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Directly compare flood early warning skill between Original vs Downscaled rainfall inputs.
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
    sample_dates: Optional[pd.Series] = None,
    save_path: Optional[str] = "results/downscaling/rainfall_comparison.png"
) -> plt.Figure:
    """
    Standard 2-panel scatter and time-series comparison (maintained for backward compatibility).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    sample_size = min(5000, len(y_true_rf))
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(y_true_rf), size=sample_size, replace=False)
    y_t_samp = y_true_rf[sample_indices]
    y_d_samp = y_downscaled_rf[sample_indices]
    
    axes[0].scatter(y_t_samp, y_d_samp, alpha=0.35, color="#0077b6", edgecolors="none", s=18)
    max_val = max(float(np.max(y_t_samp)), float(np.max(y_d_samp)), 10.0)
    axes[0].plot([0, max_val], [0, max_val], "r--", lw=1.5, label="1:1 Perfect Agreement")
    axes[0].set_title("Station Rainfall: Observed vs Downscaled", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Observed Station Rainfall (mm)", fontsize=11)
    axes[0].set_ylabel("Downscaled Station Rainfall (mm)", fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    axes[1].plot(range(60), y_t_samp[:60], label="Observed", color="#0077b6", lw=2)
    axes[1].plot(range(60), y_d_samp[:60], label="Downscaled", color="#e63946", lw=1.8, linestyle="--")
    axes[1].set_title("Sample 60-Day Temporal Profile", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Time Step (Days)", fontsize=11)
    axes[1].set_ylabel("Rainfall (mm)", fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        
    return fig


def plot_comprehensive_downscaling_audit(
    y_observed: np.ndarray,
    y_coarse: np.ndarray,
    y_downscaled: np.ndarray,
    save_path: Optional[str] = "results/downscaling/downscaling_audit_comparison.png",
    extra_save_paths: Optional[List[str]] = None
) -> plt.Figure:
    """
    Generate publication-grade 4-panel scientific comparison figure:
      Panel (a): Observed vs Coarse Regional Rainfall (Scatter with 1:1 line)
      Panel (b): Observed vs Downscaled Station Rainfall (Scatter with 1:1 line)
      Panel (c): 60-day Time-Series Comparison (Observed vs Coarse vs Downscaled)
      Panel (d): Bias Comparison across IMD Rainfall Intensity Categories
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    
    # Subsampling for scatter to ensure clean readability
    sample_size = min(6000, len(y_observed))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(y_observed), size=sample_size, replace=False)
    
    y_obs_s = y_observed[indices]
    y_crs_s = y_coarse[indices]
    y_dwn_s = y_downscaled[indices]
    
    max_val = max(float(np.max(y_obs_s)), float(np.max(y_crs_s)), float(np.max(y_dwn_s)), 10.0)
    
    # Panel (a): Observed vs Coarse
    m_coarse = evaluate_downscaling_quality(y_observed, y_coarse)
    axes[0, 0].scatter(y_obs_s, y_crs_s, alpha=0.25, color="#457b9d", edgecolors="none", s=16)
    axes[0, 0].plot([0, max_val], [0, max_val], "r--", lw=1.5, label="1:1 Perfect Agreement")
    axes[0, 0].set_title(
        f"(a) Observed Station vs Coarse District Mean\n"
        f"MAE={m_coarse['mae_mm']:.2f}mm | RMSE={m_coarse['rmse_mm']:.2f}mm | R²={m_coarse['r2_score']:.3f} | r={m_coarse['pearson_correlation']:.3f}",
        fontsize=11, fontweight="bold", pad=8
    )
    axes[0, 0].set_xlabel("Observed Station Rainfall (mm)", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Coarse District Rainfall (mm)", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlim(-2, max_val + 5)
    axes[0, 0].set_ylim(-2, max_val + 5)
    axes[0, 0].legend(loc="upper left", frameon=True)
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    
    # Panel (b): Observed vs Downscaled
    m_down = evaluate_downscaling_quality(y_observed, y_downscaled)
    axes[0, 1].scatter(y_obs_s, y_dwn_s, alpha=0.25, color="#2a9d8f", edgecolors="none", s=16)
    axes[0, 1].plot([0, max_val], [0, max_val], "r--", lw=1.5, label="1:1 Perfect Agreement")
    axes[0, 1].set_title(
        f"(b) Observed Station vs Downscaled Model\n"
        f"MAE={m_down['mae_mm']:.2f}mm | RMSE={m_down['rmse_mm']:.2f}mm | R²={m_down['r2_score']:.3f} | r={m_down['pearson_correlation']:.3f}",
        fontsize=11, fontweight="bold", pad=8
    )
    axes[0, 1].set_xlabel("Observed Station Rainfall (mm)", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("Downscaled Station Rainfall (mm)", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlim(-2, max_val + 5)
    axes[0, 1].set_ylim(-2, max_val + 5)
    axes[0, 1].legend(loc="upper left", frameon=True)
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    
    # Panel (c): 60-day Time-Series Comparison
    # Pick a contiguous 60-day window with active rainfall
    window_len = min(60, len(y_observed))
    axes[1, 0].plot(range(window_len), y_obs_s[:window_len], label="Observed Station", color="#1d3557", lw=2)
    axes[1, 0].plot(range(window_len), y_crs_s[:window_len], label="Coarse District Mean", color="#e76f51", lw=1.6, linestyle=":")
    axes[1, 0].plot(range(window_len), y_dwn_s[:window_len], label="Downscaled Model", color="#2a9d8f", lw=1.8, linestyle="--")
    axes[1, 0].set_title("(c) 60-Day Temporal Profile: Observed vs Coarse vs Downscaled", fontsize=11, fontweight="bold", pad=8)
    axes[1, 0].set_xlabel("Sample Step (Days)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Daily Rainfall (mm)", fontsize=10, fontweight="bold")
    axes[1, 0].legend(loc="upper right", frameon=True)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)
    
    # Panel (d): Bias Comparison across IMD Intensity Categories
    df_tiers = evaluate_intensity_tiers(y_observed, y_coarse, y_downscaled)
    categories = [
        "Dry (<2.5)", "Moderate\n(2.5-64.4)", "Heavy\n(64.5-115.5)",
        "Very Heavy\n(115.6-204.4)", "Extremely Heavy\n(>=204.5)"
    ]
    x_pos = np.arange(len(df_tiers))
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, df_tiers["Coarse_Bias_mm"], width, label="Coarse Regional Bias", color="#e76f51", alpha=0.85)
    axes[1, 1].bar(x_pos + width/2, df_tiers["Downscaled_Bias_mm"], width, label="Downscaled Model Bias", color="#2a9d8f", alpha=0.85)
    axes[1, 1].axhline(0, color="black", lw=1.0, linestyle="-")
    axes[1, 1].set_title("(d) Mean Bias across IMD Rainfall Intensity Categories", fontsize=11, fontweight="bold", pad=8)
    axes[1, 1].set_xlabel("Rainfall Intensity Category (mm/day)", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Mean Bias (mm)", fontsize=10, fontweight="bold")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(categories[:len(df_tiers)], fontsize=9)
    axes[1, 1].legend(loc="lower left", frameon=True)
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    
    plt.suptitle("Statistical Downscaling Scientific Audit: Observed vs Coarse vs Downscaled Rainfall", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    # Save targets
    save_targets = []
    if save_path:
        save_targets.append(save_path)
    if extra_save_paths:
        save_targets.extend(extra_save_paths)
        
    for path in save_targets:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        
    return fig
