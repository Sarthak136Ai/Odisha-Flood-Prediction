"""
Multi-Horizon Flood Forecasting Evaluation Engine for Odisha.

Evaluates predictive capability across expanding forecast horizons:
- Horizon 1 (T+1): Next day (24-hour lead time)
- Horizon 2 (T+2): Two days ahead (48-hour lead time)
- Horizon 3 (T+3): Three days ahead (72-hour lead time)

Strict Methodological Principles:
1. Zero Temporal Leakage: For any forecast made at date T, features strictly use
   meteorological observations up to day T. No future rainfall is ever used.
2. Identical Splits: Train (2001–2018), Val (2019–2021), Test (2022–2024).
3. Validation-Only Threshold Optimization: Optimal decision threshold T* is tuned
   strictly on the Validation set and frozen for out-of-sample Test evaluation.
4. Comprehensive Metrics: ROC-AUC, PR-AUC, Precision, Recall, F1, F2, Brier Score.
5. Degradation Analysis: Quantifies performance decay as lead time increases.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features.multi_horizon import (
    generate_multi_horizon_targets,
    get_horizon_target_name
)
from src.evaluation.feature_ablation import (
    compute_f2_score,
    find_validation_optimal_threshold,
    evaluate_predictions
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MultiHorizonEvaluationEngine:
    """
    Orchestrates training and out-of-sample testing across multiple forecast horizons.
    """
    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        optimization_metric: str = "f1"
    ):
        self.horizons = horizons or [1, 2, 3]
        self.optimization_metric = optimization_metric

    def run_multi_horizon_evaluation(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        model_factories: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Evaluate models across all forecast horizons.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training partition (2001-2018)
        val_df : pd.DataFrame
            Validation partition (2019-2021)
        test_df : pd.DataFrame
            Untouched test partition (2022-2024)
        feature_cols : list of str
            Predictor features strictly observed at or prior to time T.
        model_factories : dict
            Mapping of model_name -> callable creating fresh model instance.

        Returns
        -------
        pd.DataFrame
            Detailed metrics across all horizons and models.
        """
        # Ensure multi-horizon targets exist in all partitions
        train_proc = generate_multi_horizon_targets(train_df, horizons=self.horizons)
        val_proc = generate_multi_horizon_targets(val_df, horizons=self.horizons)
        test_proc = generate_multi_horizon_targets(test_df, horizons=self.horizons)

        results_rows = []

        for model_name, model_fn in model_factories.items():
            logger.info("=================================================================")
            logger.info(f"Running Multi-Horizon Evaluation for: {model_name}")
            logger.info("=================================================================")

            base_metrics_val = None
            base_metrics_test = None

            for h in self.horizons:
                target_col = get_horizon_target_name(h)
                lead_time_str = f"T+{h} ({h*24}h ahead)"

                logger.info(f"Evaluating Horizon {lead_time_str} using target '{target_col}'...")

                # Filter valid non-null rows for this horizon
                tr_sub = train_proc.dropna(subset=[target_col]).copy()
                va_sub = val_proc.dropna(subset=[target_col]).copy()
                te_sub = test_proc.dropna(subset=[target_col]).copy()

                X_train = tr_sub[feature_cols].copy()
                y_train = tr_sub[target_col].values.astype(int)

                X_val = va_sub[feature_cols].copy()
                y_val = va_sub[target_col].values.astype(int)

                X_test = te_sub[feature_cols].copy()
                y_test = te_sub[target_col].values.astype(int)

                # Instantiate and fit fresh model
                # Check if model supports dynamic scale_pos_weight
                num_neg = int(np.sum(y_train == 0))
                num_pos = int(np.sum(y_train == 1))
                spw = float(num_neg / max(num_pos, 1))

                model = model_fn(spw) if callable(model_fn) else model_fn()
                model.fit(X_train, y_train)

                # Validation inference & validation-only threshold tuning
                val_proba = model.predict_proba(X_val)[:, 1]
                t_opt, _ = find_validation_optimal_threshold(
                    y_val, val_proba, metric=self.optimization_metric
                )

                val_metrics = evaluate_predictions(y_val, val_proba, threshold=t_opt)

                # Out-of-sample Test inference with frozen threshold
                test_proba = model.predict_proba(X_test)[:, 1]
                test_metrics = evaluate_predictions(y_test, test_proba, threshold=t_opt)

                # Track baseline (T+1) for degradation computation
                if h == 1:
                    base_metrics_val = val_metrics
                    base_metrics_test = test_metrics

                deg_pr_auc = test_metrics["pr_auc"] - base_metrics_test["pr_auc"]
                deg_roc_auc = test_metrics["roc_auc"] - base_metrics_test["roc_auc"]
                deg_f1 = test_metrics["f1"] - base_metrics_test["f1"]
                deg_f2 = test_metrics["f2"] - base_metrics_test["f2"]

                row = {
                    "Model": model_name,
                    "Horizon_Days": h,
                    "Horizon_Label": f"T+{h}",
                    "Lead_Time_Hours": h * 24,
                    "Target_Column": target_col,
                    "Threshold_Opt": t_opt,
                    # Validation Metrics
                    "Val_ROC_AUC": val_metrics["roc_auc"],
                    "Val_PR_AUC": val_metrics["pr_auc"],
                    "Val_F1": val_metrics["f1"],
                    "Val_F2": val_metrics["f2"],
                    "Val_Precision": val_metrics["precision"],
                    "Val_Recall": val_metrics["recall"],
                    "Val_Brier_Score": val_metrics["brier_score"],
                    # Test Metrics (Strictly out-of-sample)
                    "Test_ROC_AUC": test_metrics["roc_auc"],
                    "Test_PR_AUC": test_metrics["pr_auc"],
                    "Test_F1": test_metrics["f1"],
                    "Test_F2": test_metrics["f2"],
                    "Test_Precision": test_metrics["precision"],
                    "Test_Recall": test_metrics["recall"],
                    "Test_Brier_Score": test_metrics["brier_score"],
                    "Test_TP": test_metrics["tp"],
                    "Test_FP": test_metrics["fp"],
                    "Test_TN": test_metrics["tn"],
                    "Test_FN": test_metrics["fn"],
                    "Test_Prevalence_Pct": round(float(num_pos / len(y_train)) * 100, 3),
                    # Degradation relative to T+1
                    "Delta_Test_PR_AUC": round(deg_pr_auc, 5),
                    "Delta_Test_ROC_AUC": round(deg_roc_auc, 5),
                    "Delta_Test_F1": round(deg_f1, 5),
                    "Delta_Test_F2": round(deg_f2, 5)
                }
                results_rows.append(row)

                logger.info(
                    f"  [{lead_time_str}] Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
                    f"Test PR-AUC: {test_metrics['pr_auc']:.4f} | "
                    f"Test ROC-AUC: {test_metrics['roc_auc']:.4f} | "
                    f"Test F1: {test_metrics['f1']:.4f} (Delta F1: {deg_f1:+.4f})"
                )

        results_df = pd.DataFrame(results_rows)
        return results_df


def plot_multi_horizon_degradation(
    results_df: pd.DataFrame,
    save_path_png: Optional[str] = "reports/plots/multi_horizon_degradation.png",
    save_path_svg: Optional[str] = "reports/plots/multi_horizon_degradation.svg"
) -> plt.Figure:
    """
    Generate 4-panel publication visualization showing performance decay across forecast horizons.

    Panels:
    1. Test PR-AUC Degradation curve (Primary rare-event metric).
    2. Test ROC-AUC Degradation curve (Discrimination capacity).
    3. Test F1 & F2 Score Decay.
    4. Precision vs Recall Dynamic Trade-off as lead time expands.
    """
    models = results_df["Model"].unique()
    horizons = sorted(results_df["Horizon_Days"].unique())
    x_labels = [f"T+{h}\n({h*24}h)" for h in horizons]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")

    palette = {
        "Logistic Regression": "#38bdf8",
        "XGBoost": "#f59e0b",
        "Random Forest": "#10b981",
        "Decision Tree": "#ec4899"
    }

    # Panel 1: Test PR-AUC Degradation
    ax1 = axes[0, 0]
    for model_name in models:
        sub = results_df[results_df["Model"] == model_name].sort_values("Horizon_Days")
        c = palette.get(model_name, "#38bdf8")
        ax1.plot(horizons, sub["Test_PR_AUC"], marker="o", markersize=8, lw=2.5, label=model_name, color=c)
        for h, v in zip(sub["Horizon_Days"], sub["Test_PR_AUC"]):
            ax1.text(h, v + 0.005, f"{v:.3f}", ha="center", va="bottom", color="#ffffff", fontsize=8.5, fontweight="bold")

    ax1.set_xticks(horizons)
    ax1.set_xticklabels(x_labels, fontsize=9.5, color="#f8fafc")
    ax1.set_ylabel("PR-AUC (Precision-Recall Area)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax1.set_title("PR-AUC Degradation as Forecast Horizon Expands (Primary Metric)", color="#ffffff", fontsize=11.5, fontweight="bold", pad=10)
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=9, loc="upper right")

    # Panel 2: Test ROC-AUC Degradation
    ax2 = axes[0, 1]
    for model_name in models:
        sub = results_df[results_df["Model"] == model_name].sort_values("Horizon_Days")
        c = palette.get(model_name, "#38bdf8")
        ax2.plot(horizons, sub["Test_ROC_AUC"], marker="s", markersize=8, lw=2.5, label=model_name, color=c)
        for h, v in zip(sub["Horizon_Days"], sub["Test_ROC_AUC"]):
            ax2.text(h, v + 0.004, f"{v:.3f}", ha="center", va="bottom", color="#ffffff", fontsize=8.5, fontweight="bold")

    ax2.set_xticks(horizons)
    ax2.set_xticklabels(x_labels, fontsize=9.5, color="#f8fafc")
    ax2.set_ylabel("ROC-AUC", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax2.set_title("ROC-AUC Discrimination Decay Across Lead Times", color="#ffffff", fontsize=11.5, fontweight="bold", pad=10)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=9, loc="upper right")

    # Panel 3: F1 and F2 Score Progression
    ax3 = axes[1, 0]
    for model_name in models:
        sub = results_df[results_df["Model"] == model_name].sort_values("Horizon_Days")
        c = palette.get(model_name, "#38bdf8")
        ax3.plot(horizons, sub["Test_F1"], marker="D", markersize=7, lw=2.2, label=f"{model_name} (F1)", color=c)
        ax3.plot(horizons, sub["Test_F2"], marker="^", markersize=7, lw=2.0, linestyle="--", label=f"{model_name} (F2 Safety)", color=c, alpha=0.75)

    ax3.set_xticks(horizons)
    ax3.set_xticklabels(x_labels, fontsize=9.5, color="#f8fafc")
    ax3.set_ylabel("Balanced F1 / Recall-Weighted F2", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax3.set_title("Test F1 & F2 Disaster Safety Score Decay", color="#ffffff", fontsize=11.5, fontweight="bold", pad=10)
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8.5, loc="upper right")

    # Panel 4: Precision vs Recall Decay
    ax4 = axes[1, 1]
    for model_name in models:
        sub = results_df[results_df["Model"] == model_name].sort_values("Horizon_Days")
        c = palette.get(model_name, "#38bdf8")
        ax4.plot(horizons, sub["Test_Recall"], marker="o", markersize=7, lw=2.2, label=f"{model_name} Recall", color=c)
        ax4.plot(horizons, sub["Test_Precision"], marker="x", markersize=8, lw=2.0, linestyle=":", label=f"{model_name} Precision", color=c, alpha=0.8)

    ax4.set_xticks(horizons)
    ax4.set_xticklabels(x_labels, fontsize=9.5, color="#f8fafc")
    ax4.set_ylabel("Score Rate (0.0 – 1.0)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax4.set_title("Recall vs Precision Trade-off Over Forecast Horizons", color="#ffffff", fontsize=11.5, fontweight="bold", pad=10)
    ax4.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8.5, loc="upper right")

    plt.suptitle("Odisha Flood Prediction: Multi-Horizon Forecast Degradation Analysis (T+1 to T+3)", color="#ffffff", fontsize=13.5, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path_png:
        os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PNG multi-horizon plot to {save_path_png}")

    if save_path_svg:
        os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)
        plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
        logger.info(f"Saved SVG multi-horizon plot to {save_path_svg}")

    return fig
