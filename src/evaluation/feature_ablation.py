"""
Feature Ablation Study Engine for Odisha Flood Prediction.

This module systematically evaluates the empirical contribution of distinct
hydrological, meteorological, and temporal feature groups on flood prediction performance.

Ablation Groups:
- Group A: Current Rainfall Only (2 features)
- Group B: Current Rainfall + Lag Rainfall (6 features)
- Group C: Current Rainfall + Lag + Rolling Features (19 features)
- Group D: Current Rainfall + Temporal/Seasonal Features (5 features)
- Group E: Full Baseline Feature Set (22 features)
- Group F: Full Feature Set + IMD Rainfall-Intensity Indicators (29 features)

Methodological Rigor:
- Identical chronological splits: Train (2001–2018), Val (2019–2021), Test (2022–2024).
- Zero data leakage: Preprocessing and scalers fitted strictly on training data.
- Validation-only threshold optimization: Optimal threshold T* selected using validation F1.
- Untouched test set: Test set strictly reserved for final out-of-sample evaluation.
- Fixed random seed: 42 for complete reproducibility.
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
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve
)

from src.features.rainfall_features import (
    compute_rainfall_intensity_features,
    RAINFALL_INTENSITY_COLUMNS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE GROUP REGISTRY FOR SYSTEMATIC ABLATION
# =============================================================================

# Core feature definitions
CURRENT_RAINFALL_COLS = ["Rainfall (mm)", "Rainfall_Missing"]

LAG_RAINFALL_COLS = [
    "Rainfall_Lag_1d",
    "Rainfall_Lag_2d",
    "Rainfall_Lag_3d",
    "Rainfall_Lag_7d"
]

ROLLING_RAINFALL_COLS = [
    "Rainfall_Prev_3d_Sum",
    "Rainfall_Prev_7d_Sum",
    "Rainfall_Prev_15d_Sum",
    "Rainfall_Prev_30d_Sum",
    "Rainfall_Prev_3d_Max",
    "Rainfall_Prev_7d_Max",
    "Rainfall_Prev_15d_Max",
    "Rainfall_Prev_30d_Max",
    "Rainy_Days_Prev_3d",
    "Rainy_Days_Prev_7d",
    "Rainy_Days_Prev_15d",
    "Rainy_Days_Prev_30d",
    "Consecutive_Rainy_Days_Before"
]

TEMPORAL_SEASONAL_COLS = [
    "Month_sin",
    "Month_cos",
    "Day_of_Week"
]

INTENSITY_COLS = [
    "Light_Rainfall",
    "Moderate_Rainfall",
    "Heavy_Rainfall",
    "Very_Heavy_Rainfall",
    "Extreme_Rainfall",
    "Heavy_Rain_Indicator",
    "Extreme_Rain_Indicator"
]


FEATURE_ABLATION_GROUPS = {
    "Experiment_A": {
        "name": "Current Rainfall Only",
        "features": CURRENT_RAINFALL_COLS,
        "description": "Examines isolated point-in-time daily precipitation without historical or seasonal context."
    },
    "Experiment_B": {
        "name": "Current + Lag Rainfall",
        "features": CURRENT_RAINFALL_COLS + LAG_RAINFALL_COLS,
        "description": "Evaluates short-term antecedent daily rainfall trajectory (lags 1d, 2d, 3d, 7d)."
    },
    "Experiment_C": {
        "name": "Current + Lag + Rolling Features",
        "features": CURRENT_RAINFALL_COLS + LAG_RAINFALL_COLS + ROLLING_RAINFALL_COLS,
        "description": "Evaluates cumulative catchment waterlogging, rolling sums (3d-30d), and antecedent wetness."
    },
    "Experiment_D": {
        "name": "Current Rainfall + Temporal/Seasonal",
        "features": CURRENT_RAINFALL_COLS + TEMPORAL_SEASONAL_COLS,
        "description": "Evaluates current rainfall modulated solely by calendar monsoon seasonality without lag history."
    },
    "Experiment_E": {
        "name": "Full Baseline Feature Set",
        "features": CURRENT_RAINFALL_COLS + LAG_RAINFALL_COLS + ROLLING_RAINFALL_COLS + TEMPORAL_SEASONAL_COLS,
        "description": "Standard 22-feature operational baseline combining meteorological, antecedent rolling, and seasonal cycles."
    },
    "Experiment_F": {
        "name": "Full Set + IMD Rainfall-Intensity",
        "features": (
            CURRENT_RAINFALL_COLS + LAG_RAINFALL_COLS + ROLLING_RAINFALL_COLS +
            TEMPORAL_SEASONAL_COLS + INTENSITY_COLS
        ),
        "description": "Full baseline augmented with statutory IMD discrete rainfall intensity bands and alert triggers."
    }
}


def compute_f2_score(precision: float, recall: float) -> float:
    """Calculate F2 score weighting Recall twice as heavily as Precision."""
    if (4 * precision + recall) <= 1e-10:
        return 0.0
    return float(5.0 * (precision * recall) / (4.0 * precision + recall))


def find_validation_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find optimal classification threshold using validation set only.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    prec_t = precisions[:-1]
    rec_t = recalls[:-1]

    if metric == "f1":
        denom = prec_t + rec_t + 1e-10
        f1_scores = 2.0 * (prec_t * rec_t) / denom
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx]), float(f1_scores[best_idx])
    elif metric == "f2":
        denom = (4.0 * prec_t) + rec_t + 1e-10
        f2_scores = 5.0 * (prec_t * rec_t) / denom
        best_idx = int(np.argmax(f2_scores))
        return float(thresholds[best_idx]), float(f2_scores[best_idx])
    return 0.5, 0.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """Calculate full suite of classification and calibration metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    # Probabilistic metrics
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = 0.0

    brier = float(brier_score_loss(y_true, y_proba))

    # Threshold-dependent metrics
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = compute_f2_score(prec, rec)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": round(threshold, 4),
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5),
        "f2": round(f2, 5),
        "brier_score": round(brier, 5),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


class FeatureAblationStudy:
    """
    Orchestrator for feature ablation experiment across multiple model architectures.
    """
    def __init__(
        self,
        groups: Optional[Dict[str, Dict[str, Any]]] = None,
        optimization_metric: str = "f1"
    ):
        self.groups = groups or FEATURE_ABLATION_GROUPS
        self.optimization_metric = optimization_metric

    def run_ablation(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        model_factory_dict: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Execute ablation study for all feature groups across all models.
        """
        results_rows = []

        # Ensure IMD intensity features exist across all partitions
        train_proc = compute_rainfall_intensity_features(train_df)
        val_proc = compute_rainfall_intensity_features(val_df)
        test_proc = compute_rainfall_intensity_features(test_df)

        y_train = train_proc[target_col].values.astype(int)
        y_val = val_proc[target_col].values.astype(int)
        y_test = test_proc[target_col].values.astype(int)

        for model_name, model_fn in model_factory_dict.items():
            logger.info(f"\n=======================================================")
            logger.info(f"Running Feature Ablation for Model: {model_name}")
            logger.info(f"=======================================================")

            for exp_key, group_meta in self.groups.items():
                group_name = group_meta["name"]
                features = group_meta["features"]
                num_feats = len(features)

                logger.info(f"  Training [{exp_key}] {group_name} ({num_feats} features)...")

                X_train = train_proc[features].copy()
                X_val = val_proc[features].copy()
                X_test = test_proc[features].copy()

                # Instantiate fresh model pipeline
                model = model_fn()
                model.fit(X_train, y_train)

                # Validation inference & threshold tuning
                val_proba = model.predict_proba(X_val)[:, 1]
                t_opt, _ = find_validation_optimal_threshold(
                    y_val, val_proba, metric=self.optimization_metric
                )

                val_metrics = evaluate_predictions(y_val, val_proba, threshold=t_opt)

                # Test evaluation using strictly validation-selected threshold
                test_proba = model.predict_proba(X_test)[:, 1]
                test_metrics = evaluate_predictions(y_test, test_proba, threshold=t_opt)

                row = {
                    "Model": model_name,
                    "Experiment_ID": exp_key,
                    "Group_Name": group_name,
                    "Num_Features": num_feats,
                    "Feature_List": ", ".join(features),
                    "Threshold_Opt": t_opt,
                    # Validation Metrics (used to select best group)
                    "Val_ROC_AUC": val_metrics["roc_auc"],
                    "Val_PR_AUC": val_metrics["pr_auc"],
                    "Val_F1": val_metrics["f1"],
                    "Val_F2": val_metrics["f2"],
                    "Val_Precision": val_metrics["precision"],
                    "Val_Recall": val_metrics["recall"],
                    "Val_Brier_Score": val_metrics["brier_score"],
                    # Test Metrics (strictly reserved for out-of-sample reporting)
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
                    "Test_FN": test_metrics["fn"]
                }
                results_rows.append(row)

                logger.info(
                    f"    Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f} "
                    f"--> Test F1: {test_metrics['f1']:.4f} | Test ROC-AUC: {test_metrics['roc_auc']:.4f}"
                )

        results_df = pd.DataFrame(results_rows)
        return results_df


def plot_feature_ablation_comparison(
    results_df: pd.DataFrame,
    save_path_png: Optional[str] = "reports/plots/feature_ablation_comparison.png",
    save_path_svg: Optional[str] = "reports/plots/feature_ablation_comparison.svg"
) -> plt.Figure:
    """
    Generate high-resolution 4-panel comparison visualization for feature ablation study.
    Panels:
    1. Validation PR-AUC vs Test PR-AUC progression across feature groups.
    2. Validation ROC-AUC vs Test ROC-AUC progression.
    3. F1 and F2 Score progression.
    4. Brier Calibration Score comparison (lower is better).
    """
    models = results_df["Model"].unique()
    num_models = len(models)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")

    group_order = list(FEATURE_ABLATION_GROUPS.keys())
    clean_labels = [
        "A: Rain Only\n(2 feats)",
        "B: + Lags\n(6 feats)",
        "C: + Rolling\n(19 feats)",
        "D: Rain + Season\n(5 feats)",
        "E: Full Base\n(22 feats)",
        "F: Full + IMD\n(29 feats)"
    ]

    palette = {
        "Logistic Regression": {"val": "#38bdf8", "test": "#0284c7"},
        "XGBoost": {"val": "#f59e0b", "test": "#d97706"},
        "Random Forest": {"val": "#10b981", "test": "#059669"}
    }

    # Panel 1: PR-AUC Comparison
    ax1 = axes[0, 0]
    for model_name in models:
        sub = results_df[results_df["Model"] == model_name].set_index("Experiment_ID").reindex(group_order)
        c = palette.get(model_name, {"val": "#38bdf8", "test": "#0284c7"})
        ax1.plot(range(len(group_order)), sub["Val_PR_AUC"], marker="o", lw=2.2, label=f"{model_name} (Val)", color=c["val"])
        ax1.plot(range(len(group_order)), sub["Test_PR_AUC"], marker="s", lw=2.0, linestyle="--", label=f"{model_name} (Test)", color=c["test"])

    ax1.set_xticks(range(len(group_order)))
    ax1.set_xticklabels(clean_labels, fontsize=8.5, color="#f8fafc")
    ax1.set_ylabel("PR-AUC (Precision-Recall Area)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax1.set_title("PR-AUC Across Feature Ablation Groups (Primary In-Class Metric)", color="#ffffff", fontsize=11.5, fontweight="bold", pad=10)
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8.5, loc="lower right")

    # Panel 2: ROC-AUC Comparison
    ax2 = axes[0, 1]
    for model_name in models:
        sub = results_df[results_df["Model"] == model_name].set_index("Experiment_ID").reindex(group_order)
        c = palette.get(model_name, {"val": "#38bdf8", "test": "#0284c7"})
        ax2.plot(range(len(group_order)), sub["Val_ROC_AUC"], marker="o", lw=2.2, label=f"{model_name} (Val)", color=c["val"])
        ax2.plot(range(len(group_order)), sub["Test_ROC_AUC"], marker="s", lw=2.0, linestyle="--", label=f"{model_name} (Test)", color=c["test"])

    ax2.set_xticks(range(len(group_order)))
    ax2.set_xticklabels(clean_labels, fontsize=8.5, color="#f8fafc")
    ax2.set_ylabel("ROC-AUC", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax2.set_title("ROC-AUC Progression Across Feature Groups", color="#ffffff", fontsize=11.5, fontweight="bold", pad=10)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8.5, loc="lower right")

    # Panel 3: Test F1 and F2 Score Comparison
    ax3 = axes[1, 0]
    x = np.arange(len(group_order))
    width = 0.35 if num_models <= 2 else 0.25
    for i, model_name in enumerate(models):
        sub = results_df[results_df["Model"] == model_name].set_index("Experiment_ID").reindex(group_order)
        c = palette.get(model_name, {"val": "#38bdf8", "test": "#0284c7"})
        pos = x + (i - (num_models - 1) / 2.0) * width
        bars = ax3.bar(pos, sub["Test_F1"], width, label=f"{model_name} Test F1", color=c["val"], alpha=0.9)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax3.text(b.get_x() + b.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", color="#ffffff", fontsize=7.5, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(clean_labels, fontsize=8.5, color="#f8fafc")
    ax3.set_ylabel("F1 Score", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax3.set_title("Test F1 Score by Ablation Group (Validation-Optimized Threshold)", color="#ffffff", fontsize=11.5, fontweight="bold", pad=10)
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8.5, loc="upper left")

    # Panel 4: Brier Score (Calibration Reliability, Lower is Better)
    ax4 = axes[1, 1]
    for model_name in models:
        sub = results_df[results_df["Model"] == model_name].set_index("Experiment_ID").reindex(group_order)
        c = palette.get(model_name, {"val": "#38bdf8", "test": "#0284c7"})
        ax4.plot(range(len(group_order)), sub["Test_Brier_Score"], marker="D", lw=2.2, label=f"{model_name} Brier", color=c["test"])

    ax4.set_xticks(range(len(group_order)))
    ax4.set_xticklabels(clean_labels, fontsize=8.5, color="#f8fafc")
    ax4.set_ylabel("Brier Score Loss (Lower is Better)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax4.set_title("Probability Calibration (Brier Score) Progression", color="#ffffff", fontsize=11.5, fontweight="bold", pad=10)
    ax4.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8.5, loc="upper right")

    plt.suptitle("Odisha Flood Prediction: Feature Ablation Study Benchmarks", color="#ffffff", fontsize=13.5, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path_png:
        os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PNG ablation plot to {save_path_png}")

    if save_path_svg:
        os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)
        plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
        logger.info(f"Saved SVG ablation plot to {save_path_svg}")

    return fig
