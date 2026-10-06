"""
Threshold Selection and Optimization Module for Odisha Flood Prediction Models.

Provides mathematically rigorous, configurable threshold selection strategies
tailored for extreme class imbalance (~2.6% flood rate) and early-warning operations:
1. F1 Optimization: Balances precision and recall (harmonic mean).
2. F2 / F-beta Optimization: Weights recall twice as heavily as precision (disaster-averse).
3. Recall Target: Selects the threshold achieving Recall >= target (e.g., 0.80 or 0.90) with maximum precision.
4. Precision Target: Selects the threshold achieving Precision >= target with maximum recall.
5. Youden's J Statistic: Maximizes (Sensitivity + Specificity - 1).
6. Cost-Sensitive Utility: Minimizes C_FN * FN + C_FP * FP under asymmetrical flood disaster loss.

Guarantees:
- Optimization is performed STRICTLY on validation data.
- The held-out test set (2022–2024) is never touched during threshold selection.
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
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Configurable Decision Threshold Optimizer for Flood Early Warning Models.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def calculate_f_beta(precision: np.ndarray, recall: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Calculate generalized F-beta score array from precision and recall."""
        beta_sq = beta ** 2
        denom = (beta_sq * precision) + recall + 1e-12
        return (1 + beta_sq) * (precision * recall) / denom

    def optimize_threshold(
        self,
        y_val: np.ndarray,
        val_proba: np.ndarray,
        strategy: str = "f1",
        beta: float = 2.0,
        target_recall: float = 0.80,
        target_precision: float = 0.25,
        cost_fn_ratio: float = 5.0
    ) -> Dict[str, Any]:
        """
        Optimize decision threshold on validation data according to selected strategy.
        
        Parameters:
        -----------
        y_val: Ground truth binary labels for validation set.
        val_proba: Model predicted probabilities for positive class (validation set).
        strategy: Optimization objective:
            - 'f1': Standard harmonic mean of precision and recall.
            - 'f2': Recall-prioritized F-beta (beta=2.0).
            - 'f_beta': Configurable beta parameter.
            - 'recall_target': Highest threshold with Recall >= target_recall.
            - 'precision_target': Lowest threshold with Precision >= target_precision.
            - 'youden_j': Maximize (Recall + Specificity - 1) on ROC curve.
            - 'cost_sensitive': Minimize cost with FN penalized cost_fn_ratio times FP.
            - 'default_0.5': Standard 0.50 baseline threshold.
            
        Returns:
        --------
        Dict containing optimal threshold, optimization strategy, and full validation metrics.
        """
        y_val = np.asarray(y_val).astype(int)
        val_proba = np.asarray(val_proba).astype(float)
        
        # Calculate PR and ROC curves on validation data
        precisions, recalls, pr_thresholds = precision_recall_curve(y_val, val_proba)
        # precision_recall_curve appends 1 to precision and 0 to recall; match length with thresholds
        prec_t = precisions[:-1]
        rec_t = recalls[:-1]
        
        roc_auc = float(roc_auc_score(y_val, val_proba)) if len(np.unique(y_val)) > 1 else 0.0
        pr_auc = float(average_precision_score(y_val, val_proba)) if len(np.unique(y_val)) > 1 else 0.0
        
        opt_threshold = 0.5
        opt_metric_val = 0.0
        
        if strategy == "f1":
            f1_scores = self.calculate_f_beta(prec_t, rec_t, beta=1.0)
            best_idx = int(np.argmax(f1_scores))
            opt_threshold = float(pr_thresholds[best_idx])
            opt_metric_val = float(f1_scores[best_idx])
            
        elif strategy in ["f2", "f_beta"]:
            b = 2.0 if strategy == "f2" else beta
            fb_scores = self.calculate_f_beta(prec_t, rec_t, beta=b)
            best_idx = int(np.argmax(fb_scores))
            opt_threshold = float(pr_thresholds[best_idx])
            opt_metric_val = float(fb_scores[best_idx])
            
        elif strategy == "recall_target":
            # Satisfy Recall >= target_recall and maximize Precision
            valid_indices = np.where(rec_t >= target_recall)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmax(prec_t[valid_indices])]
                opt_threshold = float(pr_thresholds[best_idx])
                opt_metric_val = float(rec_t[best_idx])
            else:
                # Fallback to maximum recall
                best_idx = int(np.argmax(rec_t))
                opt_threshold = float(pr_thresholds[best_idx])
                opt_metric_val = float(rec_t[best_idx])
                
        elif strategy == "precision_target":
            # Satisfy Precision >= target_precision and maximize Recall
            valid_indices = np.where(prec_t >= target_precision)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmax(rec_t[valid_indices])]
                opt_threshold = float(pr_thresholds[best_idx])
                opt_metric_val = float(prec_t[best_idx])
            else:
                best_idx = int(np.argmax(prec_t))
                opt_threshold = float(pr_thresholds[best_idx])
                opt_metric_val = float(prec_t[best_idx])
                
        elif strategy == "youden_j":
            fpr, tpr, roc_thresh = roc_curve(y_val, val_proba)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            opt_threshold = float(roc_thresh[best_idx])
            opt_metric_val = float(j_scores[best_idx])
            
        elif strategy == "cost_sensitive":
            # Grid search candidate thresholds to minimize total cost
            candidate_thresholds = np.linspace(0.01, 0.99, 100)
            best_cost = float("inf")
            best_t = 0.5
            for t in candidate_thresholds:
                y_pred_t = (val_proba >= t).astype(int)
                cm = confusion_matrix(y_val, y_pred_t, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                cost = (cost_fn_ratio * fn) + (1.0 * fp)
                if cost < best_cost:
                    best_cost = cost
                    best_t = float(t)
            opt_threshold = best_t
            opt_metric_val = float(best_cost)
            
        elif strategy == "default_0.5":
            opt_threshold = 0.5
            y_pred_def = (val_proba >= 0.5).astype(int)
            opt_metric_val = float(f1_score(y_val, y_pred_def, zero_division=0))
            
        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")
            
        # Ensure threshold is within reasonable bounds (avoid exactly 0.0 or 1.0 extremes)
        opt_threshold = float(np.clip(opt_threshold, 0.0001, 0.9999))
        
        # Calculate full metrics at selected threshold
        y_pred = (val_proba >= opt_threshold).astype(int)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        precision = float(precision_score(y_val, y_pred, zero_division=0))
        recall = float(recall_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        f2 = float(self.calculate_f_beta(np.array([precision]), np.array([recall]), beta=2.0)[0])
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        accuracy = float(accuracy_score(y_val, y_pred))
        brier = float(brier_score_loss(y_val, val_proba))
        
        return {
            "strategy": strategy,
            "threshold": round(opt_threshold, 4),
            "metric_value": round(opt_metric_val, 4),
            "val_f1": round(f1, 4),
            "val_f2": round(f2, 4),
            "val_precision": round(precision, 4),
            "val_recall": round(recall, 4),
            "val_specificity": round(specificity, 4),
            "val_accuracy": round(accuracy, 4),
            "val_roc_auc": round(roc_auc, 4),
            "val_pr_auc": round(pr_auc, 4),
            "val_brier_score": round(brier, 4),
            "val_tp": int(tp),
            "val_fp": int(fp),
            "val_tn": int(tn),
            "val_fn": int(fn)
        }

    def evaluate_all_strategies(
        self,
        y_val: np.ndarray,
        val_proba: np.ndarray,
        model_name: str = "Model"
    ) -> pd.DataFrame:
        """
        Evaluate a comprehensive suite of threshold strategies on validation data.
        """
        strategies = [
            ("f1", 1.0, 0.80, 0.25, 5.0, "Balanced F1 (Default)"),
            ("f2", 2.0, 0.80, 0.25, 5.0, "Disaster-Averse F2 (Recall-Weighted)"),
            ("recall_target", 1.0, 0.80, 0.25, 5.0, "High Sensitivity (Recall >= 80%)"),
            ("recall_target", 1.0, 0.90, 0.25, 5.0, "Ultra-High Sensitivity (Recall >= 90%)"),
            ("youden_j", 1.0, 0.80, 0.25, 5.0, "Youden's J (ROC Optimal)"),
            ("cost_sensitive", 1.0, 0.80, 0.25, 5.0, "Cost-Sensitive (FN Loss = 5x FP)"),
            ("default_0.5", 1.0, 0.80, 0.25, 5.0, "Standard Baseline (0.50)")
        ]
        
        rows = []
        for strat, b, r_t, p_t, c_r, label in strategies:
            res = self.optimize_threshold(
                y_val=y_val,
                val_proba=val_proba,
                strategy=strat,
                beta=b,
                target_recall=r_t,
                target_precision=p_t,
                cost_fn_ratio=c_r
            )
            rows.append({
                "Model": model_name,
                "Strategy_Label": label,
                "Strategy_Key": strat,
                "Threshold": res["threshold"],
                "Val_F1": res["val_f1"],
                "Val_F2": res["val_f2"],
                "Val_Precision": res["val_precision"],
                "Val_Recall": res["val_recall"],
                "Val_Specificity": res["val_specificity"],
                "Val_Accuracy": res["val_accuracy"],
                "Val_PR_AUC": res["val_pr_auc"],
                "Val_ROC_AUC": res["val_roc_auc"],
                "Val_Brier_Score": res["val_brier_score"],
                "TP": res["val_tp"],
                "FP": res["val_fp"],
                "TN": res["val_tn"],
                "FN": res["val_fn"]
            })
            
        return pd.DataFrame(rows)

    def plot_threshold_tradeoffs(
        self,
        model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        save_path: str = "reports/plots/threshold_tradeoff_curves.png"
    ) -> str:
        """
        Generate multi-model Precision, Recall, F1, and F2 trade-off curves across all threshold levels.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        n_models = len(model_predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
        if n_models == 1:
            axes = [axes]
            
        fig.patch.set_facecolor("#0f172a")
        
        thresholds = np.linspace(0.01, 0.99, 150)
        
        for idx, (model_name, (y_val, val_proba)) in enumerate(model_predictions.items()):
            ax = axes[idx]
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#cbd5e1")
            ax.xaxis.label.set_color("#f8fafc")
            ax.yaxis.label.set_color("#f8fafc")
            ax.title.set_color("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color("#334155")
                
            p_list, r_list, f1_list, f2_list = [], [], [], []
            for t in thresholds:
                y_pred = (val_proba >= t).astype(int)
                p = precision_score(y_val, y_pred, zero_division=0)
                r = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                f2 = self.calculate_f_beta(np.array([p]), np.array([r]), beta=2.0)[0]
                p_list.append(p)
                r_list.append(r)
                f1_list.append(f1)
                f2_list.append(f2)
                
            # Find F1 and F2 optimal points
            best_f1_idx = int(np.argmax(f1_list))
            best_f2_idx = int(np.argmax(f2_list))
            
            ax.plot(thresholds, p_list, label="Precision", color="#38bdf8", linewidth=2.0)
            ax.plot(thresholds, r_list, label="Recall", color="#f59e0b", linewidth=2.0)
            ax.plot(thresholds, f1_list, label="F1 Score", color="#10b981", linewidth=2.5)
            ax.plot(thresholds, f2_list, label="F2 Score (Recall-biased)", color="#ec4899", linewidth=2.0, linestyle="--")
            
            # Vertical marker for optimal F1
            ax.axvline(thresholds[best_f1_idx], color="#10b981", linestyle=":", alpha=0.8,
                       label=f"Opt F1 (T={thresholds[best_f1_idx]:.2f})")
            
            ax.set_title(f"{model_name}", fontsize=12, fontweight="bold", pad=8)
            ax.set_xlabel("Decision Threshold", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Score", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
            ax.set_ylim(0, 1.02)
            
            if idx == len(model_predictions) - 1:
                ax.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8, loc="upper right")
                
        plt.suptitle("Validation Decision Threshold vs Performance Trade-offs", fontsize=14, fontweight="bold", color="#ffffff", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Threshold trade-off curves saved to {save_path}")
        return save_path
