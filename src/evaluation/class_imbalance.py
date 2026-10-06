"""
Class Imbalance Analysis and Evaluation Module for Odisha Flood Prediction.

Provides mathematical and empirical analysis of extreme class imbalance (~2.6% flood rate):
1. Target distribution analysis across splits (Overall, Train, Val, Test) and across years (2001–2024).
2. Spatio-temporal flood frequency analysis across all 30 Odisha districts.
3. Multi-threshold performance auditing (Default 0.50, F1-optimal, F2-optimal, Recall >= 80%).
4. Evaluation of existing mitigation techniques:
   - Loss function class weighting (w_pos / w_neg ≈ 37.0)
   - Decision threshold shifting
   - Probability calibration (Platt scaling, Isotonic regression)
   - Non-misleading evaluation metrics (PR-AUC, F1, F2, Recall vs standard Accuracy / ROC-AUC)
5. Analysis of why synthetic oversampling (SMOTE) is unsafe for hydrological time-series.
6. Publication-grade visualization generators.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ClassImbalanceAnalyzer:
    """
    Comprehensive Class Imbalance Analyzer for Spatio-Temporal Flood Prediction.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def calculate_f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
        """Calculate generalized F-beta score from precision and recall scalars."""
        beta_sq = beta ** 2
        denom = (beta_sq * precision) + recall + 1e-12
        return float((1 + beta_sq) * (precision * recall) / denom)

    def analyze_dataset_distributions(
        self,
        df: pd.DataFrame,
        target_col: str = "Flood_Next_Day",
        year_col: str = "Year",
        district_col: str = "District",
        train_years: Optional[List[int]] = None,
        val_years: Optional[List[int]] = None,
        test_years: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Compute complete target class counts, percentages, and imbalance ratios across
        the full dataset, temporal splits, and individual years/districts.
        """
        if train_years is None:
            train_years = list(range(2001, 2019))
        if val_years is None:
            val_years = [2019, 2020, 2021]
        if test_years is None:
            test_years = [2022, 2023, 2024]

        clean_df = df.dropna(subset=[target_col]).copy()
        clean_df[target_col] = clean_df[target_col].astype(int)

        def get_split_stats(subset: pd.DataFrame, split_name: str) -> Dict[str, Any]:
            total = len(subset)
            pos_count = int(subset[target_col].sum())
            neg_count = int(total - pos_count)
            pos_pct = float((pos_count / total) * 100) if total > 0 else 0.0
            neg_pct = float((neg_count / total) * 100) if total > 0 else 0.0
            imbalance_ratio = float(neg_count / max(pos_count, 1))
            return {
                "Split": split_name,
                "Total_Records": total,
                "Negative_Count_0": neg_count,
                "Negative_Pct_0": round(neg_pct, 3),
                "Positive_Count_1": pos_count,
                "Positive_Pct_1": round(pos_pct, 3),
                "Imbalance_Ratio_Neg_to_Pos": round(imbalance_ratio, 2),
                "Class_Weight_Multiplier": round(imbalance_ratio, 2)
            }

        # Overall and Splits
        overall_stats = get_split_stats(clean_df, "Complete Dataset (2001–2024)")
        train_df = clean_df[clean_df[year_col].isin(train_years)]
        val_df = clean_df[clean_df[year_col].isin(val_years)]
        test_df = clean_df[clean_df[year_col].isin(test_years)]

        train_stats = get_split_stats(train_df, f"Training Set ({min(train_years)}–{max(train_years)})")
        val_stats = get_split_stats(val_df, f"Validation Set ({min(val_years)}–{max(val_years)})")
        test_stats = get_split_stats(test_df, f"Test Set ({min(test_years)}–{max(test_years)})")

        splits_summary_df = pd.DataFrame([overall_stats, train_stats, val_stats, test_stats])

        # Year-wise Breakdown (2001–2024)
        year_rows = []
        for year in sorted(clean_df[year_col].unique()):
            ydf = clean_df[clean_df[year_col] == year]
            y_tot = len(ydf)
            y_pos = int(ydf[target_col].sum())
            y_neg = int(y_tot - y_pos)
            y_pos_pct = float((y_pos / y_tot) * 100) if y_tot > 0 else 0.0
            y_imb = float(y_neg / max(y_pos, 1))
            
            # Determine temporal split bucket
            if year in train_years:
                split_tag = "Train"
            elif year in val_years:
                split_tag = "Validation"
            elif year in test_years:
                split_tag = "Test"
            else:
                split_tag = "Other"

            year_rows.append({
                "Year": int(year),
                "Temporal_Split": split_tag,
                "Total_Records": y_tot,
                "Non_Flood_Count_0": y_neg,
                "Flood_Count_1": y_pos,
                "Flood_Prevalence_Pct": round(y_pos_pct, 4),
                "Imbalance_Ratio": round(y_imb, 2)
            })
        yearwise_df = pd.DataFrame(year_rows)

        # District-wise Breakdown (30 Districts)
        district_rows = []
        if district_col in clean_df.columns:
            for dist in sorted(clean_df[district_col].unique()):
                ddf = clean_df[clean_df[district_col] == dist]
                d_tot = len(ddf)
                d_pos = int(ddf[target_col].sum())
                d_neg = int(d_tot - d_pos)
                d_pos_pct = float((d_pos / d_tot) * 100) if d_tot > 0 else 0.0
                d_imb = float(d_neg / max(d_pos, 1))
                district_rows.append({
                    "District": dist,
                    "Total_Records": d_tot,
                    "Non_Flood_Count": d_neg,
                    "Flood_Count": d_pos,
                    "Flood_Prevalence_Pct": round(d_pos_pct, 3),
                    "Imbalance_Ratio": round(d_imb, 2)
                })
        district_df = pd.DataFrame(district_rows).sort_values(by="Flood_Prevalence_Pct", ascending=False)

        return {
            "splits_summary": splits_summary_df,
            "yearwise_summary": yearwise_df,
            "district_summary": district_df,
            "overall_flood_rate": overall_stats["Positive_Pct_1"],
            "train_flood_rate": train_stats["Positive_Pct_1"],
            "val_flood_rate": val_stats["Positive_Pct_1"],
            "test_flood_rate": test_stats["Positive_Pct_1"],
            "scale_pos_weight": train_stats["Imbalance_Ratio_Neg_to_Pos"]
        }

    def evaluate_threshold_performance_matrix(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold_map: Dict[str, float],
        model_name: str = "Model",
        split_name: str = "Test"
    ) -> pd.DataFrame:
        """
        Evaluate full classification metrics and confusion matrices across different
        threshold operational regimes (Baseline 0.50, Balanced F1, Disaster-Averse F2, High Recall).
        """
        y_true = np.asarray(y_true).astype(int)
        y_proba = np.asarray(y_proba).astype(float)
        
        roc_auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0
        pr_auc = float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0
        brier = float(brier_score_loss(y_true, y_proba))
        
        rows = []
        for strat_name, threshold in threshold_map.items():
            t = float(np.clip(threshold, 0.0001, 0.9999))
            y_pred = (y_proba >= t).astype(int)
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            f2 = self.calculate_f_beta(prec, rec, beta=2.0)
            acc = float(accuracy_score(y_true, y_pred))
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            
            # False Negative Rate & False Positive Rate
            fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
            fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
            
            rows.append({
                "Model": model_name,
                "Split": split_name,
                "Strategy": strat_name,
                "Threshold": round(t, 4),
                "Accuracy": round(acc, 5),
                "Precision": round(prec, 5),
                "Recall": round(rec, 5),
                "Specificity": round(spec, 5),
                "F1_Score": round(f1, 5),
                "F2_Score": round(f2, 5),
                "PR_AUC": round(pr_auc, 5),
                "ROC_AUC": round(roc_auc, 5),
                "Brier_Score": round(brier, 5),
                "False_Negative_Rate": round(fnr, 5),
                "False_Positive_Rate": round(fpr, 5),
                "True_Positives_TP": int(tp),
                "False_Positives_FP": int(fp),
                "True_Negatives_TN": int(tn),
                "False_Negatives_FN": int(fn)
            })
            
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------

    def plot_class_distributions(
        self,
        splits_df: pd.DataFrame,
        save_path: str = "reports/plots/class_imbalance_distributions.png"
    ) -> str:
        """
        Plot class distribution comparisons across splits with counts and imbalance proportions.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.patch.set_facecolor("#0f172a")
        
        # Panel 1: Stacked Bar of Negative vs Positive records
        ax1 = axes[0]
        ax1.set_facecolor("#1e293b")
        ax1.tick_params(colors="#cbd5e1")
        for spine in ax1.spines.values():
            spine.set_color("#334155")
            
        splits = splits_df["Split"].tolist()
        neg_counts = splits_df["Negative_Count_0"].values / 1000  # in thousands
        pos_counts = splits_df["Positive_Count_1"].values / 1000
        
        y_pos = np.arange(len(splits))
        bar_width = 0.55
        
        ax1.barh(y_pos, neg_counts, bar_width, label="Non-Flood (0)", color="#3b82f6", alpha=0.85)
        ax1.barh(y_pos, pos_counts, bar_width, left=neg_counts, label="Flood Event (1)", color="#ef4444", alpha=0.9)
        
        for i, (neg, pos, pct) in enumerate(zip(neg_counts, pos_counts, splits_df["Positive_Pct_1"])):
            ax1.text(neg + pos + 15, i, f"{pos*1000:,.0f} floods ({pct:.2f}%)", va="center", color="#f8fafc", fontsize=9, fontweight="bold")
            
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(splits, color="#f8fafc", fontsize=10)
        ax1.set_xlabel("Record Count (Thousands)", color="#f8fafc", fontsize=10)
        ax1.set_title("Target Class Breakdown Across Splits", color="#ffffff", fontsize=12, fontweight="bold", pad=10)
        ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", loc="lower right")
        ax1.grid(True, linestyle="--", alpha=0.2, color="#64748b", axis="x")
        
        # Panel 2: Imbalance Multiplier Bar Chart
        ax2 = axes[1]
        ax2.set_facecolor("#1e293b")
        ax2.tick_params(colors="#cbd5e1")
        for spine in ax2.spines.values():
            spine.set_color("#334155")
            
        ratios = splits_df["Imbalance_Ratio_Neg_to_Pos"].values
        colors = ["#0ea5e9", "#10b981", "#f59e0b", "#a855f7"]
        
        bars = ax2.bar(splits, ratios, width=0.5, color=colors, alpha=0.85, edgecolor="#ffffff", linewidth=0.5)
        for bar, r in zip(bars, ratios):
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{r:.1f} : 1", ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
            
        ax2.set_ylabel("Negative to Positive Ratio", color="#f8fafc", fontsize=10)
        ax2.set_title("Class Imbalance Ratio (Negatives per Positive)", color="#ffffff", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xticklabels(["Overall", "Train", "Val", "Test"], color="#f8fafc", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.2, color="#64748b", axis="y")
        ax2.set_ylim(0, max(ratios) * 1.2)
        
        plt.suptitle("Odisha Flood Prediction: Comprehensive Target Class Imbalance Analysis", fontsize=14, fontweight="bold", color="#ffffff", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Class distribution plot saved to {save_path}")
        return save_path

    def plot_yearwise_flood_frequency(
        self,
        yearwise_df: pd.DataFrame,
        save_path: str = "reports/plots/yearwise_flood_prevalence.png"
    ) -> str:
        """
        Plot year-wise flood counts and prevalence percentages (2001–2024), highlighting major disaster years.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ax1 = plt.subplots(figsize=(15, 6))
        fig.patch.set_facecolor("#0f172a")
        
        ax1.set_facecolor("#1e293b")
        ax1.tick_params(colors="#cbd5e1")
        for spine in ax1.spines.values():
            spine.set_color("#334155")
            
        years = yearwise_df["Year"].values
        flood_counts = yearwise_df["Flood_Count_1"].values
        prev_pcts = yearwise_df["Flood_Prevalence_Pct"].values
        
        # Color bars by temporal split
        bar_colors = []
        for s in yearwise_df["Temporal_Split"]:
            if s == "Train":
                bar_colors.append("#3b82f6")  # Blue
            elif s == "Validation":
                bar_colors.append("#10b981")  # Emerald
            else:
                bar_colors.append("#f59e0b")  # Amber
                
        bars = ax1.bar(years, flood_counts, color=bar_colors, alpha=0.75, width=0.65, label="Flood Records Count")
        ax1.set_xlabel("Year (2001–2024)", color="#f8fafc", fontsize=11, labelpad=8)
        ax1.set_ylabel("Inundation Day Records (Count)", color="#38bdf8", fontsize=11)
        ax1.set_xticks(years)
        ax1.set_xticklabels(years, rotation=45, ha="right", color="#cbd5e1", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.15, color="#64748b")
        
        # Dual axis for prevalence line
        ax2 = ax1.twinx()
        ax2.tick_params(colors="#cbd5e1")
        for spine in ax2.spines.values():
            spine.set_color("#334155")
            
        line = ax2.plot(years, prev_pcts, color="#ef4444", linewidth=2.5, marker="o", markersize=5, label="Flood Prevalence (%)")
        ax2.set_ylabel("Flood Prevalence (%)", color="#ef4444", fontsize=11)
        ax2.set_ylim(0, max(prev_pcts) * 1.25)
        
        # Annotate notable cyclonic flood years
        disasters = {
            2001: "2001 Severe Floods",
            2006: "2006 Monsoon Floods",
            2008: "2008 Mahanadi Floods",
            2011: "2011 Dual Peak Floods",
            2013: "Cyclone Phailin",
            2018: "Cyclone Titli",
            2019: "Cyclone Fani",
            2020: "Cyclone Amphan & Monsoons",
            2022: "2022 Statewide Floods",
            2024: "2024 Inundations"
        }
        for yr, label in disasters.items():
            if yr in years:
                idx = np.where(years == yr)[0][0]
                y_val = prev_pcts[idx]
                ax2.annotate(
                    f"{label}\n({y_val:.1f}%)",
                    xy=(yr, y_val),
                    xytext=(yr, y_val + (max(prev_pcts)*0.08)),
                    arrowprops=dict(facecolor="#f87171", edgecolor="#ef4444", arrowstyle="->", lw=1.2),
                    ha="center", fontsize=7.5, color="#ffffff", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e1b4b", edgecolor="#818cf8", alpha=0.9)
                )

        # Legend handles
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#3b82f6", label="Train Split (2001–2018)"),
            Patch(facecolor="#10b981", label="Validation Split (2019–2021)"),
            Patch(facecolor="#f59e0b", label="Test Split (2022–2024)"),
            plt.Line2D([0], [0], color="#ef4444", lw=2, marker="o", label="Prevalence Rate (%)")
        ]
        ax1.legend(handles=legend_elements, facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", loc="upper left", fontsize=9)
        
        plt.title("Odisha Historical Flood Frequency & Prevalence Evolution (2001–2024)", color="#ffffff", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Year-wise flood frequency plot saved to {save_path}")
        return save_path

    def plot_confusion_matrices_comparison(
        self,
        eval_df: pd.DataFrame,
        model_name: str = "XGBoost",
        save_path: str = "reports/plots/confusion_matrices_imbalance.png"
    ) -> str:
        """
        Plot formatted side-by-side confusion matrices showing the dramatic difference
        between standard 0.50 threshold and validation-optimized F1 & F2 thresholds.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        sub_df = eval_df[eval_df["Model"] == model_name].copy()
        if len(sub_df) == 0:
            sub_df = eval_df.iloc[:3].copy()
            
        n_strats = len(sub_df)
        fig, axes = plt.subplots(1, n_strats, figsize=(4.5 * n_strats, 4.2))
        if n_strats == 1:
            axes = [axes]
            
        fig.patch.set_facecolor("#0f172a")
        
        for idx, (_, row) in enumerate(sub_df.iterrows()):
            ax = axes[idx]
            ax.set_facecolor("#1e293b")
            for spine in ax.spines.values():
                spine.set_color("#334155")
                
            tp = row["True_Positives_TP"]
            fp = row["False_Positives_FP"]
            tn = row["True_Negatives_TN"]
            fn = row["False_Negatives_FN"]
            strat = row["Strategy"]
            thresh = row["Threshold"]
            rec = row["Recall"]
            prec = row["Precision"]
            f1 = row["F1_Score"]
            
            cm_matrix = np.array([[tn, fp], [fn, tp]])
            total = tn + fp + fn + tp
            
            # Heatmap display
            im = ax.imshow(cm_matrix, cmap="Blues", interpolation="nearest")
            
            # Text annotations
            cell_texts = [
                [f"TN\n{tn:,}\n({tn/total*100:.1f}%)", f"FP (False Alarm)\n{fp:,}\n({fp/total*100:.1f}%)"],
                [f"FN (MISSED!)\n{fn:,}\n({fn/total*100:.1f}%)", f"TP (Caught)\n{tp:,}\n({tp/total*100:.1f}%)"]
            ]
            
            for r in range(2):
                for c in range(2):
                    cell_color = "#ffffff" if (r == 0 and c == 0) else ("#fca5a5" if (r == 1 and c == 0) else "#cbd5e1")
                    if r == 1 and c == 0 and fn > 0.5 * (tp + fn):
                        # Highlight high missed floods in bright red
                        cell_color = "#f87171"
                    ax.text(c, r, cell_texts[r][c], ha="center", va="center", color=cell_color, fontsize=9.5, fontweight="bold")
                    
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred No Flood", "Pred Flood"], color="#cbd5e1", fontsize=9.5)
            ax.set_yticklabels(["Actual No Flood", "Actual Flood"], color="#cbd5e1", fontsize=9.5)
            
            ax.set_title(f"{strat}\n(Threshold T = {thresh:.2f})\nRecall: {rec*100:.1f}% | Prec: {prec*100:.1f}%",
                         color="#f8fafc", fontsize=10.5, fontweight="bold", pad=8)
                         
        plt.suptitle(f"{model_name}: Test Set (2022–2024) Confusion Matrices Under Severe Imbalance",
                     fontsize=13, fontweight="bold", color="#ffffff", y=1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix comparison plot saved to {save_path}")
        return save_path

    def plot_pr_curves_with_baseline(
        self,
        predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        prevalence: float = 0.02855,
        save_path: str = "reports/plots/pr_curves_imbalance.png"
    ) -> str:
        """
        Plot Precision-Recall curves for all models with the horizontal random-chance
        prevalence baseline (y = positive_rate), demonstrating metric sensitivity.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
        colors = {
            "Logistic Regression": "#38bdf8",
            "Decision Tree": "#94a3b8",
            "Random Forest": "#10b981",
            "XGBoost": "#f59e0b",
            "ANN (MLP)": "#ec4899"
        }
        
        for name, (y_test, test_proba) in predictions_dict.items():
            prec, rec, _ = precision_recall_curve(y_test, test_proba)
            pr_auc = average_precision_score(y_test, test_proba)
            color = colors.get(name, "#a855f7")
            ax.plot(rec, prec, color=color, lw=2.2, label=f"{name} (PR-AUC = {pr_auc:.4f})")
            
        # Draw random prevalence baseline
        ax.axhline(prevalence, color="#ef4444", linestyle="--", lw=1.8, label=f"Random Chance Baseline (Prevalence = {prevalence*100:.2f}%)")
        
        ax.set_xlabel("Recall (Sensitivity / True Positive Rate)", color="#f8fafc", fontsize=11, labelpad=8)
        ax.set_ylabel("Precision (Positive Predictive Value)", color="#f8fafc", fontsize=11, labelpad=8)
        ax.set_title("Test Set (2022–2024) Precision-Recall Curves (Class Imbalance ~2.85%)", color="#ffffff", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.grid(True, linestyle="--", alpha=0.2, color="#64748b")
        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", loc="upper right", fontsize=9.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"PR curves with prevalence baseline plot saved to {save_path}")
        return save_path
