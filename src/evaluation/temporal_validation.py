"""
Temporal Walk-Forward Validation Module for Odisha Flood Prediction.

Implements rigorous, causal, walk-forward temporal cross-validation
designed specifically for flood forecasting and hydrological time series.

Key Principles:
1. Strict Temporal Order: max(train_years) < min(val_years). No future leakage.
2. Expanding Window Strategy:
   - Fold 1: Train 2001–2016 (16 yrs), Val 2017 (1 yr)
   - Fold 2: Train 2001–2017 (17 yrs), Val 2018 (1 yr)
   - Fold 3: Train 2001–2018 (18 yrs), Val 2019 (1 yr)
   - Fold 4: Train 2001–2019 (19 yrs), Val 2020 (1 yr)
   - Fold 5: Train 2001–2020 (20 yrs), Val 2021 (1 yr)
3. Strict Isolation: Scalers and class weights fit ONLY on training fold.
4. Separate Held-Out Test Set: 2022–2024 reserved strictly for final testing;
   never used during walk-forward validation or threshold/model selection.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Generator
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)

from src.data.load_data import load_combined_data, load_config
from src.evaluation.metrics import calculate_metrics, find_optimal_threshold
from src.evaluation.calibration import evaluate_calibration
from src.models.logistic_regression import create_logistic_regression_pipeline
from src.models.decision_tree import create_decision_tree_model
from src.models.random_forest import create_random_forest_model
from src.models.xgboost_model import create_xgboost_model
from src.models.ann_model import create_ann_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TemporalWalkForwardSplitter:
    """
    Chronological Walk-Forward Temporal Splitter.
    Generates non-overlapping, causal training and validation splits.
    """
    def __init__(
        self,
        start_year: int = 2001,
        initial_train_end_year: int = 2016,
        val_step_years: int = 1,
        n_folds: int = 5,
        test_years: Optional[List[int]] = None,
        year_col: str = "Year"
    ):
        self.start_year = start_year
        self.initial_train_end_year = initial_train_end_year
        self.val_step_years = val_step_years
        self.n_folds = n_folds
        self.test_years = test_years if test_years is not None else [2022, 2023, 2024]
        self.year_col = year_col

    def generate_fold_definitions(self) -> List[Dict[str, Any]]:
        """Generate structured fold definitions without data dependencies."""
        folds = []
        for fold_idx in range(self.n_folds):
            train_end = self.initial_train_end_year + (fold_idx * self.val_step_years)
            val_start = train_end + 1
            val_end = val_start + self.val_step_years - 1
            
            train_years = list(range(self.start_year, train_end + 1))
            val_years = list(range(val_start, val_end + 1))
            
            # Validation: ensure no overlap with test years
            overlap_with_test = set(val_years).intersection(set(self.test_years))
            if overlap_with_test:
                raise ValueError(
                    f"Fold {fold_idx + 1} validation years {val_years} overlap with "
                    f"held-out test years {self.test_years}!"
                )
            
            folds.append({
                "fold_id": fold_idx + 1,
                "fold_name": f"Fold_{fold_idx + 1}",
                "train_years": train_years,
                "val_years": val_years,
                "train_span": f"{min(train_years)}–{max(train_years)}",
                "val_span": f"{min(val_years)}" if len(val_years) == 1 else f"{min(val_years)}–{max(val_years)}"
            })
        return folds

    def split(
        self, df: pd.DataFrame
    ) -> Generator[Tuple[int, pd.DataFrame, pd.DataFrame, Dict[str, Any]], None, None]:
        """
        Yield (fold_id, train_df, val_df, fold_meta) for each chronological fold.
        """
        if self.year_col not in df.columns:
            raise KeyError(f"Year column '{self.year_col}' not found in dataframe columns: {df.columns.tolist()}")

        fold_defs = self.generate_fold_definitions()
        for fold in fold_defs:
            fold_id = fold["fold_id"]
            train_years = fold["train_years"]
            val_years = fold["val_years"]
            
            train_df = df[df[self.year_col].isin(train_years)].copy()
            val_df = df[df[self.year_col].isin(val_years)].copy()
            
            # Verification: zero leakage
            self.verify_no_leakage(train_df, val_df, fold)
            
            fold_meta = {
                **fold,
                "n_train_samples": len(train_df),
                "n_val_samples": len(val_df),
            }
            yield fold_id, train_df, val_df, fold_meta

    def verify_no_leakage(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, fold_info: Dict[str, Any]
    ) -> None:
        """Verify strict chronological integrity and absence of future data leakage."""
        if train_df.empty:
            raise ValueError(f"Train split for {fold_info['fold_name']} is empty!")
        if val_df.empty:
            raise ValueError(f"Validation split for {fold_info['fold_name']} is empty!")
            
        max_train_year = train_df[self.year_col].max()
        min_val_year = val_df[self.year_col].min()
        
        if max_train_year >= min_val_year:
            raise ValueError(
                f"Temporal leakage detected in {fold_info['fold_name']}: "
                f"max_train_year ({max_train_year}) >= min_val_year ({min_val_year})"
            )
            
        # Check index overlap
        overlap_indices = set(train_df.index).intersection(set(val_df.index))
        if overlap_indices:
            raise ValueError(
                f"Index leakage detected in {fold_info['fold_name']}: "
                f"{len(overlap_indices)} overlapping row indices between train and val splits."
            )
            
        # Check held-out test contamination
        test_in_train = set(train_df[self.year_col]).intersection(set(self.test_years))
        test_in_val = set(val_df[self.year_col]).intersection(set(self.test_years))
        if test_in_train or test_in_val:
            raise ValueError(
                f"Contamination of held-out test years {self.test_years} in {fold_info['fold_name']}! "
                f"Found in train: {test_in_train}, in val: {test_in_val}"
            )


class TemporalWalkForwardValidator:
    """
    Orchestrates walk-forward temporal cross-validation across multiple model architectures.
    Computes per-fold metrics, aggregates mean/std, and generates reports and plots.
    """
    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: str = "config.yaml",
        random_state: int = 42
    ):
        self.config = config or load_config(config_path)
        self.random_state = random_state
        self.target_col = self.config["target"]["name"]
        
        # Feature columns (pure meteorological + temporal)
        self.feature_cols = (
            self.config["features"]["temporal_cols"] +
            self.config["features"]["rainfall_cols"]
        )
        if self.config.get("features", {}).get("same_day_flood_col"):
            self.feature_cols.append(self.config["features"]["same_day_flood_col"])
            
        self.test_years = self.config["split"].get("test_years", [2022, 2023, 2024])
        self.splitter = TemporalWalkForwardSplitter(
            start_year=2001,
            initial_train_end_year=2016,
            val_step_years=1,
            n_folds=5,
            test_years=self.test_years,
            year_col="Year"
        )
        
        self.fold_results: List[Dict[str, Any]] = []
        self.summary_results: Optional[pd.DataFrame] = None
        self.roc_pr_curve_data: Dict[str, List[Dict[str, Any]]] = {}

    def get_fresh_model_instance(
        self, model_name: str, y_train: np.ndarray
    ) -> Any:
        """
        Instantiate a fresh model instance with strictly isolated training parameters.
        """
        # Dynamic scale_pos_weight calculated strictly on fold training data
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = float(n_neg / max(n_pos, 1))

        if model_name == "Logistic Regression":
            return create_logistic_regression_pipeline(
                max_iter=self.config["models"]["logistic_regression"]["max_iter"],
                class_weight=self.config["models"]["logistic_regression"]["class_weight"],
                random_state=self.random_state
            )
        elif model_name == "Decision Tree":
            return create_decision_tree_model(
                max_depth=self.config["models"].get("decision_tree", {}).get("max_depth", 8),
                min_samples_leaf=self.config["models"].get("decision_tree", {}).get("min_samples_leaf", 50),
                class_weight=self.config["models"].get("decision_tree", {}).get("class_weight", "balanced"),
                random_state=self.random_state
            )
        elif model_name == "Random Forest":
            return create_random_forest_model(
                n_estimators=self.config["models"]["random_forest"]["n_estimators"],
                max_depth=self.config["models"]["random_forest"]["max_depth"],
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=self.config["models"]["random_forest"]["n_jobs"]
            )
        elif model_name == "XGBoost":
            return create_xgboost_model(
                n_estimators=self.config["models"]["xgboost"]["n_estimators"],
                max_depth=self.config["models"]["xgboost"]["max_depth"],
                learning_rate=self.config["models"]["xgboost"]["learning_rate"],
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                n_jobs=self.config["models"]["xgboost"]["n_jobs"]
            )
        elif model_name == "ANN (MLP)":
            return create_ann_model(
                hidden_layer_sizes=tuple(self.config["models"]["ann"]["hidden_layer_sizes"]),
                activation=self.config["models"]["ann"]["activation"],
                batch_size=4096,
                max_iter=30,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def evaluate_fold_model(
        self,
        model_name: str,
        fold_meta: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fit a model on a single temporal fold training set and evaluate on validation year.
        """
        fold_id = fold_meta["fold_id"]
        logger.info(f"[{model_name}] Training Fold {fold_id} (Train: {fold_meta['train_span']} -> Val: {fold_meta['val_span']})...")
        
        # Instantiate clean model
        model = self.get_fresh_model_instance(model_name, y_train)
        model.fit(X_train, y_train)
        
        # Predict probabilities on validation fold
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        # Find optimal threshold on validation fold to maximize F1 under class imbalance
        opt_thresh, opt_f1 = find_optimal_threshold(y_val, y_val_proba, metric="f1")
        
        # Compute metrics at optimal threshold
        metrics_opt = calculate_metrics(y_val, y_val_proba, threshold=opt_thresh, prefix="")
        
        # Compute metrics at standard 0.5 threshold
        metrics_def = calculate_metrics(y_val, y_val_proba, threshold=0.5, prefix="def_")
        
        # Calibration statistics
        cal_stats = evaluate_calibration(y_val, y_val_proba, n_bins=10)
        
        # Store curve points for plotting
        prec_arr, rec_arr, _ = precision_recall_curve(y_val, y_val_proba)
        fpr_arr, tpr_arr, _ = roc_curve(y_val, y_val_proba)
        
        fold_entry = {
            "Model": model_name,
            "Fold": fold_id,
            "Fold_Name": fold_meta["fold_name"],
            "Train_Span": fold_meta["train_span"],
            "Val_Span": fold_meta["val_span"],
            "Train_Samples": len(X_train),
            "Val_Samples": len(X_val),
            "Val_Flood_Events": int(np.sum(y_val)),
            "Val_Flood_Rate_Pct": round(float(np.mean(y_val) * 100), 3),
            "Optimal_Threshold": round(float(opt_thresh), 4),
            "ROC_AUC": metrics_opt["roc_auc"],
            "PR_AUC": metrics_opt["pr_auc"],
            "F1": metrics_opt["f1"],
            "Precision": metrics_opt["precision"],
            "Recall": metrics_opt["recall"],
            "Specificity": metrics_opt["specificity"],
            "Accuracy": metrics_opt["accuracy"],
            "Brier_Score": cal_stats["brier_score"],
            "ECE": round(cal_stats["expected_calibration_error"], 4),
            "F1_at_0.5": metrics_def["def_f1"],
            "Precision_at_0.5": metrics_def["def_precision"],
            "Recall_at_0.5": metrics_def["def_recall"],
            "TP": metrics_opt["tp"],
            "FP": metrics_opt["fp"],
            "TN": metrics_opt["tn"],
            "FN": metrics_opt["fn"],
        }
        
        logger.info(
            f"  -> Fold {fold_id} ({fold_meta['val_span']}) Result: "
            f"ROC-AUC={fold_entry['ROC_AUC']:.4f}, PR-AUC={fold_entry['PR_AUC']:.4f}, "
            f"F1={fold_entry['F1']:.4f} (Rec={fold_entry['Recall']:.4f}, Prec={fold_entry['Precision']:.4f}), "
            f"Brier={fold_entry['Brier_Score']:.4f}"
        )
        
        return {
            "fold_entry": fold_entry,
            "curve_data": {
                "fold_id": fold_id,
                "precision": prec_arr,
                "recall": rec_arr,
                "fpr": fpr_arr,
                "tpr": tpr_arr,
                "pr_auc": metrics_opt["pr_auc"],
                "roc_auc": metrics_opt["roc_auc"]
            }
        }

    def run_validation(
        self,
        df: pd.DataFrame,
        models_to_run: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Execute full walk-forward validation across all 5 chronological folds.
        """
        # Clean dataframe: drop rows where target is missing
        df_clean = df.dropna(subset=[self.target_col]).copy()
        
        all_models = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "ANN (MLP)"]
        selected_models = models_to_run if models_to_run is not None else all_models
        
        logger.info(f"Starting Walk-Forward Temporal Cross-Validation across {len(selected_models)} architectures...")
        logger.info(f"Total Available Historical Observations: {len(df_clean):,} rows (2001–2024)")
        
        self.fold_results = []
        self.roc_pr_curve_data = {m: [] for m in selected_models}
        
        # Iterate over chronological folds
        for fold_id, train_df, val_df, fold_meta in self.splitter.split(df_clean):
            logger.info(f"\n{'='*65}\nEXECUTING FOLD {fold_id}/5: Train ({fold_meta['train_span']}, {len(train_df):,} rows) -> Val ({fold_meta['val_span']}, {len(val_df):,} rows)\n{'='*65}")
            
            X_train = train_df[self.feature_cols]
            y_train = train_df[self.target_col].values.astype(int)
            
            X_val = val_df[self.feature_cols]
            y_val = val_df[self.target_col].values.astype(int)
            
            for model_name in selected_models:
                res = self.evaluate_fold_model(model_name, fold_meta, X_train, y_train, X_val, y_val)
                self.fold_results.append(res["fold_entry"])
                self.roc_pr_curve_data[model_name].append(res["curve_data"])
                
        results_df = pd.DataFrame(self.fold_results)
        
        # Compute aggregate mean and standard deviation per model
        summary_rows = []
        metrics_to_agg = ["ROC_AUC", "PR_AUC", "F1", "Precision", "Recall", "Specificity", "Accuracy", "Brier_Score", "ECE"]
        
        for model_name in selected_models:
            m_df = results_df[results_df["Model"] == model_name]
            agg_entry: Dict[str, Any] = {
                "Model": model_name,
                "Folds_Evaluated": len(m_df)
            }
            for metric in metrics_to_agg:
                mean_val = float(m_df[metric].mean())
                std_val = float(m_df[metric].std())
                agg_entry[f"{metric}_Mean"] = round(mean_val, 4)
                agg_entry[f"{metric}_Std"] = round(std_val, 4)
                agg_entry[f"{metric}_Summary"] = f"{mean_val:.4f} ± {std_val:.4f}"
            summary_rows.append(agg_entry)
            
        self.summary_results = pd.DataFrame(summary_rows)
        
        # Sort summary by Mean PR_AUC and F1
        self.summary_results = self.summary_results.sort_values(
            by=["PR_AUC_Mean", "F1_Mean"], ascending=False
        ).reset_index(drop=True)
        
        summary_dict = {
            "strategy": "Expanding Walk-Forward Temporal Cross-Validation",
            "folds": self.splitter.generate_fold_definitions(),
            "held_out_test_years": self.test_years,
            "models_evaluated": selected_models,
            "features_used": self.feature_cols,
            "target": self.target_col,
            "summary_table": self.summary_results.to_dict(orient="records"),
            "fold_details": self.fold_results
        }
        
        return results_df, self.summary_results, summary_dict

    def save_artifacts(
        self,
        results_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        summary_dict: Dict[str, Any],
        reports_dir: str = "reports",
        results_metrics_dir: str = "results/metrics"
    ) -> Dict[str, str]:
        """
        Save detailed results, summary tables, and JSON metadata.
        """
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(results_metrics_dir, exist_ok=True)
        
        # 1. reports/temporal_validation_results.csv
        p_rep_results = os.path.join(reports_dir, "temporal_validation_results.csv")
        results_df.to_csv(p_rep_results, index=False)
        
        # 2. reports/temporal_validation_summary.csv
        p_rep_summary = os.path.join(reports_dir, "temporal_validation_summary.csv")
        summary_df.to_csv(p_rep_summary, index=False)
        
        # 3. reports/temporal_validation_summary.json
        p_rep_json = os.path.join(reports_dir, "temporal_validation_summary.json")
        with open(p_rep_json, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
            
        # 4. Mirror into results/metrics
        p_res_results = os.path.join(results_metrics_dir, "temporal_validation_results.csv")
        p_res_summary = os.path.join(results_metrics_dir, "temporal_validation_summary.csv")
        results_df.to_csv(p_res_results, index=False)
        summary_df.to_csv(p_res_summary, index=False)
        
        logger.info(f"Temporal validation artifacts saved to {reports_dir} and {results_metrics_dir}")
        return {
            "detailed_csv": p_rep_results,
            "summary_csv": p_rep_summary,
            "summary_json": p_rep_json
        }

    def generate_visualizations(
        self,
        results_df: pd.DataFrame,
        plots_dir: str = "reports/plots"
    ) -> Dict[str, str]:
        """
        Generate visual evaluation charts across walk-forward folds.
        """
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Fold-by-Fold Performance Comparison (PR-AUC, ROC-AUC, F1, Brier)
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.patch.set_facecolor("#0f172a")
        for ax in axes.flat:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#cbd5e1")
            ax.xaxis.label.set_color("#f8fafc")
            ax.yaxis.label.set_color("#f8fafc")
            ax.title.set_color("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color("#334155")
                
        models = results_df["Model"].unique()
        palette = {
            "Logistic Regression": "#38bdf8",
            "XGBoost": "#10b981",
            "Random Forest": "#f59e0b",
            "Decision Tree": "#ec4899",
            "ANN (MLP)": "#a855f7"
        }
        
        metrics = [
            ("PR_AUC", "Precision-Recall AUC (PR-AUC) across Folds", axes[0, 0]),
            ("ROC_AUC", "Receiver Operating Characteristic AUC (ROC-AUC)", axes[0, 1]),
            ("F1", "Optimal F1-Score across Folds", axes[1, 0]),
            ("Brier_Score", "Brier Reliability Score (Lower is Better)", axes[1, 1])
        ]
        
        folds = sorted(results_df["Fold"].unique())
        fold_labels = [f"F{f}\n({results_df[results_df['Fold']==f]['Val_Span'].iloc[0]})" for f in folds]
        
        for metric, title, ax in metrics:
            for model_name in models:
                m_data = results_df[results_df["Model"] == model_name].sort_values("Fold")
                color = palette.get(model_name, "#94a3b8")
                ax.plot(
                    folds, m_data[metric], 
                    marker="o", linewidth=2.5, markersize=8, 
                    label=model_name, color=color
                )
            ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
            ax.set_xticks(folds)
            ax.set_xticklabels(fold_labels, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
            ax.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=9)
            
        plt.suptitle(
            "Odisha Flood Prediction: Walk-Forward Temporal Validation (2017–2021)",
            fontsize=15, fontweight="bold", color="#ffffff", y=0.98
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        folds_plot_path = os.path.join(plots_dir, "temporal_validation_folds.png")
        plt.savefig(folds_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        
        # 2. Precision-Recall Curves Across Folds (for best model or all models)
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        ax.xaxis.label.set_color("#f8fafc")
        ax.yaxis.label.set_color("#f8fafc")
        ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
        fold_colors = ["#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#a855f7"]
        
        # Plot for Logistic Regression (Pure Met Champion)
        lr_curves = self.roc_pr_curve_data.get("Logistic Regression", [])
        for idx, cdata in enumerate(lr_curves):
            f_id = cdata["fold_id"]
            color = fold_colors[(f_id - 1) % len(fold_colors)]
            val_span = results_df[results_df["Fold"] == f_id]["Val_Span"].iloc[0]
            ax.plot(
                cdata["recall"], cdata["precision"],
                linewidth=2.2, color=color,
                label=f"Fold {f_id} ({val_span}) - PR-AUC: {cdata['pr_auc']:.4f}"
            )
            
        ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
        ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
        ax.set_title("Walk-Forward Precision-Recall Curves (Logistic Regression)", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        ax.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=10)
        
        pr_plot_path = os.path.join(plots_dir, "temporal_validation_pr_curves.png")
        plt.savefig(pr_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Visualizations saved to {folds_plot_path} and {pr_plot_path}")
        return {
            "folds_plot": folds_plot_path,
            "pr_curves_plot": pr_plot_path
        }


def format_validation_summary_table(summary_df: pd.DataFrame) -> str:
    """Format summary DataFrame into a markdown table."""
    cols_to_show = [
        "Model", "ROC_AUC_Summary", "PR_AUC_Summary", "F1_Summary",
        "Recall_Summary", "Precision_Summary", "Brier_Score_Summary"
    ]
    sub_df = summary_df[cols_to_show].copy()
    sub_df.columns = [
        "Model Architecture", "ROC-AUC (Mean ± Std)", "PR-AUC (Mean ± Std)",
        "F1 (Mean ± Std)", "Recall (Mean ± Std)", "Precision (Mean ± Std)", "Brier Score (Mean ± Std)"
    ]
    return sub_df.to_markdown(index=False)
