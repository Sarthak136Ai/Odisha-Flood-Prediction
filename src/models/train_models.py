"""
Comprehensive training, calibration, and temporal evaluation pipeline for flood prediction models.
Evaluates baseline models (Logistic Regression, Decision Tree) and stronger ensembles (Random Forest, XGBoost),
along with ANN comparison, probability calibration curves, Brier reliability scores, and year-wise stability.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from src.data.load_data import load_combined_data, load_config
from src.evaluation.metrics import calculate_metrics, find_optimal_threshold
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.roc_curve import plot_roc_curves
from src.evaluation.precision_recall import plot_precision_recall_curves
from src.evaluation.calibration import evaluate_calibration, plot_reliability_diagrams
from src.evaluation.year_wise_eval import evaluate_year_wise_performance

from src.models.logistic_regression import create_logistic_regression_pipeline
from src.models.decision_tree import create_decision_tree_model
from src.models.random_forest import create_random_forest_model
from src.models.xgboost_model import create_xgboost_model
from src.models.ann_model import create_ann_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_temporal_datasets(
    df: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], str]:
    """
    Split dataset chronologically into Train (2001-2018), Val (2019-2021), and Test (2022-2024).
    Strips ID columns and isolates target Flood_Next_Day.
    """
    target_col = config["target"]["name"]
    # Drop rows where target is missing (e.g. boundary day)
    df_clean = df.dropna(subset=[target_col]).copy()
    
    feature_cols = (
        config["features"]["temporal_cols"] +
        config["features"]["rainfall_cols"] +
        [config["features"]["same_day_flood_col"]]
    )
    
    train_years = config["split"]["train_years"]
    val_years = config["split"]["val_years"]
    test_years = config["split"]["test_years"]
    
    train_df = df_clean[df_clean["Year"].isin(train_years)].copy()
    val_df = df_clean[df_clean["Year"].isin(val_years)].copy()
    test_df = df_clean[df_clean["Year"].isin(test_years)].copy()
    
    logger.info(f"Chronological temporal split completed:")
    logger.info(f"  Train Set ({min(train_years)}-{max(train_years)}): {len(train_df):,} rows (Flood rate: {train_df[target_col].mean()*100:.3f}%)")
    logger.info(f"  Validation Set ({min(val_years)}-{max(val_years)}): {len(val_df):,} rows (Flood rate: {val_df[target_col].mean()*100:.3f}%)")
    logger.info(f"  Test Set ({min(test_years)}-{max(test_years)}): {len(test_df):,} rows (Flood rate: {test_df[target_col].mean()*100:.3f}%)")
    
    return train_df, val_df, test_df, feature_cols, target_col


def train_and_evaluate_all_models(
    config_path: str = "config.yaml"
) -> Dict[str, Any]:
    """
    Train and evaluate Logistic Regression, Decision Tree, Random Forest, XGBoost, and ANN.
    """
    config = load_config(config_path)
    combined_path = config["paths"]["combined_data_path"]
    results_metrics_dir = config["paths"]["results_metrics_dir"]
    results_plots_dir = config["paths"]["results_plots_dir"]
    models_dir = config["paths"]["flood_models_dir"]
    
    os.makedirs(results_metrics_dir, exist_ok=True)
    os.makedirs(results_plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info(f"Loading combined dataset from {combined_path}...")
    df = load_combined_data(combined_path, parse_dates=False)
    
    train_df, val_df, test_df, feature_cols, target_col = prepare_temporal_datasets(df, config)
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values.astype(int)
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].values.astype(int)
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values.astype(int)
    
    # Calculate scale_pos_weight for XGBoost
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = float(num_neg / (num_pos + 1e-5))
    logger.info(f"Training Class Imbalance: {num_neg:,} Negatives vs {num_pos:,} Positives (scale_pos_weight = {scale_pos_weight:.2f})")
    
    # Instantiate models (Baselines + Ensembles + ANN)
    models = {
        "Logistic Regression": create_logistic_regression_pipeline(
            max_iter=config["models"]["logistic_regression"]["max_iter"],
            class_weight=config["models"]["logistic_regression"]["class_weight"],
            random_state=config["project"]["random_seed"]
        ),
        "Decision Tree": create_decision_tree_model(
            max_depth=config["models"].get("decision_tree", {}).get("max_depth", 8),
            min_samples_leaf=config["models"].get("decision_tree", {}).get("min_samples_leaf", 50),
            class_weight=config["models"].get("decision_tree", {}).get("class_weight", "balanced"),
            random_state=config["project"]["random_seed"]
        ),
        "Random Forest": create_random_forest_model(
            n_estimators=config["models"]["random_forest"]["n_estimators"],
            max_depth=config["models"]["random_forest"]["max_depth"],
            class_weight=config["models"]["random_forest"]["class_weight"],
            random_state=config["project"]["random_seed"],
            n_jobs=config["models"]["random_forest"]["n_jobs"]
        ),
        "XGBoost": create_xgboost_model(
            n_estimators=config["models"]["xgboost"]["n_estimators"],
            max_depth=config["models"]["xgboost"]["max_depth"],
            learning_rate=config["models"]["xgboost"]["learning_rate"],
            scale_pos_weight=scale_pos_weight,
            random_state=config["project"]["random_seed"],
            n_jobs=config["models"]["xgboost"]["n_jobs"]
        ),
        "ANN (MLP)": create_ann_model(
            hidden_layer_sizes=tuple(config["models"]["ann"]["hidden_layer_sizes"]),
            activation=config["models"]["ann"]["activation"],
            max_iter=config["models"]["ann"]["max_iter"],
            random_state=config["project"]["random_seed"]
        )
    }
    
    comparison_rows = []
    trained_models = {}
    val_predictions = {}
    test_predictions = {}
    calibration_metrics = {}
    
    for name, model in models.items():
        logger.info(f"\n{'='*50}\nTraining model: {name}...\n{'='*50}")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on Validation
        val_proba = model.predict_proba(X_val)[:, 1]
        val_predictions[name] = (y_val, val_proba)
        
        # Determine optimal threshold on validation set
        opt_thresh, opt_f1 = find_optimal_threshold(y_val, val_proba, metric="f1")
        logger.info(f"[{name}] Optimal Threshold on Validation Set: {opt_thresh:.4f} (Val F1: {opt_f1:.4f})")
        
        # Metrics on Validation (at optimal threshold)
        val_metrics = calculate_metrics(y_val, val_proba, threshold=opt_thresh, prefix="val_")
        
        # Predict on Unseen Test Set (2022-2024)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_predictions[name] = (y_test, test_proba)
        
        # Probability Calibration Statistics
        cal_stats = evaluate_calibration(y_test, test_proba, n_bins=10)
        calibration_metrics[name] = cal_stats
        
        # Metrics on Test at optimal validation threshold
        test_metrics_opt = calculate_metrics(y_test, test_proba, threshold=opt_thresh, prefix="test_")
        
        row = {
            "Model": name,
            "Optimal_Threshold": opt_thresh,
            "Val_Accuracy": val_metrics["val_accuracy"],
            "Val_Precision": val_metrics["val_precision"],
            "Val_Recall": val_metrics["val_recall"],
            "Val_F1": val_metrics["val_f1"],
            "Val_ROC_AUC": val_metrics["val_roc_auc"],
            "Val_PR_AUC": val_metrics["val_pr_auc"],
            "Test_ROC_AUC": test_metrics_opt["test_roc_auc"],
            "Test_PR_AUC": test_metrics_opt["test_pr_auc"],
            "Test_Accuracy": test_metrics_opt["test_accuracy"],
            "Test_Precision": test_metrics_opt["test_precision"],
            "Test_Recall": test_metrics_opt["test_recall"],
            "Test_Specificity": test_metrics_opt["test_specificity"],
            "Test_F1": test_metrics_opt["test_f1"],
            "Test_Brier_Score": cal_stats["brier_score"],
            "Test_Calibration_ECE": round(cal_stats["expected_calibration_error"], 4),
            "Test_TP": test_metrics_opt["test_tp"],
            "Test_FP": test_metrics_opt["test_fp"],
            "Test_TN": test_metrics_opt["test_tn"],
            "Test_FN": test_metrics_opt["test_fn"],
        }
        comparison_rows.append(row)
        logger.info(f"[{name}] Test Results: ROC-AUC={row['Test_ROC_AUC']:.4f}, PR-AUC={row['Test_PR_AUC']:.4f}, F1={row['Test_F1']:.4f}, Recall={row['Test_Recall']:.4f}, Precision={row['Test_Precision']:.4f}, Brier={row['Test_Brier_Score']:.4f}")
        
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = os.path.join(results_metrics_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"Model comparison table saved to {comparison_csv_path}")
    
    # Plot ROC & PR Curves for Test Set
    roc_plot_path = os.path.join(results_plots_dir, "roc_curve.png")
    pr_plot_path = os.path.join(results_plots_dir, "precision_recall_curve.png")
    cal_plot_path = os.path.join(results_plots_dir, "calibration_curves.png")
    
    plot_roc_curves(test_predictions, title="Test Set (2022-2024) ROC Curves", save_path=roc_plot_path)
    plot_precision_recall_curves(test_predictions, title="Test Set (2022-2024) Precision-Recall Curves", save_path=pr_plot_path)
    plot_reliability_diagrams(test_predictions, title="Probability Calibration & Reliability Diagram (Test 2022-2024)", save_path=cal_plot_path)
    
    # Identify Best Model (by Test PR-AUC & F1)
    best_row = comparison_df.sort_values(by=["Test_PR_AUC", "Test_F1"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    best_threshold = float(best_row["Optimal_Threshold"])
    
    logger.info(f"\n{'*'*60}\nBEST PERFORMING MODEL: {best_model_name} (PR-AUC: {best_row['Test_PR_AUC']:.4f}, ROC-AUC: {best_row['Test_ROC_AUC']:.4f}, F1: {best_row['Test_F1']:.4f})\n{'*'*60}")
    
    # Plot best model confusion matrix
    y_test_best, y_proba_best = test_predictions[best_model_name]
    y_pred_best = (y_proba_best >= best_threshold).astype(int)
    cm_path = os.path.join(results_plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        y_test_best,
        y_pred_best,
        title=f"Confusion Matrix - {best_model_name} (Test 2022-2024)",
        save_path=cm_path
    )
    
    # Compute Year-Wise Stability Metrics for Best Model
    year_wise_df = evaluate_year_wise_performance(
        df=test_df,
        y_true=y_test_best,
        y_proba=y_proba_best,
        threshold=best_threshold,
        year_col="Year"
    )
    year_wise_csv_path = os.path.join(results_metrics_dir, "year_wise_metrics.csv")
    year_wise_df.to_csv(year_wise_csv_path, index=False)
    logger.info(f"Year-wise metrics saved to {year_wise_csv_path}")
    
    # Save Best Model Artifact & Metadata
    model_save_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, model_save_path)
    
    metadata = {
        "model_name": best_model_name,
        "optimal_threshold": best_threshold,
        "features": feature_cols,
        "risk_thresholds": config["risk_thresholds"],
        "train_years": config["split"]["train_years"],
        "val_years": config["split"]["val_years"],
        "test_years": config["split"]["test_years"],
        "test_metrics": best_row.to_dict(),
        "brier_score": float(best_row["Test_Brier_Score"]),
        "calibration_ece": float(best_row["Test_Calibration_ECE"])
    }
    
    metadata_save_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Best model artifact saved to {model_save_path}")
    logger.info(f"Model metadata saved to {metadata_save_path}")
    
    return {
        "comparison_df": comparison_df,
        "best_model_name": best_model_name,
        "metadata": metadata,
        "year_wise_df": year_wise_df
    }
