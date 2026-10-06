"""
Audit and Multi-Strategy Threshold Selection Runner for Odisha Flood Prediction Models.

Demonstrates and verifies:
1. Threshold optimization is performed STRICTLY on validation data (2019–2021).
2. The final test set (2022–2024) is evaluated ONLY using validation-selected thresholds.
3. Multiple operational strategies are computed: F1 (balanced), F2 (disaster-averse),
   Recall Target (>=80%, >=90%), Youden's J, and Cost-Sensitive.
4. Generates model_thresholds.json, CSV summaries, and tradeoff plots.
"""

import sys
import os
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_combined_data, load_config
from src.evaluation.metrics import calculate_metrics
from src.evaluation.threshold_optimization import ThresholdOptimizer
from src.models.train_models import prepare_temporal_datasets
from src.models.logistic_regression import create_logistic_regression_pipeline
from src.models.decision_tree import create_decision_tree_model
from src.models.random_forest import create_random_forest_model
from src.models.xgboost_model import create_xgboost_model
from src.models.ann_model import create_ann_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Audit and optimize decision thresholds for flood prediction models.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Path to reports directory")
    parser.add_argument("--models-dir", type=str, default="models/flood_prediction", help="Path to models directory")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 75)
    logger.info("ODISHA FLOOD PREDICTION: MULTI-STRATEGY THRESHOLD SELECTION & AUDIT")
    logger.info("=" * 75)

    config = load_config(args.config)
    dataset_path = config["paths"]["combined_data_path"]
    
    logger.info(f"Loading dataset from {dataset_path}...")
    df = load_combined_data(dataset_path, parse_dates=False)
    
    # Chronological Temporal Split: Train (2001-2018), Val (2019-2021), Test (2022-2024)
    train_df, val_df, test_df, feature_cols, target_col = prepare_temporal_datasets(df, config)
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values.astype(int)
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].values.astype(int)
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values.astype(int)
    
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    
    # Instantiate models
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
            class_weight="balanced",
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
            batch_size=4096,
            max_iter=30,
            random_state=config["project"]["random_seed"]
        )
    }

    optimizer = ThresholdOptimizer(random_state=config["project"]["random_seed"])
    
    val_predictions = {}
    test_predictions = {}
    audit_summary_rows = []
    model_thresholds_metadata = {}

    for model_name, model in models.items():
        logger.info(f"\n--- Training {model_name} on Train Set (2001–2018, {len(X_train):,} rows) ---")
        model.fit(X_train, y_train)
        
        # Predict on Validation (2019-2021)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_predictions[model_name] = (y_val, val_proba)
        
        # Predict on Test (2022-2024)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_predictions[model_name] = (y_test, test_proba)
        
        # 1. Optimize on Validation using F1 (Balanced Operational Target)
        f1_opt = optimizer.optimize_threshold(y_val, val_proba, strategy="f1")
        
        # 2. Optimize on Validation using F2 (Disaster-Averse Recall Target)
        f2_opt = optimizer.optimize_threshold(y_val, val_proba, strategy="f2")
        
        # 3. Optimize on Validation using High Sensitivity (Recall >= 0.80)
        rec80_opt = optimizer.optimize_threshold(y_val, val_proba, strategy="recall_target", target_recall=0.80)
        
        # Evaluate on Test Set using Validation-Selected Thresholds ONLY
        test_metrics_f1 = calculate_metrics(y_test, test_proba, threshold=f1_opt["threshold"], prefix="test_")
        test_metrics_f2 = calculate_metrics(y_test, test_proba, threshold=f2_opt["threshold"], prefix="test_")
        test_metrics_rec80 = calculate_metrics(y_test, test_proba, threshold=rec80_opt["threshold"], prefix="test_")
        
        logger.info(f"[{model_name}] F1-Optimal Threshold: {f1_opt['threshold']:.4f} -> Val F1: {f1_opt['val_f1']:.4f}, Test F1: {test_metrics_f1['test_f1']:.4f} (Rec={test_metrics_f1['test_recall']:.4f}, Prec={test_metrics_f1['test_precision']:.4f})")
        logger.info(f"[{model_name}] F2-Optimal Threshold: {f2_opt['threshold']:.4f} -> Val F2: {f2_opt['val_f2']:.4f}, Test Recall: {test_metrics_f2['test_recall']:.4f} (Prec={test_metrics_f2['test_precision']:.4f})")
        
        # Store metadata
        model_thresholds_metadata[model_name] = {
            "model_name": model_name,
            "balanced_f1_strategy": {
                "selected_threshold": f1_opt["threshold"],
                "optimization_metric": "f1",
                "val_f1": f1_opt["val_f1"],
                "val_precision": f1_opt["val_precision"],
                "val_recall": f1_opt["val_recall"],
                "val_f2": f1_opt["val_f2"],
                "val_pr_auc": f1_opt["val_pr_auc"],
                "val_roc_auc": f1_opt["val_roc_auc"],
                "test_f1_at_val_threshold": test_metrics_f1["test_f1"],
                "test_recall_at_val_threshold": test_metrics_f1["test_recall"],
                "test_precision_at_val_threshold": test_metrics_f1["test_precision"]
            },
            "disaster_averse_f2_strategy": {
                "selected_threshold": f2_opt["threshold"],
                "optimization_metric": "f2",
                "val_f1": f2_opt["val_f1"],
                "val_f2": f2_opt["val_f2"],
                "val_precision": f2_opt["val_precision"],
                "val_recall": f2_opt["val_recall"],
                "test_f2_at_val_threshold": float(5 * (test_metrics_f2["test_precision"] * test_metrics_f2["test_recall"]) / (4 * test_metrics_f2["test_precision"] + test_metrics_f2["test_recall"] + 1e-10)),
                "test_recall_at_val_threshold": test_metrics_f2["test_recall"],
                "test_precision_at_val_threshold": test_metrics_f2["test_precision"]
            },
            "high_sensitivity_strategy": {
                "selected_threshold": rec80_opt["threshold"],
                "optimization_metric": "recall_target_0.80",
                "val_recall": rec80_opt["val_recall"],
                "val_precision": rec80_opt["val_precision"],
                "test_recall_at_val_threshold": test_metrics_rec80["test_recall"],
                "test_precision_at_val_threshold": test_metrics_rec80["test_precision"]
            }
        }
        
        # Detailed multi-strategy dataframe
        strat_df = optimizer.evaluate_all_strategies(y_val, val_proba, model_name=model_name)
        for _, s_row in strat_df.iterrows():
            t_val = s_row["Threshold"]
            t_test_m = calculate_metrics(y_test, test_proba, threshold=t_val, prefix="Test_")
            s_dict = s_row.to_dict()
            s_dict.update({
                "Test_F1": t_test_m["Test_f1"],
                "Test_Recall": t_test_m["Test_recall"],
                "Test_Precision": t_test_m["Test_precision"],
                "Test_Accuracy": t_test_m["Test_accuracy"],
                "Test_Specificity": t_test_m["Test_specificity"]
            })
            audit_summary_rows.append(s_dict)

    audit_summary_df = pd.DataFrame(audit_summary_rows)
    
    # Save CSV artifacts
    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    
    csv_report_path = os.path.join(args.reports_dir, "threshold_audit_summary.csv")
    csv_results_path = os.path.join("results/metrics", "threshold_selection_audit.csv")
    json_path = os.path.join(args.models_dir, "model_thresholds.json")
    
    audit_summary_df.to_csv(csv_report_path, index=False)
    audit_summary_df.to_csv(csv_results_path, index=False)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_thresholds_metadata, f, indent=2)
        
    logger.info(f"Threshold audit CSV saved to {csv_report_path}")
    logger.info(f"Model thresholds JSON metadata saved to {json_path}")
    
    # Generate Visualizations
    plot_path = os.path.join(args.reports_dir, "plots", "threshold_tradeoff_curves.png")
    optimizer.plot_threshold_tradeoffs(val_predictions, save_path=plot_path)

    # Format print summary
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION AUDIT SUMMARY TABLE (Validation-Tuned -> Test Evaluated)")
    print("=" * 80)
    display_cols = ["Model", "Strategy_Label", "Threshold", "Val_F1", "Val_F2", "Val_Recall", "Val_Precision", "Test_F1", "Test_Recall", "Test_Precision"]
    print(audit_summary_df[display_cols].to_string(index=False))
    print("=" * 80 + "\n")
    
    elapsed = time.time() - start_time
    logger.info(f"Threshold audit completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
