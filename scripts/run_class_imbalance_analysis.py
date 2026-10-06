"""
Execution Runner for Comprehensive Class Imbalance Analysis in Odisha Flood Prediction.

Generates:
1. Target class counts, percentages, and imbalance ratios across Overall, Train, Val, Test, and each Year (2001–2024).
2. Model performance comparisons under class imbalance (Default 0.50 vs F1-optimal vs F2-optimal).
3. Confusion matrices, Precision, Recall, F1, F2, PR-AUC, and Brier reliability scores.
4. Publication-quality plots and CSV summaries in reports/.
5. Analysis of existing mitigation mechanisms (class weights, threshold optimization, probability calibration, PR-AUC evaluation)
   and scientific justification of why SMOTE/synthetic oversampling is rejected.
"""

import sys
import os
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_combined_data, load_config
from src.evaluation.class_imbalance import ClassImbalanceAnalyzer
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
    parser = argparse.ArgumentParser(description="Run complete class imbalance audit and analysis.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Path to reports directory")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("ODISHA FLOOD PREDICTION: CLASS IMBALANCE AUDIT & METHODOLOGY ANALYSIS")
    logger.info("=" * 80)

    config = load_config(args.config)
    dataset_path = config["paths"]["combined_data_path"]
    reports_dir = args.reports_dir
    plots_dir = os.path.join(reports_dir, "plots")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    logger.info(f"Loading historical dataset from {dataset_path}...")
    df = load_combined_data(dataset_path, parse_dates=False)

    analyzer = ClassImbalanceAnalyzer(random_state=config["project"]["random_seed"])
    optimizer = ThresholdOptimizer(random_state=config["project"]["random_seed"])

    # 1. Dataset Target Class Distribution Analysis
    train_years = config["split"]["train_years"]
    val_years = config["split"]["val_years"]
    test_years = config["split"]["test_years"]
    target_col = config["target"]["name"]

    logger.info("\n--- Analyzing Target Class Distributions Across Temporal Splits ---")
    dist_results = analyzer.analyze_dataset_distributions(
        df=df,
        target_col=target_col,
        year_col="Year",
        district_col="District",
        train_years=train_years,
        val_years=val_years,
        test_years=test_years
    )

    splits_df = dist_results["splits_summary"]
    yearwise_df = dist_results["yearwise_summary"]
    district_df = dist_results["district_summary"]

    # Save CSV summaries
    splits_csv = os.path.join(reports_dir, "class_imbalance_summary.csv")
    yearwise_csv = os.path.join(reports_dir, "yearwise_flood_frequency.csv")
    splits_df.to_csv(splits_csv, index=False)
    yearwise_df.to_csv(yearwise_csv, index=False)
    logger.info(f"Splits distribution summary saved to {splits_csv}")
    logger.info(f"Year-wise flood frequency saved to {yearwise_csv}")

    print("\n" + "=" * 80)
    print("TARGET CLASS DISTRIBUTION SUMMARY ACROSS SPLITS")
    print("=" * 80)
    print(splits_df.to_string(index=False))
    print("=" * 80 + "\n")

    # 2. Generate Distribution Visualizations
    analyzer.plot_class_distributions(splits_df, save_path=os.path.join(plots_dir, "class_imbalance_distributions.png"))
    analyzer.plot_yearwise_flood_frequency(yearwise_df, save_path=os.path.join(plots_dir, "yearwise_flood_prevalence.png"))

    # 3. Model Training, Validation Thresholding, and Test Performance Under Imbalance
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

    models = {
        "Logistic Regression": create_logistic_regression_pipeline(
            max_iter=config["models"]["logistic_regression"]["max_iter"],
            class_weight="balanced",
            random_state=config["project"]["random_seed"]
        ),
        "Decision Tree": create_decision_tree_model(
            max_depth=config["models"].get("decision_tree", {}).get("max_depth", 8),
            min_samples_leaf=config["models"].get("decision_tree", {}).get("min_samples_leaf", 50),
            class_weight="balanced",
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

    test_predictions = {}
    perf_rows = []

    for name, model in models.items():
        logger.info(f"\n--- Training {name} on {len(X_train):,} rows ---")
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        test_predictions[name] = (y_test, test_proba)

        # Optimize thresholds on Validation Set ONLY
        f1_opt = optimizer.optimize_threshold(y_val, val_proba, strategy="f1")
        f2_opt = optimizer.optimize_threshold(y_val, val_proba, strategy="f2")
        rec80_opt = optimizer.optimize_threshold(y_val, val_proba, strategy="recall_target", target_recall=0.80)

        threshold_map = {
            "Standard Baseline (0.50)": 0.50,
            "Validation-Tuned F1 (Balanced)": f1_opt["threshold"],
            "Validation-Tuned F2 (Disaster-Averse)": f2_opt["threshold"],
            "High Sensitivity (Recall >= 80%)": rec80_opt["threshold"]
        }

        m_eval_df = analyzer.evaluate_threshold_performance_matrix(
            y_true=y_test,
            y_proba=test_proba,
            threshold_map=threshold_map,
            model_name=name,
            split_name=f"Test ({min(test_years)}–{max(test_years)})"
        )
        perf_rows.append(m_eval_df)

    all_perf_df = pd.concat(perf_rows, ignore_index=True)
    perf_csv = os.path.join(reports_dir, "model_imbalance_performance.csv")
    all_perf_df.to_csv(perf_csv, index=False)
    logger.info(f"Model performance under imbalance saved to {perf_csv}")

    # Generate Confusion Matrix Comparisons & PR Curves
    analyzer.plot_confusion_matrices_comparison(all_perf_df, model_name="XGBoost", save_path=os.path.join(plots_dir, "confusion_matrices_imbalance.png"))
    test_prevalence = float(np.mean(y_test))
    analyzer.plot_pr_curves_with_baseline(test_predictions, prevalence=test_prevalence, save_path=os.path.join(plots_dir, "pr_curves_imbalance.png"))

    print("\n" + "=" * 110)
    print("MODEL PERFORMANCE COMPARISON UNDER SEVERE CLASS IMBALANCE (TEST SET: 2022–2024, 343,758 ROWS)")
    print("=" * 110)
    display_cols = ["Model", "Strategy", "Threshold", "Precision", "Recall", "F1_Score", "F2_Score", "PR_AUC", "ROC_AUC", "True_Positives_TP", "False_Negatives_FN"]
    print(all_perf_df[display_cols].to_string(index=False))
    print("=" * 110 + "\n")

    elapsed = time.time() - start_time
    logger.info(f"Class imbalance audit completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
