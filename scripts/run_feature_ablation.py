"""
Command-line runner for Feature Ablation Study in Odisha Flood Prediction.

This script:
1. Loads the historical 2001-2024 dataset.
2. Performs walk-forward chronological splitting:
   - Train: 2001–2018 (2,063,989 records)
   - Val:   2019–2021 (344,140 records)
   - Test:  2022–2024 (343,758 records)
3. Evaluates 6 distinct feature ablation experiments:
   - Experiment A: Current rainfall only (2 features)
   - Experiment B: Current rainfall + lag rainfall (6 features)
   - Experiment C: Current rainfall + lag + rolling features (19 features)
   - Experiment D: Current rainfall + temporal/seasonal (5 features)
   - Experiment E: Full baseline feature set (22 features)
   - Experiment F: Full feature set + IMD rainfall-intensity indicators (29 features)
4. Uses identical models (Logistic Regression baseline and XGBoost champion),
   preprocessing, temporal splits, threshold optimization (validation-only), and random seed (42).
5. Reports ROC-AUC, PR-AUC, Precision, Recall, F1, F2, and Brier Score.
6. Saves detailed results to CSV and generates publication-grade multi-panel plots.
"""

import os
import sys
import argparse
import logging
import time
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_combined_data, load_config
from src.models.train_models import prepare_temporal_datasets
from src.models.logistic_regression import create_logistic_regression_pipeline
from src.models.xgboost_model import create_xgboost_model
from src.evaluation.feature_ablation import (
    FeatureAblationStudy,
    FEATURE_ABLATION_GROUPS,
    plot_feature_ablation_comparison
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Feature Ablation Study for Flood Prediction.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "f2"], help="Metric for validation threshold tuning.")
    parser.add_argument("--output-csv", type=str, default="reports/feature_ablation_results.csv", help="Primary results CSV destination.")
    parser.add_argument("--plots-dir", type=str, default="reports/plots", help="Directory for plots.")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("================================================================================")
    logger.info("Starting Odisha Flood Feature Ablation Study")
    logger.info("================================================================================")

    # 1. Load configuration and dataset
    config = load_config(args.config)
    data_path = config["paths"]["combined_data_path"]
    logger.info(f"Loading combined historical dataset from: {data_path}")
    df = load_combined_data(data_path, parse_dates=False)

    # 2. Chronological temporal split
    train_df, val_df, test_df, feature_cols, target_col = prepare_temporal_datasets(df, config)
    y_train = train_df[target_col].values.astype(int)

    # Compute scale_pos_weight on Train set only (zero leakage)
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = float(num_neg / (num_pos + 1e-5))
    logger.info(f"Train split imbalance: {num_neg:,} Neg vs {num_pos:,} Pos (scale_pos_weight = {scale_pos_weight:.2f})")

    # 3. Define Model Factories
    model_factories = {
        "Logistic Regression": lambda: create_logistic_regression_pipeline(
            max_iter=config["models"]["logistic_regression"].get("max_iter", 1000),
            class_weight=config["models"]["logistic_regression"].get("class_weight", "balanced"),
            random_state=config["project"]["random_seed"]
        ),
        "XGBoost": lambda: create_xgboost_model(
            n_estimators=config["models"]["xgboost"].get("n_estimators", 200),
            max_depth=config["models"]["xgboost"].get("max_depth", 6),
            learning_rate=config["models"]["xgboost"].get("learning_rate", 0.05),
            scale_pos_weight=scale_pos_weight,
            random_state=config["project"]["random_seed"],
            n_jobs=config["models"]["xgboost"].get("n_jobs", -1)
        )
    }

    # 4. Instantiate & Run Feature Ablation Study
    study = FeatureAblationStudy(
        groups=FEATURE_ABLATION_GROUPS,
        optimization_metric=args.metric
    )

    logger.info(f"Evaluating {len(FEATURE_ABLATION_GROUPS)} feature groups across {len(model_factories)} models...")
    results_df = study.run_ablation(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=target_col,
        model_factory_dict=model_factories
    )

    # 5. Save Results CSV to reports and results directories
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved primary results CSV to: {args.output_csv}")

    # Also mirror to results/metrics/
    metrics_mirror = os.path.join(config["paths"]["results_metrics_dir"], "feature_ablation_results.csv")
    os.makedirs(os.path.dirname(metrics_mirror), exist_ok=True)
    results_df.to_csv(metrics_mirror, index=False)
    logger.info(f"Mirrored results CSV to: {metrics_mirror}")

    # 6. Generate Publication Comparison Plots
    os.makedirs(args.plots_dir, exist_ok=True)
    plot_png = os.path.join(args.plots_dir, "feature_ablation_comparison.png")
    plot_svg = os.path.join(args.plots_dir, "feature_ablation_comparison.svg")
    plot_feature_ablation_comparison(results_df, save_path_png=plot_png, save_path_svg=plot_svg)

    # Also mirror PNG to results/plots/
    plots_mirror = os.path.join(config["paths"]["results_plots_dir"], "feature_ablation_comparison.png")
    os.makedirs(os.path.dirname(plots_mirror), exist_ok=True)
    plot_feature_ablation_comparison(results_df, save_path_png=plots_mirror, save_path_svg=None)

    elapsed = time.time() - start_time
    logger.info("================================================================================")
    logger.info(f"Feature Ablation Study Completed in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    logger.info("================================================================================")

    # Print summary markdown table
    summary_cols = [
        "Model", "Experiment_ID", "Group_Name", "Num_Features", "Threshold_Opt",
        "Val_PR_AUC", "Val_ROC_AUC", "Val_F1", "Val_F2",
        "Test_PR_AUC", "Test_ROC_AUC", "Test_F1", "Test_F2", "Test_Brier_Score"
    ]
    try:
        print(results_df[summary_cols].to_markdown(index=False))
    except Exception:
        print(results_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
