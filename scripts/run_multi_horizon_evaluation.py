"""
Command-line runner for Multi-Horizon Flood Forecasting Evaluation in Odisha.

This script:
1. Loads the historical 2001–2024 dataset (2,752,252 records).
2. Performs walk-forward chronological splitting:
   - Train: 2001–2018 (2,063,989 records)
   - Val:   2019–2021 (344,140 records)
   - Test:  2022–2024 (343,758 records)
3. Generates and verifies multi-day forecast horizon targets:
   - Flood_T+1: 24h lead time (next-day flood)
   - Flood_T+2: 48h lead time (two days ahead)
   - Flood_T+3: 72h lead time (three days ahead)
4. Evaluates both the linear baseline (Logistic Regression) and the production
   champion (XGBoost) across all horizons using strictly past observations (time <= T).
5. Tunes decision thresholds strictly on Validation data (frozen for Test evaluation).
6. Computes ROC-AUC, PR-AUC, Precision, Recall, F1, F2, Brier Score, and degradation vs T+1.
7. Saves comprehensive CSV tables and publication-grade degradation plots.
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
from src.features.rainfall_features import compute_rainfall_intensity_features
from src.features.multi_horizon import (
    generate_multi_horizon_targets,
    verify_target_alignment
)
from src.models.logistic_regression import create_logistic_regression_pipeline
from src.models.xgboost_model import create_xgboost_model
from src.evaluation.multi_horizon_evaluation import (
    MultiHorizonEvaluationEngine,
    plot_multi_horizon_degradation
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Horizon Flood Forecasting Evaluation.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--horizons", type=str, default="1,2,3", help="Comma-separated forecast horizons in days.")
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "f2"], help="Metric for validation threshold tuning.")
    parser.add_argument("--output-csv", type=str, default="reports/multi_horizon_evaluation.csv", help="Primary results CSV destination.")
    parser.add_argument("--plots-dir", type=str, default="reports/plots", help="Directory for degradation plots.")
    args = parser.parse_args()

    start_time = time.time()
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    logger.info("================================================================================")
    logger.info(f"Starting Multi-Horizon Flood Forecasting Evaluation (Horizons: {horizons})")
    logger.info("================================================================================")

    # 1. Load config and historical data
    config = load_config(args.config)
    data_path = config["paths"]["combined_data_path"]
    logger.info(f"Loading combined historical dataset from: {data_path}")
    df = load_combined_data(data_path, parse_dates=False)

    # Ensure intensity features exist if configured
    df = compute_rainfall_intensity_features(df)

    # Define feature set (strictly predictors observed at time <= T)
    feature_cols = (
        config["features"]["temporal_cols"] +
        config["features"]["rainfall_cols"]
    )
    if "rainfall_intensity_cols" in config.get("features", {}):
        for col in config["features"]["rainfall_intensity_cols"]:
            if col in df.columns and col not in feature_cols:
                feature_cols.append(col)

    logger.info(f"Using {len(feature_cols)} strictly past/current predictor features (Time <= T)")

    # 2. Chronological Walk-Forward Splits
    train_years = config["split"]["train_years"]
    val_years = config["split"]["val_years"]
    test_years = config["split"]["test_years"]

    train_df = df[df["Year"].isin(train_years)].copy()
    val_df = df[df["Year"].isin(val_years)].copy()
    test_df = df[df["Year"].isin(test_years)].copy()

    logger.info(f"Temporal splits: Train={len(train_df):,} rows, Val={len(val_df):,} rows, Test={len(test_df):,} rows")

    # 3. Mathematical Alignment Audit for Each Horizon
    logger.info("Conducting strict mathematical alignment audits...")
    for h in horizons:
        df_target = generate_multi_horizon_targets(train_df.head(10000), horizons=[h])
        audit = verify_target_alignment(df_target, horizon=h)
        logger.info(
            f"  [Horizon T+{h}] Audit Passed: {audit['is_perfectly_aligned']} | "
            f"Valid Rows: {audit['valid_target_rows']} | Value Mismatches: {audit['value_mismatches']}"
        )

    # 4. Model Factories
    model_factories = {
        "Logistic Regression": lambda spw: create_logistic_regression_pipeline(
            max_iter=config["models"]["logistic_regression"].get("max_iter", 1000),
            class_weight=config["models"]["logistic_regression"].get("class_weight", "balanced"),
            random_state=config["project"]["random_seed"]
        ),
        "XGBoost": lambda spw: create_xgboost_model(
            n_estimators=config["models"]["xgboost"].get("n_estimators", 200),
            max_depth=config["models"]["xgboost"].get("max_depth", 6),
            learning_rate=config["models"]["xgboost"].get("learning_rate", 0.05),
            scale_pos_weight=spw,
            random_state=config["project"]["random_seed"],
            n_jobs=config["models"]["xgboost"].get("n_jobs", -1)
        )
    }

    # 5. Execute Multi-Horizon Evaluation Engine
    engine = MultiHorizonEvaluationEngine(horizons=horizons, optimization_metric=args.metric)
    results_df = engine.run_multi_horizon_evaluation(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        model_factories=model_factories
    )

    # 6. Save Results CSV to reports and results directories
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved primary results CSV to: {args.output_csv}")

    metrics_mirror = os.path.join(config["paths"]["results_metrics_dir"], "multi_horizon_evaluation.csv")
    os.makedirs(os.path.dirname(metrics_mirror), exist_ok=True)
    results_df.to_csv(metrics_mirror, index=False)
    logger.info(f"Mirrored results CSV to: {metrics_mirror}")

    # 7. Generate Publication Degradation Plots
    os.makedirs(args.plots_dir, exist_ok=True)
    plot_png = os.path.join(args.plots_dir, "multi_horizon_degradation.png")
    plot_svg = os.path.join(args.plots_dir, "multi_horizon_degradation.svg")
    plot_multi_horizon_degradation(results_df, save_path_png=plot_png, save_path_svg=plot_svg)

    plots_mirror = os.path.join(config["paths"]["results_plots_dir"], "multi_horizon_degradation.png")
    os.makedirs(os.path.dirname(plots_mirror), exist_ok=True)
    plot_multi_horizon_degradation(results_df, save_path_png=plots_mirror, save_path_svg=None)

    elapsed = time.time() - start_time
    logger.info("================================================================================")
    logger.info(f"Multi-Horizon Forecasting Evaluation Completed in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    logger.info("================================================================================")

    # Print summary comparison table
    summary_cols = [
        "Model", "Horizon_Label", "Lead_Time_Hours", "Threshold_Opt",
        "Test_ROC_AUC", "Test_PR_AUC", "Test_Precision", "Test_Recall", "Test_F1", "Test_F2",
        "Delta_Test_PR_AUC", "Delta_Test_ROC_AUC", "Delta_Test_F1"
    ]
    print("\n### Multi-Horizon Forecast Degradation Comparison Table\n")
    try:
        print(results_df[summary_cols].to_markdown(index=False))
    except Exception:
        print(results_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
