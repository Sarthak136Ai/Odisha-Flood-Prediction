"""
Script to execute Walk-Forward Temporal Validation across all 5 chronological folds (2017–2021).
Generates detailed fold-by-fold results, aggregated summary metrics, plots, and markdown tables.
"""

import sys
import os
import time
import argparse
import logging
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_combined_data, load_config
from src.evaluation.temporal_validation import (
    TemporalWalkForwardValidator,
    format_validation_summary_table
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Walk-Forward Temporal Cross-Validation for Odisha Flood Prediction.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--models", nargs="+", default=None, help="List of models to evaluate (e.g. 'Logistic Regression' 'Decision Tree' 'Random Forest' 'XGBoost' 'ANN (MLP)')")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory for reports output")
    parser.add_argument("--plots-dir", type=str, default="reports/plots", help="Directory for plots output")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("ODISHA FLOOD PREDICTION: TEMPORAL WALK-FORWARD VALIDATION")
    logger.info("=" * 70)

    config = load_config(args.config)
    dataset_path = config["paths"]["combined_data_path"]
    
    logger.info(f"Loading historical processed dataset from {dataset_path}...")
    df = load_combined_data(dataset_path, parse_dates=False)
    logger.info(f"Loaded {len(df):,} total records.")

    validator = TemporalWalkForwardValidator(config=config, random_state=config["project"]["random_seed"])
    
    results_df, summary_df, summary_dict = validator.run_validation(df, models_to_run=args.models)
    
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD TEMPORAL VALIDATION SUMMARY (Mean ± Std across 5 Folds)")
    logger.info("=" * 70)
    try:
        table_str = format_validation_summary_table(summary_df)
        print("\n" + table_str + "\n")
    except Exception as e:
        print(summary_df[["Model", "ROC_AUC_Summary", "PR_AUC_Summary", "F1_Summary", "Brier_Score_Summary"]])

    # Save CSV and JSON artifacts
    paths = validator.save_artifacts(
        results_df=results_df,
        summary_df=summary_df,
        summary_dict=summary_dict,
        reports_dir=args.reports_dir,
        results_metrics_dir=config["paths"]["results_metrics_dir"]
    )

    # Generate Visualizations
    plot_paths = validator.generate_visualizations(results_df=results_df, plots_dir=args.plots_dir)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"Temporal validation completed successfully in {elapsed:.2f} seconds.")
    logger.info(f"Detailed Results: {paths['detailed_csv']}")
    logger.info(f"Summary Table:    {paths['summary_csv']}")
    logger.info(f"Summary Metadata: {paths['summary_json']}")
    logger.info(f"Folds Plot:       {plot_paths['folds_plot']}")
    logger.info(f"PR Curves Plot:   {plot_paths['pr_curves_plot']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
