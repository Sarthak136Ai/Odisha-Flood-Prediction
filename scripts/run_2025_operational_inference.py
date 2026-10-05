"""
CLI runner script for 2025 Unseen Operational Inference Pipeline.
Validates raw 2025 rainfall data, stitches warm-up historical context,
executes the frozen flood prediction model, and generates predictions with top risk drivers.
"""

import os
import sys
import json
import pandas as pd

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.unseen_2025_pipeline import Operational2025Pipeline


def main():
    print("=" * 70)
    print("2025 UNSEEN OPERATIONAL INFERENCE PIPELINE")
    print("=" * 70)

    raw_2025_path = "data/raw/operational/2025.csv"
    output_pred_path = "data/predictions/2025_flood_risk_predictions.csv"

    if not os.path.exists(raw_2025_path):
        print(f"Error: Raw 2025 operational dataset not found at {raw_2025_path}")
        sys.exit(1)

    print(f"Loading raw operational dataset: {raw_2025_path}...")
    df_raw = pd.read_csv(raw_2025_path)

    pipeline = Operational2025Pipeline()
    print("Executing validation, warm-up feature engineering, and frozen model inference...")

    df_preds, summary = pipeline.run_pipeline(
        raw_df_2025=df_raw,
        output_csv_path=output_pred_path
    )

    print("\n--- VALIDATION REPORT ---")
    val = summary["validation"]
    print(f"Status: {val['validation_status']}")
    print(f"Total Rows: {val['row_count']:,}")
    print(f"Date Range: {val['date_range']}")
    print(f"Districts: {val['districts_count']} (Non-standard: {len(val['non_standard_districts'])})")
    print(f"Blocks/Stations: {val['blocks_count']}")
    print(f"Missing Rainfall: {val['missing_rainfall_count']}")
    print(f"Duplicates: {val['duplicate_count']}")
    if val["warnings"]:
        print(f"Warnings: {val['warnings']}")

    print("\n--- WARM-UP & FEATURE ENGINEERING ---")
    warm = summary["warmup"]
    print(f"Warm-up Applied: {warm['warmup_applied']} ({warm['warmup_rows_used']:,} historical Dec 2024 rows stitched)")
    print(f"Insufficient Context Warning: {warm['insufficient_context_warning']}")

    print("\n--- OPERATIONAL PREDICTIONS SUMMARY ---")
    print(f"Total Output Predictions: {summary['total_predictions']:,}")
    print(f"High Risk Days (>=70%): {summary['high_risk_count']:,}")
    print(f"Moderate Risk Days (30-70%): {summary['moderate_risk_count']:,}")
    print(f"Low Risk Days (<30%): {summary['low_risk_count']:,}")
    print(f"Output File: {output_pred_path}")

    print("\nSample Generated Predictions:")
    cols_display = ["District", "Block/Station", "Date", "Rainfall (mm)", "Predicted_Probability", "Risk_Level", "Top_Risk_Driver_1", "Top_Risk_Driver_2"]
    print(df_preds[cols_display].head(10).to_string(index=False))
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
