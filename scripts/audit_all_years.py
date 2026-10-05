"""
Script to execute the complete Phase 2 24-Year Dataset Audit.
Saves machine-readable audit reports to results/metrics/.
"""

import os
import sys
import json
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.validate_data import audit_all_years, check_cross_year_continuity, EXPECTED_COLUMNS


def main():
    print("=" * 70)
    print("PHASE 2: 24-YEAR DATASET AUDIT (2001 - 2024)")
    print("=" * 70)

    source_dir = "Rain_new_updated"
    metrics_dir = "results/metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Scanning source directory: {source_dir}")
    print(f"Expected schema columns ({len(EXPECTED_COLUMNS)} cols):")
    for i, col in enumerate(EXPECTED_COLUMNS, 1):
        print(f"  {i:02d}. {col}")
    print("-" * 70)

    summary_df, detailed_report = audit_all_years(source_dir)
    continuity_report = check_cross_year_continuity(source_dir)

    summary_csv_path = os.path.join(metrics_dir, "audit_summary_2001_2024.csv")
    detailed_json_path = os.path.join(metrics_dir, "audit_detailed_report.json")
    continuity_json_path = os.path.join(metrics_dir, "cross_year_continuity.json")

    summary_df.to_csv(summary_csv_path, index=False)
    with open(detailed_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_report, f, indent=2)
    with open(continuity_json_path, "w", encoding="utf-8") as f:
        json.dump(continuity_report, f, indent=2)

    print("\n--- AUDIT SUMMARY TABLE ---")
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY ACROSS 2001-2024:")
    print(f"Total Years Analyzed: {len(summary_df)}")
    print(f"Total Rows: {summary_df['Rows'].sum():,}")
    print(f"Total File Size: {summary_df['File_Size_MB'].sum():.2f} MB")
    print(f"Total Null Values: {summary_df['Total_Nulls'].sum()}")
    print(f"Total Exact Duplicates: {summary_df['Duplicates'].sum()}")
    print(f"Total Location-Date Duplicates: {summary_df['Loc_Date_Dups'].sum()}")
    print(f"Total Flood Occurred Events (1): {summary_df['Flood_Occurred_1'].sum():,} ({summary_df['Flood_Occurred_1'].sum() / summary_df['Rows'].sum() * 100:.3f}%)")
    print(f"Total Flood Next Day Events (1): {summary_df['Flood_Next_Day_1'].sum():,} ({summary_df['Flood_Next_Day_1'].sum() / summary_df['Rows'].sum() * 100:.3f}%)")
    print(f"Schema match across all years: {summary_df['Schema_Match'].all()}")
    print(f"Reports saved to:")
    print(f"  - {summary_csv_path}")
    print(f"  - {detailed_json_path}")
    print(f"  - {continuity_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
