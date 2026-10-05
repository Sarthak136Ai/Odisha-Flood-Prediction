"""
Script to combine all 24 yearly CSV files (2001-2024) into a unified dataset.
Performs post-combination integrity checks and writes verification metrics.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.combine_data import combine_all_years, STANDARD_DISTRICTS
from src.data.validate_data import EXPECTED_COLUMNS


def verify_combined_dataset(df: pd.DataFrame, output_path: str) -> dict:
    """Run thorough verification suite on the combined dataset."""
    print("\n" + "=" * 70)
    print("RUNNING POST-COMBINATION INTEGRITY VERIFICATION")
    print("=" * 70)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Check columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    
    # Check duplicates
    exact_dups = int(df.duplicated().sum())
    loc_date_dups = int(df.duplicated(subset=["District", "Block/Station", "Date"]).sum())
    
    # Check districts
    unique_districts = sorted(df["District"].unique())
    non_standard_districts = [d for d in unique_districts if d not in STANDARD_DISTRICTS]
    
    # Check dates
    df["Date_Parsed"] = pd.to_datetime(df["Date"])
    min_date = str(df["Date_Parsed"].min().date())
    max_date = str(df["Date_Parsed"].max().date())
    
    # Check targets & occurrences
    fo_1 = int((df["Flood_Occurred"] == 1).sum())
    fnd_1 = int((df["Flood_Next_Day"] == 1).sum())
    fnd_nulls = int(df["Flood_Next_Day"].isnull().sum())
    
    # Null summary
    nulls = {k: int(v) for k, v in df.isnull().sum().to_dict().items() if v > 0}
    
    report = {
        "file_path": output_path,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "schema_match": (len(missing_cols) == 0 and len(extra_cols) == 0),
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "exact_duplicates": exact_dups,
        "location_date_duplicates": loc_date_dups,
        "unique_districts_count": len(unique_districts),
        "unique_districts": unique_districts,
        "non_standard_districts": non_standard_districts,
        "unique_blocks_count": int(df["Block/Station"].nunique()),
        "min_date": min_date,
        "max_date": max_date,
        "max_rainfall_mm": float(df["Rainfall (mm)"].max()),
        "flood_occurred_1_count": fo_1,
        "flood_occurred_rate": round(fo_1 / total_rows, 5),
        "flood_next_day_1_count": fnd_1,
        "flood_next_day_null_count": fnd_nulls,
        "flood_next_day_rate": round(fnd_1 / (total_rows - fnd_nulls) if (total_rows - fnd_nulls) > 0 else 0, 5),
        "columns_with_nulls": nulls,
        "status": "VERIFIED" if (loc_date_dups == 0 and len(non_standard_districts) == 0 and len(missing_cols) == 0) else "FAILED"
    }
    
    print(f"Total Rows: {total_rows:,}")
    print(f"Total Columns: {total_cols} (Schema Match: {report['schema_match']})")
    print(f"Duplicates (Exact): {exact_dups}")
    print(f"Duplicates (Location-Date): {loc_date_dups}")
    print(f"Unique Districts ({len(unique_districts)}): Standard 30 districts verified = {len(non_standard_districts) == 0}")
    print(f"Unique Blocks/Stations: {report['unique_blocks_count']}")
    print(f"Date Range: {min_date} to {max_date}")
    print(f"Max Rainfall: {report['max_rainfall_mm']} mm")
    print(f"Flood Occurred Events: {fo_1:,} ({report['flood_occurred_rate'] * 100:.3f}%)")
    print(f"Flood Next Day Events: {fnd_1:,} ({report['flood_next_day_rate'] * 100:.3f}%)")
    print(f"Flood Next Day Nulls (Last observation day): {fnd_nulls}")
    print(f"Columns with nulls: {nulls}")
    print(f"Verification Status: {report['status']}")
    print("=" * 70)
    
    return report


def main():
    source_dir = "Rain_new_updated"
    output_path = "data/combined/Odisha_Flood_2001_2024.csv"
    metrics_dir = "results/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    combined_df = combine_all_years(source_dir=source_dir, output_path=output_path, recompute_continuity=True)
    report = verify_combined_dataset(combined_df, output_path)
    
    report_path = os.path.join(metrics_dir, "combined_data_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Verification report saved to: {report_path}")


if __name__ == "__main__":
    main()
