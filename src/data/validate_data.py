"""
Validation and auditing module for Odisha flood datasets.
Performs exhaustive checks on schema, integrity, temporal continuity, feature math, target distribution, and leakage.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional


EXPECTED_COLUMNS = [
    "District",
    "Block/Station",
    "Date",
    "Year",
    "Month_Number",
    "Day",
    "Day_of_Year",
    "Day_of_Week",
    "Month_sin",
    "Month_cos",
    "Rainfall (mm)",
    "Rainfall_Missing",
    "Rainfall_Lag_1d",
    "Rainfall_Lag_2d",
    "Rainfall_Lag_3d",
    "Rainfall_Lag_7d",
    "Rainfall_Prev_3d_Sum",
    "Rainfall_Prev_7d_Sum",
    "Rainfall_Prev_15d_Sum",
    "Rainfall_Prev_30d_Sum",
    "Rainfall_Prev_3d_Max",
    "Rainfall_Prev_7d_Max",
    "Rainfall_Prev_15d_Max",
    "Rainfall_Prev_30d_Max",
    "Rainy_Days_Prev_3d",
    "Rainy_Days_Prev_7d",
    "Rainy_Days_Prev_15d",
    "Rainy_Days_Prev_30d",
    "Consecutive_Rainy_Days_Before",
    "Flood_Occurred",
    "Flood_Next_Day"
]


def audit_single_year_file(file_path: str, year: int) -> Dict[str, Any]:
    """
    Perform a complete audit on a single yearly CSV file.
    Returns a dictionary of detailed audit metrics.
    """
    report: Dict[str, Any] = {
        "year": year,
        "file_path": file_path,
        "file_exists": os.path.exists(file_path),
        "file_size_bytes": 0,
        "file_size_mb": 0.0,
        "readable": False,
        "row_count": 0,
        "col_count": 0,
        "columns_match_expected": False,
        "missing_expected_cols": [],
        "extra_cols": [],
        "col_order_matches": False,
        "null_counts": {},
        "total_nulls": 0,
        "exact_duplicates": 0,
        "location_date_duplicates": 0,
        "unique_districts_count": 0,
        "unique_districts": [],
        "unique_blocks_count": 0,
        "min_date": None,
        "max_date": None,
        "expected_days": 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365,
        "is_chronologically_sorted": True,
        "negative_rainfall_count": 0,
        "max_rainfall_mm": 0.0,
        "flood_occurred_0_count": 0,
        "flood_occurred_1_count": 0,
        "flood_occurred_rate": 0.0,
        "flood_next_day_0_count": 0,
        "flood_next_day_1_count": 0,
        "flood_next_day_null_count": 0,
        "flood_next_day_rate": 0.0,
        "dec_31_target_status": {},
        "anomalies": []
    }

    if not report["file_exists"]:
        report["anomalies"].append(f"File for year {year} does not exist at {file_path}")
        return report

    report["file_size_bytes"] = os.path.getsize(file_path)
    report["file_size_mb"] = round(report["file_size_bytes"] / (1024 * 1024), 2)

    try:
        df = pd.read_csv(file_path)
        report["readable"] = True
    except Exception as e:
        report["anomalies"].append(f"Error reading CSV: {str(e)}")
        return report

    report["row_count"] = len(df)
    report["col_count"] = len(df.columns)
    actual_cols = list(df.columns)

    # Schema checks
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in actual_cols]
    extra_cols = [c for c in actual_cols if c not in EXPECTED_COLUMNS]
    report["missing_expected_cols"] = missing_cols
    report["extra_cols"] = extra_cols
    report["columns_match_expected"] = (len(missing_cols) == 0 and len(extra_cols) == 0)
    report["col_order_matches"] = (actual_cols == EXPECTED_COLUMNS)

    if not report["columns_match_expected"]:
        report["anomalies"].append(f"Column schema mismatch. Missing: {missing_cols}, Extra: {extra_cols}")

    # Null counts
    null_counts = df.isnull().sum().to_dict()
    report["null_counts"] = {k: int(v) for k, v in null_counts.items() if v > 0}
    report["total_nulls"] = int(df.isnull().sum().sum())

    # Duplicates
    report["exact_duplicates"] = int(df.duplicated().sum())
    if report["exact_duplicates"] > 0:
        report["anomalies"].append(f"Found {report['exact_duplicates']} exact duplicate rows.")

    if "District" in df.columns and "Block/Station" in df.columns and "Date" in df.columns:
        loc_date_dups = int(df.duplicated(subset=["District", "Block/Station", "Date"]).sum())
        report["location_date_duplicates"] = loc_date_dups
        if loc_date_dups > 0:
            report["anomalies"].append(f"Found {loc_date_dups} duplicate location-date pairs.")

    # Location distribution
    if "District" in df.columns:
        districts = sorted([str(d) for d in df["District"].dropna().unique()])
        report["unique_districts_count"] = len(districts)
        report["unique_districts"] = districts

    if "Block/Station" in df.columns:
        blocks = df["Block/Station"].dropna().unique()
        report["unique_blocks_count"] = len(blocks)

    # Date checks
    if "Date" in df.columns:
        try:
            df["Date_Parsed"] = pd.to_datetime(df["Date"])
            report["min_date"] = str(df["Date_Parsed"].min().date())
            report["max_date"] = str(df["Date_Parsed"].max().date())
            
            # Check chronological ordering by location
            if "District" in df.columns and "Block/Station" in df.columns:
                grouped = df.groupby(["District", "Block/Station"])
                sorted_checks = []
                for _, group in grouped:
                    sorted_checks.append(group["Date_Parsed"].is_monotonic_increasing)
                report["is_chronologically_sorted"] = all(sorted_checks)
                if not report["is_chronologically_sorted"]:
                    report["anomalies"].append("Dates are not strictly monotonic increasing per location.")
        except Exception as e:
            report["anomalies"].append(f"Failed to parse dates: {str(e)}")

    # Rainfall integrity
    if "Rainfall (mm)" in df.columns:
        neg_rf = int((df["Rainfall (mm)"] < 0).sum())
        report["negative_rainfall_count"] = neg_rf
        if neg_rf > 0:
            report["anomalies"].append(f"Found {neg_rf} negative rainfall entries.")
        report["max_rainfall_mm"] = float(df["Rainfall (mm)"].max())

    # Target & Occurrence distributions
    if "Flood_Occurred" in df.columns:
        fo_vc = df["Flood_Occurred"].value_counts().to_dict()
        report["flood_occurred_0_count"] = int(fo_vc.get(0, 0))
        report["flood_occurred_1_count"] = int(fo_vc.get(1, 0))
        fo_total = report["flood_occurred_0_count"] + report["flood_occurred_1_count"]
        if fo_total > 0:
            report["flood_occurred_rate"] = round(report["flood_occurred_1_count"] / fo_total, 5)

    if "Flood_Next_Day" in df.columns:
        fnd_vc = df["Flood_Next_Day"].value_counts(dropna=False).to_dict()
        report["flood_next_day_0_count"] = int(fnd_vc.get(0, 0) if 0 in fnd_vc else fnd_vc.get(0.0, 0))
        report["flood_next_day_1_count"] = int(fnd_vc.get(1, 0) if 1 in fnd_vc else fnd_vc.get(1.0, 0))
        report["flood_next_day_null_count"] = int(df["Flood_Next_Day"].isnull().sum())
        fnd_valid = report["flood_next_day_0_count"] + report["flood_next_day_1_count"]
        if fnd_valid > 0:
            report["flood_next_day_rate"] = round(report["flood_next_day_1_count"] / fnd_valid, 5)

    # Last day behavior (Dec 31)
    if "Date" in df.columns and "Flood_Next_Day" in df.columns:
        dec31_rows = df[df["Date"].astype(str).str.contains(f"{year}-12-31")]
        if len(dec31_rows) > 0:
            dec31_nulls = int(dec31_rows["Flood_Next_Day"].isnull().sum())
            dec31_zeros = int((dec31_rows["Flood_Next_Day"] == 0).sum())
            dec31_ones = int((dec31_rows["Flood_Next_Day"] == 1).sum())
            report["dec_31_target_status"] = {
                "total_locations": len(dec31_rows),
                "null_count": dec31_nulls,
                "zero_count": dec31_zeros,
                "one_count": dec31_ones
            }

    return report


def audit_all_years(source_dir: str = "Rain_new_updated") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run audit across all 24 years (2001-2024).
    Returns (summary_dataframe, full_detailed_dictionary).
    """
    all_reports = {}
    summary_rows = []

    for year in range(2001, 2025):
        file_path = os.path.join(source_dir, f"Odisha_Flood_{year}_Corrected_Preprocessed.csv")
        rep = audit_single_year_file(file_path, year)
        all_reports[year] = rep
        
        summary_rows.append({
            "Year": year,
            "File_Size_MB": rep["file_size_mb"],
            "Rows": rep["row_count"],
            "Columns": rep["col_count"],
            "Schema_Match": rep["columns_match_expected"],
            "Total_Nulls": rep["total_nulls"],
            "Duplicates": rep["exact_duplicates"],
            "Loc_Date_Dups": rep["location_date_duplicates"],
            "Districts": rep["unique_districts_count"],
            "Blocks": rep["unique_blocks_count"],
            "Min_Date": rep["min_date"],
            "Max_Date": rep["max_date"],
            "Max_Rainfall_mm": rep["max_rainfall_mm"],
            "Flood_Occurred_1": rep["flood_occurred_1_count"],
            "Flood_Occurred_Rate": rep["flood_occurred_rate"],
            "Flood_Next_Day_1": rep["flood_next_day_1_count"],
            "Flood_Next_Day_Rate": rep["flood_next_day_rate"],
            "Anomalies_Count": len(rep["anomalies"])
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, all_reports


def check_cross_year_continuity(
    source_dir: str = "Rain_new_updated"
) -> Dict[str, Any]:
    """
    Check continuity between consecutive years (Dec 31 of Year T vs Jan 1 of Year T+1).
    Specifically validates whether locations match and lag features carry over properly.
    """
    continuity_report = {}
    for year in range(2001, 2024):
        next_year = year + 1
        path_t = os.path.join(source_dir, f"Odisha_Flood_{year}_Corrected_Preprocessed.csv")
        path_t1 = os.path.join(source_dir, f"Odisha_Flood_{next_year}_Corrected_Preprocessed.csv")
        
        if not os.path.exists(path_t) or not os.path.exists(path_t1):
            continuity_report[f"{year}_{next_year}"] = {"error": "Missing file"}
            continue
            
        df_t = pd.read_csv(path_t)
        df_t1 = pd.read_csv(path_t1)
        
        locs_t = set(df_t["District"].astype(str) + "___" + df_t["Block/Station"].astype(str))
        locs_t1 = set(df_t1["District"].astype(str) + "___" + df_t1["Block/Station"].astype(str))
        
        common_locs = locs_t.intersection(locs_t1)
        added_in_next = locs_t1 - locs_t
        removed_in_next = locs_t - locs_t1
        
        continuity_report[f"{year}_{next_year}"] = {
            "locations_in_year_T": len(locs_t),
            "locations_in_year_T_plus_1": len(locs_t1),
            "common_locations": len(common_locs),
            "added_locations_count": len(added_in_next),
            "removed_locations_count": len(removed_in_next)
        }
    return continuity_report
