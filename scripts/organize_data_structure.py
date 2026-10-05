"""
Script to reorganize and build the standardized data architecture:
├── data/
│   ├── raw/
│   │   ├── historical/
│   │   │   ├── 2001.csv ... 2024.csv
│   │   └── operational/
│   │       └── 2025.csv
│   ├── processed/
│   │   ├── historical_2001_2024.csv
│   │   └── operational_2025.csv
│   └── predictions/
│       └── 2025_flood_risk_predictions.csv
"""

import os
import shutil
import joblib
import json
import logging
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.combine_data import (
    standardize_district_name,
    clean_single_year_df,
    recompute_cross_year_features,
    STANDARD_DISTRICTS
)
from src.features.rainfall_features import ClimatologicalAnomalyEngine, enrich_rainfall_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_data_directory_structure():
    logger.info("Setting up standardized data directory tree...")
    
    raw_hist_dir = "data/raw/historical"
    raw_ops_dir = "data/raw/operational"
    proc_dir = "data/processed"
    pred_dir = "data/predictions"
    
    os.makedirs(raw_hist_dir, exist_ok=True)
    os.makedirs(raw_ops_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    # 1. Copy and rename raw historical files 2001-2024
    logger.info("1. Populating data/raw/historical/ (2001.csv ... 2024.csv)...")
    for yr in range(2001, 2025):
        src_file = f"Rain_new_updated/Odisha_Flood_{yr}_Corrected_Preprocessed.csv"
        dst_file = os.path.join(raw_hist_dir, f"{yr}.csv")
        if os.path.exists(src_file) and not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)
            logger.info(f"  Copied {src_file} -> {dst_file}")
            
    # 2. Extract and format raw operational 2025 data
    logger.info("2. Creating data/raw/operational/2025.csv...")
    ops_excel_path = "data/operational/Blockwise_Daily_Rainfall_2025_Merged.xlsx"
    raw_2025_csv = os.path.join(raw_ops_dir, "2025.csv")
    
    if os.path.exists(ops_excel_path):
        df_2025_raw = pd.read_excel(ops_excel_path)
        # Rename Block -> Block/Station if needed
        if "Block" in df_2025_raw.columns and "Block/Station" not in df_2025_raw.columns:
            df_2025_raw.rename(columns={"Block": "Block/Station"}, inplace=True)
            
        df_2025_raw["District"] = df_2025_raw["District"].apply(standardize_district_name)
        df_2025_raw["Block/Station"] = df_2025_raw["Block/Station"].astype(str).str.strip()
        df_2025_raw["Date"] = pd.to_datetime(df_2025_raw["Date"]).dt.strftime("%Y-%m-%d")
        df_2025_raw["Rainfall (mm)"] = df_2025_raw["Rainfall (mm)"].fillna(0.0).astype(float)
        
        # Save raw 2025.csv
        df_2025_raw.to_csv(raw_2025_csv, index=False)
        logger.info(f"  Saved {raw_2025_csv} ({len(df_2025_raw):,} rows)")
    else:
        logger.warning(f"  Source Excel {ops_excel_path} not found!")

    # 3. Populate data/processed/historical_2001_2024.csv
    logger.info("3. Populating data/processed/historical_2001_2024.csv...")
    proc_hist_path = os.path.join(proc_dir, "historical_2001_2024.csv")
    existing_comb_path = "data/combined/Odisha_Flood_2001_2024.csv"
    
    if os.path.exists(existing_comb_path) and not os.path.exists(proc_hist_path):
        shutil.copy2(existing_comb_path, proc_hist_path)
        logger.info(f"  Copied {existing_comb_path} -> {proc_hist_path}")
    elif not os.path.exists(proc_hist_path):
        logger.error(f"  Processed historical file missing at {existing_comb_path}")

    # 4. Generate data/processed/operational_2025.csv with continuous boundary lags
    logger.info("4. Generating data/processed/operational_2025.csv with continuous lags from 2024 boundary...")
    proc_ops_path = os.path.join(proc_dir, "operational_2025.csv")
    
    # Load late 2024 history (Dec 2024) to ensure initial January 2025 lag calculations are 100% physically exact
    df_hist = pd.read_csv(proc_hist_path)
    late_2024 = df_hist[df_hist["Year"] == 2024].copy()
    late_2024 = late_2024[late_2024["Month_Number"] >= 11] # Nov & Dec 2024
    
    df_2025 = pd.read_csv(raw_2025_csv)
    df_2025["Date_Parsed"] = pd.to_datetime(df_2025["Date"])
    df_2025["Year"] = df_2025["Date_Parsed"].dt.year
    df_2025["Month_Number"] = df_2025["Date_Parsed"].dt.month
    df_2025["Day"] = df_2025["Date_Parsed"].dt.day
    df_2025["Day_of_Year"] = df_2025["Date_Parsed"].dt.dayofyear
    df_2025["Day_of_Week"] = df_2025["Date_Parsed"].dt.dayofweek
    df_2025["Month_sin"] = np.sin(2 * np.pi * df_2025["Month_Number"] / 12)
    df_2025["Month_cos"] = np.cos(2 * np.pi * df_2025["Month_Number"] / 12)
    df_2025["Rainfall_Missing"] = 0
    df_2025["Flood_Occurred"] = 0 # Baseline ground truth flag unlabelled for unseen operational period
    
    # Concatenate late 2024 + 2025 to compute lags across the boundary
    combined_seq = pd.concat([
        late_2024[["District", "Block/Station", "Date", "Rainfall (mm)", "Flood_Occurred"]],
        df_2025[["District", "Block/Station", "Date", "Rainfall (mm)", "Flood_Occurred"]]
    ], ignore_index=True)
    
    combined_seq["Date_Parsed"] = pd.to_datetime(combined_seq["Date"])
    combined_seq = combined_seq.sort_values(by=["District", "Block/Station", "Date_Parsed"]).reset_index(drop=True)
    
    grouped = combined_seq.groupby(["District", "Block/Station"])
    combined_seq["Rainfall_Lag_1d"] = grouped["Rainfall (mm)"].shift(1)
    combined_seq["Rainfall_Lag_2d"] = grouped["Rainfall (mm)"].shift(2)
    combined_seq["Rainfall_Lag_3d"] = grouped["Rainfall (mm)"].shift(3)
    combined_seq["Rainfall_Lag_7d"] = grouped["Rainfall (mm)"].shift(7)
    
    combined_seq["Rainfall_Prev_3d_Sum"] = grouped["Rainfall (mm)"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).sum())
    combined_seq["Rainfall_Prev_7d_Sum"] = grouped["Rainfall (mm)"].transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).sum())
    combined_seq["Rainfall_Prev_15d_Sum"] = grouped["Rainfall (mm)"].transform(lambda s: s.shift(1).rolling(window=15, min_periods=1).sum())
    combined_seq["Rainfall_Prev_30d_Sum"] = grouped["Rainfall (mm)"].transform(lambda s: s.shift(1).rolling(window=30, min_periods=1).sum())
    
    combined_seq["Rainfall_Prev_3d_Max"] = grouped["Rainfall (mm)"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).max())
    combined_seq["Rainfall_Prev_7d_Max"] = grouped["Rainfall (mm)"].transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).max())
    combined_seq["Rainfall_Prev_15d_Max"] = grouped["Rainfall (mm)"].transform(lambda s: s.shift(1).rolling(window=15, min_periods=1).max())
    combined_seq["Rainfall_Prev_30d_Max"] = grouped["Rainfall (mm)"].transform(lambda s: s.shift(1).rolling(window=30, min_periods=1).max())
    
    combined_seq["_rainy"] = (combined_seq["Rainfall (mm)"] >= 2.5).astype(float)
    combined_seq["Rainy_Days_Prev_3d"] = combined_seq.groupby(["District", "Block/Station"])["_rainy"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).sum())
    combined_seq["Rainy_Days_Prev_7d"] = combined_seq.groupby(["District", "Block/Station"])["_rainy"].transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).sum())
    combined_seq["Rainy_Days_Prev_15d"] = combined_seq.groupby(["District", "Block/Station"])["_rainy"].transform(lambda s: s.shift(1).rolling(window=15, min_periods=1).sum())
    combined_seq["Rainy_Days_Prev_30d"] = combined_seq.groupby(["District", "Block/Station"])["_rainy"].transform(lambda s: s.shift(1).rolling(window=30, min_periods=1).sum())
    
    # Consecutive rainy days
    def get_consec_rainy(s):
        shifted = s.shift(1).fillna(0)
        res = []
        c = 0
        for val in shifted:
            if val >= 2.5:
                c += 1
            else:
                c = 0
            res.append(c)
        return pd.Series(res, index=s.index)
        
    combined_seq["Consecutive_Rainy_Days_Before"] = grouped["Rainfall (mm)"].transform(get_consec_rainy)
    
    # Filter back only 2025 records
    combined_seq_2025 = combined_seq[combined_seq["Date_Parsed"].dt.year == 2025].copy()
    
    # Merge temporal columns back
    final_2025_df = pd.merge(
        df_2025[["District", "Block/Station", "Date", "Year", "Month_Number", "Day", "Day_of_Year", "Day_of_Week", "Month_sin", "Month_cos", "Rainfall (mm)", "Rainfall_Missing", "Flood_Occurred"]],
        combined_seq_2025.drop(columns=["Date_Parsed", "_rainy", "Rainfall (mm)", "Flood_Occurred"]),
        on=["District", "Block/Station", "Date"],
        how="left"
    )
    
    final_2025_df.fillna(0.0, inplace=True)
    final_2025_df.to_csv(proc_ops_path, index=False)
    logger.info(f"  Saved {proc_ops_path} ({len(final_2025_df):,} rows)")
    
    # 5. Run Frozen Model Inference to generate data/predictions/2025_flood_risk_predictions.csv
    logger.info("5. Generating data/predictions/2025_flood_risk_predictions.csv using frozen model...")
    pred_path = os.path.join(pred_dir, "2025_flood_risk_predictions.csv")
    
    model_path = "models/flood_prediction/best_model.pkl"
    meta_path = "models/flood_prediction/model_metadata.json"
    
    if os.path.exists(model_path) and os.path.exists(meta_path):
        model = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        feat_cols = meta["features"]
        opt_thresh = meta.get("optimal_threshold", 0.5)
        risk_thresh = meta.get("risk_thresholds", {"low_max": 0.30, "moderate_max": 0.70})
        
        X_2025 = final_2025_df[feat_cols]
        probas_2025 = model.predict_proba(X_2025)[:, 1]
        
        preds_df = final_2025_df[["District", "Block/Station", "Date", "Year", "Month_Number", "Day", "Rainfall (mm)", "Rainfall_Prev_7d_Sum"]].copy()
        preds_df["Predicted_Flood_Probability"] = np.round(probas_2025, 4)
        preds_df["Predicted_Flood_Next_Day"] = (probas_2025 >= opt_thresh).astype(int)
        
        def assign_risk(p):
            if p >= risk_thresh["moderate_max"]:
                return "HIGH"
            elif p >= risk_thresh["low_max"]:
                return "MODERATE"
            else:
                return "LOW"
                
        preds_df["Predicted_Risk_Level"] = preds_df["Predicted_Flood_Probability"].apply(assign_risk)
        
        preds_df.to_csv(pred_path, index=False)
        logger.info(f"  Saved {pred_path} ({len(preds_df):,} rows)")
    else:
        logger.warning("  Model artifact not found for 2025 predictions generation.")
        
    logger.info("Standardized data directory setup completed successfully!")


if __name__ == "__main__":
    build_data_directory_structure()
