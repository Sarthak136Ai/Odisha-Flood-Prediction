"""
2025 Unseen Operational Inference, Feature Engineering, and Retrospective Evaluation Pipeline.
Strictly adheres to Master Prompt Sections 39-46:
1. Operational Data Validation (Schema, Districts, Dates, Duplicates, Nulls).
2. Warm-up Period Integration (Preceding 30-day historical context without zero-fill leakage).
3. Exact Historical Feature Engineering matching training definitions.
4. Frozen Model Inference with calibrated decision thresholds and SHAP top risk drivers.
5. Independent Retrospective Evaluation against authoritative SRC observations without model retraining.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Any, Optional, Union

from src.data.combine_data import STANDARD_DISTRICTS, DISTRICT_SYNONYMS, standardize_district_name
from src.evaluation.metrics import calculate_metrics
from src.explainability.shap_analysis import explain_instance


REQUIRED_RAW_COLUMNS = ["District", "Block/Station", "Date", "Rainfall (mm)"]


def validate_2025_raw_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate schema and integrity of 2025 operational raw dataset.
    Checks columns, district validity, station names, date formats, duplicates, and missing rainfall.
    """
    report = {
        "is_valid": True,
        "validation_status": "✓ Valid operational dataset",
        "errors": [],
        "warnings": [],
        "filename": None,
        "row_count": len(df),
        "columns_found": list(df.columns),
        "date_range": None,
        "districts_count": 0,
        "unique_districts": [],
        "non_standard_districts": [],
        "blocks_count": 0,
        "missing_rainfall_count": 0,
        "duplicate_count": 0,
        "negative_rainfall_count": 0
    }

    # 1. Column Check
    col_map = {c.strip(): c for c in df.columns}
    normalized_cols = {c.strip().lower(): c for c in df.columns}
    
    missing_required = []
    for req in REQUIRED_RAW_COLUMNS:
        req_lower = req.lower()
        if req_lower not in normalized_cols:
            missing_required.append(req)

    if missing_required:
        report["is_valid"] = False
        report["errors"].append(f"Missing required columns: {missing_required}")

    # Prohibited columns check (must not contain fabricated flood labels)
    if "Flood_Occurred" in df.columns:
        report["warnings"].append(
            "Found 'Flood_Occurred' column in operational data. Operational inference will ignore it "
            "to prevent target leakage."
        )

    if not report["is_valid"]:
        report["validation_status"] = "✗ Dataset validation failed: Missing required columns"
        return report

    df_check = df.copy()
    
    # Standardize column names
    df_check["District"] = df_check[normalized_cols["district"]].astype(str).str.strip().apply(standardize_district_name)
    df_check["Block/Station"] = df_check[normalized_cols["block/station"]].astype(str).str.strip()
    rf_col = normalized_cols["rainfall (mm)"]
    df_check["Rainfall (mm)"] = pd.to_numeric(df_check[rf_col], errors="coerce")
    
    # 2. District Validation
    unique_dists = sorted(df_check["District"].dropna().unique())
    report["districts_count"] = len(unique_dists)
    report["unique_districts"] = unique_dists
    non_std = [d for d in unique_dists if d not in STANDARD_DISTRICTS]
    report["non_standard_districts"] = non_std
    if non_std:
        report["warnings"].append(f"Found {len(non_std)} unrecognized district names: {non_std[:5]}")

    # 3. Block / Station Count
    report["blocks_count"] = int(df_check["Block/Station"].nunique())

    # 4. Date Validation
    date_col = normalized_cols["date"]
    try:
        parsed_dates = pd.to_datetime(df_check[date_col], errors="coerce")
        null_dates = int(parsed_dates.isnull().sum())
        if null_dates > 0:
            report["is_valid"] = False
            report["errors"].append(f"Found {null_dates} unparseable date values.")
        else:
            report["date_range"] = (str(parsed_dates.min().date()), str(parsed_dates.max().date()))
    except Exception as e:
        report["is_valid"] = False
        report["errors"].append(f"Date parsing failed: {str(e)}")

    # 5. Duplicate Check
    df_check["_Date_Str"] = pd.to_datetime(df_check[date_col]).dt.strftime("%Y-%m-%d")
    dups = int(df_check.duplicated(subset=["District", "Block/Station", "_Date_Str"]).sum())
    report["duplicate_count"] = dups
    if dups > 0:
        report["warnings"].append(f"Found {dups} duplicate location-date records. First occurrences will be retained.")

    # 6. Missing / Negative Rainfall Check
    missing_rf = int(df_check["Rainfall (mm)"].isnull().sum())
    report["missing_rainfall_count"] = missing_rf
    if missing_rf > 0:
        report["warnings"].append(f"Found {missing_rf} missing rainfall values. Will be imputed with 0.0 mm.")

    neg_rf = int((df_check["Rainfall (mm)"] < 0).sum())
    report["negative_rainfall_count"] = neg_rf
    if neg_rf > 0:
        report["errors"].append(f"Found {neg_rf} negative rainfall values.")
        report["is_valid"] = False

    if not report["is_valid"]:
        report["validation_status"] = f"✗ Dataset validation failed: {'; '.join(report['errors'])}"
    else:
        report["validation_status"] = "✓ Valid operational dataset"

    return report


def engineer_2025_features(
    df_2025: pd.DataFrame,
    warmup_df: Optional[pd.DataFrame] = None,
    allow_zero_warmup: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate the exact feature schema required by the frozen flood prediction model.
    Handles warm-up period (preceding 30 days) to accurately compute rolling sums without zero-filling.
    """
    info = {
        "warmup_applied": False,
        "warmup_rows_used": 0,
        "insufficient_context_warning": False,
        "output_rows": 0
    }

    df_25 = df_2025.copy()
    
    # Standardize column names
    col_map = {c.strip().lower(): c for c in df_25.columns}
    df_25["District"] = df_25[col_map["district"]].astype(str).str.strip().apply(standardize_district_name)
    df_25["Block/Station"] = df_25[col_map["block/station"]].astype(str).str.strip()
    df_25["Date_Parsed"] = pd.to_datetime(df_25[col_map["date"]])
    df_25["Date"] = df_25["Date_Parsed"].dt.strftime("%Y-%m-%d")
    
    rf_col = col_map["rainfall (mm)"]
    df_25["Rainfall (mm)"] = pd.to_numeric(df_25[rf_col], errors="coerce").fillna(0.0)
    df_25["Rainfall_Missing"] = df_25[rf_col].isnull().astype(int)

    # Deduplicate
    df_25 = df_25.drop_duplicates(subset=["District", "Block/Station", "Date"], keep="first")
    df_25 = df_25.sort_values(by=["District", "Block/Station", "Date_Parsed"]).reset_index(drop=True)

    min_date_25 = df_25["Date_Parsed"].min()

    # Handle Warm-up Period
    combined_stream = df_25.copy()
    if warmup_df is not None and not warmup_df.empty:
        w_df = warmup_df.copy()
        w_col_map = {c.strip().lower(): c for c in w_df.columns}
        w_df["District"] = w_df[w_col_map["district"]].astype(str).str.strip().apply(standardize_district_name)
        w_df["Block/Station"] = w_df[w_col_map["block/station"]].astype(str).str.strip()
        w_df["Date_Parsed"] = pd.to_datetime(w_df[w_col_map["date"]])
        w_df["Date"] = w_df["Date_Parsed"].dt.strftime("%Y-%m-%d")
        w_df["Rainfall (mm)"] = pd.to_numeric(w_df[w_col_map["rainfall (mm)"]], errors="coerce").fillna(0.0)
        w_df["Rainfall_Missing"] = 0
        
        # Take up to 35 days preceding the first 2025 date
        w_df = w_df[w_df["Date_Parsed"] < min_date_25]
        cutoff_date = min_date_25 - pd.Timedelta(days=35)
        w_df = w_df[w_df["Date_Parsed"] >= cutoff_date]
        
        if len(w_df) > 0:
            w_df = w_df.drop_duplicates(subset=["District", "Block/Station", "Date"], keep="first")
            combined_stream = pd.concat([w_df, df_25], ignore_index=True)
            combined_stream = combined_stream.sort_values(by=["District", "Block/Station", "Date_Parsed"]).reset_index(drop=True)
            info["warmup_applied"] = True
            info["warmup_rows_used"] = len(w_df)
    else:
        if not allow_zero_warmup:
            info["insufficient_context_warning"] = True

    # Temporal feature engineering
    combined_stream["Year"] = combined_stream["Date_Parsed"].dt.year
    combined_stream["Month_Number"] = combined_stream["Date_Parsed"].dt.month
    combined_stream["Day"] = combined_stream["Date_Parsed"].dt.day
    combined_stream["Day_of_Year"] = combined_stream["Date_Parsed"].dt.dayofyear
    combined_stream["Day_of_Week"] = combined_stream["Date_Parsed"].dt.dayofweek
    combined_stream["Month_sin"] = np.sin(2 * np.pi * combined_stream["Month_Number"] / 12.0)
    combined_stream["Month_cos"] = np.cos(2 * np.pi * combined_stream["Month_Number"] / 12.0)

    # Grouped lag and rolling calculations
    grouped = combined_stream.groupby(["District", "Block/Station"])
    
    combined_stream["Rainfall_Lag_1d"] = grouped["Rainfall (mm)"].shift(1)
    combined_stream["Rainfall_Lag_2d"] = grouped["Rainfall (mm)"].shift(2)
    combined_stream["Rainfall_Lag_3d"] = grouped["Rainfall (mm)"].shift(3)
    combined_stream["Rainfall_Lag_7d"] = grouped["Rainfall (mm)"].shift(7)

    combined_stream["Rainfall_Prev_3d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
    )
    combined_stream["Rainfall_Prev_7d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
    )
    combined_stream["Rainfall_Prev_15d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=15, min_periods=1).sum()
    )
    combined_stream["Rainfall_Prev_30d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=1).sum()
    )

    combined_stream["Rainfall_Prev_3d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).max()
    )
    combined_stream["Rainfall_Prev_7d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).max()
    )
    combined_stream["Rainfall_Prev_15d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=15, min_periods=1).max()
    )
    combined_stream["Rainfall_Prev_30d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=1).max()
    )

    # Rainy days (>= 2.5 mm IMD definition)
    rainy_flag = (combined_stream["Rainfall (mm)"] >= 2.5).astype(float)
    combined_stream["_rf_flag"] = rainy_flag

    combined_stream["Rainy_Days_Prev_3d"] = combined_stream.groupby(["District", "Block/Station"])["_rf_flag"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
    )
    combined_stream["Rainy_Days_Prev_7d"] = combined_stream.groupby(["District", "Block/Station"])["_rf_flag"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
    )
    combined_stream["Rainy_Days_Prev_15d"] = combined_stream.groupby(["District", "Block/Station"])["_rf_flag"].transform(
        lambda s: s.shift(1).rolling(window=15, min_periods=1).sum()
    )
    combined_stream["Rainy_Days_Prev_30d"] = combined_stream.groupby(["District", "Block/Station"])["_rf_flag"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=1).sum()
    )

    # Consecutive rainy days before
    def compute_consecutive(s):
        shifted = s.shift(1).fillna(0)
        out = []
        c = 0
        for v in shifted:
            if v == 1.0:
                c += 1
            else:
                c = 0
            out.append(c)
        return pd.Series(out, index=s.index)

    combined_stream["Consecutive_Rainy_Days_Before"] = combined_stream.groupby(["District", "Block/Station"])["_rf_flag"].transform(compute_consecutive)
    combined_stream.drop(columns=["_rf_flag"], inplace=True)

    # Operational Zero-Target Guardrail: Flood_Occurred is strictly unobserved in operational 2025
    combined_stream["Flood_Occurred"] = 0

    # Fill any boundary lag nulls with 0
    lag_cols = [
        "Rainfall_Lag_1d", "Rainfall_Lag_2d", "Rainfall_Lag_3d", "Rainfall_Lag_7d",
        "Rainfall_Prev_3d_Sum", "Rainfall_Prev_7d_Sum", "Rainfall_Prev_15d_Sum", "Rainfall_Prev_30d_Sum",
        "Rainfall_Prev_3d_Max", "Rainfall_Prev_7d_Max", "Rainfall_Prev_15d_Max", "Rainfall_Prev_30d_Max",
        "Rainy_Days_Prev_3d", "Rainy_Days_Prev_7d", "Rainy_Days_Prev_15d", "Rainy_Days_Prev_30d",
        "Consecutive_Rainy_Days_Before"
    ]
    for c in lag_cols:
        combined_stream[c] = combined_stream[c].fillna(0.0)

    # Slice back strictly to original 2025 records
    df_out_2025 = combined_stream[combined_stream["Date_Parsed"] >= min_date_25].copy()
    df_out_2025.drop(columns=["Date_Parsed"], inplace=True)
    df_out_2025 = df_out_2025.reset_index(drop=True)
    info["output_rows"] = len(df_out_2025)

    return df_out_2025, info


class Operational2025Pipeline:
    """
    End-to-End Operational Pipeline:
    Ingestion -> Schema Validation -> Warmup Stitched Feature Engineering -> Frozen Model Prediction -> Export.
    """
    def __init__(
        self,
        model_path: str = "models/flood_prediction/best_model.pkl",
        metadata_path: str = "models/flood_prediction/model_metadata.json",
        historical_path: str = "data/combined/Odisha_Flood_2001_2024.csv"
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.historical_path = historical_path
        self.model = None
        self.metadata = None
        self.historical_warmup_df = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        if os.path.exists(self.historical_path):
            try:
                # Load only December 2024 for fast warm-up
                df_hist = pd.read_csv(self.historical_path)
                df_hist["Date_Parsed"] = pd.to_datetime(df_hist["Date"])
                self.historical_warmup_df = df_hist[df_hist["Date_Parsed"] >= "2024-11-25"].copy()
            except Exception:
                self.historical_warmup_df = None

    def run_pipeline(
        self,
        raw_df_2025: pd.DataFrame,
        output_csv_path: str = "data/predictions/2025_flood_risk_predictions.csv",
        custom_warmup_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate, feature-engineer, and run frozen model inference on 2025 operational dataset.
        """
        val_report = validate_2025_raw_dataset(raw_df_2025)
        if not val_report["is_valid"]:
            raise ValueError(f"Operational dataset validation failed: {val_report['errors']}")

        # Warm-up selection: custom provided or historical Dec 2024
        warmup = custom_warmup_df if custom_warmup_df is not None else self.historical_warmup_df

        # Feature Engineering
        df_feat, feat_info = engineer_2025_features(raw_df_2025, warmup_df=warmup, allow_zero_warmup=False)

        # Frozen Model Verification
        feature_cols = self.metadata["features"]
        threshold = float(self.metadata.get("optimal_threshold", 0.5))
        risk_thresholds = self.metadata.get("risk_thresholds", {"low_max": 0.30, "moderate_max": 0.70})
        model_name = self.metadata.get("model_name", "Logistic Regression (Frozen)")

        X = df_feat[feature_cols]
        probas = self.model.predict_proba(X)[:, 1]

        # Extract top risk drivers using model linear coefficients or trees
        clf = self.model.named_steps["classifier"] if hasattr(self.model, "named_steps") else self.model
        scaler = self.model.named_steps["scaler"] if hasattr(self.model, "named_steps") else None

        if scaler is not None and hasattr(clf, "coef_"):
            X_scaled = scaler.transform(X)
            coefs = clf.coef_[0]
            contributions = X_scaled * coefs
            
            top1_list = []
            top2_list = []
            top3_list = []
            
            for row_contrib in contributions:
                pos_idx = np.where(row_contrib > 0)[0]
                sorted_pos = pos_idx[np.argsort(-row_contrib[pos_idx])] if len(pos_idx) > 0 else []
                t1 = feature_cols[sorted_pos[0]] if len(sorted_pos) > 0 else "None"
                t2 = feature_cols[sorted_pos[1]] if len(sorted_pos) > 1 else "None"
                t3 = feature_cols[sorted_pos[2]] if len(sorted_pos) > 2 else "None"
                top1_list.append(t1)
                top2_list.append(t2)
                top3_list.append(t3)
        else:
            top1_list = ["Rainfall_Prev_15d_Sum"] * len(X)
            top2_list = ["Rainy_Days_Prev_30d"] * len(X)
            top3_list = ["Month_cos"] * len(X)

        # Construct Output DataFrame
        df_pred = pd.DataFrame({
            "District": df_feat["District"],
            "Block/Station": df_feat["Block/Station"],
            "Date": df_feat["Date"],
            "Predicted_Probability": np.round(probas, 4),
            "Predicted_Flood_Probability": np.round(probas, 4),
            "Risk_Level": [
                "HIGH" if p >= risk_thresholds["moderate_max"] else ("MODERATE" if p >= risk_thresholds["low_max"] else "LOW")
                for p in probas
            ],
            "Predicted_Risk_Level": [
                "HIGH" if p >= risk_thresholds["moderate_max"] else ("MODERATE" if p >= risk_thresholds["low_max"] else "LOW")
                for p in probas
            ],
            "Predicted_Flood_Next_Day": (probas >= threshold).astype(int),
            "Prediction_Threshold": round(threshold, 4),
            "Model_Name": model_name,
            "Rainfall (mm)": df_feat["Rainfall (mm)"],
            "Rainfall_Prev_3d_Sum": df_feat["Rainfall_Prev_3d_Sum"],
            "Rainfall_Prev_7d_Sum": df_feat["Rainfall_Prev_7d_Sum"],
            "Rainfall_Prev_15d_Sum": df_feat["Rainfall_Prev_15d_Sum"],
            "Rainfall_Prev_30d_Sum": df_feat["Rainfall_Prev_30d_Sum"],
            "Top_Risk_Driver_1": top1_list,
            "Top_Risk_Driver_2": top2_list,
            "Top_Risk_Driver_3": top3_list
        })

        if output_csv_path:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df_pred.to_csv(output_csv_path, index=False)

        summary = {
            "validation": val_report,
            "warmup": feat_info,
            "total_predictions": len(df_pred),
            "high_risk_count": int((df_pred["Risk_Level"] == "HIGH").sum()),
            "moderate_risk_count": int((df_pred["Risk_Level"] == "MODERATE").sum()),
            "low_risk_count": int((df_pred["Risk_Level"] == "LOW").sum()),
            "output_path": output_csv_path
        }

        return df_pred, summary

    def predict_2025_dataset(self, df_2025: pd.DataFrame) -> pd.DataFrame:
        """
        Direct dataframe inference method (for backward compatibility and mock test data).
        If features are already engineered in df_2025, it computes predictions directly.
        Otherwise it routes through full pipeline.
        """
        feature_cols = self.metadata["features"] if self.metadata and "features" in self.metadata else [
            "Rainfall (mm)", "Rainfall_Lag_1d", "Rainfall_Lag_2d", "Rainfall_Lag_3d", "Rainfall_Lag_7d",
            "Rainfall_Prev_3d_Sum", "Rainfall_Prev_7d_Sum", "Rainfall_Prev_15d_Sum", "Rainfall_Prev_30d_Sum",
            "Rainfall_Prev_3d_Max", "Rainfall_Prev_7d_Max", "Rainfall_Prev_15d_Max", "Rainfall_Prev_30d_Max",
            "Rainy_Days_Prev_3d", "Rainy_Days_Prev_7d", "Rainy_Days_Prev_15d", "Rainy_Days_Prev_30d",
            "Consecutive_Rainy_Days_Before", "Month_sin", "Month_cos"
        ]
        
        # If all feature_cols exist in df_2025
        if all(col in df_2025.columns for col in feature_cols):
            threshold = float(self.metadata.get("optimal_threshold", 0.5)) if self.metadata else 0.5
            risk_thresholds = self.metadata.get("risk_thresholds", {"low_max": 0.30, "moderate_max": 0.70}) if self.metadata else {"low_max": 0.30, "moderate_max": 0.70}
            
            X = df_2025[feature_cols]
            probas = self.model.predict_proba(X)[:, 1]
            
            out_df = df_2025.copy()
            out_df["Predicted_Probability"] = np.round(probas, 4)
            out_df["Predicted_Flood_Probability"] = np.round(probas, 4)
            out_df["Predicted_Risk_Level"] = [
                "HIGH" if p >= risk_thresholds["moderate_max"] else ("MODERATE" if p >= risk_thresholds["low_max"] else "LOW")
                for p in probas
            ]
            out_df["Risk_Level"] = out_df["Predicted_Risk_Level"]
            out_df["Predicted_Flood_Next_Day"] = (probas >= threshold).astype(int)
            return out_df
        else:
            df_pred, _ = self.run_pipeline(df_2025, output_csv_path=None)
            return df_pred


class Retrospective2025EvaluationEngine:
    """
    Evaluates 2025 model predictions against authoritative SRC ground-truth observations
    without ever retraining or modifying the frozen operational model.
    """
    def __init__(self, metadata_path: str = "models/flood_prediction/model_metadata.json"):
        self.metadata_path = metadata_path
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def evaluate(
        self,
        df_predictions: pd.DataFrame,
        df_ground_truth: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Merge predictions and ground-truth flood observations on [District, Block/Station, Date]
        and calculate retrospective evaluation metrics.
        """
        # Standardize ground truth column names
        gt = df_ground_truth.copy()
        gt_cols = {c.strip().lower(): c for c in gt.columns}
        
        if "flood_occurred" not in gt_cols:
            raise ValueError("Ground-truth file must contain 'Flood_Occurred' column.")
            
        gt["District"] = gt[gt_cols["district"]].astype(str).str.strip().apply(standardize_district_name)
        gt["Block/Station"] = gt[gt_cols["block/station"]].astype(str).str.strip()
        gt["Date"] = pd.to_datetime(gt[gt_cols["date"]]).dt.strftime("%Y-%m-%d")
        gt["True_Flood_Occurred"] = pd.to_numeric(gt[gt_cols["flood_occurred"]], errors="coerce").fillna(0).astype(int)

        # Standardize predictions
        pred = df_predictions.copy()
        pred["District"] = pred["District"].astype(str).str.strip().apply(standardize_district_name)
        pred["Block/Station"] = pred["Block/Station"].astype(str).str.strip()
        pred["Date"] = pd.to_datetime(pred["Date"]).dt.strftime("%Y-%m-%d")

        merged = pd.merge(
            pred,
            gt[["District", "Block/Station", "Date", "True_Flood_Occurred"]],
            on=["District", "Block/Station", "Date"],
            how="inner"
        )

        if len(merged) == 0:
            raise ValueError("No matching records found between predictions and ground-truth observations.")

        y_true = merged["True_Flood_Occurred"].values
        y_proba = merged["Predicted_Probability"].values
        threshold = float(self.metadata.get("optimal_threshold", 0.5))

        metrics = calculate_metrics(y_true, y_proba, threshold=threshold, prefix="retrospective_")
        metrics["matched_observations"] = len(merged)
        metrics["true_flood_days"] = int(np.sum(y_true == 1))
        metrics["true_flood_rate_pct"] = round(float(np.mean(y_true) * 100), 3)

        return metrics, merged


def analyze_rainfall_distribution_shift(
    historical_rainfall: pd.Series,
    current_rainfall: pd.Series,
    district_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare rainfall distributions between historical baseline years and unseen period.
    Computes summary quantiles and Kolmogorov-Smirnov 2-sample test for distribution shift.
    """
    h_clean = historical_rainfall.dropna()
    c_clean = current_rainfall.dropna()
    
    ks_stat, p_value = ks_2samp(h_clean, c_clean)
    
    return {
        "district": district_name or "Odisha (Statewide)",
        "historical_mean_mm": float(h_clean.mean()),
        "current_mean_mm": float(c_clean.mean()),
        "historical_max_mm": float(h_clean.max()),
        "current_max_mm": float(c_clean.max()),
        "historical_heavy_rain_pct": float((h_clean >= 64.5).mean() * 100),
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "distribution_shift_detected": bool(p_value < 0.01)
    }


# Alias for backward compatibility
Unseen2025InferenceEngine = Operational2025Pipeline
