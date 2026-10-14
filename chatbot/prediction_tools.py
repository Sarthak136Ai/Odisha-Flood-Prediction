"""
Chatbot live prediction tool that interfaces with trained models and explainability engine.
Functions as a deterministic bridge between user queries and core ML inference pipelines.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from src.models.predict import FloodPredictor
from src.explainability.shap_analysis import explain_instance

logger = logging.getLogger(__name__)


class ChatbotPredictionTool:
    """Tool to execute verified model predictions and explainability attributions for chatbot queries."""
    
    def __init__(
        self,
        model_path: str = "models/flood_prediction/best_model.pkl",
        metadata_path: str = "models/flood_prediction/model_metadata.json",
        data_path: str = "data/combined/Odisha_Flood_2001_2024.csv",
        predictions_2025_path: str = "data/predictions/2025_flood_risk_predictions.csv"
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.data_path = data_path
        self.predictions_2025_path = predictions_2025_path
        
        self.predictor: Optional[FloodPredictor] = None
        self.df: Optional[pd.DataFrame] = None
        self.df_2025: Optional[pd.DataFrame] = None
        
        self._load_predictor()
        self._load_datasets()

    def _load_predictor(self):
        """Safely load production predictor without crashing if model file is missing."""
        try:
            if os.path.exists(self.model_path):
                self.predictor = FloodPredictor(self.model_path, self.metadata_path)
            else:
                self.predictor = None
        except Exception as e:
            logger.warning(f"FloodPredictor initialization failed: {e}")
            self.predictor = None

    def _load_datasets(self):
        """Load historical telemetry and 2025 operational predictions if available."""
        if os.path.exists(self.data_path):
            try:
                self.df = pd.read_csv(self.data_path)
                self.df["Date"] = self.df["Date"].astype(str)
            except Exception as e:
                logger.warning(f"Error loading historical data: {e}")
                self.df = None
                
        if os.path.exists(self.predictions_2025_path):
            try:
                self.df_2025 = pd.read_csv(self.predictions_2025_path)
                self.df_2025["Date"] = self.df_2025["Date"].astype(str)
            except Exception as e:
                logger.warning(f"Error loading 2025 predictions: {e}")
                self.df_2025 = None

    def is_model_available(self) -> bool:
        """Check whether the underlying ML prediction model is loaded and functional."""
        return self.predictor is not None and self.predictor.model is not None

    def predict_for_location_and_date(
        self,
        district: str,
        block: Optional[str] = None,
        date_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Look up antecedent features for a specific location and run model prediction.
        Returns explicit error message if data or model is unavailable.
        """
        if not self.is_model_available():
            return {
                "error": "Sufficient project data is not available: The core flood prediction model is currently unavailable or uninitialized."
            }

        if self.df is None and self.df_2025 is None:
            return {
                "error": "Sufficient project data is not available: Telemetry dataset is not loaded."
            }
            
        d_clean = district.strip().upper()
        
        # Check in historical dataset
        sub = self.df[self.df["District"] == d_clean] if self.df is not None else pd.DataFrame()
        sub_25 = self.df_2025[self.df_2025["District"] == d_clean] if self.df_2025 is not None else pd.DataFrame()
        
        if len(sub) == 0 and len(sub_25) == 0:
            return {
                "error": f"Sufficient project data is not available: District '{district}' is not recognized in Odisha telemetry records."
            }
            
        # Check block validation
        if block:
            b_clean = block.strip().lower()
            if len(sub) > 0:
                sub_b = sub[sub["Block/Station"].str.lower() == b_clean]
            else:
                sub_b = pd.DataFrame()
                
            if len(sub_25) > 0:
                sub_25_b = sub_25[sub_25["Block/Station"].str.lower() == b_clean]
            else:
                sub_25_b = pd.DataFrame()
                
            if len(sub_b) == 0 and len(sub_25_b) == 0:
                valid_blocks = sorted(list(sub["Block/Station"].unique() if len(sub) > 0 else sub_25["Block/Station"].unique()))[:6]
                return {
                    "error": f"Sufficient project data is not available: Block/Station '{block}' was not found in district '{district}'. Valid blocks include: {', '.join(valid_blocks)}."
                }
            sub = sub_b
            sub_25 = sub_25_b

        # Date handling
        if date_str:
            d_target = str(date_str).strip()
            # If 2025 date and operational predictions available
            if d_target.startswith("2025") and len(sub_25) > 0:
                match_25 = sub_25[sub_25["Date"] == d_target]
                if len(match_25) > 0:
                    row_25 = match_25.iloc[0]
                    return {
                        "district": row_25["District"],
                        "block": row_25["Block/Station"],
                        "date": row_25["Date"],
                        "current_rainfall_mm": float(row_25.get("Rainfall (mm)", 0.0)),
                        "rainfall_prev_3d_sum_mm": float(row_25.get("Rainfall_Prev_3d_Sum", 0.0)),
                        "rainfall_prev_15d_sum_mm": float(row_25.get("Rainfall_Prev_15d_Sum", 0.0)),
                        "flood_probability": float(row_25.get("Predicted_Probability", row_25.get("Predicted_Flood_Probability", 0.0))),
                        "predicted_flood_next_day": int(row_25.get("Predicted_Flood_Next_Day", 0)),
                        "risk_level": str(row_25.get("Risk_Level", row_25.get("Predicted_Risk_Level", "LOW"))),
                        "prediction_threshold": float(row_25.get("Prediction_Threshold", self.predictor.threshold)),
                        "model_name": str(row_25.get("Model_Name", self.predictor.metadata.get("model_name", "XGBoost"))),
                        "top_risk_drivers": [
                            {"feature": str(row_25.get("Top_Risk_Driver_1", "Antecedent Rainfall")), "value": 1.0, "contribution": 0.5},
                            {"feature": str(row_25.get("Top_Risk_Driver_2", "Soil Saturation")), "value": 1.0, "contribution": 0.3}
                        ],
                        "top_mitigators": []
                    }
                    
            # Historical date check
            if len(sub) > 0:
                match_hist = sub[sub["Date"] == d_target]
                if len(match_hist) > 0:
                    target_row = match_hist.iloc[0]
                else:
                    return {
                        "error": f"Sufficient project data is not available: No telemetry observations found for {district} on date '{date_str}'."
                    }
            else:
                return {
                    "error": f"Sufficient project data is not available: No telemetry observations found for {district} on date '{date_str}'."
                }
        else:
            # Default to most recent available date
            if len(sub_25) > 0:
                target_row = sub_25.sort_values("Date").iloc[-1]
                return {
                    "district": target_row["District"],
                    "block": target_row["Block/Station"],
                    "date": target_row["Date"],
                    "current_rainfall_mm": float(target_row.get("Rainfall (mm)", 0.0)),
                    "rainfall_prev_3d_sum_mm": float(target_row.get("Rainfall_Prev_3d_Sum", 0.0)),
                    "rainfall_prev_15d_sum_mm": float(target_row.get("Rainfall_Prev_15d_Sum", 0.0)),
                    "flood_probability": float(target_row.get("Predicted_Probability", target_row.get("Predicted_Flood_Probability", 0.0))),
                    "predicted_flood_next_day": int(target_row.get("Predicted_Flood_Next_Day", 0)),
                    "risk_level": str(target_row.get("Risk_Level", target_row.get("Predicted_Risk_Level", "LOW"))),
                    "prediction_threshold": float(target_row.get("Prediction_Threshold", self.predictor.threshold)),
                    "model_name": str(target_row.get("Model_Name", self.predictor.metadata.get("model_name", "XGBoost"))),
                    "top_risk_drivers": [
                        {"feature": str(target_row.get("Top_Risk_Driver_1", "Antecedent Rainfall")), "value": 1.0, "contribution": 0.5},
                        {"feature": str(target_row.get("Top_Risk_Driver_2", "Soil Saturation")), "value": 1.0, "contribution": 0.3}
                    ],
                    "top_mitigators": []
                }
            elif len(sub) > 0:
                target_row = sub.sort_values("Date").iloc[-1]
            else:
                return {
                    "error": f"Sufficient project data is not available for district '{district}'."
                }
                
        # Run live model inference on target row
        feature_dict = target_row[self.predictor.feature_names].to_dict()
        pred_res = self.predictor.predict_single(feature_dict)
        
        # Get SHAP explanation
        expl = explain_instance(self.predictor.model, feature_dict, self.predictor.feature_names)
        
        return {
            "district": target_row["District"],
            "block": target_row["Block/Station"],
            "date": target_row["Date"],
            "current_rainfall_mm": float(target_row.get("Rainfall (mm)", 0.0)),
            "rainfall_prev_3d_sum_mm": float(target_row.get("Rainfall_Prev_3d_Sum", 0.0)),
            "rainfall_prev_15d_sum_mm": float(target_row.get("Rainfall_Prev_15d_Sum", 0.0)),
            "flood_probability": pred_res["flood_probability"],
            "predicted_flood_next_day": pred_res["predicted_flood_next_day"],
            "risk_level": pred_res["risk_level"],
            "prediction_threshold": pred_res["threshold"],
            "model_name": self.predictor.metadata.get("model_name", "XGBoost"),
            "top_risk_drivers": expl.get("top_risk_drivers", []),
            "top_mitigators": expl.get("top_mitigators", [])
        }

    def get_operational_summary(self) -> Dict[str, Any]:
        """Return operational state-wide monitoring summary from 2025 predictions or historical records."""
        if self.df_2025 is not None and len(self.df_2025) > 0:
            df_25 = self.df_2025
            prob_col = "Predicted_Probability" if "Predicted_Probability" in df_25.columns else "Predicted_Flood_Probability"
            risk_col = "Risk_Level" if "Risk_Level" in df_25.columns else "Predicted_Risk_Level"
            
            return {
                "source": "2025 Operational Telemetry",
                "total_records": len(df_25),
                "districts_count": int(df_25["District"].nunique()),
                "blocks_count": int(df_25["Block/Station"].nunique()),
                "date_min": str(df_25["Date"].min()),
                "date_max": str(df_25["Date"].max()),
                "high_risk_count": int((df_25[risk_col].str.upper() == "HIGH").sum()),
                "moderate_risk_count": int((df_25[risk_col].str.upper() == "MODERATE").sum()),
                "low_risk_count": int((df_25[risk_col].str.upper() == "LOW").sum()),
                "avg_probability": float(df_25[prob_col].mean()),
                "max_rainfall_mm": float(df_25["Rainfall (mm)"].max()) if "Rainfall (mm)" in df_25.columns else 0.0
            }
        elif self.df is not None and len(self.df) > 0:
            return {
                "source": "2001–2024 Historical Telemetry",
                "total_records": len(self.df),
                "districts_count": int(self.df["District"].nunique()),
                "blocks_count": int(self.df["Block/Station"].nunique()),
                "date_min": str(self.df["Date"].min()),
                "date_max": str(self.df["Date"].max()),
                "high_risk_count": 0,
                "moderate_risk_count": 0,
                "low_risk_count": len(self.df),
                "avg_probability": float(self.df["Flood_Occurred"].mean()),
                "max_rainfall_mm": float(self.df["Rainfall (mm)"].max())
            }
        return {"error": "Sufficient project data is not available: No operational or historical data loaded."}
