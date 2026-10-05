"""
Chatbot live prediction tool that interfaces with trained models and explainability engine.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.models.predict import FloodPredictor
from src.explainability.shap_analysis import explain_instance


class ChatbotPredictionTool:
    """Tool to execute verified model predictions and explanations for chatbot queries."""
    
    def __init__(
        self,
        model_path: str = "models/flood_prediction/best_model.pkl",
        metadata_path: str = "models/flood_prediction/model_metadata.json",
        data_path: str = "data/combined/Odisha_Flood_2001_2024.csv"
    ):
        self.predictor = FloodPredictor(model_path, metadata_path)
        self.data_path = data_path
        self.df = None
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
            self.df["Date"] = self.df["Date"].astype(str)

    def predict_for_location_and_date(
        self,
        district: str,
        block: Optional[str] = None,
        date_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Look up historical antecedent features for a specific location and run model prediction.
        """
        if self.df is None:
            return {"error": "Dataset not available for feature lookup."}
            
        d_clean = district.strip().upper()
        sub = self.df[self.df["District"] == d_clean]
        if len(sub) == 0:
            return {"error": f"District '{district}' not found in records."}
            
        if block:
            b_clean = block.strip()
            sub_b = sub[sub["Block/Station"].str.lower() == b_clean.lower()]
            if len(sub_b) > 0:
                sub = sub_b
                
        if date_str:
            sub_d = sub[sub["Date"] == date_str]
            if len(sub_d) > 0:
                target_row = sub_d.iloc[0]
            else:
                # Default to most recent matching date
                target_row = sub.sort_values("Date").iloc[-1]
        else:
            target_row = sub.sort_values("Date").iloc[-1]
            
        feature_dict = target_row[self.predictor.feature_names].to_dict()
        pred_res = self.predictor.predict_single(feature_dict)
        
        # Get explanation
        expl = explain_instance(self.predictor.model, feature_dict, self.predictor.feature_names)
        
        return {
            "district": target_row["District"],
            "block": target_row["Block/Station"],
            "date": target_row["Date"],
            "current_rainfall_mm": float(target_row["Rainfall (mm)"]),
            "rainfall_prev_3d_sum_mm": float(target_row["Rainfall_Prev_3d_Sum"]),
            "rainfall_prev_15d_sum_mm": float(target_row["Rainfall_Prev_15d_Sum"]),
            "flood_probability": pred_res["flood_probability"],
            "predicted_flood_next_day": pred_res["predicted_flood_next_day"],
            "risk_level": pred_res["risk_level"],
            "top_risk_drivers": expl["top_risk_drivers"],
            "top_mitigators": expl["top_mitigators"]
        }

    def predict_custom_scenario(
        self,
        district: str,
        rainfall_mm: float,
        rainfall_prev_3d_sum: float,
        rainfall_prev_7d_sum: float,
        rainfall_prev_15d_sum: float,
        month: int = 8,
        day: int = 15,
        flood_occurred_today: int = 0
    ) -> Dict[str, Any]:
        """Run simulation for a user-specified custom meteorological scenario."""
        month_sin = float(np.sin(2 * np.pi * month / 12.0))
        month_cos = float(np.cos(2 * np.pi * month / 12.0))
        day_of_year = int((month - 1) * 30 + day)
        
        feature_dict = {
            "Year": 2024,
            "Month_Number": month,
            "Day": day,
            "Day_of_Year": day_of_year,
            "Day_of_Week": 3,
            "Month_sin": month_sin,
            "Month_cos": month_cos,
            "Rainfall (mm)": rainfall_mm,
            "Rainfall_Missing": 0,
            "Rainfall_Lag_1d": rainfall_mm * 0.8,
            "Rainfall_Lag_2d": rainfall_mm * 0.6,
            "Rainfall_Lag_3d": rainfall_mm * 0.4,
            "Rainfall_Lag_7d": rainfall_mm * 0.2,
            "Rainfall_Prev_3d_Sum": rainfall_prev_3d_sum,
            "Rainfall_Prev_7d_Sum": rainfall_prev_7d_sum,
            "Rainfall_Prev_15d_Sum": rainfall_prev_15d_sum,
            "Rainfall_Prev_30d_Sum": rainfall_prev_15d_sum * 1.5,
            "Rainfall_Prev_3d_Max": max(rainfall_mm, rainfall_prev_3d_sum / 3.0),
            "Rainfall_Prev_7d_Max": max(rainfall_mm, rainfall_prev_7d_sum / 7.0),
            "Rainfall_Prev_15d_Max": max(rainfall_mm, rainfall_prev_15d_sum / 15.0),
            "Rainfall_Prev_30d_Max": max(rainfall_mm, rainfall_prev_15d_sum / 15.0),
            "Rainy_Days_Prev_3d": 2.0 if rainfall_prev_3d_sum > 10 else 0.0,
            "Rainy_Days_Prev_7d": 5.0 if rainfall_prev_7d_sum > 25 else 0.0,
            "Rainy_Days_Prev_15d": 10.0 if rainfall_prev_15d_sum > 50 else 0.0,
            "Rainy_Days_Prev_30d": 18.0 if rainfall_prev_15d_sum > 100 else 2.0,
            "Consecutive_Rainy_Days_Before": 3.0 if rainfall_mm > 5.0 else 0.0,
            "Flood_Occurred": flood_occurred_today
        }
        
        pred_res = self.predictor.predict_single(feature_dict)
        expl = explain_instance(self.predictor.model, feature_dict, self.predictor.feature_names)
        
        return {
            "scenario": {
                "district": district,
                "input_rainfall_mm": rainfall_mm,
                "input_prev_3d_sum_mm": rainfall_prev_3d_sum,
                "input_prev_15d_sum_mm": rainfall_prev_15d_sum
            },
            "flood_probability": pred_res["flood_probability"],
            "predicted_flood_next_day": pred_res["predicted_flood_next_day"],
            "risk_level": pred_res["risk_level"],
            "top_risk_drivers": expl["top_risk_drivers"],
            "top_mitigators": expl["top_mitigators"]
        }
