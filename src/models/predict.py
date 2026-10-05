"""
Inference and prediction engine for Odisha flood early warning.
Loads trained model artifact, checks feature alignment, produces calibrated probabilities,
and assigns transparent risk levels.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional


class FloodPredictor:
    """Production predictor for next-day flood forecasting."""
    
    def __init__(
        self,
        model_path: str = "models/flood_prediction/best_model.pkl",
        metadata_path: str = "models/flood_prediction/model_metadata.json"
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = {}
        self.feature_names = []
        self.threshold = 0.5
        self.risk_thresholds = {
            "low_max": 0.30,
            "moderate_max": 0.70,
            "high_min": 0.70
        }
        self.load()

    def load(self):
        """Load trained model artifact and metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
            
        self.model = joblib.load(self.model_path)
        
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata.get("features", [])
                self.threshold = self.metadata.get("optimal_threshold", 0.5)
                self.risk_thresholds = self.metadata.get("risk_thresholds", self.risk_thresholds)

    def predict_probability(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Calculate next-day flood probabilities for input features."""
        if isinstance(X, pd.DataFrame) and self.feature_names:
            # Ensure all required features are present
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")
            X_mat = X[self.feature_names]
        else:
            X_mat = X
            
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_mat)[:, 1]
        else:
            proba = self.model.predict(X_mat)
        return np.asarray(proba)

    def get_risk_level(self, probability: float) -> str:
        """Assign risk level category based on calibrated thresholds."""
        if probability < self.risk_thresholds["low_max"]:
            return "Low"
        elif probability < self.risk_thresholds["moderate_max"]:
            return "Moderate"
        else:
            return "High"

    def predict_single(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Produce structured prediction output for a single location observation."""
        df_input = pd.DataFrame([feature_dict])
        proba = float(self.predict_probability(df_input)[0])
        pred_class = int(proba >= self.threshold)
        risk_level = self.get_risk_level(proba)
        
        return {
            "flood_probability": round(proba, 4),
            "predicted_flood_next_day": pred_class,
            "risk_level": risk_level,
            "classification_threshold": self.threshold,
            "model_type": self.metadata.get("model_name", "Trained ML Model"),
        }
