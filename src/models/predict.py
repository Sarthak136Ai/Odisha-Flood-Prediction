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
        metadata_path: str = "models/flood_prediction/model_metadata.json",
        use_calibrated: bool = False,
        calibrated_model_path: str = "models/flood_prediction/calibrated_best_model.pkl",
        calibrated_metadata_path: str = "models/flood_prediction/calibrated_metadata.json"
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.use_calibrated = use_calibrated
        self.calibrated_model_path = calibrated_model_path
        self.calibrated_metadata_path = calibrated_metadata_path
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
        target_model_path = self.calibrated_model_path if (self.use_calibrated and os.path.exists(self.calibrated_model_path)) else self.model_path
        target_meta_path = self.calibrated_metadata_path if (self.use_calibrated and os.path.exists(self.calibrated_metadata_path)) else self.metadata_path

        if not os.path.exists(target_model_path):
            # Fallback to standard model path if calibrated model not found
            target_model_path = self.model_path
            target_meta_path = self.metadata_path

        if not os.path.exists(target_model_path):
            raise FileNotFoundError(f"Trained model not found at {target_model_path}")
            
        self.model = joblib.load(target_model_path)
        
        if os.path.exists(target_meta_path):
            with open(target_meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata.get("features", [])
                self.threshold = self.metadata.get("optimal_threshold", 0.5)
                if self.use_calibrated and "calibrated_risk_thresholds" in self.metadata:
                    self.risk_thresholds = self.metadata["calibrated_risk_thresholds"]
                else:
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

    def get_intensity_classification(self, rainfall_mm: Optional[float]) -> Dict[str, int]:
        """Classify rainfall into IMD meteorological intensity categories."""
        from src.features.rainfall_features import RainfallIntensityEngine
        return RainfallIntensityEngine.classify_single_rainfall(rainfall_mm)

    def predict_single(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Produce structured prediction output for a single location observation."""
        df_input = pd.DataFrame([feature_dict])
        proba = float(self.predict_probability(df_input)[0])
        pred_class = int(proba >= self.threshold)
        risk_level = self.get_risk_level(proba)
        
        output = {
            "flood_probability": round(proba, 4),
            "predicted_flood_next_day": pred_class,
            "risk_level": risk_level,
            "classification_threshold": self.threshold,
            "model_type": self.metadata.get("model_name", "Trained ML Model"),
        }
        
        rf = feature_dict.get("Rainfall (mm)")
        if rf is not None:
            output["rainfall_intensity"] = self.get_intensity_classification(rf)
            
        return output

