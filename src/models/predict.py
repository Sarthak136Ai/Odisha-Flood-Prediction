"""
Inference and prediction engine for Odisha flood early warning.
Loads trained model artifact together with its metadata bundle, checks feature schema alignment,
produces calibrated probabilities, and assigns transparent risk levels.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional

from src.models.metadata import (
    SchemaValidationError,
    ModelBundle,
    load_model_with_metadata,
    validate_feature_schema
)


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
        self.bundle: Optional[ModelBundle] = None
        self.model: Any = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names: list = []
        self.threshold: float = 0.5
        self.risk_thresholds: Dict[str, float] = {
            "low_max": 0.30,
            "moderate_max": 0.70,
            "high_min": 0.70
        }
        self.load()

    def load(self):
        """Load trained model artifact together with its metadata."""
        target_model_path = self.calibrated_model_path if (self.use_calibrated and os.path.exists(self.calibrated_model_path)) else self.model_path
        target_meta_path = self.calibrated_metadata_path if (self.use_calibrated and os.path.exists(self.calibrated_metadata_path)) else self.metadata_path

        if not os.path.exists(target_model_path):
            # Fallback to standard model path if calibrated model not found
            target_model_path = self.model_path
            target_meta_path = self.metadata_path

        if not os.path.exists(target_model_path):
            raise FileNotFoundError(f"Trained model not found at {target_model_path}")
            
        # Load model together with metadata into a ModelBundle
        meta_to_load = target_meta_path if (target_meta_path and os.path.exists(target_meta_path)) else None
        self.bundle = load_model_with_metadata(target_model_path, meta_to_load)
        
        self.model = self.bundle.model
        self.metadata = self.bundle.metadata
        self.feature_names = self.bundle.features
        self.threshold = self.bundle.threshold
        
        if self.use_calibrated and "calibrated_risk_thresholds" in self.metadata:
            self.risk_thresholds = self.metadata["calibrated_risk_thresholds"]
        else:
            self.risk_thresholds = self.metadata.get("risk_thresholds", self.risk_thresholds)

    @classmethod
    def from_bundle(cls, bundle: ModelBundle) -> "FloodPredictor":
        """Instantiate a FloodPredictor directly from an existing ModelBundle."""
        predictor = cls.__new__(cls)
        predictor.bundle = bundle
        predictor.model = bundle.model
        predictor.metadata = bundle.metadata
        predictor.feature_names = bundle.features
        predictor.threshold = bundle.threshold
        predictor.risk_thresholds = bundle.risk_thresholds
        predictor.model_path = bundle.model_path or ""
        predictor.metadata_path = bundle.metadata_path or ""
        predictor.use_calibrated = (bundle.calibration_method != "none")
        predictor.calibrated_model_path = bundle.model_path or ""
        predictor.calibrated_metadata_path = bundle.metadata_path or ""
        return predictor

    @classmethod
    def load_with_metadata(
        cls,
        model_path: str = "models/flood_prediction/best_model.pkl",
        metadata_path: Optional[str] = None
    ) -> "FloodPredictor":
        """Explicit helper to load model and metadata together."""
        bundle = load_model_with_metadata(model_path, metadata_path)
        return cls.from_bundle(bundle)

    def validate_input(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Validate input dataset against model schema prior to inference.
        
        Raises
        ------
        SchemaValidationError
            If features are missing, non-numeric, or dimensional mismatch occurs.
        """
        if self.bundle is not None:
            return self.bundle.validate_schema(X)
        return validate_feature_schema(X, self.feature_names)

    def predict_probability(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Calculate next-day flood probabilities for input features after strict schema validation.
        Prevents execution if input schema does not match model expectations.
        """
        X_validated = self.validate_input(X)
            
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_validated)[:, 1]
        else:
            proba = self.model.predict(X_validated)
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
