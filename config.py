"""
Configuration module for the Odisha Flood Intelligence & Early Warning System Flask Application.
"""

import os
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Flask application configuration class."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "odisha-flood-intelligence-secret-key-2026")
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "yes"]
    
    # Upload configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
    ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024 # 32 MB max upload
    
    # Data Paths
    HISTORICAL_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "historical")
    OPERATIONAL_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "operational", "2025.csv")
    PROCESSED_HISTORICAL_PATH = os.path.join(BASE_DIR, "data", "processed", "historical_2001_2024.csv")
    COMBINED_HISTORICAL_PATH = os.path.join(BASE_DIR, "data", "combined", "Odisha_Flood_2001_2024.csv")
    PROCESSED_OPERATIONAL_PATH = os.path.join(BASE_DIR, "data", "processed", "operational_2025.csv")
    PREDICTIONS_2025_PATH = os.path.join(BASE_DIR, "data", "predictions", "2025_flood_risk_predictions.csv")
    
    # Model Paths
    MODEL_PATH = os.path.join(BASE_DIR, "models", "flood_prediction", "best_model.pkl")
    METADATA_PATH = os.path.join(BASE_DIR, "models", "flood_prediction", "model_metadata.json")
    DOWNSCALING_MODEL_PATH = os.path.join(BASE_DIR, "models", "downscaling", "downscaling_model.pkl")
    
    # Metrics & Results
    MODEL_COMPARISON_PATH = os.path.join(BASE_DIR, "results", "metrics", "model_comparison.csv")
    SHAP_IMPORTANCE_PATH = os.path.join(BASE_DIR, "results", "metrics", "shap_feature_importance.csv")
    DOWNSCALING_METRICS_PATH = os.path.join(BASE_DIR, "results", "downscaling", "downscaling_metrics.csv")
    PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
    
    # Risk Thresholds
    RISK_THRESHOLDS = {
        "low_max": 0.30,
        "moderate_max": 0.70,
        "high_min": 0.70
    }
    OPTIMAL_THRESHOLD = 0.8073
