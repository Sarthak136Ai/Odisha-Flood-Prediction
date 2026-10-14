"""
Training module for statistical downscaling regression models.
Maps coarse-resolution regional meteorological drivers (district spatial aggregates) to block-scale station rainfall.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from typing import Dict, Any, Tuple, List

from src.downscaling.evaluate_downscaling import evaluate_downscaling_quality


def train_downscaling_model(
    train_df: pd.DataFrame,
    features: List[str] = [
        "coarse_rainfall_mean",
        "coarse_rainfall_std",
        "Month_sin",
        "Month_cos",
        "Day_of_Year"
    ],
    target: str = "Rainfall (mm)",
    save_path: str = "models/downscaling/downscaling_model.pkl"
) -> Tuple[HistGradientBoostingRegressor, Dict[str, float]]:
    """
    Train a HistGradientBoostingRegressor to downscale coarse district rainfall signals
    to block/station point rainfall.
    Trained strictly on historical train period (2001-2018).
    """
    X_train = train_df[features]
    y_train = train_df[target].values
    
    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=8,
        learning_rate=0.08,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Train set evaluation
    y_pred_train = model.predict(X_train)
    y_pred_train = np.clip(y_pred_train, 0.0, None)
    
    quality = evaluate_downscaling_quality(y_train, y_pred_train)
    metrics = {
        "train_mae": quality["mae_mm"],
        "train_rmse": quality["rmse_mm"],
        "train_r2": quality["r2_score"],
        "train_bias": quality["bias_mm"],
        "train_pearson_r": quality["pearson_correlation"]
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    
    return model, metrics
