"""
Training module for statistical downscaling regression models.
Maps coarse-resolution regional meteorological drivers to fine-scale station rainfall.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple, List


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
    Train a HistGradientBoostingRegressor to downscale coarse rainfall signals to local station rainfall.
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
    
    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "train_r2": float(r2_score(y_train, y_pred_train))
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    
    return model, metrics
