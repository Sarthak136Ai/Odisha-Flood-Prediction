"""
Application of statistical downscaling model to reconstruct block-scale rainfall
and regenerate hydrologic flood prediction features without lookahead leakage.
"""

from typing import List, Optional
import pandas as pd
import numpy as np


def generate_downscaled_flood_features(
    df_coarse: pd.DataFrame,
    downscaling_model,
    downscaling_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Predict station-level rainfall from coarse inputs and regenerate rolling flood features.
    Strictly preserves chronological ordering and prevents future lookahead leakage by using
    shift(1) on all antecedent hydrologic features.
    """
    df_out = df_coarse.copy()
    
    # Determine features
    if downscaling_features is None:
        if hasattr(downscaling_model, "feature_names_in_"):
            downscaling_features = list(downscaling_model.feature_names_in_)
        else:
            downscaling_features = [
                "coarse_rainfall_mean",
                "coarse_rainfall_std",
                "Month_sin",
                "Month_cos",
                "Day_of_Year"
            ]
            
    # Predict downscaled rainfall
    raw_preds = downscaling_model.predict(df_out[downscaling_features])
    df_out["Downscaled_Rainfall (mm)"] = np.clip(raw_preds, 0.0, None)
    
    # Save original rainfall as reference and replace 'Rainfall (mm)' with downscaled
    df_out["Original_Rainfall (mm)"] = df_out["Rainfall (mm)"]
    df_out["Rainfall (mm)"] = df_out["Downscaled_Rainfall (mm)"]
    
    # Recompute lag and rolling features from downscaled rainfall
    df_out["Date_Parsed"] = pd.to_datetime(df_out["Date"])
    df_out = df_out.sort_values(by=["District", "Block/Station", "Date_Parsed"]).reset_index(drop=True)
    
    grouped = df_out.groupby(["District", "Block/Station"])
    
    # Lagged daily values
    df_out["Rainfall_Lag_1d"] = grouped["Rainfall (mm)"].shift(1).fillna(0.0)
    df_out["Rainfall_Lag_2d"] = grouped["Rainfall (mm)"].shift(2).fillna(0.0)
    df_out["Rainfall_Lag_3d"] = grouped["Rainfall (mm)"].shift(3).fillna(0.0)
    df_out["Rainfall_Lag_7d"] = grouped["Rainfall (mm)"].shift(7).fillna(0.0)
    
    # Rolling sums
    df_out["Rainfall_Prev_3d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
    ).fillna(0.0)
    df_out["Rainfall_Prev_7d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
    ).fillna(0.0)
    df_out["Rainfall_Prev_15d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=15, min_periods=1).sum()
    ).fillna(0.0)
    df_out["Rainfall_Prev_30d_Sum"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=1).sum()
    ).fillna(0.0)
    
    # Rolling maxima
    df_out["Rainfall_Prev_3d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).max()
    ).fillna(0.0)
    df_out["Rainfall_Prev_7d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).max()
    ).fillna(0.0)
    df_out["Rainfall_Prev_15d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=15, min_periods=1).max()
    ).fillna(0.0)
    df_out["Rainfall_Prev_30d_Max"] = grouped["Rainfall (mm)"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=1).max()
    ).fillna(0.0)
    
    # Rainy day counts (IMD threshold: >= 2.5 mm)
    rainy_flag = (df_out["Rainfall (mm)"] >= 2.5).astype(float)
    df_out["_rainy_flag"] = rainy_flag
    
    df_out["Rainy_Days_Prev_3d"] = df_out.groupby(["District", "Block/Station"])["_rainy_flag"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
    ).fillna(0.0)
    df_out["Rainy_Days_Prev_7d"] = df_out.groupby(["District", "Block/Station"])["_rainy_flag"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
    ).fillna(0.0)
    df_out["Rainy_Days_Prev_15d"] = df_out.groupby(["District", "Block/Station"])["_rainy_flag"].transform(
        lambda s: s.shift(1).rolling(window=15, min_periods=1).sum()
    ).fillna(0.0)
    df_out["Rainy_Days_Prev_30d"] = df_out.groupby(["District", "Block/Station"])["_rainy_flag"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=1).sum()
    ).fillna(0.0)
    
    # Recompute consecutive rainy days before
    def _compute_consecutive(series: pd.Series) -> pd.Series:
        shifted = series.shift(1).fillna(0.0)
        counts = []
        c = 0.0
        for v in shifted:
            if v >= 1.0:
                c += 1.0
            else:
                c = 0.0
            counts.append(c)
        return pd.Series(counts, index=series.index)
        
    df_out["Consecutive_Rainy_Days_Before"] = df_out.groupby(["District", "Block/Station"])["_rainy_flag"].transform(_compute_consecutive)
    
    df_out.drop(columns=["_rainy_flag", "Date_Parsed"], inplace=True)
    return df_out
