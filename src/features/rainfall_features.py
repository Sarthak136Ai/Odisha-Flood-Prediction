"""
Advanced feature engineering module for Odisha Flood Intelligence System.
Implements spatial/seasonal baseline Rainfall Anomaly and IMD extreme precipitation indicators.
All baselines are computed strictly on training data to prevent temporal leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class ClimatologicalAnomalyEngine:
    """
    Computes spatial-seasonal expected rainfall baselines from training data
    and calculates rainfall anomalies for any dataset.
    """
    def __init__(self):
        self.district_month_baseline_: Dict[Tuple[str, int], float] = {}
        self.global_month_baseline_: Dict[int, float] = {}
        self.is_fitted_ = False

    def fit(self, train_df: pd.DataFrame, rainfall_col: str = "Rainfall (mm)"):
        """
        Fit historical district-month climatological expectations on training partition.
        """
        # District x Month expected daily rainfall
        grouped = train_df.groupby(["District", "Month_Number"])[rainfall_col].mean().to_dict()
        self.district_month_baseline_ = grouped
        
        # Fallback Month expected daily rainfall
        self.global_month_baseline_ = train_df.groupby("Month_Number")[rainfall_col].mean().to_dict()
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, rainfall_col: str = "Rainfall (mm)") -> pd.Series:
        """
        Calculate rainfall anomaly = Current Rainfall - Historical Expected Baseline.
        """
        if not self.is_fitted_:
            raise ValueError("ClimatologicalAnomalyEngine must be fitted on training data before transforming.")
            
        def get_baseline(row):
            dist = row["District"]
            month = int(row["Month_Number"])
            return self.district_month_baseline_.get((dist, month), self.global_month_baseline_.get(month, 0.0))
            
        expected_rainfall = df.apply(get_baseline, axis=1)
        anomaly = df[rainfall_col] - expected_rainfall
        return anomaly.rename("Rainfall_Anomaly")


def enrich_rainfall_features(df: pd.DataFrame, anomaly_engine: Optional[ClimatologicalAnomalyEngine] = None) -> pd.DataFrame:
    """
    Add technical rainfall features including:
    - 7-day and 30-day rolling averages
    - IMD threshold extreme rainfall indicators (>= 64.5mm Heavy, >= 115.6mm Very Heavy)
    - Rainfall Anomaly (if anomaly engine is provided)
    """
    df = df.copy()
    
    # 1. Rolling averages
    if "Rainfall_Prev_7d_Sum" in df.columns:
        df["Rainfall_Prev_7d_Avg"] = df["Rainfall_Prev_7d_Sum"] / 7.0
    if "Rainfall_Prev_30d_Sum" in df.columns:
        df["Rainfall_Prev_30d_Avg"] = df["Rainfall_Prev_30d_Sum"] / 30.0
        
    # 2. IMD Meteorological Extreme Rainfall Indicators
    # IMD definitions: Heavy: 64.5 to 115.5 mm, Very Heavy: 115.6 to 204.4 mm, Extremely Heavy: >= 204.5 mm
    if "Rainfall (mm)" in df.columns:
        df["Is_Heavy_Rain"] = (df["Rainfall (mm)"] >= 64.5).astype(int)
        df["Is_Very_Heavy_Rain"] = (df["Rainfall (mm)"] >= 115.6).astype(int)
        
    # 3. Rainfall Anomaly
    if anomaly_engine is not None and anomaly_engine.is_fitted_:
        df["Rainfall_Anomaly"] = anomaly_engine.transform(df)
        
    return df
