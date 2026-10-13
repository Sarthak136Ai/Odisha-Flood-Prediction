"""
Advanced feature engineering module for Odisha Flood Intelligence System.
Implements spatial/seasonal baseline Rainfall Anomaly and IMD rainfall-intensity indicators.

Threshold Standards & Scientific Citations:
All rainfall intensity thresholds are defined according to statutory meteorological standards
established by the India Meteorological Department (IMD), Ministry of Earth Sciences,
Government of India, and adopted by the Special Relief Commissioner (SRC), Government of Odisha:
- Trace / Very Light Rain: 0.1 to 2.4 mm
- Rainy Day (Statutory IMD Cutoff): >= 2.5 mm
- Light Rainfall: 2.5 to 15.5 mm
- Moderate Rainfall: 15.6 to 64.4 mm
- Heavy Rainfall: 64.5 to 115.5 mm (Yellow/Orange Alert threshold)
- Very Heavy Rainfall: 115.6 to 204.4 mm (Orange/Red Alert threshold)
- Extremely Heavy Rainfall: >= 204.5 mm (Red Alert / Catastrophic Flood hazard threshold)

All feature transformations are strictly causal and chronological, using only past and
present rainfall observations on observation day T (or antecedent lags), guaranteeing zero
future information leakage. Missing or invalid values are strictly mapped to non-active (0).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Any


# =============================================================================
# IMD METEOROLOGICAL RAINFALL INTENSITY CONSTANTS (24-Hour Accumulation in mm)
# =============================================================================
IMD_DRY_CUTOFF = 0.0
IMD_TRACE_MAX = 2.4              # 0.1 - 2.4 mm: Trace / Very Light Rain
IMD_RAINY_DAY_MIN = 2.5          # >= 2.5 mm: Statutory IMD threshold for a "Rainy Day"
IMD_LIGHT_MIN = 2.5              # 2.5 - 15.5 mm: Light Rainfall
IMD_LIGHT_MAX = 15.5
IMD_MODERATE_MIN = 15.6          # 15.6 - 64.4 mm: Moderate Rainfall
IMD_MODERATE_MAX = 64.4
IMD_HEAVY_MIN = 64.5             # 64.5 - 115.5 mm: Heavy Rainfall
IMD_HEAVY_MAX = 115.5
IMD_VERY_HEAVY_MIN = 115.6       # 115.6 - 204.4 mm: Very Heavy Rainfall
IMD_VERY_HEAVY_MAX = 204.4
IMD_EXTREME_MIN = 204.5          # >= 204.5 mm: Extremely Heavy Rainfall


# Canonical list of rainfall intensity feature column names
RAINFALL_INTENSITY_COLUMNS = [
    "Light_Rainfall",
    "Moderate_Rainfall",
    "Heavy_Rainfall",
    "Very_Heavy_Rainfall",
    "Extreme_Rainfall",
    "Heavy_Rain_Indicator",
    "Extreme_Rain_Indicator",
]


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


class RainfallIntensityEngine:
    """
    Engine for generating standardized IMD rainfall-intensity indicators.
    Guarantees:
    - Pure causal computation: uses only observation day T rainfall telemetry.
    - Zero future lookahead.
    - Robust handling of missing (NaN), None, or corrupted negative rainfall values.
    - Strict adherence to official India Meteorological Department (IMD) intensity thresholds.
    """

    @staticmethod
    def classify_single_rainfall(rainfall_mm: Optional[float]) -> Dict[str, int]:
        """
        Classify a single rainfall scalar into all IMD intensity indicators.
        Returns a dictionary of binary indicators {0, 1}.
        """
        if rainfall_mm is None or np.isnan(rainfall_mm) or rainfall_mm < 0:
            return {
                "Light_Rainfall": 0,
                "Moderate_Rainfall": 0,
                "Heavy_Rainfall": 0,
                "Very_Heavy_Rainfall": 0,
                "Extreme_Rainfall": 0,
                "Heavy_Rain_Indicator": 0,
                "Extreme_Rain_Indicator": 0,
                "Is_Heavy_Rain": 0,
                "Is_Very_Heavy_Rain": 0,
                "Is_Extreme_Rain": 0,
                "Rain_Light": 0,
                "Rain_Moderate": 0,
                "Rain_Heavy": 0,
                "Rain_Very_Heavy": 0,
                "Rain_Extreme": 0,
            }

        rf = float(rainfall_mm)
        
        is_light = int(IMD_LIGHT_MIN <= rf <= IMD_LIGHT_MAX)
        is_moderate = int(IMD_MODERATE_MIN <= rf <= IMD_MODERATE_MAX)
        is_heavy = int(IMD_HEAVY_MIN <= rf <= IMD_HEAVY_MAX)
        is_very_heavy = int(IMD_VERY_HEAVY_MIN <= rf <= IMD_VERY_HEAVY_MAX)
        is_extreme = int(rf >= IMD_EXTREME_MIN)

        # Cumulative alert warning triggers
        heavy_ind = int(rf >= IMD_HEAVY_MIN)          # Heavy, Very Heavy, or Extreme
        extreme_ind = int(rf >= IMD_EXTREME_MIN)      # Extreme only
        very_heavy_ind = int(rf >= IMD_VERY_HEAVY_MIN) # Very Heavy or Extreme

        return {
            # Canonical feature names
            "Light_Rainfall": is_light,
            "Moderate_Rainfall": is_moderate,
            "Heavy_Rainfall": is_heavy,
            "Very_Heavy_Rainfall": is_very_heavy,
            "Extreme_Rainfall": is_extreme,
            "Heavy_Rain_Indicator": heavy_ind,
            "Extreme_Rain_Indicator": extreme_ind,
            # Backward-compatible & concise aliases
            "Is_Heavy_Rain": heavy_ind,
            "Is_Very_Heavy_Rain": very_heavy_ind,
            "Is_Extreme_Rain": extreme_ind,
            "Rain_Light": is_light,
            "Rain_Moderate": is_moderate,
            "Rain_Heavy": is_heavy,
            "Rain_Very_Heavy": is_very_heavy,
            "Rain_Extreme": is_extreme,
        }

    @classmethod
    def transform(
        cls,
        df: pd.DataFrame,
        rainfall_col: str = "Rainfall (mm)",
        missing_flag_col: Optional[str] = "Rainfall_Missing"
    ) -> pd.DataFrame:
        """
        Compute all IMD rainfall-intensity indicators across a DataFrame.
        Missing (NaN) or negative rainfall values are safely masked to 0.
        """
        df_out = df.copy()
        
        if rainfall_col not in df_out.columns:
            # If rainfall column is absent, fill with 0s
            for col in RAINFALL_INTENSITY_COLUMNS:
                df_out[col] = 0
            df_out["Is_Heavy_Rain"] = 0
            df_out["Is_Very_Heavy_Rain"] = 0
            df_out["Is_Extreme_Rain"] = 0
            df_out["Rain_Light"] = 0
            df_out["Rain_Moderate"] = 0
            df_out["Rain_Heavy"] = 0
            df_out["Rain_Very_Heavy"] = 0
            df_out["Rain_Extreme"] = 0
            return df_out

        rf_series = pd.to_numeric(df_out[rainfall_col], errors="coerce")
        
        # Valid mask: non-null and non-negative
        valid_mask = rf_series.notnull() & (rf_series >= 0)
        if missing_flag_col and missing_flag_col in df_out.columns:
            valid_mask = valid_mask & (df_out[missing_flag_col] == 0)

        rf_clean = rf_series.where(valid_mask, other=-1.0)

        # Mutually exclusive categorical bins
        light_series = (rf_clean >= IMD_LIGHT_MIN) & (rf_clean <= IMD_LIGHT_MAX)
        mod_series = (rf_clean >= IMD_MODERATE_MIN) & (rf_clean <= IMD_MODERATE_MAX)
        heavy_series = (rf_clean >= IMD_HEAVY_MIN) & (rf_clean <= IMD_HEAVY_MAX)
        vheavy_series = (rf_clean >= IMD_VERY_HEAVY_MIN) & (rf_clean <= IMD_VERY_HEAVY_MAX)
        ext_series = (rf_clean >= IMD_EXTREME_MIN)

        # Cumulative alert triggers
        heavy_ind_series = (rf_clean >= IMD_HEAVY_MIN)
        vheavy_ind_series = (rf_clean >= IMD_VERY_HEAVY_MIN)
        ext_ind_series = (rf_clean >= IMD_EXTREME_MIN)

        # Canonical features
        df_out["Light_Rainfall"] = light_series.astype(int)
        df_out["Moderate_Rainfall"] = mod_series.astype(int)
        df_out["Heavy_Rainfall"] = heavy_series.astype(int)
        df_out["Very_Heavy_Rainfall"] = vheavy_series.astype(int)
        df_out["Extreme_Rainfall"] = ext_series.astype(int)
        df_out["Heavy_Rain_Indicator"] = heavy_ind_series.astype(int)
        df_out["Extreme_Rain_Indicator"] = ext_ind_series.astype(int)

        # Backward compatibility and concise aliases
        df_out["Is_Heavy_Rain"] = heavy_ind_series.astype(int)
        df_out["Is_Very_Heavy_Rain"] = vheavy_ind_series.astype(int)
        df_out["Is_Extreme_Rain"] = ext_ind_series.astype(int)
        df_out["Rain_Light"] = df_out["Light_Rainfall"]
        df_out["Rain_Moderate"] = df_out["Moderate_Rainfall"]
        df_out["Rain_Heavy"] = df_out["Heavy_Rainfall"]
        df_out["Rain_Very_Heavy"] = df_out["Very_Heavy_Rainfall"]
        df_out["Rain_Extreme"] = df_out["Extreme_Rainfall"]

        return df_out


def compute_rainfall_intensity_features(
    df: pd.DataFrame,
    rainfall_col: str = "Rainfall (mm)",
    missing_flag_col: Optional[str] = "Rainfall_Missing"
) -> pd.DataFrame:
    """
    Convenience function to compute IMD rainfall intensity features on a DataFrame.
    """
    return RainfallIntensityEngine.transform(
        df=df,
        rainfall_col=rainfall_col,
        missing_flag_col=missing_flag_col
    )


def enrich_rainfall_features(
    df: pd.DataFrame,
    anomaly_engine: Optional[ClimatologicalAnomalyEngine] = None
) -> pd.DataFrame:
    """
    Add advanced technical rainfall features including:
    - 7-day and 30-day rolling averages
    - IMD rainfall-intensity indicators (Light, Moderate, Heavy, Very Heavy, Extreme)
    - IMD Heavy-rain and Extreme-rain warning indicators
    - Rainfall Anomaly (if climatological anomaly engine is provided)
    """
    df = df.copy()

    # 1. IMD Meteorological Rainfall Intensity Indicators
    df = RainfallIntensityEngine.transform(df, rainfall_col="Rainfall (mm)")

    # 2. Rolling averages
    if "Rainfall_Prev_7d_Sum" in df.columns:
        df["Rainfall_Prev_7d_Avg"] = df["Rainfall_Prev_7d_Sum"] / 7.0
    if "Rainfall_Prev_30d_Sum" in df.columns:
        df["Rainfall_Prev_30d_Avg"] = df["Rainfall_Prev_30d_Sum"] / 30.0

    # 3. Rainfall Anomaly
    if anomaly_engine is not None and anomaly_engine.is_fitted_:
        df["Rainfall_Anomaly"] = anomaly_engine.transform(df)

    return df
