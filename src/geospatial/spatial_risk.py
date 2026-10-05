"""
Geospatial analysis and risk mapping module for Odisha districts and blocks.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Verified latitude and longitude centers for all 30 standard districts of Odisha
ODISHA_DISTRICT_COORDINATES = {
    "ANGUL": {"lat": 20.8398, "lon": 85.1013, "hq": "Angul"},
    "BALANGIR": {"lat": 20.7107, "lon": 83.4851, "hq": "Balangir"},
    "BALASORE": {"lat": 21.4934, "lon": 86.9135, "hq": "Balasore"},
    "BARGARH": {"lat": 21.3333, "lon": 83.6167, "hq": "Bargarh"},
    "BHADRAK": {"lat": 21.0543, "lon": 86.4955, "hq": "Bhadrak"},
    "BOUDH": {"lat": 20.8400, "lon": 84.3200, "hq": "Boudh"},
    "CUTTACK": {"lat": 20.4625, "lon": 85.8828, "hq": "Cuttack"},
    "DEOGARH": {"lat": 21.5333, "lon": 84.7333, "hq": "Deogarh"},
    "DHENKANAL": {"lat": 20.6667, "lon": 85.6000, "hq": "Dhenkanal"},
    "GAJAPATI": {"lat": 18.8100, "lon": 84.1600, "hq": "Paralakhemundi"},
    "GANJAM": {"lat": 19.3800, "lon": 85.0500, "hq": "Chhatrapur"},
    "JAGATSINGHPUR": {"lat": 20.2667, "lon": 86.1667, "hq": "Jagatsinghpur"},
    "JAJPUR": {"lat": 20.8500, "lon": 86.3333, "hq": "Panikoili / Jajpur"},
    "JHARSUGUDA": {"lat": 21.8500, "lon": 84.0167, "hq": "Jharsuguda"},
    "KALAHANDI": {"lat": 19.9100, "lon": 83.1100, "hq": "Bhawanipatna"},
    "KANDHAMAL": {"lat": 20.1400, "lon": 84.1400, "hq": "Phulbani"},
    "KENDRAPARA": {"lat": 20.5000, "lon": 86.4200, "hq": "Kendrapara"},
    "KEONJHAR": {"lat": 21.6300, "lon": 85.5800, "hq": "Kendujhar"},
    "KHORDHA": {"lat": 20.1800, "lon": 85.6200, "hq": "Bhubaneswar / Khordha"},
    "KORAPUT": {"lat": 18.8100, "lon": 82.7100, "hq": "Koraput"},
    "MALKANGIRI": {"lat": 18.3400, "lon": 81.8900, "hq": "Malkangiri"},
    "MAYURBHANJ": {"lat": 21.9300, "lon": 86.7300, "hq": "Baripada"},
    "NAWARANGPUR": {"lat": 19.2300, "lon": 82.5500, "hq": "Nabarangpur"},
    "NAYAGARH": {"lat": 20.1300, "lon": 85.1000, "hq": "Nayagarh"},
    "NUAPADA": {"lat": 20.8300, "lon": 82.5200, "hq": "Nuapada"},
    "PURI": {"lat": 19.8135, "lon": 85.8312, "hq": "Puri"},
    "RAYAGADA": {"lat": 19.1700, "lon": 83.4200, "hq": "Rayagada"},
    "SAMBALPUR": {"lat": 21.4700, "lon": 83.9700, "hq": "Sambalpur"},
    "SUBARNAPUR": {"lat": 20.8300, "lon": 83.9100, "hq": "Sonepur"},
    "SUNDARGARH": {"lat": 22.1200, "lon": 84.0300, "hq": "Sundargarh / Rourkela"}
}


def calculate_district_historical_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 24-year cumulative flood risk and precipitation summary per district."""
    summary = df.groupby("District").agg(
        total_observations=("Date", "count"),
        total_flood_days=("Flood_Occurred", "sum"),
        total_rainfall_mm=("Rainfall (mm)", "sum"),
        max_single_day_rain=("Rainfall (mm)", "max"),
        avg_annual_rainfall=("Rainfall (mm)", lambda s: s.sum() / 24.0)
    ).reset_index()
    
    summary["flood_frequency_pct"] = (summary["total_flood_days"] / summary["total_observations"]) * 100
    
    # Add Coordinates
    summary["latitude"] = summary["District"].map(lambda d: ODISHA_DISTRICT_COORDINATES.get(d, {}).get("lat", 20.5))
    summary["longitude"] = summary["District"].map(lambda d: ODISHA_DISTRICT_COORDINATES.get(d, {}).get("lon", 84.5))
    summary["headquarters"] = summary["District"].map(lambda d: ODISHA_DISTRICT_COORDINATES.get(d, {}).get("hq", d))
    
    # Sort by historical flood frequency
    summary = summary.sort_values(by="total_flood_days", ascending=False).reset_index(drop=True)
    return summary


def get_odisha_spatial_data() -> Dict[str, Any]:
    """Get district geographic metadata dictionary."""
    return ODISHA_DISTRICT_COORDINATES
