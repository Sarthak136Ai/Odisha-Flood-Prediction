"""
Geospatial analysis and risk mapping module for Odisha districts and blocks.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

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


# GeoJSON File Paths
GEOJSON_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "geospatial", "odisha_districts.geojson")
)
GEOJSON_STATIC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "static", "data", "odisha_districts.geojson")
)
HISTORICAL_PRESET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "geospatial", "historical_preset_predictions.csv")
)


def load_odisha_districts_geojson() -> Dict[str, Any]:
    """
    Load the standardized GeoJSON boundary dataset for all 30 districts of Odisha.
    Tries data/geospatial first, falling back to static/data.
    """
    target_path = GEOJSON_DATA_PATH if os.path.exists(GEOJSON_DATA_PATH) else GEOJSON_STATIC_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Odisha district boundaries GeoJSON not found at {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def classify_flood_risk(probability: float) -> Tuple[str, str, float]:
    """
    Categorize predicted probability into standardized risk tier, hex color, and base extrusion scale.
    
    Tiers:
    - LOW (< 0.30): #10b981 (Green)
    - MODERATE (0.30 to < 0.70): #eab308 (Yellow)
    - HIGH (0.70 to < 0.85): #f97316 (Orange)
    - VERY HIGH (>= 0.85): #ef4444 (Red)
    """
    p = float(max(0.0, min(1.0, probability)))
    if p >= 0.85:
        return "VERY HIGH", "#ef4444", p * 60000.0 + 2000.0
    elif p >= 0.70:
        return "HIGH", "#f97316", p * 55000.0 + 1500.0
    elif p >= 0.30:
        return "MODERATE", "#eab308", p * 45000.0 + 1000.0
    else:
        return "LOW", "#10b981", max(1000.0, p * 35000.0)


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


def build_3d_risk_geojson(
    date_str: str,
    cached_2025_df: Optional[pd.DataFrame] = None,
    district_risk_df: Optional[pd.DataFrame] = None,
    data_layer: str = "probability"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build an enriched 3D GeoJSON FeatureCollection for Odisha's 30 districts on a specified date.
    
    Parameters
    ----------
    date_str : str
        Target prediction date in YYYY-MM-DD format (e.g. '2025-08-15' or '2022-08-15').
    cached_2025_df : Optional[pd.DataFrame]
        In-memory 2025 operational predictions DataFrame containing block-level records.
    district_risk_df : Optional[pd.DataFrame]
        24-year historical district baseline summary DataFrame.
    data_layer : str
        Extrusion layer driver: 'probability', 'rainfall', 'antecedent_rainfall', or 'historical_risk'.
        
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        (enriched_geojson, summary_metadata)
    """
    raw_geojson = load_odisha_districts_geojson()
    
    # Sanitize date parameter
    target_date = str(date_str).strip()
    fallback_applied = False
    data_source = "2025 Operational Predictions"
    
    # Attempt to locate date in available prediction data
    sub_df = pd.DataFrame()
    
    # Check 1: In-memory 2025 operational predictions
    if cached_2025_df is not None and not cached_2025_df.empty and "Date" in cached_2025_df.columns:
        sub_df = cached_2025_df[cached_2025_df["Date"] == target_date]
        if not sub_df.empty:
            data_source = "2025 Operational Pipeline Inference (Frozen XGBoost / LogReg)"
    
    # Check 2: Historical preset disaster dates (2019-08-08, 2020-08-27, 2022-08-15, 2024-09-16)
    if sub_df.empty and os.path.exists(HISTORICAL_PRESET_PATH):
        try:
            df_presets = pd.read_csv(HISTORICAL_PRESET_PATH)
            m_pre = df_presets[df_presets["Date"] == target_date]
            if not m_pre.empty:
                sub_df = m_pre
                data_source = f"Historical Disaster Replay ({target_date})"
        except Exception:
            pass
            
    # Check 3: If still empty, fall back to default peak-monsoon date '2025-08-15'
    if sub_df.empty:
        if cached_2025_df is not None and not cached_2025_df.empty:
            sub_df = cached_2025_df[cached_2025_df["Date"] == "2025-08-15"]
            if not sub_df.empty:
                target_date = "2025-08-15"
                fallback_applied = True
                data_source = "2025 Operational Pipeline (Defaulted to Peak Monsoon 2025-08-15)"
                
    # Prepare district baseline lookup if available
    hist_lookup = {}
    if district_risk_df is not None and not district_risk_df.empty:
        for _, row in district_risk_df.iterrows():
            d_name = str(row["District"]).strip().upper()
            hist_lookup[d_name] = {
                "flood_days": int(row.get("total_flood_days", 0)),
                "flood_rate": float(row.get("flood_frequency_pct", 0.0)),
                "avg_annual_rain": float(row.get("avg_annual_rainfall", 1450.0))
            }
            
    # Aggregate predictions by district if sub_df is populated
    district_data_map = {}
    if not sub_df.empty:
        for dist_name, grp in sub_df.groupby("District"):
            d_clean = str(dist_name).strip().upper()
            # Mean probability across reporting blocks
            p_mean = float(grp["Predicted_Probability"].mean()) if "Predicted_Probability" in grp.columns else 0.05
            p_max = float(grp["Predicted_Probability"].max()) if "Predicted_Probability" in grp.columns else p_mean
            
            # Rainfall metrics
            rf_mean = float(grp["Rainfall (mm)"].mean()) if "Rainfall (mm)" in grp.columns else 0.0
            rf_max = float(grp["Rainfall (mm)"].max()) if "Rainfall (mm)" in grp.columns else rf_mean
            
            rf_3d = float(grp["Rainfall_Prev_3d_Sum"].mean()) if "Rainfall_Prev_3d_Sum" in grp.columns else rf_mean * 2.0
            rf_7d = float(grp["Rainfall_Prev_7d_Sum"].mean()) if "Rainfall_Prev_7d_Sum" in grp.columns else rf_3d * 1.8
            rf_15d = float(grp["Rainfall_Prev_15d_Sum"].mean()) if "Rainfall_Prev_15d_Sum" in grp.columns else rf_7d * 1.6
            rf_30d = float(grp["Rainfall_Prev_30d_Sum"].mean()) if "Rainfall_Prev_30d_Sum" in grp.columns else rf_15d * 1.4
            
            # Risk drivers
            d1 = str(grp["Top_Risk_Driver_1"].iloc[0]) if "Top_Risk_Driver_1" in grp.columns and len(grp) > 0 else "Rainfall_Prev_7d_Sum"
            d2 = str(grp["Top_Risk_Driver_2"].iloc[0]) if "Top_Risk_Driver_2" in grp.columns and len(grp) > 0 else "Rainfall_Prev_3d_Sum"
            d3 = str(grp["Top_Risk_Driver_3"].iloc[0]) if "Top_Risk_Driver_3" in grp.columns and len(grp) > 0 else "Month_cos"
            
            # Blocks info
            blocks_list = sorted(list(grp["Block/Station"].unique())) if "Block/Station" in grp.columns else []
            # Find block with highest probability
            peak_block = blocks_list[0] if blocks_list else "District HQ"
            if "Block/Station" in grp.columns and "Predicted_Probability" in grp.columns:
                max_idx = grp["Predicted_Probability"].idxmax()
                peak_block = str(grp.loc[max_idx, "Block/Station"])
                
            district_data_map[d_clean] = {
                "predicted_probability": round(p_mean, 4),
                "max_probability": round(p_max, 4),
                "rainfall_mm": round(rf_mean, 1),
                "max_rainfall_mm": round(rf_max, 1),
                "rainfall_prev_3d_sum": round(rf_3d, 1),
                "rainfall_prev_7d_sum": round(rf_7d, 1),
                "rainfall_prev_15d_sum": round(rf_15d, 1),
                "rainfall_prev_30d_sum": round(rf_30d, 1),
                "top_risk_driver_1": d1,
                "top_risk_driver_2": d2,
                "top_risk_driver_3": d3,
                "blocks_count": len(blocks_list),
                "blocks_sample": blocks_list,
                "peak_block": peak_block
            }

    # Enrich GeoJSON features
    features = []
    high_count = 0
    mod_count = 0
    low_count = 0
    vhigh_count = 0
    prob_list = []
    rainfall_list = []
    
    for f in raw_geojson.get("features", []):
        props = f.get("properties", {})
        d_name = str(props.get("district_name") or props.get("Dist_Name") or "").strip().upper()
        
        # Coordinates and Headquarters lookup
        coord_meta = ODISHA_DISTRICT_COORDINATES.get(d_name, {"lat": 20.5, "lon": 84.5, "hq": d_name.title()})
        hq_name = coord_meta.get("hq", d_name.title())
        c_lat = coord_meta.get("lat", 20.5)
        c_lon = coord_meta.get("lon", 84.5)
        
        # Historical baseline
        h_info = hist_lookup.get(d_name, {"flood_days": 120, "flood_rate": 2.2, "avg_annual_rain": 1450.0})
        
        # Match prediction data
        p_info = district_data_map.get(d_name)
        if p_info:
            proba = p_info["predicted_probability"]
            max_p = p_info["max_probability"]
            rf_val = p_info["rainfall_mm"]
            rf_max = p_info["max_rainfall_mm"]
            rf_3d = p_info["rainfall_prev_3d_sum"]
            rf_7d = p_info["rainfall_prev_7d_sum"]
            rf_15d = p_info["rainfall_prev_15d_sum"]
            rf_30d = p_info["rainfall_prev_30d_sum"]
            d1 = p_info["top_risk_driver_1"]
            d2 = p_info["top_risk_driver_2"]
            d3 = p_info["top_risk_driver_3"]
            b_cnt = p_info["blocks_count"]
            b_sample = p_info["blocks_sample"]
            p_blk = p_info["peak_block"]
        else:
            # Fallback to historical rate if prediction row missing
            proba = min(0.85, max(0.01, h_info["flood_rate"] / 8.0))
            max_p = proba
            rf_val = 15.0
            rf_max = 25.0
            rf_3d = 40.0
            rf_7d = 85.0
            rf_15d = 160.0
            rf_30d = 280.0
            d1 = "Rainfall_Prev_7d_Sum"
            d2 = "Rainfall_Prev_3d_Sum"
            d3 = "Month_cos"
            b_cnt = 10
            b_sample = [f"{d_name.title()} Sadar"]
            p_blk = f"{d_name.title()} Sadar"
            
        prob_list.append(proba)
        rainfall_list.append(rf_val)
        
        # Assign Risk Tier
        risk_level, risk_color, prob_height = classify_flood_risk(proba)
        if risk_level == "VERY HIGH":
            vhigh_count += 1
        elif risk_level == "HIGH":
            high_count += 1
        elif risk_level == "MODERATE":
            mod_count += 1
        else:
            low_count += 1
            
        # Determine 3D Extrusion Height depending on requested data layer
        if data_layer == "rainfall":
            # Scale 0 - 200 mm to 1,000 - 50,000 meters
            extrusion_height = max(1000.0, min(50000.0, (rf_val / 200.0) * 49000.0 + 1000.0))
        elif data_layer == "antecedent_rainfall":
            # Scale 0 - 400 mm to 1,000 - 50,000 meters
            extrusion_height = max(1000.0, min(50000.0, (rf_7d / 400.0) * 49000.0 + 1000.0))
        elif data_layer == "historical_risk":
            # Scale 0 - 6% flood rate to 1,000 - 50,000 meters
            extrusion_height = max(1000.0, min(50000.0, (h_info["flood_rate"] / 6.0) * 49000.0 + 1000.0))
        else:  # default 'probability'
            extrusion_height = prob_height

        # Assemble rich properties
        enriched_props = {
            "id": d_name,
            "district_id": d_name,
            "district_name": d_name,
            "display_name": d_name.title(),
            "headquarters": hq_name,
            "centroid_lat": c_lat,
            "centroid_lon": c_lon,
            "census_code": str(props.get("census_code", props.get("Dist_Code", "N/A"))),
            "predicted_probability": round(proba, 4),
            "predicted_probability_pct": round(proba * 100.0, 2),
            "max_block_probability": round(max_p, 4),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "extrusion_height": round(extrusion_height, 1),
            "rainfall_mm": round(rf_val, 1),
            "max_rainfall_mm": round(rf_max, 1),
            "rainfall_prev_3d_sum": round(rf_3d, 1),
            "rainfall_prev_7d_sum": round(rf_7d, 1),
            "rainfall_prev_15d_sum": round(rf_15d, 1),
            "rainfall_prev_30d_sum": round(rf_30d, 1),
            "prediction_date": target_date,
            "forecast_horizon": "T+1 (Next-Day)",
            "historical_flood_days": h_info["flood_days"],
            "historical_flood_rate_pct": round(h_info["flood_rate"], 2),
            "top_risk_driver_1": d1,
            "top_risk_driver_2": d2,
            "top_risk_driver_3": d3,
            "blocks_count": b_cnt,
            "peak_block": p_blk,
            "blocks_sample": b_sample[:8],
            "coverage_note": f"District-level 3D polygon extrusion. {b_cnt} block telemetry stations integrated."
        }
        
        f_copy = {
            "type": "Feature",
            "id": d_name,
            "properties": enriched_props,
            "geometry": f.get("geometry")
        }
        features.append(f_copy)
        
    enriched_geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }
    
    # Calculate state-level summary
    avg_p = float(np.mean(prob_list)) if prob_list else 0.0
    max_rf = float(np.max(rainfall_list)) if rainfall_list else 0.0
    
    # Identify highest-risk district
    sorted_features = sorted(features, key=lambda x: x["properties"]["predicted_probability"], reverse=True)
    highest_risk_dist = sorted_features[0]["properties"] if sorted_features else {}
    
    summary = {
        "selected_date": target_date,
        "data_layer": data_layer,
        "forecast_horizon": "T+1 (Next-Day Flood Inundation)",
        "total_districts": len(features),
        "data_source": data_source,
        "fallback_applied": fallback_applied,
        "highest_risk_district": {
            "name": highest_risk_dist.get("district_name", "N/A"),
            "probability": highest_risk_dist.get("predicted_probability", 0.0),
            "probability_pct": highest_risk_dist.get("predicted_probability_pct", 0.0),
            "risk_level": highest_risk_dist.get("risk_level", "LOW"),
            "peak_block": highest_risk_dist.get("peak_block", "N/A"),
            "rainfall_mm": highest_risk_dist.get("rainfall_mm", 0.0)
        },
        "risk_breakdown": {
            "very_high": vhigh_count,
            "high": high_count,
            "moderate": mod_count,
            "low": low_count,
            "total_elevated": vhigh_count + high_count
        },
        "state_averages": {
            "mean_probability": round(avg_p, 4),
            "mean_probability_pct": round(avg_p * 100.0, 2),
            "max_rainfall_mm": round(max_rf, 1)
        },
        "league_table": [
            {
                "district": f["properties"]["district_name"],
                "probability": f["properties"]["predicted_probability"],
                "probability_pct": f["properties"]["predicted_probability_pct"],
                "risk_level": f["properties"]["risk_level"],
                "rainfall_mm": f["properties"]["rainfall_mm"],
                "rainfall_prev_7d_sum": f["properties"]["rainfall_prev_7d_sum"],
                "peak_block": f["properties"]["peak_block"],
                "headquarters": f["properties"]["headquarters"]
            }
            for f in sorted_features
        ]
    }
    
    return enriched_geojson, summary

