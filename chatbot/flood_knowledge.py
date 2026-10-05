"""
Flood Knowledge Base for Odisha meteorological, hydrological, and scientific domain facts.
"""

from typing import Optional, Dict, List

ODISHA_FLOOD_KNOWLEDGE = {
    "river_basins": {
        "Mahanadi": "The largest river basin in Odisha, draining 42% of the state. Major flood drivers include Hirakud Dam discharges, intense monsoon depressions over Chhattisgarh and western Odisha.",
        "Brahmani": "Second largest river basin, originating in Jharkhand (South Koel and Sankh rivers) and joining Baitarani near Dhamra before entering Bay of Bengal. Prone to flash floods in Jajpur and Kendrapara.",
        "Baitarani": "Originates from Keonjhar plateau. Rapid runoff leads to severe inundation in Anandapur, Akhuapada, and Jajpur.",
        "Subarnarekha": "Interstate river flowing through Jharkhand, West Bengal, and northern Odisha (Balasore district). Severe floods occur during cyclonic rainfall.",
        "Rushikulya": "Major river in southern Odisha (Ganjam district), prone to rapid flash floods following Bay of Bengal low-pressure systems."
    },
    "rainfall_thresholds_imd": {
        "Very Light Rain": "0.1 to 2.4 mm/day",
        "Light Rain": "2.5 to 7.5 mm/day",
        "Moderate Rain": "7.6 to 35.5 mm/day",
        "Rather Heavy Rain": "35.6 to 64.4 mm/day",
        "Heavy Rain": "64.5 to 115.5 mm/day",
        "Very Heavy Rain": "115.6 to 204.4 mm/day",
        "Extremely Heavy Rain": ">= 204.5 mm/day"
    },
    "scientific_concepts": {
        "rainfall_downscaling": "A computational method to derive fine-scale, high-resolution local weather information (e.g., station-level 1km rainfall) from coarse-scale global climate models or satellite grids (e.g., 25km–100km). It bridges the spatial gap between global atmospheric forecasts and local catchment hydrology.",
        "flood_probability": "The mathematical likelihood (between 0.0 and 1.0 or 0% to 100%) that a location will experience inundation / flood conditions on the subsequent day, calculated from cumulative rainfall, soil saturation proxies, and seasonal factors.",
        "explainable_ai_shap": "SHAP (SHapley Additive exPlanations) uses game-theoretic principles to allocate fair credit to each input variable (e.g., 15-day rainfall sum, current flood status) for pushing a prediction toward or away from flood risk.",
        "class_imbalance": "In natural hazard forecasting, flood days are rare events (~2.64% of days in Odisha). Machine learning models must use balanced class weights, scale_pos_weight, and threshold calibration rather than raw accuracy to avoid missing critical disaster events."
    },
    "vulnerable_districts": [
        "Puri", "Cuttack", "Kendrapara", "Jagatsinghpur", "Jajpur", "Bhadrak", "Balasore", "Ganjam", "Subarnapur", "Boudh"
    ]
}


def get_knowledge_answer(query: str) -> Optional[str]:
    """Retrieve verified domain knowledge answer matching query keywords."""
    q_lower = query.lower()
    
    if "downscal" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["scientific_concepts"]["rainfall_downscaling"]
    elif "shap" in q_lower or "explain" in q_lower or "feature importance" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["scientific_concepts"]["explainable_ai_shap"]
    elif "probability" in q_lower or "risk level" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["scientific_concepts"]["flood_probability"]
    elif "imbalance" in q_lower or "pr-auc" in q_lower or "metric" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["scientific_concepts"]["class_imbalance"]
    elif "mahanadi" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["river_basins"]["Mahanadi"]
    elif "brahmani" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["river_basins"]["Brahmani"]
    elif "baitarani" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["river_basins"]["Baitarani"]
    elif "imd" in q_lower or "heavy rain" in q_lower or "threshold" in q_lower:
        return "IMD Rainfall Classifications: " + ", ".join([f"{k}: {v}" for k, v in ODISHA_FLOOD_KNOWLEDGE["rainfall_thresholds_imd"].items()])
    elif "vulnerable" in q_lower or "high risk district" in q_lower:
        return f"Historically most flood-prone coastal and deltaic districts in Odisha include: {', '.join(ODISHA_FLOOD_KNOWLEDGE['vulnerable_districts'])}."
        
    return None
