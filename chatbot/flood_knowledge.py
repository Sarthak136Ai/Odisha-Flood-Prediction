"""
Flood Knowledge Base for Odisha meteorological, hydrological, and scientific domain facts.
Provides grounded reference definitions and facts for the chatbot interface layer.
"""

from typing import Optional, Dict, List

ODISHA_FLOOD_KNOWLEDGE = {
    "river_basins": {
        "Mahanadi": "The largest river basin in Odisha, draining ~42% of the state. Major flood drivers include upstream discharges from Hirakud Dam (Sambalpur), intense monsoon depressions over Chhattisgarh, and deltaic confluence bottlenecks in Cuttack, Puri, and Jagatsinghpur.",
        "Brahmani": "The second largest river basin in Odisha, formed by the confluence of the Sankh and South Koel rivers in Sundargarh. Prone to flash floods affecting Jajpur and Kendrapara before discharging into the Bay of Bengal near Dhamra.",
        "Baitarani": "Originates in the Gonasika hills of Keonjhar. Fast-draining river with steep upstream gradients causing rapid inundation in Anandapur, Akhuapada, and Jajpur.",
        "Subarnarekha": "Interstate river flowing through Jharkhand, West Bengal, and northern Odisha (Balasore district). Severe floods occur during cyclonic rainfall and upstream Galudih barrage releases.",
        "Rushikulya": "Major river in southern Odisha (Ganjam district), characterized by rapid flash floods following Bay of Bengal low-pressure depressions.",
        "Budhabalanga": "Originates from Similipal hills in Mayurbhanj district; causes coastal waterlogging in Balasore and Baripada during monsoon downpours."
    },
    "rainfall_thresholds_imd": {
        "No Rain / Dry": "0.0 mm/day",
        "Very Light / Trace": "0.1 to 2.4 mm/day (below statutory IMD rainy-day threshold)",
        "Light Rain": "2.5 to 15.5 mm/day (meets IMD rainy-day criterion)",
        "Moderate Rain": "15.6 to 64.4 mm/day (steady soil moisture saturation)",
        "Heavy Rain": "64.5 to 115.5 mm/day (significant surface runoff, Yellow/Orange alert)",
        "Very Heavy Rain": "115.6 to 204.4 mm/day (catchment inundation, Orange alert)",
        "Extremely Heavy Rain": ">= 204.5 mm/day (severe flood emergency, Red alert)"
    },
    "scientific_concepts": {
        "rainfall_downscaling": "A statistical spatial disaggregation method that maps coarse-scale regional precipitation aggregates (district spatial means of ~5,190 km²) to local block rain gauge stations (~440 km²). It bridges the spatial granularity gap between coarse weather forecasts and local block hydrology without claiming unverified high-resolution gridded NWP.",
        "flood_probability": "The mathematical likelihood (0.0 to 1.0, or 0% to 100%) that an administrative block station will experience next-day inundation (Flood_Next_Day = 1), computed from antecedent cumulative rainfall, soil saturation proxies, and seasonal cycles.",
        "explainable_ai_shap": "SHAP (SHapley Additive exPlanations) uses cooperative game theory to quantify the exact contribution of each meteorological variable (e.g., 15-day rainfall sum, current precipitation) to the model's log-odds prediction for a specific date and station.",
        "class_imbalance": "In Odisha, flood occurrences are rare events (~2.64% of total daily observations). The model handles this 36.9:1 imbalance through loss weighting, decision threshold optimization, and probability calibration rather than synthetic oversampling (SMOTE), which avoids temporal leakage.",
        "disaster_risk_reduction": "Disaster risk reduction (DRR) prioritizes High Recall over Precision because False Negatives (missed floods with no warning) carry catastrophic consequences for human life and property, whereas precautionary False Positives are manageable alerts."
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
    elif "probability" in q_lower and ("what is" in q_lower or "how is" in q_lower or "definition" in q_lower):
        return ODISHA_FLOOD_KNOWLEDGE["scientific_concepts"]["flood_probability"]
    elif "imbalance" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["scientific_concepts"]["class_imbalance"]
    elif "drr" in q_lower or "disaster risk" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["scientific_concepts"]["disaster_risk_reduction"]
    elif "mahanadi" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["river_basins"]["Mahanadi"]
    elif "brahmani" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["river_basins"]["Brahmani"]
    elif "baitarani" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["river_basins"]["Baitarani"]
    elif "subarnarekha" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["river_basins"]["Subarnarekha"]
    elif "rushikulya" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["river_basins"]["Rushikulya"]
    elif "budhabalanga" in q_lower:
        return ODISHA_FLOOD_KNOWLEDGE["river_basins"]["Budhabalanga"]
    elif "river" in q_lower or "basin" in q_lower:
        basins = list(ODISHA_FLOOD_KNOWLEDGE["river_basins"].keys())
        return f"Major flood-prone river basins in Odisha include: {', '.join(basins)}. You can ask about any specific basin (e.g., 'Tell me about the Mahanadi basin')."
    elif "imd" in q_lower or "rainfall classification" in q_lower or "heavy rain" in q_lower or "rain category" in q_lower:
        return "Statutory IMD 24-Hour Rainfall Classifications:\n" + "\n".join([f"- **{k}**: {v}" for k, v in ODISHA_FLOOD_KNOWLEDGE["rainfall_thresholds_imd"].items()])
    elif "vulnerable" in q_lower or "flood prone" in q_lower or "high risk district" in q_lower:
        return f"Historically most flood-prone coastal, deltaic, and riverine districts in Odisha include: {', '.join(ODISHA_FLOOD_KNOWLEDGE['vulnerable_districts'])}."
        
    return None
