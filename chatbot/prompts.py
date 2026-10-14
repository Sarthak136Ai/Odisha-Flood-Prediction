"""
System prompt guidelines, formatting helpers, and grounded response templates for the Odisha Flood Chatbot.
The chatbot functions as an interface layer, retrieving and interpreting verified model outputs,
metrics, and datasets without inventing numerical values.
"""

from typing import Dict, Any, List, Optional


SYSTEM_PROMPT_TEMPLATE = """
You are an interface layer for the Odisha Flood Early Warning and Disaster Intelligence System.
Your answers are grounded strictly in:
1. Current machine learning model predictions (FloodPredictor)
2. Project datasets (2001–2024 historical telemetry and 2025 operational stream)
3. Documented model evaluation metrics (ROC-AUC, PR-AUC, Recall, Precision, Brier score)
4. Formal risk level definitions (LOW, MODERATE, HIGH)
5. Documented hydrological facts and IMD standards for Odisha.

Rules:
1. You are an interface layer, not the core predictive model.
2. Never invent or hallucinate numerical predictions, probabilities, or rainfall amounts.
3. If information is unavailable, explicitly state: "Sufficient project data is not available."
4. Always reference actual model metadata, thresholds, and metrics rather than hardcoded approximations.
"""


def format_prediction_response(res: Dict[str, Any]) -> str:
    """Format model prediction result into clear markdown."""
    prob = res.get("flood_probability", 0.0)
    prob_pct = prob * 100
    risk = res.get("risk_level", "Low")
    risk_badge = "🔴 **HIGH RISK**" if risk.upper() == "HIGH" else ("🟡 **MODERATE RISK**" if risk.upper() == "MODERATE" else "🟢 **LOW RISK**")
    model_name = res.get("model_name", "Production Model")
    threshold = res.get("prediction_threshold", 0.8073)
    
    msg = f"### Flood Prediction for **{res.get('district', 'Odisha')}**"
    if res.get("block"):
        msg += f" (Block: *{res['block']}*)"
    if res.get("date"):
        msg += f" — Date: `{res['date']}`"
        
    msg += f"\n\n- **Estimated Next-Day Flood Probability**: `{prob_pct:.1f}%`"
    msg += f"\n- **Risk Classification**: {risk_badge}"
    msg += f"\n- **Inference Model**: `{model_name}` (Classification Alert Threshold: `{threshold:.4f}`)"
    msg += f"\n- **Station Rainfall Telemetry**: Current `{res.get('current_rainfall_mm', 0.0):.1f} mm` | Prev 3-Day Sum `{res.get('rainfall_prev_3d_sum_mm', 0.0):.1f} mm` | Prev 15-Day Sum `{res.get('rainfall_prev_15d_sum_mm', 0.0):.1f} mm`\n\n"
    
    top_drivers = res.get("top_risk_drivers", [])
    if top_drivers:
        msg += "#### Key Factors Influencing Prediction (Top Risk Amplifiers):\n"
        for d in top_drivers[:3]:
            val_str = f"{d['value']:.1f}" if isinstance(d['value'], (int, float)) else str(d['value'])
            msg += f"- **{d['feature']}** ({val_str}): +{d['contribution']:.3f} log-odds\n"
            
    top_mitigators = res.get("top_mitigators", [])
    if top_mitigators:
        msg += "\n#### Key Risk Mitigating Factors:\n"
        for m in top_mitigators[:2]:
            val_str = f"{m['value']:.1f}" if isinstance(m['value'], (int, float)) else str(m['value'])
            msg += f"- **{m['feature']}** ({val_str}): {m['contribution']:.3f} log-odds\n"
            
    return msg


def format_risk_definition_response(risk_thresholds: Optional[Dict[str, float]] = None) -> str:
    """Format project risk level definitions grounded in model thresholds."""
    low_max = risk_thresholds.get("low_max", 0.30) if risk_thresholds else 0.30
    mod_max = risk_thresholds.get("moderate_max", 0.70) if risk_thresholds else 0.70
    
    return (
        "### Project Flood Risk Level Definitions\n\n"
        "The Odisha Flood Prediction System classifies next-day flood risk into three standardized operational tiers:\n\n"
        f"1. 🔴 **HIGH RISK** (Probability $\\ge {mod_max * 100:.0f}\\%$ or calibrated posterior $\\ge 20\\%$):\n"
        "   - **Meaning**: Indicates critical hydrologic soil saturation paired with intense current or forecasted rainfall, representing severe risk of inundation within the next 24 hours.\n"
        "   - **Operational Action**: Triggers urgent administrative alert, reservoir gate pre-discharge coordination, and ODRAF/NDRF rescue unit staging.\n\n"
        f"2. 🟡 **MODERATE RISK** (Probability between ${low_max * 100:.0f}\\%$ and ${mod_max * 100:.0f}\\%$):\n"
        "   - **Meaning**: Elevated antecedent precipitation and river swelling; high potential for localized waterlogging in low-lying blocks.\n"
        "   - **Operational Action**: Heightened vigilance, hourly river gauge telemetry monitoring, and precautionary advisory for vulnerable riverbank settlements.\n\n"
        f"3. 🟢 **LOW RISK** (Probability $< {low_max * 100:.0f}\\%$):\n"
        "   - **Meaning**: Precipitation levels and antecedent moisture remain within normal catchment drainage capacity with no imminent flood threat.\n"
        "   - **Operational Action**: Standard civic routine and continuous baseline monitoring."
    )


def format_model_info_response(metadata: Dict[str, Any]) -> str:
    """Format model identity and architecture information from verified metadata."""
    model_name = metadata.get("model_name", "XGBoost")
    threshold = metadata.get("optimal_threshold", 0.8073)
    train_years = metadata.get("train_years", [2001, 2018])
    features = metadata.get("features", [])
    
    start_yr = min(train_years) if train_years else 2001
    end_yr = max(train_years) if train_years else 2018
    
    return (
        f"### Flood Prediction Model Architecture\n\n"
        f"- **Primary Production Model**: `{model_name}` (Champion Model)\n"
        f"- **Optimal Decision Threshold ($T^*$)**: `{threshold:.4f}`\n"
        f"- **Historical Training Baseline**: `{start_yr}–{end_yr}` ({len(train_years)} historical years, 2,063,989 station records)\n"
        f"- **Input Feature Count**: `{len(features)}` engineered meteorological and temporal features\n"
        f"- **Benchmark Family**: The system evaluates 5 architectures: Logistic Regression, Decision Tree, Random Forest, XGBoost, and Artificial Neural Network (ANN/MLP).\n"
        f"- **Role of Chatbot**: The chatbot serves exclusively as an **interface layer** for user queries, routing requests to the frozen `{model_name}` inference pipeline without modifying model weights."
    )


def format_recall_and_metrics_response(metadata: Dict[str, Any], metrics_df: Optional[Any] = None) -> str:
    """Format model recall and evaluation metrics from verified test results."""
    test_metrics = metadata.get("test_metrics", {})
    model_name = metadata.get("model_name", "XGBoost")
    
    recall = test_metrics.get("Test_Recall", 0.33945)
    precision = test_metrics.get("Test_Precision", 0.19046)
    roc_auc = test_metrics.get("Test_ROC_AUC", 0.83876)
    pr_auc = test_metrics.get("Test_PR_AUC", 0.15926)
    f1 = test_metrics.get("Test_F1", 0.24401)
    brier = test_metrics.get("Test_Brier_Score", 0.14445)
    tp = test_metrics.get("Test_TP", 3331)
    fn = test_metrics.get("Test_FN", 6482)
    
    msg = (
        f"### Model Recall & Test Performance Metrics\n\n"
        f"Evaluated on the **untouched chronological test partition (2022–2024, 343,758 observations)** for `{model_name}`:\n\n"
        f"- **Test Recall (Sensitivity)**: **`{recall * 100:.2f}%`** (detected `{tp:,}` flood events)\n"
        f"- **Test Precision**: `{precision * 100:.2f}%`\n"
        f"- **Test F1-Score**: `{f1:.4f}`\n"
        f"- **Test ROC-AUC**: `{roc_auc:.4f}`\n"
        f"- **Test PR-AUC**: `{pr_auc:.4f}`\n"
        f"- **Test Brier Score**: `{brier:.4f}`\n\n"
        "#### Why Recall is Prioritized in Flood Forecasting:\n"
        f"In disaster risk reduction (DRR), **False Negatives** (`{fn:,}` missed floods in test partition) carry catastrophic life-safety consequences because communities receive zero warning before inundation. "
        "A high recall ensures that active flood events are detected in advance, even if it requires accepting precautionary false alarms (False Positives)."
    )
    return msg


def format_rainfall_features_response(features: List[str]) -> str:
    """Format engineered rainfall feature inventory."""
    return (
        "### Engineered Rainfall & Meteorological Features\n\n"
        "The flood forecasting pipeline uses a 22+ feature taxonomy engineered to capture surface runoff, soil saturation, and monsoon seasonality:\n\n"
        "1. **Current & Immediate Rainfall**:\n"
        "   - `Rainfall (mm)`: 24-hour recorded precipitation at block telemetry station.\n"
        "   - `Rainfall_Missing`: Binary indicator (1 if sensor reported missing/NaN, 0 otherwise).\n\n"
        "2. **Antecedent Rainfall Lags (Hydrologic Travel Time)**:\n"
        "   - `Rainfall_Lag_1d`, `Rainfall_Lag_2d`, `Rainfall_Lag_3d`, `Rainfall_Lag_7d`: Captures river wave propagation from upstream tributaries.\n\n"
        "3. **Cumulative Rolling Sums (Catchment Soil Saturation)**:\n"
        "   - `Rainfall_Prev_3d_Sum`, `Rainfall_Prev_7d_Sum`, `Rainfall_Prev_15d_Sum`, `Rainfall_Prev_30d_Sum`: Multi-day rainfall accumulation acting as a proxy for soil moisture storage capacity.\n\n"
        "4. **Peak Precipitation Maxima (Convective Surges)**:\n"
        "   - `Rainfall_Prev_3d_Max`, `Rainfall_Prev_7d_Max`, `Rainfall_Prev_15d_Max`, `Rainfall_Prev_30d_Max`: Detects isolated cloudbursts.\n\n"
        "5. **Antecedent Wetness & Rainy Days Frequency**:\n"
        "   - `Rainy_Days_Prev_3d`, `Rainy_Days_Prev_7d`, `Rainy_Days_Prev_15d`, `Rainy_Days_Prev_30d`, `Consecutive_Rainy_Days_Before`: Count of days with rainfall $\\ge 2.5\\text{ mm}$ (statutory IMD rainy day threshold).\n\n"
        "6. **Monsoon Seasonality & IMD Intensity Categories**:\n"
        "   - `Month_sin`, `Month_cos`, `Day_of_Year`, plus statutory IMD tiers (`Light`, `Moderate`, `Heavy`, `Very Heavy`, `Extreme`)."
    )
