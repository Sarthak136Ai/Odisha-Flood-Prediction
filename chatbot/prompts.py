"""
System prompt guidelines and formatting helpers for the Odisha Flood Chatbot.
"""

SYSTEM_PROMPT_TEMPLATE = """
You are the Odisha Flood Early Warning and Climate Assistant.
Your answers must be grounded strictly in the verified 24-year historical dataset (2001-2024),
the trained machine learning early-warning models, explainability attributions (SHAP),
and verified hydrological knowledge of Odisha.

Rules:
1. Never invent or hallucinate flood probabilities or rainfall amounts.
2. For prediction questions, always state the calculated probability, risk level (Low, Moderate, High), and key meteorological drivers.
3. For historical questions, retrieve exact metrics from the dataset.
4. For general concepts, provide scientific, clear definitions.
"""


def format_prediction_response(res: dict) -> str:
    """Format model prediction result into clear markdown."""
    prob_pct = res["flood_probability"] * 100
    risk = res["risk_level"]
    risk_badge = "🔴 **HIGH RISK**" if risk == "High" else ("🟡 **MODERATE RISK**" if risk == "Moderate" else "🟢 **LOW RISK**")
    
    msg = f"### Flood Prediction for **{res.get('district', 'Odisha')}**"
    if res.get("block"):
        msg += f" (Block: *{res['block']}*)"
    if res.get("date"):
        msg += f" — Date: `{res['date']}`"
        
    msg += f"\n\n- **Estimated Next-Day Flood Probability**: `{prob_pct:.1f}%`\n"
    msg += f"- **Risk Classification**: {risk_badge}\n"
    msg += f"- **Recent Rainfall Status**: Current `{res.get('current_rainfall_mm', 0.0):.1f} mm` | Prev 3-Day Sum `{res.get('rainfall_prev_3d_sum_mm', 0.0):.1f} mm` | Prev 15-Day Sum `{res.get('rainfall_prev_15d_sum_mm', 0.0):.1f} mm`\n\n"
    
    if res.get("top_risk_drivers"):
        msg += "#### Key Risk Amplifiers (Top Drivers):\n"
        for d in res["top_risk_drivers"][:3]:
            msg += f"- **{d['feature']}** ({d['value']:.1f}): +{d['contribution']:.3f} log-odds\n"
            
    if res.get("top_mitigators"):
        msg += "\n#### Key Risk Mitigators:\n"
        for m in res["top_mitigators"][:2]:
            msg += f"- **{m['feature']}** ({m['value']:.1f}): {m['contribution']:.3f} log-odds\n"
            
    return msg
