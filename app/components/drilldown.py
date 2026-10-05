"""
District & Block Drilldown Risk Panel Component with SHAP Explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from chatbot.prediction_tools import ChatbotPredictionTool


def render_drilldown_component(df: pd.DataFrame, pred_tool: ChatbotPredictionTool):
    """Render comprehensive District and Block Risk Drilldown Panel."""
    st.markdown("### 🔍 Spatio-Temporal Risk Drilldown: District & Block Station Intelligence")
    st.markdown(
        "Hierarchical drilldown answering: **WHERE is the risk? WHEN is the risk? HOW HIGH is the probability? WHY is the model producing this risk?**"
    )
    
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    
    with col_sel1:
        districts = sorted(df["District"].unique())
        sel_district = st.selectbox("1. Select District:", districts, index=districts.index("CUTTACK") if "CUTTACK" in districts else 0, key="dd_dist")
        
    with col_sel2:
        district_blocks = sorted(df[df["District"] == sel_district]["Block/Station"].unique())
        sel_block = st.selectbox("2. Select Block / Station:", district_blocks, key="dd_block")
        
    with col_sel3:
        available_dates = sorted(df[(df["District"] == sel_district) & (df["Block/Station"] == sel_block)]["Date"].unique(), reverse=True)
        default_d = "2024-08-15" if "2024-08-15" in available_dates else available_dates[0]
        sel_date = st.selectbox("3. Select Target Date:", available_dates, index=available_dates.index(default_d) if default_d in available_dates else 0, key="dd_date")
        
    st.markdown("---")
    
    # Run Prediction & Feature Retrieval
    with st.spinner("Retrieving station meteorology and computing explainable AI attributions..."):
        res = pred_tool.predict_for_location_and_date(
            district=sel_district,
            block=sel_block,
            date_str=sel_date
        )
        
    prob_pct = res["flood_probability"] * 100
    risk = res["risk_level"]
    color = "#d9534f" if risk == "High" else ("#f0ad4e" if risk == "Moderate" else "#5cb85c")
    
    # Header Risk Summary
    row_kpi1, row_kpi2, row_kpi3, row_kpi4, row_kpi5 = st.columns(5)
    with row_kpi1:
        st.metric("Location", f"{sel_block}, {sel_district}")
    with row_kpi2:
        st.metric("Date", sel_date)
    with row_kpi3:
        st.metric("Flood Probability", f"{prob_pct:.1f}%")
    with row_kpi4:
        st.metric("Risk Level", risk)
    with row_kpi5:
        st.metric("Early Warning Class", "FLOOD ALERT (1)" if res["predicted_flood_next_day"] == 1 else "NO FLOOD (0)")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown("#### 🌧️ Station Meteorological Indicators (Input State)")
        
        # Look up station record in df
        rec = df[(df["District"] == sel_district) & (df["Block/Station"] == sel_block) & (df["Date"] == sel_date)]
        if not rec.empty:
            r = rec.iloc[0]
            
            m_df = pd.DataFrame([
                {"Indicator": "Today's Rainfall (24-Hr)", "Value": f"{r.get('Rainfall (mm)', 0.0):.1f} mm", "Category": "Precipitation"},
                {"Indicator": "Prior 3-Day Cumulative Rainfall", "Value": f"{r.get('Rainfall_Prev_3d_Sum', 0.0):.1f} mm", "Category": "Accumulation"},
                {"Indicator": "Prior 7-Day Cumulative Rainfall", "Value": f"{r.get('Rainfall_Prev_7d_Sum', 0.0):.1f} mm", "Category": "Accumulation"},
                {"Indicator": "Prior 15-Day Cumulative Rainfall", "Value": f"{r.get('Rainfall_Prev_15d_Sum', 0.0):.1f} mm", "Category": "Accumulation"},
                {"Indicator": "Prior 30-Day Cumulative Rainfall", "Value": f"{r.get('Rainfall_Prev_30d_Sum', 0.0):.1f} mm", "Category": "Accumulation"},
                {"Indicator": "Consecutive Rainy Days Before", "Value": f"{int(r.get('Consecutive_Rainy_Days_Before', 0))} days", "Category": "Persistence"},
                {"Indicator": "Rainy Days in Past 30 Days", "Value": f"{int(r.get('Rainy_Days_Prev_30d', 0))} days", "Category": "Persistence"},
                {"Indicator": "Active Flood Inundation Today", "Value": "YES" if r.get("Flood_Occurred", 0) == 1 else "No", "Category": "Hydrology"}
            ])
            st.dataframe(m_df, use_container_width=True, hide_index=True)
        else:
            st.info("Direct station record not found; using calibrated real-time estimation.")
            
    with col_right:
        st.markdown("#### 🧠 Explainable AI: Why is the Model Producing this Risk?")
        st.markdown(f"Game-theoretic SHAP / coefficient attributions for **{sel_block}** on **{sel_date}**:")
        
        drivers = res.get("top_risk_drivers", [])
        mitigators = res.get("top_mitigators", [])
        
        feature_names = []
        contributions = []
        colors = []
        
        for d in drivers[:4]:
            feature_names.append(d["feature"])
            contributions.append(d["contribution"])
            colors.append("#e53e3e") # Red for risk amplification
            
        for m in mitigators[:4]:
            feature_names.append(m["feature"])
            contributions.append(m["contribution"])
            colors.append("#38a169") # Green for risk mitigation
            
        if contributions:
            fig_shap = go.Figure(go.Bar(
                x=contributions,
                y=feature_names,
                orientation="h",
                marker_color=colors,
                text=[f"{c:+.3f}" for c in contributions],
                textposition="auto"
            ))
            fig_shap.update_layout(
                title="<b>Local Factor Contribution (Log-Odds Impact)</b>",
                xaxis=dict(title="Contribution to Flood Probability (+ = Increases Risk, - = Lowers Risk)"),
                yaxis=dict(autorange="reversed"),
                height=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            
            st.success(
                f"💡 **Physical Explanation**: The prediction for {sel_block} is primarily driven by "
                f"**{drivers[0]['feature'] if drivers else 'antecedent rainfall'}** (+{drivers[0]['contribution']:.2f}) "
                f"amplifying flood potential, mitigated by **{mitigators[0]['feature'] if mitigators else 'seasonal trends'}**."
            )
