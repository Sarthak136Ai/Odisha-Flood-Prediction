"""
Prediction UI component for Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from chatbot.prediction_tools import ChatbotPredictionTool
from src.data.combine_data import STANDARD_DISTRICTS


def render_prediction_component(pred_tool: ChatbotPredictionTool, df: pd.DataFrame):
    """Render interactive next-day flood forecasting interface."""
    st.markdown("### 🔮 Real-Time Next-Day Flood Forecaster & Risk Simulator")
    st.markdown(
        "Generate calibrated next-day flood probability, risk severity classification, "
        "and SHAP-grounded factor attributions."
    )
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("1. Location & Meteorological Inputs")
        
        mode = st.radio("Input Mode:", ["Historical Date Look-up", "Custom Weather Simulation"], horizontal=True)
        
        selected_district = st.selectbox("Select District:", STANDARD_DISTRICTS, index=6) # Cuttack by default
        
        # Get blocks for selected district
        district_blocks = sorted(df[df["District"] == selected_district]["Block/Station"].unique())
        selected_block = st.selectbox("Select Block / Station:", district_blocks)
        
        if mode == "Historical Date Look-up":
            available_dates = sorted(df[(df["District"] == selected_district) & (df["Block/Station"] == selected_block)]["Date"].unique(), reverse=True)
            default_date = "2024-08-15" if "2024-08-15" in available_dates else available_dates[0]
            selected_date = st.selectbox("Select Date (2001–2024):", available_dates, index=available_dates.index(default_date) if default_date in available_dates else 0)
            
            run_btn = st.button("🚀 Compute Flood Risk Forecast", use_container_width=True, type="primary")
            
            if run_btn:
                with st.spinner("Executing calibrated inference pipeline..."):
                    res = pred_tool.predict_for_location_and_date(
                        district=selected_district,
                        block=selected_block,
                        date_str=selected_date
                    )
                    st.session_state["last_pred_result"] = res
        else:
            st.markdown("#### Scenario Meteorological Sliders:")
            sim_rain = st.slider("Today's Rainfall (mm):", 0.0, 350.0, 85.0, 1.0)
            sim_prev_3d = st.slider("Previous 3-Day Cumulative Rainfall (mm):", 0.0, 600.0, 190.0, 5.0)
            sim_prev_7d = st.slider("Previous 7-Day Cumulative Rainfall (mm):", 0.0, 900.0, 310.0, 10.0)
            sim_prev_15d = st.slider("Previous 15-Day Cumulative Rainfall (mm):", 0.0, 1200.0, 480.0, 10.0)
            sim_flood_today = st.radio("Flood Occurred Today?", [0, 1], format_func=lambda x: "Yes (Active Inundation)" if x==1 else "No (Normal)")
            sim_month = st.selectbox("Simulated Month:", list(range(1, 13)), index=7, format_func=lambda m: ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m])
            
            run_btn = st.button("🚀 Run Scenario Simulation", use_container_width=True, type="primary")
            if run_btn:
                with st.spinner("Simulating scenario..."):
                    res = pred_tool.predict_custom_scenario(
                        district=selected_district,
                        rainfall_mm=sim_rain,
                        rainfall_prev_3d_sum=sim_prev_3d,
                        rainfall_prev_7d_sum=sim_prev_7d,
                        rainfall_prev_15d_sum=sim_prev_15d,
                        month=sim_month,
                        flood_occurred_today=sim_flood_today
                    )
                    res["district"] = selected_district
                    res["block"] = selected_block
                    st.session_state["last_pred_result"] = res
                    
    with col2:
        st.subheader("2. Early Warning Prediction Results")
        
        if "last_pred_result" in st.session_state:
            res = st.session_state["last_pred_result"]
            prob_pct = res["flood_probability"] * 100
            risk = res["risk_level"]
            
            # Risk Colors
            color = "#d9534f" if risk == "High" else ("#f0ad4e" if risk == "Moderate" else "#5cb85c")
            
            # Gauge Indicator
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Next-Day Flood Risk: <b>{risk.upper()}</b>", 'font': {'size': 20, 'color': color}},
                number={'suffix': "%", 'font': {'size': 32, 'color': color}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prob_pct
                    }
                }
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Metric Tiles
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Target", "FLOOD (1)" if res["predicted_flood_next_day"] == 1 else "NO FLOOD (0)")
            m2.metric("Probability", f"{prob_pct:.1f}%")
            m3.metric("Risk Category", risk)
            
            # Top Attributions
            st.markdown("#### 🧠 Factor Attributions & Explainability")
            
            drivers = res.get("top_risk_drivers", [])
            mitigators = res.get("top_mitigators", [])
            
            if drivers or mitigators:
                contrib_items = []
                for d in drivers[:3]:
                    contrib_items.append({"Feature": d["feature"], "Effect (+ Risk)": round(d["contribution"], 3), "Type": "Amplifier (Risk Up)"})
                for m in mitigators[:3]:
                    contrib_items.append({"Feature": m["feature"], "Effect (+ Risk)": round(m["contribution"], 3), "Type": "Mitigator (Risk Down)"})
                    
                df_c = pd.DataFrame(contrib_items)
                st.dataframe(df_c, use_container_width=True, hide_index=True)
        else:
            st.info("👈 Select location parameters on the left and click **Compute Flood Risk Forecast** to view early warning results.")
