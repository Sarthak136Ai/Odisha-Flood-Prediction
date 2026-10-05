"""
Controlled What-If Scenario Simulator Component.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from chatbot.prediction_tools import ChatbotPredictionTool
from src.data.combine_data import STANDARD_DISTRICTS


def render_what_if_simulator(df: pd.DataFrame, pred_tool: ChatbotPredictionTool):
    """Render interactive What-If scenario simulation tool."""
    st.markdown("### 🧪 Controlled What-If Scenario Simulator")
    
    st.info(
        "ℹ️ **Scientific Transparency Note**: This tool performs **machine learning model sensitivity simulation**. "
        "It evaluates how the trained statistical model responds to hypothetical changes in precipitation and soil moisture accumulation. "
        "It is not a 2D hydrodynamic flood routing simulation."
    )
    
    col_sim_ctrl, col_sim_out = st.columns([1, 1], gap="large")
    
    with col_sim_ctrl:
        st.markdown("#### 1. Baseline Selection & Synthetic Overrides")
        
        sim_dist = st.selectbox("Target District:", STANDARD_DISTRICTS, index=6, key="wi_dist")
        blocks = sorted(df[df["District"] == sim_dist]["Block/Station"].unique())
        sim_block = st.selectbox("Target Block / Station:", blocks, key="wi_block")
        
        sim_month = st.selectbox(
            "Simulated Monsoon Month:",
            list(range(1, 13)),
            index=7, # August
            format_func=lambda m: ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][m],
            key="wi_month"
        )
        
        st.markdown("##### Precipitation Parameters:")
        sim_rain = st.slider("Simulated 24-Hr Rainfall (mm):", 0.0, 400.0, 120.0, 2.0, key="wi_rain")
        sim_3d = st.slider("Simulated 3-Day Cumulative Rainfall (mm):", 0.0, 700.0, 240.0, 5.0, key="wi_3d")
        sim_7d = st.slider("Simulated 7-Day Cumulative Rainfall (mm):", 0.0, 1000.0, 410.0, 10.0, key="wi_7d")
        sim_15d = st.slider("Simulated 15-Day Cumulative Rainfall (mm):", 0.0, 1500.0, 620.0, 10.0, key="wi_15d")
        sim_active_flood = st.checkbox("Simulate Pre-Existing Flood Inundation Today", value=False, key="wi_flood")
        
        btn_run_sim = st.button("🚀 Execute Scenario Simulation", use_container_width=True, type="primary", key="wi_run")
        
    with col_sim_out:
        st.markdown("#### 2. Simulation Outcome & Sensitivity Response")
        
        if btn_run_sim or "sim_result" in st.session_state:
            if btn_run_sim:
                with st.spinner("Computing model response across parameter space..."):
                    res = pred_tool.predict_custom_scenario(
                        district=sim_dist,
                        rainfall_mm=sim_rain,
                        rainfall_prev_3d_sum=sim_3d,
                        rainfall_prev_7d_sum=sim_7d,
                        rainfall_prev_15d_sum=sim_15d,
                        month=sim_month,
                        flood_occurred_today=1 if sim_active_flood else 0
                    )
                    st.session_state["sim_result"] = res
                    
            res = st.session_state["sim_result"]
            prob = res["flood_probability"]
            risk = res["risk_level"]
            prob_pct = prob * 100
            
            color = "#d9534f" if risk == "High" else ("#f0ad4e" if risk == "Moderate" else "#5cb85c")
            
            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Simulated Risk: <b>{risk.upper()}</b>", 'font': {'size': 20, 'color': color}},
                number={'suffix': "%", 'font': {'size': 32, 'color': color}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': prob_pct}
                }
            ))
            fig_g.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_g, use_container_width=True)
            
            # Sensitivity Summary
            st.markdown("##### 📈 Rainfall Sensitivity Sweep (0mm to 300mm)")
            
            # Generate sensitivity curve
            sweep_rainfalls = np.linspace(0, 300, 20)
            sweep_probs = []
            for r_val in sweep_rainfalls:
                r_res = pred_tool.predict_custom_scenario(
                    district=sim_dist,
                    rainfall_mm=r_val,
                    rainfall_prev_3d_sum=sim_3d,
                    rainfall_prev_7d_sum=sim_7d,
                    rainfall_prev_15d_sum=sim_15d,
                    month=sim_month,
                    flood_occurred_today=1 if sim_active_flood else 0
                )
                sweep_probs.append(r_res["flood_probability"] * 100)
                
            fig_sweep = go.Figure()
            fig_sweep.add_trace(go.Scatter(
                x=sweep_rainfalls,
                y=sweep_probs,
                mode="lines+markers",
                name="Flood Probability (%)",
                line=dict(color="#3182ce", width=3)
            ))
            fig_sweep.add_vline(x=sim_rain, line_dash="dash", line_color="#e53e3e", annotation_text=f"Current Sim ({sim_rain} mm)")
            fig_sweep.add_hline(y=70, line_dash="dot", line_color="orange", annotation_text="High Risk Threshold (70%)")
            fig_sweep.update_layout(
                title=f"<b>Risk Sensitivity Curve for {sim_block} ({sim_dist})</b>",
                xaxis=dict(title="Simulated 24-Hr Precipitation (mm)"),
                yaxis=dict(title="Estimated Flood Probability (%)", range=[0, 105]),
                height=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_sweep, use_container_width=True)
        else:
            st.info("👈 Adjust the hypothetical precipitation sliders on the left and click **Execute Scenario Simulation**.")
