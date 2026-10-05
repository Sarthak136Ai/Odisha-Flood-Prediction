"""
Main Streamlit Application for Odisha Flood Intelligence & Early Warning System.
Spatio-temporal flood prediction using 24 years of rainfall and SRC-derived flood observations.
"""

import os
import sys
import json
import pandas as pd
import streamlit as st
from PIL import Image

# Add project root and app directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data.load_data import load_combined_data, load_config
from chatbot.chatbot import OdishaFloodChatbot
from chatbot.prediction_tools import ChatbotPredictionTool

try:
    from app.components.prediction import render_prediction_component
    from app.components.charts import render_climate_analytics, render_model_comparison_chart
    from app.components.map import render_odisha_map
    from app.components.chatbot_ui import render_chatbot_component
    from app.components.timeline import render_historical_timeline_component
    from app.components.unseen_2025 import render_unseen_2025_component
    from app.components.drilldown import render_drilldown_component
    from app.components.what_if import render_what_if_simulator
except ImportError:
    from components.prediction import render_prediction_component
    from components.charts import render_climate_analytics, render_model_comparison_chart
    from components.map import render_odisha_map
    from components.chatbot_ui import render_chatbot_component
    from components.timeline import render_historical_timeline_component
    from components.unseen_2025 import render_unseen_2025_component
    from components.drilldown import render_drilldown_component
    from components.what_if import render_what_if_simulator

# Streamlit Page Config
st.set_page_config(
    page_title="Odisha Flood Intelligence & Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern emergency command center styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f2b48;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 1.02rem;
        font-weight: 500;
        color: #4a5568;
        margin-bottom: 14px;
    }
    .kpi-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-left: 5px solid #2b6cb0;
        padding: 14px 18px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 6px 6px 0 0;
        color: #4a5568;
    }
    .stTabs [aria-selected="true"] {
        color: #2b6cb0 !important;
        border-bottom: 3px solid #2b6cb0 !important;
        background-color: rgba(43, 108, 176, 0.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_system_resources():
    """Load and cache combined dataset, prediction tools, chatbot engine, and evaluation artifacts."""
    config = load_config("config.yaml")
    df = load_combined_data(config["paths"]["combined_data_path"], parse_dates=False)
    
    bot = OdishaFloodChatbot(
        data_path=config["paths"]["combined_data_path"],
        model_path="models/flood_prediction/best_model.pkl",
        metadata_path="models/flood_prediction/model_metadata.json"
    )
    pred_tool = ChatbotPredictionTool(
        model_path="models/flood_prediction/best_model.pkl",
        metadata_path="models/flood_prediction/model_metadata.json",
        data_path=config["paths"]["combined_data_path"]
    )
    
    # Load Model Comparison
    comp_path = os.path.join(config["paths"]["results_metrics_dir"], "model_comparison.csv")
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()
    
    # Load Year-Wise Metrics
    yw_path = os.path.join(config["paths"]["results_metrics_dir"], "year_wise_metrics.csv")
    yw_df = pd.read_csv(yw_path) if os.path.exists(yw_path) else pd.DataFrame()
    
    # Load Downscaling Metrics
    down_path = os.path.join(config["paths"]["results_downscaling_dir"], "downscaling_metrics.csv")
    down_df = pd.read_csv(down_path) if os.path.exists(down_path) else pd.DataFrame()
    
    # Load Feature Importance
    imp_path = os.path.join(config["paths"]["results_metrics_dir"], "feature_importance.csv")
    imp_df = pd.read_csv(imp_path) if os.path.exists(imp_path) else pd.DataFrame()
    
    return config, df, bot, pred_tool, comp_df, yw_df, down_df, imp_df


def main():
    config, df, bot, pred_tool, comp_df, yw_df, down_df, imp_df = get_system_resources()
    
    # Header
    st.markdown('<div class="main-title">🌊 Odisha Flood Intelligence & Early Warning System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Spatio-temporal flood prediction using 24 years of rainfall and SRC-derived flood observations (2001–2024)</div>', unsafe_allow_html=True)
    
    # Top KPI Metrics Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Historical Dataset", "2001–2024 (24 Yrs)", "2.75M Observations")
    with k2:
        st.metric("Spatial Coverage", "30 Districts", "354 Block Stations")
    with k3:
        st.metric("Best Model PR-AUC", "0.8306", "Test Set (2022–2024)")
    with k4:
        st.metric("Best Model ROC-AUC", "0.9693", "LogReg Balanced")
    with k5:
        st.metric("2025 Inference Engine", "Active (Frozen)", "Strict Zero Leakage")
        
    st.markdown("---")
    
    # Tab Navigation
    tabs = st.tabs([
        "🗺️ Odisha Risk Map",
        "🔍 District & Block Drilldown",
        "⏱️ Historical Timeline & Replay",
        "🔮 Real-Time Forecaster",
        "🧪 What-If Simulator",
        "🛰️ 2025 Unseen Risk Monitor",
        "🏆 Model Benchmarks & Calibration",
        "🧠 Explainable AI (SHAP)",
        "🌧️ Rainfall Downscaling",
        "💬 AI Flood Assistant",
        "📑 System Architecture"
    ])
    
    # 1. Odisha Risk Map
    with tabs[0]:
        render_odisha_map(df)
        
    # 2. District & Block Drilldown
    with tabs[1]:
        render_drilldown_component(df, pred_tool)
        
    # 3. Historical Timeline & Event Replay
    with tabs[2]:
        render_historical_timeline_component(df, pred_tool)
        
    # 4. Real-Time Forecaster
    with tabs[3]:
        render_prediction_component(pred_tool, df)
        
    # 5. What-If Simulator
    with tabs[4]:
        render_what_if_simulator(df, pred_tool)
        
    # 6. 2025 Unseen Operational Monitor
    with tabs[5]:
        render_unseen_2025_component(df, pred_tool)
        
    # 7. Model Benchmarks & Calibration
    with tabs[6]:
        st.markdown("### 🏆 Multi-Model Chronological Benchmarking & Reliability")
        st.markdown(
            "Evaluated strictly on future unseen historical test years (**2022–2024**, `343,758` records) "
            "with threshold calibration performed on the disjoint validation period (**2019–2021**)."
        )
        
        if not comp_df.empty:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            render_model_comparison_chart(comp_df)
            
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            if os.path.exists("results/plots/roc_curve.png"):
                st.image("results/plots/roc_curve.png", caption="Test ROC Curves (2022-2024)", use_container_width=True)
        with col_c2:
            if os.path.exists("results/plots/precision_recall_curve.png"):
                st.image("results/plots/precision_recall_curve.png", caption="Test Precision-Recall Curves", use_container_width=True)
        with col_c3:
            if os.path.exists("results/plots/calibration_curves.png"):
                st.image("results/plots/calibration_curves.png", caption="Reliability Diagram & Calibration", use_container_width=True)
                
        # Year-Wise Performance Breakdown
        st.markdown("#### 📅 Temporal Stability: Year-by-Year Performance Breakdown (Test Period)")
        if not yw_df.empty:
            st.dataframe(yw_df, use_container_width=True, hide_index=True)
            
    # 8. Explainable AI (SHAP)
    with tabs[7]:
        st.markdown("### 🧠 Explainable AI & Game-Theoretic SHAP Attributions")
        st.markdown(
            "Uncovering the meteorological and temporal mechanisms that govern the model's early-warning outputs."
        )
        
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            if os.path.exists("results/plots/shap_summary.png"):
                st.image("results/plots/shap_summary.png", caption="Global SHAP Beeswarm Distribution", use_container_width=True)
        with col_x2:
            if os.path.exists("results/plots/feature_importance.png"):
                st.image("results/plots/feature_importance.png", caption="Standardized Linear Coefficients / Relative Importance", use_container_width=True)
                
        if not imp_df.empty:
            st.markdown("#### Feature Influence Ranking")
            st.dataframe(imp_df, use_container_width=True, hide_index=True)
            
    # 9. Statistical Downscaling
    with tabs[8]:
        st.markdown("### 🌧️ Statistical Rainfall Downscaling Framework")
        st.markdown(
            "Bridging coarse regional weather driver inputs to high-resolution local station precipitation, "
            "and verifying the preservation of early warning flood detection skill."
        )
        
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            if os.path.exists("results/downscaling/rainfall_comparison.png"):
                st.image("results/downscaling/rainfall_comparison.png", caption="Observed vs Downscaled Station Precipitation", use_container_width=True)
        with col_d2:
            st.markdown("#### Flood Early Warning Impact Comparison")
            if not down_df.empty:
                st.dataframe(down_df, use_container_width=True, hide_index=True)
            st.success(
                "✅ **Key Finding**: Downscaled precipitation preserved flood early warning performance with zero degradation "
                "(PR-AUC `0.8402` vs `0.8306`, F1 `0.8750` vs `0.8748`), demonstrating operational feasibility for low-resolution forecast ingestion."
            )
            
    # 10. AI Flood Assistant
    with tabs[9]:
        render_chatbot_component(bot)
        
    # 11. System Architecture
    with tabs[10]:
        st.markdown("### 📑 System Architecture, Reproducibility & Methodology")
        st.markdown("""
        #### System Core Dimensions:
        1. **WHERE (Spatial Intelligence)**: 30 standard districts, 354 block stations, and historical vulnerability league tables.
        2. **WHEN (Temporal Intelligence)**: Date-indexed continuous lag and accumulation windows (1d, 2d, 3d, 7d, 15d, 30d sums/maxima, consecutive wet spells).
        3. **HOW HIGH (Probability-Calibrated Risk)**: Calibrated probabilities categorized into `LOW` (<30%), `MODERATE` (30–70%), and `HIGH` (≥70%) risk alerts based on validation optimization.
        4. **WHY (Explainable AI)**: Exact SHAP game-theoretic value attributions and log-odds coefficient factor attributions for every single prediction.
        
        #### Strict Scientific Guardrails:
        - **Zero Future Lookahead / Leakage**: Temporal split (Train: 2001–2018, Validation: 2019–2021, Test: 2022–2024). Target `Flood_Next_Day` is strictly observed on day T+1.
        - **2025 Unseen Protocol**: Frozen model inference with transparent academic disclaimer; no speculative accuracy claims without ground-truth labels.
        - **Class Imbalance Optimization**: Minority flood class (~2.8%) optimized using balanced class weights, threshold tuning, and PR-AUC prioritization.
        """)


if __name__ == "__main__":
    main()
