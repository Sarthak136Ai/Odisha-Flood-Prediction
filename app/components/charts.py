"""
Interactive Plotly charting components for Odisha Flood Prediction dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any


def render_climate_analytics(df: pd.DataFrame):
    """Render 24-year precipitation and climate trends."""
    st.markdown("### 📊 24-Year Rainfall & Monsoon Dynamics (2001–2024)")
    
    # 1. Annual Rainfall vs Flood Events
    annual_df = df.groupby("Year").agg(
        total_rainfall_mm=("Rainfall (mm)", "sum"),
        avg_rainfall_mm=("Rainfall (mm)", "mean"),
        total_flood_events=("Flood_Occurred", "sum")
    ).reset_index()
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=annual_df["Year"],
        y=annual_df["total_flood_events"],
        name="Annual Flood Events (Days)",
        marker_color="#d9534f",
        yaxis="y1"
    ))
    fig1.add_trace(go.Scatter(
        x=annual_df["Year"],
        y=annual_df["total_rainfall_mm"] / 1000.0,
        name="Cumulative Rainfall (Thousand mm)",
        mode="lines+markers",
        marker=dict(size=8, color="#1f77b4"),
        line=dict(width=3, color="#1f77b4"),
        yaxis="y2"
    ))
    fig1.update_layout(
        title="<b>Annual Historical Inundation vs Total Precipitation (2001–2024)</b>",
        xaxis=dict(title="Year", tickmode="linear"),
        yaxis=dict(title="Total Flood Events (Station-Days)", side="left", showgrid=True),
        yaxis2=dict(title="Cumulative Rainfall (k mm)", side="right", overlaying="y", showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly Seasonality
        monthly_df = df.groupby("Month_Number").agg(
            mean_rain=("Rainfall (mm)", "mean"),
            total_floods=("Flood_Occurred", "sum")
        ).reset_index()
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_df["Month_Name"] = monthly_df["Month_Number"].apply(lambda m: months[m-1])
        
        fig2 = px.bar(
            monthly_df,
            x="Month_Name",
            y="mean_rain",
            color="total_floods",
            color_continuous_scale="Viridis",
            labels={"mean_rain": "Daily Mean Rain (mm)", "total_floods": "Total Flood Days", "Month_Name": "Month"},
            title="<b>Monthly Monsoon Precipitation Cycle</b>"
        )
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        
    with col2:
        # Top Vulnerable Districts
        dist_df = df.groupby("District").agg(
            total_floods=("Flood_Occurred", "sum")
        ).reset_index().sort_values(by="total_floods", ascending=True).tail(12)
        
        fig3 = px.bar(
            dist_df,
            x="total_floods",
            y="District",
            orientation="h",
            color="total_floods",
            color_continuous_scale="Reds",
            labels={"total_floods": "Flood Incidents (Days)", "District": "District"},
            title="<b>Top 12 Most Flood-Prone Districts (2001–2024)</b>"
        )
        fig3.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)


def render_model_comparison_chart(comp_df: pd.DataFrame):
    """Render interactive model performance comparison bar chart."""
    st.markdown("### 📈 Multi-Model Chronological Benchmark (Test 2022–2024)")
    
    metrics = ["Test_ROC_AUC", "Test_PR_AUC", "Test_F1", "Test_Recall", "Test_Precision"]
    
    df_melt = comp_df.melt(
        id_vars=["Model"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )
    df_melt["Metric"] = df_melt["Metric"].str.replace("Test_", "")
    
    fig = px.bar(
        df_melt,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="<b>Comparative Performance on Strictly Future Unseen Years (2022–2024)</b>",
        color_discrete_sequence=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"],
        text_auto=".3f"
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1.05], title="Score / AUC"),
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
