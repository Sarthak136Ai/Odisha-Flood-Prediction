"""
Historical Flood Timeline, Event Replay, and Prediction vs Actual Observation Module.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

from chatbot.prediction_tools import ChatbotPredictionTool


# Notable historical flood spells documented in Odisha State Disaster Management reports
HISTORICAL_NOTABLE_EVENTS = {
    "August 2022 Mahanadi & Subarnarekha Basin Flood": {
        "district": "CUTTACK",
        "block": "Banki",
        "start_date": "2022-08-10",
        "end_date": "2022-08-25",
        "description": "Deep depression over Bay of Bengal causing heavy sustained precipitation and major inundation across downstream Mahanadi delta."
    },
    "August 2020 Heavy Monsoon Depression": {
        "district": "PURI",
        "block": "Kanas",
        "start_date": "2020-08-15",
        "end_date": "2020-08-31",
        "description": "Consecutive low-pressure systems triggered severe waterlogging and flash floods across coastal delta tracts."
    },
    "August 2019 Monsoon Inundation (Post-Fani Year)": {
        "district": "BALASORE",
        "block": "Bhograi",
        "start_date": "2019-08-01",
        "end_date": "2019-08-18",
        "description": "Extreme localized downpour causing Subarnarekha river swelling and widespread agricultural submergence."
    },
    "September 2024 Deep Depression Flood Spell": {
        "district": "JAJPUR",
        "block": "Bari",
        "start_date": "2024-09-05",
        "end_date": "2024-09-20",
        "description": "Intense late-monsoon cyclonic circulation resulting in Baitarani river gauge threshold crossings."
    }
}


def render_historical_timeline_component(df: pd.DataFrame, pred_tool: ChatbotPredictionTool):
    """Render interactive multi-view historical timeline, event replay, and observation comparisons."""
    st.markdown("### ⏱️ Historical Flood Timeline & Event Intelligence")
    st.markdown(
        "Explore 24 years of empirical rainfall, observed flood events (SRC ground-truth), "
        "and model early warning probability dynamics on synchronized time-series."
    )
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "📈 Interactive Location Timeline",
        "🔄 Historical Flood Event Replay",
        "📋 Prediction vs Actual Historical Evaluation"
    ])
    
    # -------------------------------------------------------------
    # SUB-TAB 1: Interactive Location Timeline
    # -------------------------------------------------------------
    with sub_tab1:
        st.markdown("#### Multi-Year Station Precipitation & Inundation Trajectory")
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            districts = sorted(df["District"].unique())
            sel_dist = st.selectbox("Select District (Timeline):", districts, index=districts.index("CUTTACK") if "CUTTACK" in districts else 0)
        with c2:
            blocks = sorted(df[df["District"] == sel_dist]["Block/Station"].unique())
            sel_block = st.selectbox("Select Block / Station (Timeline):", blocks)
        with c3:
            years = sorted(df["Year"].unique(), reverse=True)
            sel_year = st.selectbox("Select Year:", years, index=years.index(2024) if 2024 in years else 0)
            
        # Filter data for selected station and year
        station_df = df[
            (df["District"] == sel_dist) & 
            (df["Block/Station"] == sel_block) & 
            (df["Year"] == sel_year)
        ].sort_values(by="Date").copy()
        
        if station_df.empty:
            st.warning("No records found for the selected station and year.")
        else:
            # Create dual-axis timeline
            fig = go.Figure()
            
            # Daily Rainfall Bar
            fig.add_trace(go.Bar(
                x=station_df["Date"],
                y=station_df["Rainfall (mm)"],
                name="Daily Rainfall (mm)",
                marker_color="#3182ce",
                opacity=0.7,
                yaxis="y1"
            ))
            
            # 7-Day Antecedent Rainfall Line
            if "Rainfall_Prev_7d_Sum" in station_df.columns:
                fig.add_trace(go.Scatter(
                    x=station_df["Date"],
                    y=station_df["Rainfall_Prev_7d_Sum"],
                    name="Antecedent 7-Day Rainfall (mm)",
                    line=dict(color="#805ad5", width=2, dash="dot"),
                    yaxis="y1"
                ))
                
            # Actual Observed Flood Event Markers
            flood_days = station_df[station_df["Flood_Occurred"] == 1]
            if not flood_days.empty:
                fig.add_trace(go.Scatter(
                    x=flood_days["Date"],
                    y=[station_df["Rainfall (mm)"].max() * 0.95] * len(flood_days),
                    mode="markers+text",
                    name="Actual Observed Flood (SRC)",
                    marker=dict(symbol="triangle-up", size=14, color="#e53e3e", line=dict(width=1, color="black")),
                    text=["FLOOD"] * len(flood_days),
                    textposition="top center",
                    yaxis="y1"
                ))
                
            fig.update_layout(
                title=f"<b>Precipitation & Inundation Trajectory: {sel_block} ({sel_dist}) - Year {sel_year}</b>",
                xaxis=dict(title="Date", showgrid=True),
                yaxis=dict(title="Precipitation / Accumulation (mm)", side="left", showgrid=True),
                legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
                hovermode="x unified",
                height=450,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics for the year
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Annual Rainfall", f"{station_df['Rainfall (mm)'].sum():,.1f} mm")
            m2.metric("Peak 1-Day Downpour", f"{station_df['Rainfall (mm)'].max():.1f} mm")
            m3.metric("Rainy Days (>=2.5mm)", int((station_df['Rainfall (mm)'] >= 2.5).sum()))
            m4.metric("Observed Flood Days", int(station_df['Flood_Occurred'].sum()))
            
    # -------------------------------------------------------------
    # SUB-TAB 2: Historical Flood Event Replay
    # -------------------------------------------------------------
    with sub_tab2:
        st.markdown("#### Chronological Step-by-Step Flood Spell Replay")
        st.markdown("Replay notable historical severe flood spells to observe antecedent rainfall buildup, model probabilities, and actual flood onset.")
        
        event_choice = st.selectbox(
            "Select Notable Flood Event:",
            list(HISTORICAL_NOTABLE_EVENTS.keys())
        )
        
        event_info = HISTORICAL_NOTABLE_EVENTS[event_choice]
        st.info(f"**Description**: {event_info['description']} | **District**: `{event_info['district']}` | **Station**: `{event_info['block']}` | **Period**: `{event_info['start_date']}` to `{event_info['end_date']}`")
        
        event_df = df[
            (df["District"] == event_info["district"]) &
            (df["Block/Station"] == event_info["block"]) &
            (df["Date"] >= event_info["start_date"]) &
            (df["Date"] <= event_info["end_date"])
        ].sort_values(by="Date").reset_index(drop=True)
        
        if event_df.empty:
            st.warning("Event data not found in current dataset partition.")
        else:
            # Generate probabilities for event days
            event_records = []
            for _, row in event_df.iterrows():
                try:
                    pred_res = pred_tool.predict_for_location_and_date(
                        district=row["District"],
                        block=row["Block/Station"],
                        date_str=row["Date"]
                    )
                    prob = pred_res["flood_probability"]
                    risk = pred_res["risk_level"]
                except Exception:
                    prob = 0.5
                    risk = "Moderate"
                    
                event_records.append({
                    "Date": row["Date"],
                    "Rainfall (mm)": row["Rainfall (mm)"],
                    "7-Day Antecedent (mm)": row.get("Rainfall_Prev_7d_Sum", 0.0),
                    "Flood Probability": prob,
                    "Predicted Risk": risk,
                    "Actual Flood": "YES (Flood Observed)" if row["Flood_Occurred"] == 1 else "No"
                })
                
            e_df = pd.DataFrame(event_records)
            
            # Dual Axis Chart for Event
            fig_e = go.Figure()
            fig_e.add_trace(go.Bar(
                x=e_df["Date"],
                y=e_df["Rainfall (mm)"],
                name="Daily Rainfall (mm)",
                marker_color="#3182ce",
                yaxis="y1"
            ))
            fig_e.add_trace(go.Scatter(
                x=e_df["Date"],
                y=e_df["Flood Probability"] * 100,
                name="Model Early Warning Probability (%)",
                line=dict(color="#e53e3e", width=3),
                mode="lines+markers",
                yaxis="y2"
            ))
            
            # Add threshold line at 70% (High Risk)
            fig_e.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="High Risk Threshold (70%)", yref="y2")
            
            fig_e.update_layout(
                title=f"<b>Event Replay: {event_choice}</b>",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Daily Precipitation (mm)", side="left"),
                yaxis2=dict(title="Estimated Flood Probability (%)", side="right", overlaying="y", range=[0, 105]),
                legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
                height=420,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_e, use_container_width=True)
            
            st.dataframe(e_df, use_container_width=True, hide_index=True)
            
    # -------------------------------------------------------------
    # SUB-TAB 3: Prediction vs Actual Historical Evaluation
    # -------------------------------------------------------------
    with sub_tab3:
        st.markdown("#### Historical Prediction vs Observed Ground-Truth Verification")
        st.markdown(
            "Direct comparison between model forecasted risk category and authoritative SRC flood observations "
            "for historical validation and test dates."
        )
        
        pv1, pv2, pv3 = st.columns(3)
        with pv1:
            p_dist = st.selectbox("Filter District:", ["All"] + districts)
        with pv2:
            p_year = st.selectbox("Filter Year:", [2024, 2023, 2022, 2021, 2020, 2019])
        with pv3:
            p_only_floods = st.checkbox("Show Only Dates with Flood Activity / Alerts", value=True)
            
        sample_query = df[df["Year"] == p_year]
        if p_dist != "All":
            sample_query = sample_query[sample_query["District"] == p_dist]
            
        if p_only_floods:
            sample_query = sample_query[(sample_query["Flood_Occurred"] == 1) | (sample_query["Rainfall (mm)"] >= 50.0)]
            
        sample_query = sample_query.head(100).sort_values(by="Date", ascending=False)
        
        if sample_query.empty:
            st.info("No matching records found for the filter criteria.")
        else:
            table_rows = []
            for _, row in sample_query.iterrows():
                # Get prediction
                try:
                    res = pred_tool.predict_for_location_and_date(
                        district=row["District"],
                        block=row["Block/Station"],
                        date_str=row["Date"]
                    )
                    prob_val = res["flood_probability"]
                    risk_val = res["risk_level"]
                    pred_target = res["predicted_flood_next_day"]
                except Exception:
                    prob_val = 0.0
                    risk_val = "Low"
                    pred_target = 0
                    
                actual_next = int(row.get("Flood_Next_Day", 0)) if not pd.isna(row.get("Flood_Next_Day")) else int(row["Flood_Occurred"])
                
                # Match Status
                if pred_target == actual_next:
                    match_status = "✅ Correct (TP/TN)" if actual_next == 1 else "✅ Correct (TN)"
                else:
                    match_status = "⚠️ False Alarm (FP)" if pred_target == 1 else "❌ Missed Event (FN)"
                    
                table_rows.append({
                    "Date": row["Date"],
                    "District": row["District"],
                    "Block": row["Block/Station"],
                    "Rainfall (mm)": round(row["Rainfall (mm)"], 1),
                    "Predicted Probability": f"{prob_val*100:.1f}%",
                    "Predicted Risk": risk_val,
                    "Actual Observed Next Day": "FLOOD (1)" if actual_next == 1 else "NO FLOOD (0)",
                    "Verification Status": match_status
                })
                
            res_comp_df = pd.DataFrame(table_rows)
            st.dataframe(res_comp_df, use_container_width=True, hide_index=True)
