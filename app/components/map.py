"""
Geospatial mapping component for Streamlit dashboard supporting multiple basemaps,
Plotly Mapbox, and custom CARTO/Mapbox API key integration.
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
from typing import Dict, Any

from src.geospatial.spatial_risk import calculate_district_historical_risk, ODISHA_DISTRICT_COORDINATES


def render_odisha_map(df: pd.DataFrame):
    """Render interactive geospatial risk map of Odisha."""
    st.markdown("### 🗺️ Odisha Geospatial Flood Risk & Vulnerability Atlas")
    st.markdown(
        "Interactive spatial map displaying 24-year historical flood frequency, "
        "annual average rainfall, and spatial risk distribution across all 30 districts."
    )
    
    risk_summary = calculate_district_historical_risk(df)
    
    # Calculate Risk Categories
    q_high = risk_summary["total_flood_days"].quantile(0.66)
    q_med = risk_summary["total_flood_days"].quantile(0.33)
    
    def get_color(days):
        if days >= q_high:
            return "#e53e3e" # Red (High Risk)
        elif days >= q_med:
            return "#dd6b20" # Orange (Moderate Risk)
        else:
            return "#38a169" # Green (Low Risk)
            
    # Map controls
    ctrl_col1, ctrl_col2 = st.columns([2, 2])
    with ctrl_col1:
        map_engine = st.radio("Map Engine:", ["OpenStreetMap (Free / No Key)", "Esri Satellite / Topo (Free / No Key)", "Custom CARTO / Mapbox API Key", "Plotly Interactive Map"], horizontal=True)
    with ctrl_col2:
        carto_key = ""
        if map_engine == "Custom CARTO / Mapbox API Key":
            carto_key = st.text_input("Enter CARTO / Mapbox API Key:", type="password", placeholder="Paste your API key here (e.g. from carto.com or mapbox.com)")
            
    if map_engine == "Plotly Interactive Map":
        # Plotly Scatter Mapbox (100% Free, No Watermark)
        risk_summary["Risk_Category"] = risk_summary["total_flood_days"].apply(
            lambda d: "HIGH" if d >= q_high else ("MODERATE" if d >= q_med else "LOW")
        )
        fig_map = px.scatter_mapbox(
            risk_summary,
            lat="latitude",
            lon="longitude",
            size="total_flood_days",
            color="Risk_Category",
            color_discrete_map={"HIGH": "#e53e3e", "MODERATE": "#dd6b20", "LOW": "#38a169"},
            hover_name="District",
            hover_data={
                "headquarters": True,
                "total_flood_days": True,
                "flood_frequency_pct": ":.2f%",
                "avg_annual_rainfall": ":.1f mm",
                "latitude": False,
                "longitude": False,
                "Risk_Category": True
            },
            zoom=6.5,
            center={"lat": 20.5, "lon": 84.5},
            mapbox_style="open-street-map",
            title="<b>Odisha 30-District Flood Risk Map (Plotly Engine)</b>"
        )
        fig_map.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        
        col_map, col_table = st.columns([3, 2], gap="medium")
        with col_map:
            st.plotly_chart(fig_map, use_container_width=True)
        with col_table:
            st.markdown("#### 🏆 District Vulnerability League Table")
            st.dataframe(
                risk_summary[["District", "total_flood_days", "flood_frequency_pct", "avg_annual_rainfall", "max_single_day_rain"]].rename(
                    columns={
                        "total_flood_days": "Flood Days (24 Yrs)",
                        "flood_frequency_pct": "Flood Rate (%)",
                        "avg_annual_rainfall": "Avg Rain (mm/Yr)",
                        "max_single_day_rain": "Peak 1-Day (mm)"
                    }
                ),
                height=480,
                use_container_width=True,
                hide_index=True
            )
        return

    # Folium Map with explicit TileLayer
    m = folium.Map(
        location=[20.5, 84.5],
        zoom_start=7,
        tiles=None # Explicitly manage tiles to prevent Carto fallback
    )
    
    if map_engine == "Esri Satellite / Topo (Free / No Key)":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, USGS, NOAA",
            name="Esri World Topo Map",
            control=True
        ).add_to(m)
    elif map_engine == "Custom CARTO / Mapbox API Key" and carto_key:
        # User provided CARTO key
        folium.TileLayer(
            tiles=f"https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png?api_key={carto_key}",
            attr="&copy; CARTO &copy; OpenStreetMap",
            name="CARTO Voyager (Authenticated)",
            control=True,
            subdomains="abcd"
        ).add_to(m)
    else:
        # Standard OpenStreetMap direct HTTPS
        folium.TileLayer(
            tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
            name="OpenStreetMap",
            control=True
        ).add_to(m)
        
    for _, row in risk_summary.iterrows():
        color = get_color(row["total_flood_days"])
        radius = max(8, min(22, int(row["total_flood_days"] / 300) + 8))
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; width: 220px; line-height: 1.4;">
            <h4 style="margin: 0 0 6px 0; color: #1f77b4; font-size: 15px;"><b>{row['District']}</b></h4>
            <b>District HQ:</b> {row['headquarters']}<br>
            <b>24-Yr Flood Days:</b> <span style="color: {color}; font-weight: bold;">{row['total_flood_days']:,}</span><br>
            <b>Flood Frequency:</b> {row['flood_frequency_pct']:.2f}% of days<br>
            <b>Avg Annual Rain:</b> {row['avg_annual_rainfall']:,.1f} mm/yr<br>
            <b>Peak 1-Day Rain:</b> {row['max_single_day_rain']:.1f} mm
        </div>
        """
        
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color="#2d3748",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['District']} ({row['total_flood_days']:,} flood days)"
        ).add_to(m)
        
    col_map, col_table = st.columns([3, 2], gap="medium")
    
    with col_map:
        st_folium(m, width=650, height=520, key=f"odisha_folium_map_{map_engine}")
        
    with col_table:
        st.markdown("#### 🏆 District Vulnerability League Table")
        st.dataframe(
            risk_summary[["District", "total_flood_days", "flood_frequency_pct", "avg_annual_rainfall", "max_single_day_rain"]].rename(
                columns={
                    "total_flood_days": "Flood Days (24 Yrs)",
                    "flood_frequency_pct": "Flood Rate (%)",
                    "avg_annual_rainfall": "Avg Rain (mm/Yr)",
                    "max_single_day_rain": "Peak 1-Day (mm)"
                }
            ),
            height=480,
            use_container_width=True,
            hide_index=True
        )
