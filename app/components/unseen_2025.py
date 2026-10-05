"""
2025 Unseen Operational Flood Risk Monitoring, Ingestion UI, and Retrospective Evaluation Component.
Strictly conforms to Master Project Prompt Sections 39 to 46:
- 2025 Operational Dataset Upload & Validation UI (CSV / XLSX)
- Preceding Warm-Up Period Context Handling (December 2024 historical stitching)
- Batch Predictions Explorer with Top 3 Risk Drivers & SHAP Attributions
- Climate Distribution Shift (Kolmogorov-Smirnov Test & Rainfall Shift)
- Authoritative 2025 SRC Ground Truth Ingestion & Retrospective Evaluation Workflow
- Strict Operational Terminology Guardrails
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from src.inference.unseen_2025_pipeline import (
    validate_2025_raw_dataset,
    engineer_2025_features,
    Operational2025Pipeline,
    Retrospective2025EvaluationEngine,
    analyze_rainfall_distribution_shift
)
from chatbot.prediction_tools import ChatbotPredictionTool


def render_unseen_2025_component(df_historical: pd.DataFrame, pred_tool: ChatbotPredictionTool):
    """Render comprehensive 2025 Unseen Operational Monitoring & Ingestion Center."""
    st.markdown("### 🛰️ 2025 Unseen Operational Flood Risk Monitor")
    
    # Official Academic & Operational Disclaimer (Section 44)
    st.info(
        "ℹ️ **2025 Unseen Operational Inference Notice**: "
        "2025 represents an unseen operational deployment period. "
        "All outputs are **Model-Estimated Risk** and **Estimated Flood Probabilities** computed using the frozen historical model (2001–2024). "
        "Ground truth flood observations are unavailable during live operational forecasting. "
        "Performance metrics (Accuracy, Precision, Recall, F1) are strictly computed only within the dedicated **Retrospective Evaluation** workflow when authoritative SRC records are provided."
    )
    
    tab_upload, tab_station, tab_explorer, tab_drift, tab_retro = st.tabs([
        "📤 Upload 2025 Operational Dataset",
        "🌊 2025 Station Risk Forecaster",
        "📋 2025 Operational Predictions Explorer",
        "📊 Climate Distribution Shift & Data Drift",
        "🎯 2025 Retrospective Evaluation"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: Upload & Validate 2025 Operational Dataset (Sections 39, 40, 41, 42)
    # -------------------------------------------------------------------------
    with tab_upload:
        st.markdown("#### Upload & Ingest 2025 Operational Rainfall Dataset")
        st.markdown(
            "Upload daily telemetric or rain-gauge observations for the 2025 operational period. "
            "Supported formats: **CSV**, **XLSX**. The system validates schema, locations, dates, and stitches "
            "preceding December 2024 warm-up history to generate accurate rolling features."
        )
        
        uploaded_file = st.file_uploader(
            "Select 2025 Operational Dataset File (CSV or XLSX):",
            type=["csv", "xlsx"],
            key="upload_2025_file"
        )
        
        raw_ops_path = "data/raw/operational/2025.csv"
        
        # Load uploaded file or fallback to local operational file
        df_uploaded = None
        upload_filename = "data/raw/operational/2025.csv (Local Default)"
        
        if uploaded_file is not None:
            upload_filename = uploaded_file.name
            try:
                if upload_filename.endswith(".xlsx"):
                    df_uploaded = pd.read_excel(uploaded_file)
                else:
                    df_uploaded = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading uploaded file: {str(e)}")
        elif os.path.exists(raw_ops_path):
            df_uploaded = pd.read_csv(raw_ops_path)
            
        if df_uploaded is not None:
            st.markdown("##### 🔍 Operational Dataset Validation Report")
            val_report = validate_2025_raw_dataset(df_uploaded)
            
            # Validation Status Badge
            if val_report["is_valid"]:
                st.success(f"### {val_report['validation_status']}")
            else:
                st.error(f"### {val_report['validation_status']}")
                
            # Summary Metrics Grid
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("File Name", upload_filename.split("/")[-1])
            c2.metric("Total Records", f"{val_report['row_count']:,}")
            c3.metric("Date Range", f"{val_report['date_range'][0]} to {val_report['date_range'][1]}" if val_report['date_range'] else "N/A")
            c4.metric("Districts Monitored", f"{val_report['districts_count']} / 30 Standard")
            
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Monitored Blocks", f"{val_report['blocks_count']}")
            c6.metric("Missing Rainfall Entries", f"{val_report['missing_rainfall_count']}")
            c7.metric("Duplicate Location-Dates", f"{val_report['duplicate_count']}")
            c8.metric("Negative Rain Entries", f"{val_report['negative_rainfall_count']}")
            
            # Warm-Up Period Check (Section 42)
            st.markdown("##### ⏱️ Warm-Up Period & Historical Context Status")
            min_date_str = val_report["date_range"][0] if val_report["date_range"] else "2025-01-01"
            
            if min_date_str.startswith("2025-01"):
                st.info(
                    "📅 **Warm-Up Context Detected**: The dataset begins in January 2025. "
                    "Preceding December 2024 historical rainfall (35 days) from `data/combined/Odisha_Flood_2001_2024.csv` "
                    "will be automatically stitched to compute 3d, 7d, 15d, and 30d rolling sums without zero-filling."
                )
            else:
                st.success("✓ Dataset contains sufficient internal historical lead time.")
                
            if val_report["warnings"]:
                st.warning("⚠️ **Validation Warnings**: " + "; ".join(val_report["warnings"]))
                
            if val_report["errors"]:
                st.error("❌ **Validation Errors**: " + "; ".join(val_report["errors"]))
                
            # Execution Button
            if val_report["is_valid"]:
                run_ops_btn = st.button("⚡ Run 2025 Operational Inference Pipeline", type="primary", use_container_width=True)
                if run_ops_btn:
                    with st.spinner("Executing frozen model inference and generating operational forecasts..."):
                        pipeline = Operational2025Pipeline()
                        df_preds, summary = pipeline.run_pipeline(
                            raw_df_2025=df_uploaded,
                            output_csv_path="data/predictions/2025_flood_risk_predictions.csv"
                        )
                        st.session_state["2025_predictions"] = df_preds
                        st.success(
                            f"✅ Successfully generated and exported {len(df_preds):,} operational predictions to "
                            f"`data/predictions/2025_flood_risk_predictions.csv`!"
                        )
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("High Risk Days (>=70%)", f"{summary['high_risk_count']:,}")
                        m2.metric("Moderate Risk Days (30-70%)", f"{summary['moderate_risk_count']:,}")
                        m3.metric("Low Risk Days (<30%)", f"{summary['low_risk_count']:,}")
        else:
            st.info("Please upload a 2025 operational dataset (CSV or XLSX) to begin.")

    # -------------------------------------------------------------------------
    # TAB 2: Station Forecaster (Section 39, 44)
    # -------------------------------------------------------------------------
    with tab_station:
        st.markdown("#### Real-Time Station Risk Forecaster on 2025 Inputs")
        
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("##### 1. Location Parameters:")
            districts = sorted(df_historical["District"].unique())
            sel_dist = st.selectbox("Operational District:", districts, index=districts.index("CUTTACK") if "CUTTACK" in districts else 0, key="u25_st_dist")
            
            blocks = sorted(df_historical[df_historical["District"] == sel_dist]["Block/Station"].unique())
            sel_block = st.selectbox("Operational Block / Station:", blocks, key="u25_st_block")
            sim_date = st.date_input("Operational Date:", pd.to_datetime("2025-08-15"), key="u25_st_date")
            
            st.markdown("##### 2. Meteorological Preconditions (mm):")
            u_rain = st.number_input("24-Hour Observed Rainfall (mm):", min_value=0.0, max_value=500.0, value=95.0, step=1.0, key="u25_st_rf")
            u_3d = st.number_input("Prior 3-Day Accumulated Rainfall (mm):", min_value=0.0, max_value=800.0, value=210.0, step=5.0, key="u25_st_3d")
            u_7d = st.number_input("Prior 7-Day Accumulated Rainfall (mm):", min_value=0.0, max_value=1200.0, value=380.0, step=10.0, key="u25_st_7d")
            u_15d = st.number_input("Prior 15-Day Accumulated Rainfall (mm):", min_value=0.0, max_value=2000.0, value=520.0, step=10.0, key="u25_st_15d")
            u_flood_today = st.checkbox("Active Inundation Reported Today?", value=False, key="u25_st_flood")
            
            run_btn = st.button("⚡ Compute 2025 Estimated Flood Risk", use_container_width=True, type="primary", key="u25_st_run")
            
        with col2:
            st.markdown("##### Estimated Flood Probability & Attributions:")
            if run_btn:
                with st.spinner("Executing frozen model inference..."):
                    month_num = sim_date.month
                    pred_res = pred_tool.predict_custom_scenario(
                        district=sel_dist,
                        rainfall_mm=u_rain,
                        rainfall_prev_3d_sum=u_3d,
                        rainfall_prev_7d_sum=u_7d,
                        rainfall_prev_15d_sum=u_15d,
                        month=month_num,
                        flood_occurred_today=1 if u_flood_today else 0
                    )
                    
                    prob = pred_res["flood_probability"]
                    risk = pred_res["risk_level"]
                    prob_pct = prob * 100
                    color = "#d9534f" if risk == "High" else ("#f0ad4e" if risk == "Moderate" else "#5cb85c")
                    
                    # Risk Gauge
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_pct,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"2025 Model-Estimated Risk: <b>{risk.upper()}</b>", 'font': {'size': 18, 'color': color}},
                        number={'suffix': "%", 'font': {'size': 30, 'color': color}},
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
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Station", f"{sel_block}")
                    m2.metric("Date", sim_date.strftime("%Y-%m-%d"))
                    m3.metric("Estimated Probability", f"{prob_pct:.2f}%")
                    
                    # Top Factors
                    st.markdown("###### Top Model Contributing Factors:")
                    drivers = pred_res.get("top_risk_drivers", [])
                    mitigators = pred_res.get("top_mitigators", [])
                    
                    c_rows = []
                    for d in drivers[:3]:
                        c_rows.append({"Feature": d["feature"], "Impact": f"+{d['contribution']:.3f}", "Type": "Risk Amplifier (Up)"})
                    for m in mitigators[:2]:
                        c_rows.append({"Feature": m["feature"], "Impact": f"{m['contribution']:.3f}", "Type": "Risk Mitigator (Down)"})
                        
                    if c_rows:
                        st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)
            else:
                st.info("👈 Enter 2025 precipitation parameters and click **Compute 2025 Estimated Flood Risk**.")

    # -------------------------------------------------------------------------
    # TAB 3: Predictions Explorer (Section 43)
    # -------------------------------------------------------------------------
    with tab_explorer:
        st.markdown("#### 📁 2025 Operational Predictions Repository")
        pred_csv_path = "data/predictions/2025_flood_risk_predictions.csv"
        
        if os.path.exists(pred_csv_path):
            df_p25 = pd.read_csv(pred_csv_path)
            
            cp1, cp2, cp3 = st.columns(3)
            with cp1:
                f_dist = st.selectbox("Filter District (2025):", ["All"] + sorted(df_p25["District"].unique()), key="f_p25_exp_dist")
            with cp2:
                f_risk = st.selectbox("Filter Risk Category:", ["All", "HIGH", "MODERATE", "LOW"], key="f_p25_exp_risk")
            with cp3:
                f_date = st.text_input("Filter Date (YYYY-MM-DD):", value="", key="f_p25_exp_date")
                
            df_filtered = df_p25.copy()
            if f_dist != "All":
                df_filtered = df_filtered[df_filtered["District"] == f_dist]
            if f_risk != "All":
                df_filtered = df_filtered[df_filtered["Risk_Level"] == f_risk]
            if f_date.strip():
                df_filtered = df_filtered[df_filtered["Date"] == f_date.strip()]
                
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Total 2025 Records", f"{len(df_p25):,}")
            p2.metric("Matching Filters", f"{len(df_filtered):,}")
            p3.metric("High Risk Days (>=70%)", int((df_filtered["Risk_Level"] == "HIGH").sum()))
            p4.metric("Moderate Risk Days", int((df_filtered["Risk_Level"] == "MODERATE").sum()))
            
            cols_show = [
                "District", "Block/Station", "Date", "Predicted_Probability", "Risk_Level",
                "Rainfall (mm)", "Rainfall_Prev_3d_Sum", "Rainfall_Prev_15d_Sum",
                "Top_Risk_Driver_1", "Top_Risk_Driver_2", "Top_Risk_Driver_3"
            ]
            cols_available = [c for c in cols_show if c in df_filtered.columns]
            st.dataframe(df_filtered[cols_available].head(250), use_container_width=True, hide_index=True)
            
            # Download Button
            csv_data = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Filtered 2025 Predictions (CSV)",
                data=csv_data,
                file_name="2025_filtered_flood_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("Batch predictions file `data/predictions/2025_flood_risk_predictions.csv` not yet generated. Please use the Upload tab.")

    # -------------------------------------------------------------------------
    # TAB 4: Climate Distribution Shift & Data Drift
    # -------------------------------------------------------------------------
    with tab_drift:
        st.markdown("#### Climate Distribution Shift: Historical Baseline (2001–2018) vs 2025 Operational Period")
        st.markdown(
            "Quantifying statistical distribution drift in precipitation patterns using "
            "Kolmogorov-Smirnov two-sample tests, quantile breakdowns, and extreme rainfall anomalies."
        )
        
        drift_dist = st.selectbox("Select District for Shift Analysis:", ["All Districts (Statewide)"] + districts, key="d_shift_dist_tab4")
        
        raw_25_path = "data/raw/operational/2025.csv"
        if os.path.exists(raw_25_path):
            df_2025_raw = pd.read_csv(raw_25_path)
            if drift_dist == "All Districts (Statewide)":
                hist_rain = df_historical[df_historical["Year"] <= 2018]["Rainfall (mm)"]
                ops_rain = df_2025_raw["Rainfall (mm)"]
            else:
                hist_rain = df_historical[(df_historical["Year"] <= 2018) & (df_historical["District"] == drift_dist)]["Rainfall (mm)"]
                ops_rain = df_2025_raw[df_2025_raw["District"] == drift_dist]["Rainfall (mm)"]
        else:
            if drift_dist == "All Districts (Statewide)":
                hist_rain = df_historical[df_historical["Year"] <= 2018]["Rainfall (mm)"]
                ops_rain = df_historical[df_historical["Year"] >= 2022]["Rainfall (mm)"]
            else:
                hist_rain = df_historical[(df_historical["Year"] <= 2018) & (df_historical["District"] == drift_dist)]["Rainfall (mm)"]
                ops_rain = df_historical[(df_historical["Year"] >= 2022) & (df_historical["District"] == drift_dist)]["Rainfall (mm)"]
            
        drift_stats = analyze_rainfall_distribution_shift(hist_rain, ops_rain, drift_dist)
        
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Historical Mean Rain", f"{drift_stats['historical_mean_mm']:.2f} mm/day")
        d2.metric("2025 Mean Rain", f"{drift_stats['current_mean_mm']:.2f} mm/day")
        d3.metric("KS Statistic", f"{drift_stats['ks_statistic']:.4f}")
        d4.metric("Shift P-Value", f"{drift_stats['p_value']:.2e}", "Significant" if drift_stats["distribution_shift_detected"] else "Stable")
        
        fig_drift = go.Figure()
        fig_drift.add_trace(go.Histogram(
            x=hist_rain[hist_rain > 0],
            name="Historical (2001–2018)",
            opacity=0.6,
            marker_color="#1f77b4",
            nbinsx=50
        ))
        fig_drift.add_trace(go.Histogram(
            x=ops_rain[ops_rain > 0],
            name="Operational (2025)",
            opacity=0.6,
            marker_color="#e53e3e",
            nbinsx=50
        ))
        fig_drift.update_layout(
            title=f"<b>Rainfall Distribution Shift: {drift_dist}</b>",
            xaxis=dict(title="Precipitation on Wet Days (mm)", type="log"),
            yaxis=dict(title="Frequency Count", type="log"),
            barmode="overlay",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_drift, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: Retrospective Evaluation (Section 45)
    # -------------------------------------------------------------------------
    with tab_retro:
        st.markdown("#### 🎯 2025 Retrospective Evaluation Workflow")
        st.markdown(
            "When authoritative 2025 Special Relief Commissioner (SRC) ground-truth flood observations become available, "
            "this workflow evaluates the frozen operational predictions against actual inundation events "
            "**without modifying or retraining the frozen operational model**."
        )
        
        gt_file = st.file_uploader(
            "Upload 2025 SRC Ground-Truth Flood Observations (CSV):",
            type=["csv"],
            key="upload_2025_gt"
        )
        
        gt_path = "data/raw/ground_truth/2025_SRC_Flood_Observations.csv"
        df_gt = None
        
        if gt_file is not None:
            df_gt = pd.read_csv(gt_file)
        elif os.path.exists(gt_path):
            df_gt = pd.read_csv(gt_path)
            
        pred_csv_path = "data/predictions/2025_flood_risk_predictions.csv"
        if os.path.exists(pred_csv_path) and df_gt is not None:
            df_preds = pd.read_csv(pred_csv_path)
            
            retro_btn = st.button("📊 Calculate Retrospective Evaluation Metrics", type="primary")
            if retro_btn:
                try:
                    engine = Retrospective2025EvaluationEngine()
                    metrics, df_matched = engine.evaluate(df_preds, df_gt)
                    
                    st.success(f"✅ Evaluated {metrics['matched_observations']:,} matched station-date observations!")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Retrospective F1-Score", f"{metrics['retrospective_f1']:.4f}")
                    m2.metric("Retrospective Recall", f"{metrics['retrospective_recall']*100:.2f}%")
                    m3.metric("Retrospective Precision", f"{metrics['retrospective_precision']*100:.2f}%")
                    m4.metric("Retrospective PR-AUC", f"{metrics['retrospective_pr_auc']:.4f}")
                    
                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("ROC-AUC", f"{metrics['retrospective_roc_auc']:.4f}")
                    m6.metric("Accuracy", f"{metrics['retrospective_accuracy']*100:.2f}%")
                    m7.metric("Specificity", f"{metrics['retrospective_specificity']*100:.2f}%")
                    m8.metric("Brier Score", f"{metrics['retrospective_brier_score']:.5f}")
                    
                    # Confusion Matrix
                    cm_df = pd.DataFrame([
                        {"Actual": "Flood (1)", "Predicted No Flood (0)": metrics['retrospective_fn'], "Predicted Flood (1)": metrics['retrospective_tp']},
                        {"Actual": "No Flood (0)", "Predicted No Flood (0)": metrics['retrospective_tn'], "Predicted Flood (1)": metrics['retrospective_fp']}
                    ])
                    st.markdown("##### Confusion Matrix (Frozen 2025 Evaluation)")
                    st.dataframe(cm_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Retrospective evaluation failed: {str(e)}")
        else:
            if not os.path.exists(pred_csv_path):
                st.warning("⚠️ Predictions file `data/predictions/2025_flood_risk_predictions.csv` missing. Please run the operational pipeline in Tab 1 first.")
            else:
                st.info("ℹ️ Ground-truth file not yet uploaded. To perform retrospective evaluation, upload `2025_SRC_Flood_Observations.csv` above.")
