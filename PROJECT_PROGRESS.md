# Odisha Flood Intelligence & Early Warning System
## Master Project Progress & Architecture Tracker

**Last Updated:** 2026-10-05 14:30:00 IST  
**Current Phase:** Production Deployment & GitHub Sync  
**Overall Status:** **100% Completed & Verified** (51/51 Automated Unit & Integration Tests Passing)

---

## 🌊 Executive Summary & Capabilities

The **Odisha Flood Intelligence & Early Warning System** is an end-to-end, spatio-temporal machine learning platform and Flask disaster intelligence web application. It processes 24 years (2001–2024, 2.75M+ records) of daily rainfall telemetry and Special Relief Commissioner (SRC) flood inundation reports across all 30 standard districts and 354 block stations in Odisha, India.

### Core Pillars:
1. **Historical Benchmark Foundation (2001–2024)**:
   - 2,752,252 historical daily observations across 30 standard districts.
   - 20-feature antecedent rolling lag, accumulation, and cyclical feature matrix with zero data leakage.
   - Rigorous benchmark across 5 ML architectures with class-imbalance optimization.
   - Production Model: **Calibrated Balanced Logistic Regression** (Test ROC-AUC: 0.9693, PR-AUC: 0.8306, Optimal Threshold: 0.8345).

2. **2025 Unseen Operational Ingestion Pipeline**:
   - Dedicated ingestion for unlabelled telemetry (`data/raw/operational/2025.csv` — 114,610 daily records).
   - Automated schema validation, district name standardization, and Dec 2024 warm-up stitching.
   - Climate Distribution Shift analysis (Kolmogorov-Smirnov 2-sample test).
   - Independent Retrospective Evaluation Engine for post-season SRC ground truth verification without model retraining.

3. **Disaster Command Center Web Application**:
   - 12 interactive responsive modules built with Flask, Jinja2, Leaflet.js, and Chart.js.
   - Geospatial risk mapping, district & block drilldowns with local SHAP attributions.
   - 24-year historical timeline replay presets (August 2019, August 2020, August 2022, September 2024).
   - Controlled What-If scenario simulator with rainfall sensitivity response curves.
   - Statistical rainfall downscaling (`HistGradientBoostingRegressor`).
   - Domain-grounded AI flood assistant with anti-hallucination routing.

---

## 📋 Comprehensive Task & Implementation Matrix

| Task ID | Component / Objective | Status | Implementation Details |
| :---: | :--- | :---: | :--- |
| **1** | Inspect project structure, datasets, models, metrics & XAI | **Completed** | Standardized folder hierarchy and verified legacy models & pipelines |
| **2** | Workspace Audit & Data Sanitization | **Completed** | 24-year historical rainfall dataset audit (2,752,252 records, 30 districts) |
| **3** | Feature Engineering & Zero-Leakage Guarantee | **Completed** | Antecedent shift lags (1d, 2d, 3d, 7d), rolling sums/max (3d, 7d, 15d, 30d), anomaly engine |
| **4** | Model Training & Benchmark Suite | **Completed** | Trained & evaluated Logistic Regression, Decision Tree, Random Forest, XGBoost, ANN |
| **5** | Probability Calibration & Reliability Diagrams | **Completed** | Brier scores, reliability diagrams, and risk tier categorization (`LOW`, `MODERATE`, `HIGH`) |
| **6** | Statistical Rainfall Downscaling | **Completed** | High-resolution `HistGradientBoostingRegressor` spatial grid refinement |
| **7** | Explainable AI (SHAP Global & Local) | **Completed** | 20-feature SHAP importance rankings, summary plots, and localized risk drivers |
| **8** | Grounded AI Chatbot Engine | **Completed** | Zero-hallucination NLP query router, prediction tools, and flood domain knowledge base |
| **9** | Flask Configuration & Architecture (`config.py`) | **Completed** | Path resolution, threshold constants, cache parameters, and upload directories |
| **10** | Application Core & REST Endpoints (`app.py`) | **Completed** | Startup model caching, REST APIs, JSON handlers, and error fallbacks |
| **11** | Master Layout & CSS System (`templates/`, `static/`) | **Completed** | Dark navy disaster intelligence aesthetic, glassmorphic cards, responsive navigation |
| **12** | Command Center Dashboard (`/` & `/dashboard`) | **Completed** | State-wide KPI summary counters, spatial vulnerability overview, live alerts |
| **13** | Interactive Odisha Risk Map (`/risk-map`) | **Completed** | Leaflet.js 30-district interactive map with risk markers & league table |
| **14** | District & Block Drilldown (`/drilldown`) | **Completed** | Cascading dropdowns (District &rarr; Block &rarr; Date) with real-time SHAP breakdowns |
| **15** | Historical Timeline & Replay (`/historical`) | **Completed** | 2001–2024 time series chart, event markers, and 4 major historical replay presets |
| **16** | Real-Time Forecaster (`/forecaster`) | **Completed** | Station telemetry form, auto-lag synthesis, probability gauge, risk classification |
| **17** | What-If Scenario Simulator (`/simulator`) | **Completed** | Baseline vs modified rainfall comparison, non-linear sensitivity curve (0–250 mm) |
| **18** | 2025 Operational Risk Monitor (`/unseen-2025`) | **Completed** | File upload, Dec 2024 warm-up stitching, KS drift test, retrospective verification |
| **19** | Model Benchmarks & Calibration UI (`/benchmarks`) | **Completed** | 5-model comparison table, ROC/PR curves, and reliability curves |
| **20** | Explainable AI UI (`/explainability`) | **Completed** | Global SHAP feature influence bar charts, beeswarm plots, and hydrological findings |
| **21** | Statistical Downscaling UI (`/downscaling`) | **Completed** | Observed vs downscaled comparative metrics & spatial distribution charts |
| **22** | Grounded AI Assistant UI (`/assistant`) | **Completed** | Interactive chat interface with quick-prompt chips and markdown rendering |
| **23** | System Architecture Page (`/architecture`) | **Completed** | Dual-pipeline visual flowcharts and complete feature schema data dictionary |
| **24** | Automated Unit & Integration Test Suite | **Completed** | 51 automated tests passing across 11 test modules with 100% pass rate |
| **25** | GitHub Sync & Version Control | **Completed** | Cleanly committed and pushed to `https://github.com/Sarthak136Ai/Odisha-Flood-Prediction` |

---

## 📊 Benchmark Summary (Test Split: 2022–2024, 343,758 observations)

| Model Architecture | Optimal Threshold | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Production)** | **0.8345** | **0.9693** | **0.8306** | **0.8748** | **87.44%** | **87.52%** | **0.0111** |
| **Random Forest (Balanced Subsample)** | 0.2745 | 0.9637 | 0.8176 | 0.8452 | 84.55% | 84.48% | 0.0107 |
| **XGBoost (scale_pos_weight)** | 0.3966 | 0.9082 | 0.6151 | 0.7033 | 61.92% | 81.39% | 0.0138 |
| **Decision Tree (Baseline)** | 0.9362 | 0.6137 | 0.5107 | 0.6858 | 54.66% | 92.02% | 0.0203 |
| **ANN (MLP)** | 0.0023 | 0.9281 | 0.5992 | 0.5863 | 44.59% | 85.57% | 0.0237 |

---

## 🧪 Automated Test Verification

All **51 test cases** in `tests/` pass with 100% success rate:
- `tests/test_routes.py`: 12 Flask page route and view tests
- `tests/test_prediction.py`: 9 Model inference, forecaster & simulator tests
- `tests/test_operational_2025.py`: 6 Operational ingestion & pipeline tests
- `tests/test_unseen_2025.py`: 2 Unseen 2025 telemetry tests
- `tests/test_data.py`: 6 Data loading, schema validation & combination tests
- `tests/test_features.py`: 3 Feature engineering & lag calculation tests
- `tests/test_rainfall_features.py`: 2 Anomaly & IMD flag tests
- `tests/test_model.py`: 2 Frozen model inference tests
- `tests/test_decision_tree.py`: Baseline Decision Tree tests
- `tests/test_calibration.py`: Probability calibration tests
- `tests/test_chatbot.py`: 5 Grounded chatbot tool tests
- `tests/test_dashboard.py`: Dashboard view tests

---

## 🚀 Execution Guide

1. **Start the Web Application**:
   ```bash
   python app.py
   ```
   Open browser at: `http://127.0.0.1:5000`

2. **Run Automated Test Suite**:
   ```bash
   pytest -v
   ```

3. **Repository URL**:
   `https://github.com/Sarthak136Ai/Odisha-Flood-Prediction`
