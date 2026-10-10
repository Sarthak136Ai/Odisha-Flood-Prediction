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
| **26** | Automated Data Quality Analysis Engine | **Completed** | 15-dimension raw vs processed audit, HTML/CSV reports, 5 plots, and unit test suite |
| **27** | Walk-Forward Temporal Validation Engine | **Completed** | 5-fold expanding window cross-validation (2017–2021), strict preprocessor isolation, reports & plots |
| **28** | Multi-Strategy Threshold Selection & Audit Engine | **Completed** | Validation-only (2019–2021) threshold tuning, 7 strategies (F1, F2, Target Recall >=80%/90%, Youden's J, Cost-Sensitive), zero-leakage proof, model metadata JSON & reports |
| **29** | Class Imbalance Analysis & Mitigation Engine | **Completed** | Full 24-year target distribution analysis (2.638% flood rate, 36.91:1 imbalance ratio), split & annual tables, SMOTE rejection rationale, confusion matrices, PR-AUC analysis, and plots |
| **30** | Comprehensive Confusion Matrix Evaluation Engine | **Completed** | Full test set (2022–2024) confusion matrix decomposition (TN, FP, FN, TP, Acc, Prec, Rec, F1, F2, FNR, FPR), dynamic model discovery, individual/grid PNG & SVG plots, CSV/JSON artifacts, and DRR interpretation guide |
| **31** | Event-Based Hydrological Evaluation Engine | **Completed** | Spatiotemporal event segmentation, temporal matching, lead time calculation, 6 real historical disaster episodes (August 2022 Mahanadi, Sept 2024 Subarnarekha, May 2021 Yaas, Aug 2020 Bhadrak, Oct 2018 Titli, Sept 2011 Mahanadi), 1,221 test event benchmarks across 5 models, high-res timeline plots, and comprehensive technical report `docs/flood_event_evaluation.md` |

---

## 🌊 Test Set (2022–2024) Macro Event-Level Performance Across Models (1,221 Actual Events)

| Model Architecture | Optimized Threshold ($T^*$) | Detected Hits | Missed Events | Event Hit Rate (Recall) | Event Precision | Event F1 Score | Mean Lead Time | Precautionary False Alarms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **0.8073** | **780** | **441** | **63.88%** | **19.36%** | **0.2971** | **1.22 days** | 3,249 |
| **Logistic Regression** | 0.8503 | 706 | 515 | 57.82% | 19.74% | 0.2943 | 0.65 days | **2,871** |
| **Random Forest** | 0.7607 | 708 | 513 | 57.99% | 16.67% | 0.2589 | **1.33 days** | 3,540 |
| **Decision Tree** | 0.8265 | 693 | 528 | 56.76% | 17.97% | 0.2730 | 0.54 days | 3,163 |
| **ANN (MLP)** | 0.1075 | 821 | 400 | 67.24% | 13.54% | 0.2254 | 1.11 days | 5,242 |

---

## 🧪 Automated Test Verification

All **89 test cases** in `tests/` pass with 100% success rate:
- `tests/test_event_evaluation.py`: 9 Spatiotemporal event segmentation, lead time, hit/miss, false alarm & timeline plot tests
- `tests/test_confusion_matrix.py`: 4 Confusion matrix decomposition, metric invariance, zero-division & plot generation tests
- `tests/test_class_imbalance.py`: 5 Target distribution, split consistency, F2 behavior & empirical purity tests
- `tests/test_threshold_selection.py`: 7 Zero-leakage, multi-strategy, F2/recall-oriented & schema tests
- `tests/test_temporal_validation.py`: 6 Temporal walk-forward causal order, isolation & metric tests
- `tests/test_data_quality.py`: 7 Data quality validation rules & report integrity tests
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
