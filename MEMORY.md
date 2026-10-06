# Project Memory: Odisha Flood Intelligence & Early Warning System

## 1. System Architecture & Core Principles
- **Domain**: Spatio-temporal flood risk prediction and early warning across 30 standard districts and 354 blocks/stations in Odisha, India.
- **Data Scope**:
  - Historical Training/Benchmark: 2001–2024 (24 years, 2,752,252 daily records).
  - Unseen Operational Inference: 2025 (114,610 daily records, unobserved ground truth).
- **Target Definition**: `Flood_Next_Day` (binary, strictly next-day observation to eliminate same-day target leakage).
- **Feature Set (20 core features)**:
  - Temporal: `Year`, `Month_Number`, `Day`, `Day_of_Year`, `Day_of_Week`, `Month_sin`, `Month_cos`
  - Current & Lagged Rainfall: `Rainfall (mm)`, `Rainfall_Missing`, `Rainfall_Lag_1d`, `Rainfall_Lag_2d`, `Rainfall_Lag_3d`, `Rainfall_Lag_7d`
  - Cumulative Rolling Sums: `Rainfall_Prev_3d_Sum`, `Rainfall_Prev_7d_Sum`, `Rainfall_Prev_15d_Sum`, `Rainfall_Prev_30d_Sum`
  - Cumulative Rolling Maxima: `Rainfall_Prev_3d_Max`, `Rainfall_Prev_7d_Max`, `Rainfall_Prev_15d_Max`, `Rainfall_Prev_30d_Max`
  - Antecedent Wetness: `Rainy_Days_Prev_3d`, `Rainy_Days_Prev_7d`, `Rainy_Days_Prev_15d`, `Rainy_Days_Prev_30d`, `Consecutive_Rainy_Days_Before`
  - Same-day observation: `Flood_Occurred` (used as antecedent signal only when available; 0 in operational mode).

## 2. Frozen Production Model & Calibration
- **Model**: Balanced Logistic Regression with StandardScaler pipeline (`models/flood_prediction/best_model.pkl`).
- **Optimal Decision Threshold**: `0.8345` (tuned for balanced precision/recall under high class imbalance).
- **Risk Thresholds**:
  - `LOW`: Probability < 0.30 (Green `#10b981`)
  - `MODERATE`: 0.30 <= Probability < 0.70 (Orange/Yellow `#f59e0b`)
  - `HIGH`: Probability >= 0.70 (Red `#ef4444`)
- **Key Test Set Metrics (2022–2024, 343,758 rows)**:
  - Test ROC-AUC: `0.9693`
  - Test PR-AUC: `0.8306`
  - Test F1-Score: `0.8748`
  - Test Recall: `87.44%`
  - Test Precision: `87.52%`
  - Test Brier Score: `0.0111`
  - Test Calibration ECE: `0.0368`

## 3. Operational 2025 Rules & Constraints
- **Strict Separation**: Model is frozen; no retraining or threshold recalibration on 2025 data.
- **Warm-Up Period**: Stitches preceding 35 days (December 2024 historical rainfall) to prevent rolling boundary zero-fill artifacts.
- **No Fabricated Labels**: 2025 operational inputs do not contain `Flood_Occurred`.
- **Language Guardrails**: Label outputs as `"2025 Unseen Operational Inference"`, `"Estimated Flood Probability"`, `"Model-Estimated Risk"`.
- **Retrospective Evaluation Engine**: Independent engine (`Retrospective2025EvaluationEngine`) to merge future SRC ground-truth observations without altering the operational model.

## 4. Flask Web Application Architecture
- **Framework**: Python Flask 3.x, Jinja2 templates, Bootstrap 5, Leaflet.js, Chart.js.
- **Application Core**: `app.py`, `config.py`.
- **Core Routes**:
  - `GET /` & `GET /dashboard`: Main Flood Command Center & KPIs.
  - `GET /risk-map`: Interactive Odisha 30-District Geospatial Risk Map & League Table.
  - `GET /drilldown`: District & Block station-level inspection & real-time SHAP attribution.
  - `GET /historical`: 2001–2024 Historical Flood Timeline & Event Replay.
  - `GET /forecaster` & `POST /api/predict`: Real-time single-station forecaster.
  - `GET /simulator` & `POST /api/simulate`: Controlled What-If scenario simulation.
  - `GET /unseen-2025` & `POST /api/upload-2025`: 2025 Operational Risk Monitor & Ingestion.
  - `GET /benchmarks`: Multi-model benchmark comparisons, ROC, PR, and Calibration curves.
  - `GET /explainability`: SHAP Global & Local feature importance and attributions.
  - `GET /downscaling`: High-resolution statistical downscaling evaluation & comparison.
  - `GET /assistant` & `POST /api/assistant/query`: Grounded AI Flood Assistant.
  - `GET /architecture`: System methodology & dual-pipeline visual architecture.
- **Automated Test Suite**: 71 tests in `tests/` passing with 100% success rate (including `test_threshold_selection.py`, `test_temporal_validation.py` & `test_data_quality.py`).
- **Data Quality Audit Engine**: `src/data/data_quality.py` & `reports/data_quality_report.html` (15-dimension automated audit).
- **Temporal Walk-Forward Validation Engine**: `src/evaluation/temporal_validation.py` & `docs/temporal_validation_strategy.md` (5-fold expanding window cross-validation, 2017–2021).
- **Threshold Selection & Audit Engine**: `src/evaluation/threshold_optimization.py` & `docs/threshold_selection_strategy.md` (Validation-only 2019–2021 tuning, F1/F2/Recall>=80%/Youden's J/Cost-Sensitive, unweighted ANN vs class-weighted odds shift justification).
