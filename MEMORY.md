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
  - IMD Rainfall Intensity Features: `Light_Rainfall` (2.5-15.5mm), `Moderate_Rainfall` (15.6-64.4mm), `Heavy_Rainfall` (64.5-115.5mm), `Very_Heavy_Rainfall` (115.6-204.4mm), `Extreme_Rainfall` (>=204.5mm), `Heavy_Rain_Indicator` (>=64.5mm), `Extreme_Rain_Indicator` (>=204.5mm)
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
- **Automated Test Suite**: 100 tests in `tests/` passing with 100% success rate (including `test_feature_ablation.py`, `test_rainfall_features.py`, `test_event_evaluation.py`, `test_confusion_matrix.py`, `test_class_imbalance.py`, `test_threshold_selection.py`, `test_temporal_validation.py` & `test_data_quality.py`).
- **Data Quality Audit Engine**: `src/data/data_quality.py` & `reports/data_quality_report.html` (15-dimension automated audit).
- **Temporal Walk-Forward Validation Engine**: `src/evaluation/temporal_validation.py` & `docs/temporal_validation_strategy.md` (5-fold expanding window cross-validation, 2017–2021).
- **Threshold Selection & Audit Engine**: `src/evaluation/threshold_optimization.py` & `docs/threshold_selection_strategy.md` (Validation-only 2019–2021 tuning, F1/F2/Recall>=80%/Youden's J/Cost-Sensitive, unweighted ANN vs class-weighted odds shift justification).
- **Class Imbalance Analysis Engine**: `src/evaluation/class_imbalance.py` & `docs/class_imbalance_analysis.md` (2.638% historical prevalence, 36.91:1 imbalance ratio, multi-tier confusion matrix evaluation, loss-weighting & SMOTE rejection rationale).
- **Confusion Matrix Evaluation Engine**: `src/evaluation/confusion_matrix.py` & `reports/plots/confusion_matrices_all_models.png/svg` (Full decomposition TN/FP/FN/TP, Acc, Prec, Rec, F1, F2, FNR/Miss Rate, FPR/Fall-out, dynamic model discovery, PNG/SVG figures, CSV/JSON artifacts).
- **Event-Based Hydrological Evaluation Engine**: `src/evaluation/event_evaluation.py` & `docs/flood_event_evaluation.md` (Spatiotemporal continuous event segmentation, temporal matching, lead-time estimation, 6 real historical disaster episodes across Cuttack/Balasore/Bhadrak/Ganjam, 1,221 test event benchmarks, high-res dual-panel timeline visualizers PNG & SVG).
- **IMD Rainfall Intensity Feature Engine**: `src/features/rainfall_features.py`, `docs/data_dictionary.md` & `README.md` (Statutory IMD 24-hr criteria: Light, Moderate, Heavy, Very Heavy, Extreme, Heavy-rain indicator, Extreme-rain indicator, strict causal non-leakage and missing-data safety).
- **Formal Feature Ablation Study**: `src/evaluation/feature_ablation.py`, `scripts/run_feature_ablation.py`, `docs/feature_ablation_study.md`, `reports/feature_ablation_results.csv`, `results/metrics/feature_ablation_results.csv` & `reports/plots/feature_ablation_comparison.png/svg`. Empirically proved:
  - Point rainfall only (Exp A) achieves poor predictive capacity (ROC-AUC 0.656, PR-AUC 0.080, FN > 7,120).
  - Antecedent lags (Exp B) capture river hydrodynamics wave travel, jumping ROC-AUC to 0.775 (+11.9%).
  - Multi-day rolling accumulations (Exp C) capture basin soil saturation, lifting ROC-AUC to 0.810 and test recall to ~0.50.
  - Seasonal harmonics (Exp D) gate monsoon windows (ROC-AUC ~0.854), but produce high false alarms when ungrounded.
  - Baseline synergy (Exp E, 22 feats) achieves top validation PR-AUC (0.178) and well-calibrated Brier loss (0.140).
  - Statutory IMD intensity indicators (Exp F, 29 feats) achieve top disaster safety: F2 rises to 0.385 (LR) and 0.319 (XGBoost), cutting missed flood events by 302–359.

