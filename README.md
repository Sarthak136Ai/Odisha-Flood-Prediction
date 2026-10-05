# Odisha Flood Intelligence & Early Warning System
**Spatio-temporal flood prediction using 24 years of rainfall and SRC-derived flood observations.**

---

## 🌊 Overview
The **Odisha Flood Intelligence & Early Warning System** is an end-to-end, spatio-temporal machine learning and disaster intelligence platform built as a Flask web application and scientific ML pipeline. It analyzes 24 years (2001–2024, 2.75M+ continuous station records) of daily rainfall observations and Special Relief Commissioner (SRC) flood inundation reports across all 30 standard districts and 354 block stations in Odisha, India.

The platform provides:
1. **WHERE is the risk?** — Geospatial intelligence across Odisha, District, and Block/Station resolutions via Leaflet.js interactive maps.
2. **WHEN is the risk?** — Continuous temporal lag windows, multi-day accumulations, and next-day forecasting (`Flood_Next_Day`).
3. **HOW HIGH is the probability?** — Calibrated posterior probability estimation categorized into `LOW` (<30%), `MODERATE` (30–70%), and `HIGH` (≥70%) risk tiers.
4. **WHY is the model producing this risk?** — Game-theoretic SHAP local and global attributions grounded in physical meteorological drivers.

---

## 🚀 Key Capabilities & Web Application Modules

- **1. Flood Intelligence Command Center (`/` & `/dashboard`)**:
  - State-wide KPI summary counters (Districts, Stations, High/Moderate Risk Days).
  - Spatial vulnerability overview, 24-year precipitation trends, risk distribution charts, and live operational alerts.
- **2. Interactive Odisha Risk Map (`/risk-map`)**:
  - Interactive Leaflet.js map with 30 districts, colored risk circles, tooltips, and popups with 24-year historical summaries.
  - Searchable District Vulnerability League Table.
- **3. District & Block Drilldown (`/drilldown`)**:
  - Cascading dropdowns (District &rarr; Block &rarr; Date) with antecedent wetness signals (1d, 3d, 7d, 15d, 30d sums/max).
  - Real-time local SHAP feature breakdown table explaining risk drivers.
- **4. Historical Timeline & Replay (`/historical`)**:
  - 2001–2024 time series chart with daily precipitation, 7-day rolling sums, and flood inundation event markers.
  - Event replay presets: August 2019 (Mahanadi), August 2020 (Baitarani/Brahmani), August 2022 (Coastal Catchment), September 2024 (Subarnarekha).
- **5. Real-Time Forecaster (`/forecaster`)**:
  - Interactive prediction form for manual station telemetry entry or auto-derived lag parameters.
  - Frozen model inference (`FloodPredictor`), probability gauge, and calibrated risk level classification.
- **6. What-If Scenario Simulator (`/simulator`)**:
  - Controlled meteorological simulation comparing baseline vs. modified rainfall.
  - Dynamic non-linear sensitivity response curve (0 to 250 mm).
- **7. 2025 Unseen Operational Risk Monitor (`/unseen-2025`)**:
  - Dedicated operational ingestion pipeline for `data/raw/operational/2025.csv` (114,610 daily records).
  - CSV/XLSX file upload with automated schema validation and Dec 2024 historical warm-up stitching.
  - Climate Distribution Shift analysis (Kolmogorov-Smirnov 2-sample test).
  - Downloadable prediction repository (`2025_flood_risk_predictions.csv`).
  - Independent Retrospective Evaluation Engine for post-season SRC ground truth verification without model retraining.
- **8. Model Benchmarks & Calibration (`/benchmarks`)**:
  - Comprehensive comparison table across 5 architectures (Logistic Regression, Decision Tree, Random Forest, XGBoost, ANN).
  - ROC curves, Precision-Recall curves, and Reliability/Calibration diagrams.
- **9. Explainable AI / SHAP (`/explainability`)**:
  - Global SHAP feature importance ranking (all 20 features) and SHAP summary plots.
  - Key hydrological findings on soil saturation dominance, wet spell duration, and monsoon seasonality.
- **10. High-Resolution Statistical Downscaling (`/downscaling`)**:
  - `HistGradientBoostingRegressor` spatial grid refinement evaluation (Observed vs. Downscaled comparative metrics).
- **11. Grounded AI Flood Assistant (`/assistant`)**:
  - Domain-specialized chat assistant grounded in 24-year data and frozen ML prediction pipelines without hallucinations.
- **12. System Architecture & Methodology (`/architecture`)**:
  - Dual-pipeline visual flowcharts (Historical Benchmark vs. 2025 Operational Ingestion) and complete feature schema data dictionary.

---

## 📊 Model Performance Benchmark (Test Set: 2022–2024, 343,758 observations)

| Model Architecture | Optimal Threshold | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Production)** | **0.8345** | **0.9693** | **0.8306** | **0.8748** | **87.44%** | **87.52%** | **0.0111** |
| **Random Forest (Balanced Subsample)** | 0.2745 | 0.9637 | 0.8176 | 0.8452 | 84.55% | 84.48% | 0.0107 |
| **XGBoost (scale_pos_weight)** | 0.3966 | 0.9082 | 0.6151 | 0.7033 | 61.92% | 81.39% | 0.0138 |
| **Decision Tree (Baseline)** | 0.9362 | 0.6137 | 0.5107 | 0.6858 | 54.66% | 92.02% | 0.0203 |
| **ANN (MLP)** | 0.0023 | 0.9281 | 0.5992 | 0.5863 | 44.59% | 85.57% | 0.0237 |

---

## 📁 Project Architecture

```text
Odisha_Flood_Prediction/
├── app.py                     # Flask Web Application Core & REST API Endpoints
├── config.py                  # Flask configuration, file paths, and thresholds
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation & run guide
├── progress_tracker.md        # Task continuity and progress log
├── MEMORY.md                  # System architecture, models, and constraints memory
│
├── data/
│   ├── raw/
│   │   ├── historical/        # 24 Annual CSV files (2001.csv ... 2024.csv)
│   │   └── operational/       # 2025.csv (114,610 unobserved operational records)
│   ├── processed/             # Cleaned historical & operational feature matrices
│   └── predictions/           # 2025_flood_risk_predictions.csv
│
├── models/
│   ├── flood_prediction/      # Frozen best_model.pkl & model_metadata.json
│   └── downscaling/           # downscaling_model.pkl
│
├── results/
│   ├── metrics/               # model_comparison.csv, shap_feature_importance.csv
│   ├── plots/                 # roc_curve.png, precision_recall_curve.png, calibration_curves.png
│   └── downscaling/           # downscaling_metrics.csv, rainfall_comparison.png
│
├── src/
│   ├── data/                  # Data ingestion, schema validation, and combining
│   ├── features/              # Feature engineering & climatological anomaly engine
│   ├── models/                # Training pipelines & FloodPredictor inference
│   ├── evaluation/            # Metrics, ROC, PR, calibration, and confusion matrix
│   ├── explainability/        # SHAP global & local attributions
│   ├── inference/             # 2025 unseen operational pipeline & retrospective engine
│   └── geospatial/            # District coordinates & spatial risk calculator
│
├── chatbot/                   # Grounded NLP query parser, prediction tools, domain knowledge
├── templates/                 # Jinja2 HTML5 templates (12 pages + base layout)
├── static/
│   ├── css/                   # Custom disaster intelligence stylesheet (style.css)
│   ├── js/                    # Core main.js, Leaflet map.js, Chart.js charts.js, assistant.js
│   └── images/plots/          # Pre-computed high-resolution evaluation plots
│
└── tests/                     # 51 Automated unit and integration tests (100% passing)
    ├── test_routes.py         # Flask route tests
    ├── test_prediction.py     # Prediction & Simulation API tests
    ├── test_operational_2025.py
    ├── test_data.py
    ├── test_features.py
    └── ...
```

---

## 🛠️ Installation & Setup

1. **Clone or navigate to the repository**:
   ```bash
   cd Odisha_Flood_Prediction
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🌐 How to Run the Flask Application

Run the Flask web server directly:
```bash
python app.py
```
Or using the Flask CLI:
```bash
flask --app app run
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Running Automated Tests

Run the complete test suite (51 unit & integration tests):
```bash
pytest -v
```

---

## ⚖️ Scientific Governance & Operational Disclaimers

- **Zero Data Leakage**: All rolling window features use strictly antecedent data (e.g. `shift(1).rolling(...)`). The target `Flood_Next_Day` is evaluated for next-day flood forecasting ($T+1$) and is never used as an input feature.
- **Frozen 2025 Inference**: 2025 is treated as unseen operational telemetry. The model weights, thresholds, and calibration are frozen from the 2001–2024 training baseline.
- **Advisory Role**: Predictions are probabilistic risk estimates designed for emergency planning and do not supersede statutory warnings issued by the Indian Meteorological Department (IMD) or the Special Relief Commissioner (SRC).
