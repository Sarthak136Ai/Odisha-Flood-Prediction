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
- **10. District-to-Block Statistical Downscaling (`/downscaling`)**:
  - `HistGradientBoostingRegressor` district-to-block spatial disaggregation (~5,190 km² district scale to ~440 km² block scale).
  - Comparative benchmark across Observed vs. Coarse vs. Downscaled series (MAE, RMSE, R², Bias, Pearson correlation; see [`docs/downscaling_methodology.md`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/docs/downscaling_methodology.md)).
- **11. Grounded AI Flood Assistant (`/assistant`)**:
  - Domain-specialized chat interface layer dynamically grounded in live model predictions (`FloodPredictor`), official model metadata, and local SHAP explainability without inventing numerical values (see [`docs/chatbot_architecture.md`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/docs/chatbot_architecture.md)).
- **12. System Architecture & Methodology (`/architecture`)**:
  - Dual-pipeline visual flowcharts (Historical Benchmark vs. 2025 Operational Ingestion) and complete feature schema data dictionary.

---

## 📊 Model Performance Benchmark (Test Set: 2022–2024, 343,758 observations)

| Model Architecture | Optimal Threshold | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Pure Met)** | 0.8503 | **0.8409** | **0.2399** | **0.3226** | **41.15%** | **26.53%** | 0.1490 |
| **XGBoost (Production Champion)** | **0.8073** | **0.8388** | 0.1593 | 0.2440 | 33.95% | 19.05% | 0.1445 |
| **Random Forest (Pure Met)** | 0.7690 | 0.8220 | 0.1529 | 0.2448 | 35.10% | 18.80% | 0.1255 |
| **ANN (MLP)** | 0.1063 | 0.8245 | 0.1332 | 0.2050 | 31.02% | 15.31% | **0.0263** |
| **Decision Tree (Baseline)** | 0.8265 | 0.8012 | 0.1295 | 0.1834 | 20.09% | 16.88% | 0.1515 |

---

## 🎯 Interpreting Confusion Matrices in a Flood Early Warning Context

In hydrological disaster risk reduction (DRR), standard classification metrics like overall Accuracy can be dangerously deceptive due to extreme class imbalance (~2.6% flood rate). The Confusion Matrix decomposes every prediction on the **untouched test set (2022–2024, 343,758 observations)** into four operational outcomes:

```
                          ┌──────────────────────────┬──────────────────────────┐
                          │    PREDICTED NO FLOOD    │     PREDICTED FLOOD      │
  ┌───────────────────────┼──────────────────────────┼──────────────────────────┤
  │ ACTUAL NO FLOOD (Dry) │   TRUE NEGATIVE (TN)     │   FALSE POSITIVE (FP)    │
  │                       │   Normal Civic Flow      │   False Alarm Deployment │
  ├───────────────────────┼──────────────────────────┼──────────────────────────┤
  │ ACTUAL FLOOD (Inund.) │ ⚠️ FALSE NEGATIVE (FN) ⚠️│    TRUE POSITIVE (TP)    │
  │                       │   MISSED FLOOD EVENT     │    CAUGHT FLOOD EVENT    │
  └───────────────────────┴──────────────────────────┴──────────────────────────┘
```

### 🚨 Operational Cost Asymmetry: Why False Negatives are Critical

| Quadrant | Operational Impact in Flood Early Warning | Civil & Financial Consequences |
|---|---|---|
| **True Negatives (TN)** | Correctly identified dry/safe days | Seamless normal civic activities; zero emergency mobilization expense. |
| **False Positives (FP)** | "False Alarms" — Model predicted flood, but no inundation occurred | Precautionary deployment of emergency teams (ODRAF/NDRF), minor evacuation inconvenience, resource standby cost. **Zero loss of life.** |
| **⚠️ False Negatives (FN)** | **"Missed Floods" — Model predicted safe dry conditions, but catastrophic inundation occurred** | **CRITICAL FAILURE**: Zero advance warning issued to civil authorities or citizens. Leads to unevacuated floodplains, trapped populations, infrastructure destruction, and loss of life. |
| **True Positives (TP)** | "Caught Floods" — Timely advance warning issued | Early reservoir pre-discharge, prompt evacuation of vulnerable villages, pre-positioning of rescue boats and relief supplies. **Lives and assets saved.** |

### 📐 Key Diagnostic Ratios for Disaster Management

1. **Recall / Sensitivity ($=\frac{\text{TP}}{\text{TP} + \text{FN}}$)**: The percentage of all actual flood events successfully forecasted. In high-risk cyclonic events, this should be maximized.
2. **Miss Rate / False Negative Rate ($\text{FNR} = \frac{\text{FN}}{\text{TP} + \text{FN}} = 1 - \text{Recall}$)**: The proportion of flood events that struck without warning. **Minimizing FNR is the platform's primary safety constraint.**
3. **Fall-Out / False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}} = 1 - \text{Specificity}$)**: The false alarm rate on non-flood days, tracking civil alert fatigue.
4. **$F_2$ Score ($= 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$)**: Disaster-averse metric weighting Recall twice as heavily as Precision to reflect the asymmetric penalty of missed floods.

---

## 📁 Project Architecture

```text
Odisha_Flood_Prediction/
├── app.py                     # Flask Web Application Core & REST API Endpoints
├── config.py                  # Flask configuration, file paths, and thresholds
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation & run guide
├── PROJECT_PROGRESS.md        # Comprehensive master progress & architecture log
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
│   ├── metrics/               # model_comparison.csv, confusion_matrix_evaluation.csv
│   ├── plots/                 # roc_curve.png, precision_recall_curve.png, confusion_matrix_*.png/svg
│   └── downscaling/           # downscaling_metrics.csv, rainfall_comparison.png
│
├── reports/                   # Automated data quality, temporal validation & confusion matrix reports
│
├── src/
│   ├── data/                  # Data ingestion, schema validation, and combining
│   ├── features/              # Feature engineering & climatological anomaly engine
│   ├── models/                # Training pipelines & FloodPredictor inference
│   ├── evaluation/            # Metrics, ROC, PR, calibration, confusion matrix & threshold optimization
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
└── tests/                     # 80 Automated unit and integration tests (100% passing)
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

Run the complete test suite across all 15 test modules:
```bash
pytest -v
```

## 🌧️ IMD Rainfall Intensity Classification & Feature Engineering

The feature engineering pipeline computes standardized rainfall-intensity indicators based on statutory 24-hour criteria established by the **India Meteorological Department (IMD)**, Ministry of Earth Sciences, Government of India:

| Category / Indicator | IMD 24-Hour Range ($R$) | Physical & Meteorological Meaning | Feature Name |
| :--- | :---: | :--- | :--- |
| **No Rain / Dry** | $R = 0.0\text{ mm}$ | Completely dry; normal civic conditions. | — |
| **Trace / Very Light** | $0.1 \le R \le 2.4\text{ mm}$ | Below statutory rainy-day threshold ($< 2.5\text{ mm}$). | — |
| **Light Rainfall** | $2.5 \le R \le 15.5\text{ mm}$ | Meets IMD rainy-day criterion; minimal surface runoff. | `Light_Rainfall` (`Rain_Light`) |
| **Moderate Rainfall** | $15.6 \le R \le 64.4\text{ mm}$ | Steady precipitation; soil moisture saturation begins. | `Moderate_Rainfall` (`Rain_Moderate`) |
| **Heavy Rainfall** | $64.5 \le R \le 115.5\text{ mm}$ | Significant surface runoff; urban/deltaic waterlogging. | `Heavy_Rainfall` (`Rain_Heavy`) |
| **Very Heavy Rainfall** | $115.6 \le R \le 204.4\text{ mm}$ | Catchment inundation; river swelling (Orange Alert). | `Very_Heavy_Rainfall` (`Rain_Very_Heavy`) |
| **Extremely Heavy Rainfall** | $R \ge 204.5\text{ mm}$ | Catastrophic deluge; major river spates (Red Alert). | `Extreme_Rainfall` (`Rain_Extreme`) |
| **Heavy Rain Indicator** | $R \ge 64.5\text{ mm}$ | Cumulative warning trigger: Heavy, Very Heavy, or Extreme. | `Heavy_Rain_Indicator` (`Is_Heavy_Rain`) |
| **Extreme Rain Indicator** | $R \ge 204.5\text{ mm}$ | Highest statutory disaster warning trigger. | `Extreme_Rain_Indicator` (`Is_Extreme_Rain`) |

### Key Properties:
- **Zero Future Information**: Generated strictly from day $T$ rainfall telemetry (or antecedent lags). No future day $T+1$ information is used.
- **Robust Missing Data Handling**: If `Rainfall (mm)` is missing (`NaN` or `Rainfall_Missing == 1`), all intensity indicators strictly default to $0$. Negative values are treated as $0$.
- **Train/Test Invariance**: IMD thresholds are fixed physical constants, ensuring complete consistency across training, validation, test, and unseen operational sets.
- **Complete Documentation**: Full data dictionary is available in [`docs/data_dictionary.md`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/docs/data_dictionary.md).

---

## ⏱️ Multi-Horizon Flood Forecasting (T+1, T+2, T+3 Lead Times)

The platform supports configurable multi-day forecast horizons to empower phased disaster risk reduction:
- **Flood_T+1 (24-Hour Lead)**: Next-day warning (equivalent to `Flood_Next_Day`). Highest precision ($24.7\%$), optimized for community evacuation and tactical rescue boat staging.
- **Flood_T+2 (48-Hour Lead)**: Two-day ahead warning. Strong discrimination ($\text{ROC-AUC} = 0.835$), ideal for reservoir gate pre-discharge and relief shelter mobilization.
- **Flood_T+3 (72-Hour Lead)**: Three-day ahead warning. High recall retention ($> 38\%$), ideal for state-wide ODRAF/NDRF battalion pre-deployment.

### Performance Across Forecast Horizons (Untouched Test Set 2022–2024):
| Model | Horizon | Lead Time | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | F2 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **T+1** | 24 Hours | **0.8406** | **0.2383** | **0.2467** | **0.4481** | **0.3182** | **0.3852** |
| **Logistic Regression** | **T+2** | 48 Hours | 0.8312 | 0.2059 | 0.2445 | 0.3884 | 0.3001 | 0.3475 |
| **Logistic Regression** | **T+3** | 72 Hours | 0.8227 | 0.1939 | 0.1973 | 0.4168 | 0.2678 | 0.3409 |
| **XGBoost (Champion)** | **T+1** | 24 Hours | **0.8470** | **0.1625** | **0.1927** | **0.3465** | **0.2477** | **0.2988** |
| **XGBoost (Champion)** | **T+2** | 48 Hours | 0.8353 | 0.1329 | 0.1703 | 0.3241 | 0.2233 | 0.2745 |
| **XGBoost (Champion)** | **T+3** | 72 Hours | 0.8232 | 0.1248 | 0.1453 | 0.3897 | 0.2117 | 0.2916 |

- **Zero Temporal Leakage**: Predictions at day $T$ strictly use rainfall telemetry up to day $T$.
- **Detailed Report**: See [`docs/multi_horizon_forecasting.md`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/docs/multi_horizon_forecasting.md) and [`reports/plots/multi_horizon_degradation.png`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/multi_horizon_degradation.png).

---

## 🔬 District-to-Block Statistical Rainfall Downscaling

To bridge the spatial granularity gap between coarse regional precipitation signals (district spatial aggregates averaging ~5,190 km²) and localized rain gauge stations (~440 km² per block), a dedicated gradient-boosted regression model (`HistGradientBoostingRegressor`) maps macro rainfall to block telemetry.

### Meteorological Benchmark (Untouched Test Set 2022–2024, 343,758 observations):
| Precipitation Series | MAE (mm) | RMSE (mm) | R² Score | Mean Bias (mm) | Pearson r |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original Coarse Regional Rainfall (District Mean)** | **2.7156** | **7.7189** | **0.6161** | **-0.0000** | **0.7849** |
| **Statistical Downscaled Rainfall (Model Estimate)** | 2.7472 | 7.8264 | 0.6053 | +0.0103 | 0.7780 |

### Key Scientific Findings:
- **Spatial Granularity Definition**: The model predicts **block-level station rain gauge point telemetry** from district spatial aggregates. It is strictly not termed "high-resolution gridded NWP".
- **Downstream Flood Model Benefit**: Feeding downscaled rainfall into the frozen flood early warning model increases Test PR-AUC from **0.1593** to **0.1813** (+0.0220) and improves precision from **19.05%** to **22.96%** (+3.91%), as spatial averaging filters hyper-localized gauge sensor noise.
- **Zero Leakage**: Trained strictly on 2001–2018; evaluated on validation (2019–2021) and test (2022–2024) with strictly antecedent rolling features (`shift(1)`).
- **Full Scientific Audit**: Detailed methodology, IMD intensity-tier breakdowns, and limitations are documented in [`docs/downscaling_methodology.md`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/docs/downscaling_methodology.md).

---

## 🤖 Grounded AI Flood Assistant & Chatbot Interface Layer

The conversational assistant (`/assistant`) operates strictly as an **interpretive interface layer**, not as the underlying numerical forecasting model. It bridges the gap between raw machine learning artifacts and civil decision-makers via a deterministic grounded retrieval architecture.

### Key Architectural Standards:
1. **Interface Layer Separation**: The assistant never computes mathematical predictions itself or invents numerical outputs. Every numerical value (probabilities, risk tiers, model recall, precision, decision thresholds) is queried dynamically from live artifacts:
   - **Live Model Inference**: `src/models/predict.py` (`FloodPredictor` and `calibrated_best_model.pkl`)
   - **Official Model Benchmarks**: `models/flood_prediction/model_metadata.json` and `reports/model_comparison.csv`
   - **Operational Telemetry**: `reports/2025_flood_risk_predictions.csv`
   - **Feature Attributions**: Real-time local TreeSHAP decompositions via `explain_instance()`
2. **Deterministic Guardrails & Error Responses**:
   - When project data or telemetry for a requested station, block, or date is missing, the chatbot explicitly states:
     `"Sufficient project data is not available: ..."`
   - If the ML model artifact is offline or uninitialized, it explicitly returns:
     `"Sufficient project data is not available: The flood prediction model artifact is currently offline or uninitialized."`
   - Unrecognized queries fall back to a safe domain assistance menu without fabricating facts.
3. **Removal of Unverified Claims**:
   - All legacy claims of being "hallucination-free" have been eliminated and replaced with verifiable **grounded retrieval with guarded fallbacks**.
4. **Complete Architecture & Verification**: Full technical documentation is available in [`docs/chatbot_architecture.md`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/docs/chatbot_architecture.md) and verified by 17 unit and integration tests in `tests/test_chatbot.py`.

---

## ⚖️ Scientific Governance & Operational Disclaimers

- **Zero Data Leakage**: All rolling window features use strictly antecedent data (e.g. `shift(1).rolling(...)`). The target `Flood_Next_Day` is evaluated for next-day flood forecasting ($T+1$) and is never used as an input feature.
- **Frozen 2025 Inference**: 2025 is treated as unseen operational telemetry. The model weights, thresholds, and calibration are frozen from the 2001–2024 training baseline.
- **Advisory Role**: Predictions are probabilistic risk estimates designed for emergency planning and do not supersede statutory warnings issued by the Indian Meteorological Department (IMD) or the Special Relief Commissioner (SRC).
