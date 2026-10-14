# Chatbot Interface Layer Architecture & Grounding Strategy

## 1. Executive Summary & Architectural Role

The **Odisha Flood AI Assistant** functions strictly as an **interpretive interface layer**, not as the underlying machine learning model or hydrologic simulation engine.

```
┌────────────────────────────────────────────────────────────────────────┐
│ USER INTERFACE (Web UI / Assistant Chat / REST API Endpoint)           │
│   • /assistant, /api/assistant/query                                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Natural Language Query
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ CHATBOT INTERFACE LAYER (chatbot/chatbot.py)                           │
│   • Intent Routing & Semantic Parsing                                  │
│   • Named Entity Extraction (Districts, Blocks, Dates)                 │
│   • Grounded Retrieval & Parameter Extraction                          │
│   • Missing-Data Fallback & Strict Validation Gateways                 │
└──────────────────┬─────────────────┬───────────────────┬───────────────┘
                   │                 │                   │
                   ▼                 ▼                   ▼
┌─────────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│ CORE PREDICTION ENGINE  │ │ DATA QUERY ENGINE│ │ METADATA & METRICS   │
│ • FloodPredictor        │ │ • 24-Yr Dataset  │ │ • model_metadata.json│
│ • Local SHAP Attrib.    │ │ • Operational '25│ │ • model_comparison  │
│ • models/best_model.pkl │ │ • data/combined/ │ │ • results/metrics/   │
└─────────────────────────┘ └──────────────────┘ └──────────────────────┘
```

The chatbot's role is to interpret natural language user queries and deterministically interface with:
1. **Live Model Predictions**: `FloodPredictor` and `ChatbotPredictionTool`
2. **Project Datasets**: 24-year historical telemetry (2001–2024) and 2025 operational predictions
3. **Model Evaluation Metrics**: Out-of-sample test results (ROC-AUC, PR-AUC, Recall, Precision, Brier score) from `model_metadata.json`
4. **Standardized Risk Definitions**: `LOW`, `MODERATE`, and `HIGH` risk boundaries
5. **Documented Flood Information**: River basin hydrologic properties and IMD rainfall classification criteria.

---

## 2. Elimination of Unverified Claims & Strict Grounding

> [!IMPORTANT]
> **Scientific Integrity Standard**:
> In accordance with scientific auditing guidelines, unproven absolute claims such as *"hallucination-free"* or *"zero hallucination"* have been purged from the codebase, templates, and documentation.
>
> In their place, the chatbot enforces a **deterministic grounded-response strategy**:
> - Numerical predictions, probabilities, and rainfall sums are **never synthesized or guessed**.
> - All outputs are dynamically looked up from verified model artifacts or dataset records.
> - When information is missing, out-of-range, or outside project scope, the system explicitly returns:
>   `"Sufficient project data is not available..."`

---

## 3. Supported Query Classes & Live Retrieval Mechanisms

### 3.1 Model Identity & Architecture
- **Sample Queries**: *"Which model is being used?"*, *"What model is this?"*, *"What algorithm is used?"*
- **Retrieval Mechanism**: Reads directly from `model_metadata.json`:
  - Primary Champion Model: `XGBoost`
  - Optimal Decision Threshold: `0.8073`
  - Historical Baseline: 18 years (2001–2018, 2,063,989 station records)
  - Benchmark Family: 5 architectures (Logistic Regression, Decision Tree, Random Forest, XGBoost, ANN).

### 3.2 Model Recall & Evaluation Metrics
- **Sample Queries**: *"What is the model's recall?"*, *"What are the model metrics?"*, *"What is the precision and F1?"*
- **Retrieval Mechanism**: Dynamically extracts out-of-sample test metrics on the untouched test partition (2022–2024, 343,758 observations):
  - Test Recall: **`33.95%`** (detects 3,331 flood occurrences)
  - Test Precision: **`19.05%`**
  - Test ROC-AUC: **`0.8388`**
  - Test PR-AUC: **`0.1593`**
  - Test Brier Score: **`0.1445`**
- **Domain Rationale**: Explains why disaster risk reduction (DRR) prioritizes High Recall over Precision—False Negatives represent unannounced flood inundations that threaten lives, whereas False Positives represent manageable precautionary alerts.

### 3.3 Risk Definitions & Actionable Tiers
- **Sample Queries**: *"What does HIGH risk mean?"*, *"What does MODERATE risk mean?"*, *"Explain risk levels"*
- **Retrieval Mechanism**: Formulates response from `risk_thresholds`:
  - 🔴 **HIGH RISK** ($\ge 70\%$ probability, or calibrated posterior $\ge 20\%$): Severe inundation risk within 24 hours. Action: Immediate administrative warning, ODRAF/NDRF pre-staging, and community evacuation.
  - 🟡 **MODERATE RISK** ($30\% - 70\%$ probability): Swollen tributaries and waterlogging risk in low-lying blocks. Action: Heightened vigilance, hourly river gauge tracking.
  - 🟢 **LOW RISK** ($< 30\%$ probability): Normal drainage capacity, no imminent threat.

### 3.4 Engineered Rainfall Features
- **Sample Queries**: *"What rainfall features are used?"*, *"Which features are in the model?"*
- **Retrieval Mechanism**: Dynamically inspects `metadata["features"]` and details:
  1. *Immediate Rainfall*: `Rainfall (mm)`, `Rainfall_Missing`
  2. *Antecedent Lags*: `Rainfall_Lag_1d`, `Rainfall_Lag_2d`, `Rainfall_Lag_3d`, `Rainfall_Lag_7d`
  3. *Rolling Cumulative Sums*: `Rainfall_Prev_3d_Sum`, `Rainfall_Prev_7d_Sum`, `Rainfall_Prev_15d_Sum`, `Rainfall_Prev_30d_Sum` (soil saturation proxies)
  4. *Rolling Maxima*: `Rainfall_Prev_3d_Max`, `Rainfall_Prev_7d_Max`, `Rainfall_Prev_15d_Max`, `Rainfall_Prev_30d_Max` (convective burst indicators)
  5. *Antecedent Wetness*: `Rainy_Days_Prev_3d`, `Rainy_Days_Prev_7d`, `Rainy_Days_Prev_15d`, `Rainy_Days_Prev_30d`, `Consecutive_Rainy_Days_Before`
  6. *Monsoon Seasonality & IMD Indicators*: `Month_sin`, `Month_cos`, `Day_of_Year`, plus statutory IMD categories (`Light`, `Moderate`, `Heavy`, `Very Heavy`, `Extreme`).

### 3.5 Factors Influencing Predictions (SHAP Explainability)
- **Sample Queries**: *"Which factors influenced the prediction for Cuttack?"*, *"Why did the model predict flood?"*
- **Retrieval Mechanism**: Invokes `explain_instance()` on the model pipeline to calculate local game-theoretic SHAP contributions (top risk amplifiers and top mitigating factors with exact feature values and log-odds contributions).

### 3.6 Current Flood Risk & Probability
- **Sample Queries**: *"What is the current flood risk?"*, *"What is the predicted probability?"*
- **Retrieval Mechanism**:
  - *Location Specified*: Evaluates `predict_for_location_and_date()` for that district/block and outputs the exact probability, risk tier, telemetry values, and decision threshold.
  - *State-Wide General*: Pulls active state-wide monitoring statistics (354 stations across 30 districts, count of High/Moderate/Low risk stations, average probability) from `predictions_2025_flood_risk_predictions.csv` or historical records.

---

## 4. Strict Validation & Edge-Case Handling

The chatbot implements guarded validation filters that eliminate hallucinated or invented answers:

| Scenario / Edge Case | System Behavior | Returned Message Pattern |
| :--- | :--- | :--- |
| **Unsupported Out-of-Domain Query** (e.g. *"Who won the World Cup?"*) | Detects non-hydrological domain query | `"Sufficient project data is not available to answer this query. As an interface layer for the Odisha Flood Prediction System..."` |
| **Invalid District / Station** (e.g. *"Predict flood for Atlantis"*) | Validates against 30 standard Odisha districts | `"Sufficient project data is not available: District 'Atlantis' is not recognized in Odisha telemetry records."` |
| **Invalid Block** (e.g. *"Predict flood for Cuttack block Gotham"*) | Validates against blocks in target district | `"Sufficient project data is not available: Block/Station 'Gotham' was not found in district 'Cuttack'."` |
| **Invalid Date** (e.g. *"Predict flood for Cuttack on 2099-12-31"*) | Validates against available data window (2001–2025) | `"Sufficient project data is not available: No telemetry observations found for Cuttack on date '2099-12-31'."` |
| **Model Unavailable** (Missing or uninitialized `.pkl`) | Detects missing predictor artifact | `"Sufficient project data is not available: The core flood prediction model is currently unavailable or uninitialized."` |
| **Missing Telemetry Record** | Detects empty subset | `"Sufficient project data is not available for district '...'"` |

---

## 5. Automated Verification Test Suite

The chatbot interface layer is covered by **17 automated unit and integration tests** in [`tests/test_chatbot.py`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/tests/test_chatbot.py):

1. `test_current_flood_risk_general`: General state-wide risk summary retrieval
2. `test_current_flood_risk_specific_location`: Localized risk prediction for Cuttack
3. `test_predicted_probability_general`: Explanation of predicted probability and decision threshold
4. `test_factors_influenced_prediction_general`: Global SHAP factor ranking
5. `test_factors_influenced_prediction_specific`: Localized SHAP risk drivers for Puri
6. `test_what_does_high_risk_mean`: High, Moderate, Low risk definitions and DRR actions
7. `test_which_model_is_being_used`: Dynamic model name and architecture retrieval
8. `test_model_recall`: Exact test recall retrieval and DRR tradeoff explanation
9. `test_rainfall_features_used`: Inventory of 22+ engineered rainfall features
10. `test_unsupported_questions`: Strict out-of-domain fallback without hallucination
11. `test_missing_prediction_data`: Explicit error reporting on historical telemetry gaps
12. `test_invalid_station_or_block`: Invalid district and invalid block error handling
13. `test_invalid_date`: Out-of-window date handling
14. `test_model_unavailable`: Fail-soft behavior when model artifact is missing
15. `test_historical_monthly_rainfall`: Grounded monthly rainfall aggregates
16. `test_district_24yr_summary`: 24-year district historical climate summaries
17. `test_downscaling_knowledge_query`: Grounded downscaling definition (district-to-block disaggregation).
