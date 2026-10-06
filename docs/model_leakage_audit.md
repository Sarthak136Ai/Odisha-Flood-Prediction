# Scientific Investigation & Model Leakage Audit Report
**Project:** Odisha Flood Intelligence & Early Warning System  
**Document Path:** `docs/model_leakage_audit.md`  
**Investigation Date:** 2026-10-06  
**Status:** Investigation Completed & Leakage Conclusively Identified  

---

## 1. Executive Summary & Core Conclusion

An objective, data-driven scientific investigation was conducted into the unusually high flood prediction metrics (ROC-AUC $\approx 0.969$, PR-AUC $\approx 0.831$, F1 $\approx 0.875$) reported for Logistic Regression.

### Core Finding:
The extraordinarily high ROC-AUC of $\sim 0.969$ was driven almost entirely by **Autoregressive Target Persistence Leakage** caused by the inclusion of **`Flood_Occurred` ($Y_t$)** as an input feature to forecast **`Flood_Next_Day` ($Y_{t+1}$)**.

1. **Hydrological Persistence Reality**:
   In historical disaster records, floods are continuous multi-day inundation events:
   - When a block is flooded on day $T$ (`Flood_Occurred = 1`), the empirical probability that it remains flooded on day $T+1$ (`Flood_Next_Day = 1`) is **$92.14\%$**.
   - When a block is dry on day $T$ (`Flood_Occurred = 0`), the probability of a new flood onset on day $T+1$ is only **$0.20\%$**.
   - The Pearson correlation between `Flood_Occurred` and `Flood_Next_Day` is **$r = 0.9193$**.
2. **Single-Feature Proof**:
   Training a Logistic Regression model using **only** `Flood_Occurred` (and zero meteorological data) yields:
   - **Validation ROC-AUC: $0.9619$**, **Test ROC-AUC: $0.9356$**, **Test F1: $0.8750$**.
3. **Operational Failure / Invalidation**:
   In real-time operational deployment (e.g. 2025 unseen telemetry in `data/raw/operational/2025.csv`), real-time flood inundation labels do not exist at time $T$. In `src/inference/unseen_2025_pipeline.py`, `Flood_Occurred` was hardcoded to $0$, causing a massive covariate distribution shift and blinding the model.
4. **True Pure Meteorological Performance**:
   When `Flood_Occurred` is removed and models are trained on genuine meteorological antecedents (daily rainfall, 1d/2d/3d/7d lags, 3d/7d/15d/30d rolling sums and max, wet spells, seasonal cycles):
   - **XGBoost (Pure Met)**: Test ROC-AUC = **$0.8467$**, Test PR-AUC = **$0.1696$**, Test F1 = **$0.2511$** (at high-recall operational threshold).
   - **Logistic Regression (Pure Met)**: Test ROC-AUC = **$0.8223$**, Test PR-AUC = **$0.2438$**, Test Recall = **$74.17\%$**.
   - **Random Forest (Pure Met)**: Test ROC-AUC = **$0.8157$**, Test PR-AUC = **$0.1644$**.

A PR-AUC of **$0.244$** on an imbalanced dataset with a **$2.85\%$** base positive rate represents an **$\approx 8.5\times$ precision enrichment** over random baseline, reflecting true physical early-warning predictive power.

---

## 2. Comprehensive 12-Point Leakage Checklist

| # | Inspection Category | Status | Detailed Empirical Findings |
| :--- | :--- | :---: | :--- |
| **1** | **Target Leakage** | ❌ **LEAKAGE FOUND** | `Flood_Occurred` ($Y_t$) was included in the feature set (`config.yaml` line 75). Because floods persist for multiple days, $Y_t$ has a $0.919$ correlation with $Y_{t+1}$, artificially inflating ROC-AUC to $0.969$. |
| **2** | **Temporal Leakage** | ✅ **PASSED** | Data splitting is strictly chronological: Train (2001–2018), Val (2019–2021), Test (2022–2024). Date overlap between splits is exactly $0$. |
| **3** | **Future Rainfall Information** | ✅ **PASSED** | Lags and rolling accumulation windows use antecedent shifts (`shift(1)`, `shift(2)`, etc.). No future day rainfall $T+1, T+2$ is accessible at time $T$. |
| **4** | **Rolling-Window Implementation** | ✅ **PASSED** | `Rainfall_Prev_Wd_Sum` applies `s.shift(1).rolling(W, min_periods=1).sum()` grouped strictly by `[District, Block/Station]`. Day $T$ is excluded from the window. |
| **5** | **Lag Feature Construction** | ✅ **PASSED** | Lags are strictly grouped per station. No cross-station lag pollution. Initial boundary nulls in 2001 are filled with $0.0$. |
| **6** | **`Flood_Next_Day` Construction** | ✅ **PASSED** | Correctly generated via `grouped["Flood_Occurred"].shift(-1)` per station. Final day of 2024 with null target is cleanly dropped. |
| **7** | **Duplicate Records** | ✅ **PASSED** | Zero duplicate records on `[District, Block/Station, Date]` across the entire 2.75M record matrix. |
| **8** | **Station/Date Duplication** | ✅ **PASSED** | Standard 30 districts and authentic 354 block stations verified. No cross-split leakage. |
| **9** | **Train/Test Overlap** | ✅ **PASSED** | Train (2001–2018: 2,063,989 rows), Val (2019–2021: 344,140 rows), Test (2022–2024: 343,758 rows). Zero temporal overlap. |
| **10** | **Preprocessing Leakage** | ✅ **PASSED** | `StandardScaler` is inside `sklearn.pipeline.Pipeline` and fitted strictly on `X_train`. |
| **11** | **Threshold Leakage** | ✅ **PASSED** | Classification threshold is tuned on the **Validation Set** ($2019–2021$) and applied out-of-sample to the Test Set. |
| **12** | **Suspicious / Non-Stationary Features** | ⚠️ **FLAGGED** | Raw `Year` ($2001 \dots 2024$) was included in `temporal_cols`, causing non-stationary linear extrapolation. |

---

## 3. Suspicious Feature Analysis Table

| Feature Name | Correlation with Target ($r$) | Logistic Reg Coef | Status | Physical / Statistical Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **`Flood_Occurred`** | **$+0.9193$** | **Huge ($\text{SHAP} \approx 0.35$)** | ❌ **UNSAFE (Leaked)** | Represents ground-truth flood state at day $T$. Teaches the model multi-day persistence instead of rainfall onset forecasting. Unobserved at real-time inference time. |
| **`Year`** | $-0.0516$ | $-0.0279$ | ⚠️ **UNSAFE (Extrapolation)** | Raw integer year cannot generalize to future unseen years ($2025+$). Imposes an artificial linear trend across decades. |
| **`Month_Number` / `Day_of_Year`** | $+0.0715$ / $+0.0694$ | $+57.51$ / $-58.22$ | ⚠️ **REDUNDANT (Collinear)** | Highly collinear with smooth trigonometric calendar features (`Month_sin`, `Month_cos`), causing exploding inverse coefficients in unregularized linear models. |
| **`Rainfall_Prev_15d_Sum`** | $+0.2381$ | $+0.3547$ | ✅ **SAFE** | Antecedent soil saturation indicator. Physically grounded hydrological driver. |
| **`Rainfall_Prev_7d_Sum`** | $+0.2309$ | $+0.0901$ | ✅ **SAFE** | Antecedent intermediate accumulation. Physically grounded. |
| **`Rainfall_Prev_30d_Sum`** | $+0.2306$ | $+0.0157$ | ✅ **SAFE** | Long-term catchment antecedent wetness. Physically grounded. |
| **`Rainy_Days_Prev_30d`** | $+0.2140$ | $-0.0153$ | ✅ **SAFE** | Antecedent wet spell frequency. Physically grounded. |
| **`Rainfall (mm)`** | $+0.1759$ | $+0.2575$ | ✅ **SAFE** | Same-day 24-hour rainfall measured antecedent to $T+1$ flood. Physically grounded. |
| **`Rainfall_Lag_1d`** | $+0.1572$ | $+0.0753$ | ✅ **SAFE** | 1-day antecedent rainfall. Physically grounded. |
| **`Month_sin` / `Month_cos`** | $-0.1737$ / $-0.0932$ | $-3.3188$ / $-1.1508$ | ✅ **SAFE** | Smooth continuous seasonal monsoon cycle. Continuous cyclical mapping $[0, 2\pi]$. |

---

## 4. Empirical Ablation Evidence

To isolate the exact contribution of each feature group, controlled ablation experiments were executed using identical 24-year chronological splits:

```
Split Breakdown:
  • Train (2001–2018): 2,063,989 rows | Positive Flood Rate: 2.502%
  • Validation (2019–2021): 344,140 rows | Positive Flood Rate: 3.243%
  • Test (2022–2024): 343,758 rows | Positive Flood Rate: 2.855%
```

### Ablation Experiment Results (Logistic Regression Pipeline):
| Experiment Configuration | Feat Count | Opt Thresh | Val ROC-AUC | Val PR-AUC | Val F1 | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Full Baseline (With `Flood_Occurred` & `Year`)** | 27 | 0.8345 | **0.9729** | **0.8924** | **0.9261** | **0.9693** | **0.8306** | **0.8748** | 87.43% | 87.51% |
| **2. Without `Flood_Occurred` (With `Year`)** | 26 | 0.2271 | 0.8448 | 0.1258 | 0.1935 | 0.8199 | 0.2486 | 0.1488 | 73.80% | 8.28% |
| **3. Pure Meteorological & Seasonal (No `Flood_Occurred`, No `Year`)** | 25 | 0.4074 | 0.8455 | 0.1285 | 0.1934 | **0.8223** | **0.2438** | **0.1431** | **74.17%** | **7.92%** |
| **4. Pure Rainfall Accumulations Only (No calendar, no persistence)** | 18 | 0.7828 | 0.8075 | 0.1101 | 0.1603 | 0.8199 | 0.1627 | 0.2231 | 49.87% | 14.37% |
| **5. Persistence ONLY (`Flood_Occurred` single feature)** | 1 | 0.9978 | **0.9619** | **0.8605** | **0.9263** | **0.9356** | **0.7691** | **0.8750** | 87.50% | 87.50% |

### Key Takeaways from Ablation:
1. **Experiment 5 proves causality**: Using *only* `Flood_Occurred` achieves $0.9356$ Test ROC-AUC and $0.8750$ Test F1. Adding 26 meteorological features only nudged Test ROC-AUC from $0.9356$ to $0.9693$.
2. **Experiment 3 establishes the scientific baseline**: Removing `Flood_Occurred` and `Year` reveals the true predictive power of antecedent precipitation: **Test ROC-AUC $\approx 0.822$**, **Test PR-AUC $\approx 0.244$**.

---

## 5. Multi-Model Benchmark under Pure Meteorological Architecture

With the target leakage eliminated, all machine learning architectures were re-evaluated under the true pure meteorological paradigm:

| Model Architecture | Optimal Threshold | Val ROC-AUC | Val PR-AUC | Val F1 | Test ROC-AUC | Test PR-AUC | Test Recall | Test Precision | Test Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Pure Meteorological)** | **0.7864** | **0.8638** | **0.1493** | **0.2139** | **0.8467** | **0.1696** | **40.54%** | **18.19%** | **0.1351** |
| **Random Forest (Pure Meteorological)** | 0.7372 | 0.8550 | 0.1437 | 0.2111 | 0.8157 | 0.1644 | 40.89% | 18.29% | 0.1093 |
| **Logistic Regression (Pure Meteorological)** | 0.4074 | 0.8455 | 0.1285 | 0.1934 | 0.8223 | 0.2438 | 74.17% | 7.92% | 0.1412 |

---

## 6. Corrective Actions Required

1. **Update Feature Schema (`config.yaml`)**:
   - Remove `same_day_flood_col: "Flood_Occurred"` from the active predictor feature set.
   - Remove `"Year"` from `temporal_cols`.
   - Keep smooth cyclical seasonal features (`Month_sin`, `Month_cos`) and drop collinear integer calendar features.
2. **Retrain All Models (`src/models/train_models.py`)**:
   - Re-fit Logistic Regression, Decision Tree, Random Forest, XGBoost, and ANN on pure meteorological features.
   - Update champion model selection to sort strictly on **Validation Set metrics** (`Val_PR_AUC`, `Val_F1`).
3. **Regenerate Artifacts & Plots**:
   - Update `models/flood_prediction/best_model.pkl` and `model_metadata.json`.
   - Update `results/metrics/model_comparison.csv`, ROC/PR curves, reliability diagrams, and confusion matrices.
   - Re-run `src/inference/unseen_2025_pipeline.py` to generate calibrated pure meteorological predictions for 2025.
4. **Update Documentation & Test Suite**:
   - Synchronize `README.md`, `PROJECT_PROGRESS.md`, and test assertions with the scientifically verified performance baseline.
