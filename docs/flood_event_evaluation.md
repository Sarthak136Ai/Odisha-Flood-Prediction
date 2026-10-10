# Odisha Flood Prediction: Event-Based Hydrological Evaluation Report

## Executive Summary

Standard machine-learning evaluation in flood forecasting assesses predictions on an **isolated day/record basis** (e.g., sample-level ROC-AUC, accuracy, daily precision, daily recall). While daily metrics measure point-in-time classification fidelity, they fail to answer the critical operational question faced by the **Odisha State Disaster Management Authority (OSDMA)** and **Special Relief Commissioner (SRC)**:

> *"Did the early-warning system detect the multi-day flood disaster before inundation onset, and with how many days of actionable advance lead time?"*

To address this gap, this document defines and implements a **second evaluation layer: Event-Based Evaluation**. This framework segments continuous time series into discrete hydrological episodes, performs temporal matching against model alert clusters, and computes event hit rates, advance lead times, missed inundation days, and precautionary false-alarm clusters.

All empirical evaluations in this report are computed using **real ground-truth telemetry from the 2001–2024 historical dataset** across 354 monitoring stations and 30 districts in Odisha. No synthetic or fabricated disaster episodes are used.

---

## 1. Ground-Truth Target Construction & Hydrological Mechanics

### 1.1 `Flood_Occurred` (Observed Ground Truth on Day $T$)
`Flood_Occurred` is a binary telemetry indicator ($y \in \{0, 1\}$) constructed directly from historical daily damage and inundation reports compiled by the Special Relief Commissioner (SRC), Revenue and Disaster Management Department, Government of Odisha:
* **$y = 1$**: Waterlogging or riverine inundation exceeded critical damage thresholds in the designated station/block on date $T$.
* **$y = 0$**: No destructive inundation reported.

### 1.2 `Flood_Next_Day` (Forecasting Target on Day $T+1$)
In the dataset pipeline, the primary prediction target `Flood_Next_Day` is constructed via a chronological backward shift grouped by monitoring station:
$$\text{Flood\_Next\_Day}_{s, T} = \text{Flood\_Occurred}_{s, T+1}$$

Features available on day $T$ (such as daily precipitation $R_T$, antecedent lags $R_{T-1}, R_{T-2}, R_{T-3}, R_{T-7}$, and rolling catchment accumulations $\sum_{k=0}^{29} R_{T-k}$) are mapped to predict whether a flood will occur on day $T+1$.

```
Telemetry Timeline:
Day T-7 ... Day T-1        Day T                       Day T+1
[Antecedent Rainfall] ---> [Daily Forecast Run] -----> [Ground-Truth Inundation]
                           P(Flood_Next_Day >= T*)     Flood_Occurred = 1 ?
                           ========================>
                           Nominal 1-Day Lead Time
```

This target formulation inherently equips the model to deliver a **nominal 1-day advance warning**. When persistent heavy precipitation triggers high model probabilities ahead of river cresting, the advance warning lead time expands beyond 24 hours.

---

## 2. Reproducible Flood-Event Grouping Methodology

### 2.1 Continuous Event Segmentation Algorithm
A discrete flood event is defined as a sequence of consecutive flood days with hydrological continuity.

$$\text{Let } S = \{(t_1, y_1), (t_2, y_2), \dots, (t_N, y_N)\} \text{ be a daily time series for station/district } s.$$

1. **Active Day Identification**: Any day $t_i$ where $y_i = 1$ (or predicted probability $P(y_i) \ge T^*$) is flagged as active.
2. **Gap Tolerance ($\Delta_{\text{max}} = 1\text{ day}$)**: In monsoonal river basins, brief 1-day dips in river level or localized precipitation do not indicate that the flood disaster has concluded. Two flood clusters separated by $\le 1$ dry day are merged into a single continuous flood episode $E = [t_{\text{start}}, t_{\text{end}}]$.
3. **Event Duration**:
   $$\text{Duration}(E) = (t_{\text{end}} - t_{\text{start}}) + 1 \text{ days}$$
4. **Minimum Duration**: Events with duration $\ge 1\text{ day}$ are cataloged.

### 2.2 Temporal Event Matching & Metric Definitions
For an actual ground-truth flood event $E_{\text{act}} = [t_{\text{start}}, t_{\text{end}}]$ and model alert clusters $\{E_{\text{pred}, j}\}$:

```
Actual Event:                      [======== Inundation Window ========]
                                   t_start                             t_end
Matching Tolerance:  [-- Lead Win --]
                     t_start - 2d

Alert Scenario A:    [=========== Predicted Alert ===========]  ===> HIT (Lead Time = 2d)
Alert Scenario B:                  [===== Predicted Alert =====]  ===> HIT (Lead Time = 0d)
Alert Scenario C:    [--- Alert ---]                              ===> FALSE ALARM (Standby)
Alert Scenario D:    (No Alerts Issued)                           ===> MISSED EVENT
```

| Metric | Mathematical Definition | Operational Meaning |
| :--- | :--- | :--- |
| **Detection Condition (Hit)** | $\exists j \text{ s.t. } t'_{\text{start}, j} \le t_{\text{end}} \text{ and } t'_{\text{end}, j} \ge t_{\text{start}} - 2\text{ days}$ | Alert overlap or advance warning prior to flood onset. |
| **Detection Lead Time** | $\Delta t_{\text{lead}} = \max(0, t_{\text{start}} - t'_{\text{first\_alert}})$ | Number of advance warning days before flood onset. |
| **Missed Days** | $\sum_{t \in E_{\text{act}}} \mathbb{I}(\hat{y}_t = 0)$ | Days during actual flood where alert was absent. |
| **Event Coverage %** | $(1 - \frac{\text{Missed Days}}{\text{Actual Duration}}) \times 100\%$ | Proportion of flood duration successfully covered. |
| **Missed Event (FN Event)** | No matching predicted alert in $[t_{\text{start}} - 2, t_{\text{end}}]$ | Dangerous failure to warn civil authorities. |
| **False Alarm Event (FA Event)** | Predicted cluster $E_{\text{pred}}$ with no actual flood overlap | Precautionary mobilization; no actual damage. |
| **Event Hit Rate (Event Recall)** | $\frac{\text{Total Hits}}{\text{Total Actual Events}}$ | Proportion of real disaster events warned. |
| **Event Precision** | $\frac{\text{Total Hits}}{\text{Total Hits} + \text{Total False Alarm Clusters}}$ | Confidence that an issued event alert is verified. |
| **Event F1 Score** | $2 \times \frac{\text{Event Precision} \times \text{Event Recall}}{\text{Event Precision} + \text{Event Recall}}$ | Harmonized balance between safety and false alerts. |

---

## 3. Major Historical Disaster Events Evaluation (Real Ground Truth)

The table below evaluates the production champion model (**XGBoost**, validation-optimized threshold $T^* = 0.8073$) on six major historical flood disasters spanning the **Test Set (2022–2024)**, **Validation Set (2019–2021)**, and **Training Set (2001–2018)**.

### Historical Disaster Event Evaluation Table

| Event Name | Temporal Split | Focal District | Actual Start | Actual End | Actual Duration | Detected | Lead Time | Missed Days | Coverage % |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **August 2022 Mahanadi Basin Floods** | Test (2022–2024) | Cuttack | 2022-08-10 | 2022-08-18 | 9 days | **YES (HIT)** | **1 day** | 3 days | **66.67%** |
| **September 2024 Subarnarekha Inundations** | Test (2022–2024) | Balasore | 2024-09-13 | 2024-09-16 | 4 days | **YES (HIT)** | **0 days** | 3 days | **25.00%** |
| **May 2021 Cyclone Yaas Deluge** | Validation (2019–2021) | Balasore | 2021-05-25 | 2021-05-26 | 2 days | **NO (MISSED)** | 0 days | 2 days | **0.00%** |
| **August 2020 Baitarani-Brahmani Spate** | Validation (2019–2021) | Bhadrak | 2020-07-20 | 2020-08-30 | 42 days | **YES (HIT)** | **0 days** | 18 days | **57.14%** |
| **October 2018 Cyclone Titli Floods** | Train (2001–2018) | Ganjam | 2018-10-09 | 2018-10-12 | 4 days | **YES (HIT)** | **0 days** | 2 days | **50.00%** |
| **September 2011 Historic Mahanadi Deluge** | Train (2001–2018) | Cuttack | 2011-08-31 | 2011-09-13 | 14 days | **YES (HIT)** | **1 day** | 1 day | **92.86%** |

### Detailed Hydrological Diagnostic of Key Episodes

#### 1. August 2022 Mahanadi Basin Floods (Test Set — Untouched Evaluation)
* **Context**: Intense depression over Bay of Bengal caused unprecedented discharge (>12 lakh cusecs) through Mundali barrage, inundating Cuttack, Kendrapara, and Puri deltaic blocks.
* **Model Telemetry**: XGBoost issued a high-risk alert on **2022-08-09** ($P(\text{Flood}) = 0.884$), providing a **1-day advance warning** before inundation began on **2022-08-10**.
* **Outcome**: **HIT**. The model maintained high alert probabilities across 6 of the 9 flood days, achieving **66.67% event coverage**.

#### 2. September 2024 Subarnarekha River Inundation (Test Set — Untouched Evaluation)
* **Context**: Severe low-pressure system in northern Odisha / Jharkhand caused flash surges along the Subarnarekha and Budhabalanga river systems in Balasore district.
* **Model Telemetry**: Ground-truth inundation was recorded starting 2024-09-13. The model crossed the decision threshold ($T^* = 0.8073$) on 2024-09-13 ($P(\text{Flood}) = 0.821$).
* **Outcome**: **HIT (Same-day detection, 0 days lead)**. While detected on onset day, the model probability fluctuated as precipitation tapered off, resulting in 25.0% coverage.

#### 3. May 2021 Cyclone Yaas Inundations (Validation Set — Missed Event Diagnostic)
* **Context**: Very severe cyclonic storm Yaas made landfall near Dhamra port on May 26, 2021. Inundation was primarily driven by a **catastrophic 3-4m astronomical storm surge** combined with tidal breaches, rather than multi-day cumulative precipitation.
* **Model Telemetry**: Peak model probability reached $P(\text{Flood}) = 0.612$, which was below the conservative F1-optimized threshold ($T^* = 0.8073$).
* **Outcome**: **MISSED EVENT**. This real-world finding accurately highlights an operational boundary: precipitation-based machine-learning models require coupling with coastal hydrodynamic surge telemetry to detect pure tidal/surge inundations.

#### 4. September 2011 Historic Mahanadi Deluge (Train Set — Benchmark)
* **Context**: The catastrophic 2011 Mahanadi floods affected 19 districts and >34 lakh people.
* **Model Telemetry**: Heavy antecedent precipitation throughout late August caused 7-day rolling accumulations to exceed 320 mm. The model raised warnings on **2011-08-30**, 1 day ahead of official flood onset on **2011-08-31**.
* **Outcome**: **HIT (1-day lead time, 92.86% coverage, 1 missed day across 14-day duration)**.

---

## 4. Test Set (2022–2024) Macro Event-Level Performance Across Models

The table below presents the macro event-level evaluation on the **untouched 2022–2024 test set** across all 354 monitoring stations (1,221 actual multi-day flood events).

| Model Architecture | Optimized Threshold ($T^*$) | Total Actual Events | Detected Hits | Missed Events | Event Hit Rate (Recall) | Event Precision | Event F1 Score | Mean Lead Time | False Alarm Clusters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | 0.8073 | 1,221 | **780** | **441** | **63.88%** | **19.36%** | **0.2971** | **1.22 days** | 3,249 |
| **Logistic Regression** | 0.8503 | 1,221 | 706 | 515 | 57.82% | 19.74% | 0.2943 | 0.65 days | **2,871** |
| **Decision Tree** | 0.8265 | 1,221 | 693 | 528 | 56.76% | 17.97% | 0.2730 | 0.54 days | 3,163 |
| **Random Forest** | 0.7607 | 1,221 | 708 | 513 | 57.99% | 16.67% | 0.2589 | **1.33 days** | 3,540 |
| **ANN (MLP)** | 0.1075 | 1,221 | 821 | 400 | 67.24% | 13.54% | 0.2254 | 1.11 days | 5,242 |

### Key Event-Level Takeaways
1. **XGBoost achieves the best balance (Highest Event F1 = 0.2971)**: Detects **63.88% of all flood episodes** (780 events) with an average advance lead time of **1.22 days** (29.3 hours).
2. **ANN (MLP) maximizes raw detection (Event Recall = 67.24%)**: Detects 821 events, but generates 5,242 false-alarm clusters due to uncalibrated probability tails.
3. **Tree Ensemble Advance Lead Time**: Random Forest (1.33 days) and XGBoost (1.22 days) provide substantially earlier advance warning compared to linear Logistic Regression (0.65 days) because tree models effectively capture non-linear 15-day and 30-day cumulative precipitation thresholds.

---

## 5. Separation of Daily Metrics vs. Event-Level Metrics

It is essential to distinguish between **daily record metrics** and **macro event metrics**:

```
+-------------------------------------------------------------------------------+
|                       ODISHA FLOOD EVALUATION LAYERS                          |
+-------------------------------------------------------------------------------+
| LAYER 1: DAILY RECORD METRICS                 LAYER 2: EVENT-BASED METRICS    |
| (Sample-Level Evaluation)                     (Disaster-Level Evaluation)     |
| --------------------------------              ----------------------------    |
| • Evaluates each station-day independently     • Groups consecutive flood days |
| • Class Distribution: 97.14% No / 2.86% Flood  • Unit: Discrete multi-day event|
| • Daily ROC-AUC: ~0.94 - 0.97                 • Event Hit Rate (Recall): ~64% |
| • Daily Precision: ~18% - 22%                 • Event Precision: ~19%         |
| • Daily F1: ~0.28 - 0.32                      • Mean Advance Lead Time: ~1.2d |
| • Purpose: Statistical classifier ranking     • Purpose: Civil DRR Readiness  |
+-------------------------------------------------------------------------------+
```

### Why Daily Accuracy & ROC-AUC Cannot Replace Event Metrics:
1. **Temporal Clustering**: In nature, floods occur in contiguous 3-to-14 day bursts. A model that predicts day 1 and day 2 of a 5-day flood but misses days 3-5 is penalized on 60% of daily records, yet it successfully provided the civil administration with the critical early warning needed to evacuate vulnerable populations.
2. **Actionable Lead Time**: Daily evaluation treats a prediction on the peak flood day identically to a prediction 24-48 hours prior to inundation onset. Event evaluation explicitly measures the advance warning lead time $\Delta t_{\text{lead}}$.
3. **Precautionary Standby**: Emergency response teams prefer precautionary alert clusters during severe monsoonal depressions even if rivers crest just below danger levels (false alarm clusters), provided missed catastrophic floods are minimized.

---

## 6. Generated Visualizations & Artifact Index

The following high-resolution visualization artifacts have been generated in `reports/plots/`:

1. **Multi-Model Event Comparison Benchmark**:
   * PNG: [`reports/plots/event_evaluation_comparison_all_models.png`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_evaluation_comparison_all_models.png)
   * SVG: [`reports/plots/event_evaluation_comparison_all_models.svg`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_evaluation_comparison_all_models.svg)
   * *Panels: (1) Event Hit Rate vs Miss Rate %, (2) Mean Advance Lead Time (Days), (3) Precautionary False Alarm Clusters.*

2. **August 2022 Mahanadi Basin Floods (Cuttack District)**:
   * PNG: [`reports/plots/event_timeline_august_2022_mahanadi_basin_flo.png`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_august_2022_mahanadi_basin_flo.png)
   * SVG: [`reports/plots/event_timeline_august_2022_mahanadi_basin_flo.svg`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_august_2022_mahanadi_basin_flo.svg)
   * *Displays daily precipitation, 7-day rolling accumulation, predicted flood probability curve, decision threshold line, and ground truth inundation span.*

3. **September 2024 Subarnarekha River Inundations (Balasore District)**:
   * PNG: [`reports/plots/event_timeline_september_2024_subarnarekha_in.png`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_september_2024_subarnarekha_in.png)
   * SVG: [`reports/plots/event_timeline_september_2024_subarnarekha_in.svg`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_september_2024_subarnarekha_in.svg)

4. **May 2021 Cyclone Yaas Deluge (Balasore District)**:
   * PNG: [`reports/plots/event_timeline_may_2021_cyclone_yaas_and_mons.png`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_may_2021_cyclone_yaas_and_mons.png)
   * SVG: [`reports/plots/event_timeline_may_2021_cyclone_yaas_and_mons.svg`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_may_2021_cyclone_yaas_and_mons.svg)

5. **August 2020 Baitarani-Brahmani Delta Spate (Bhadrak District)**:
   * PNG: [`reports/plots/event_timeline_august_2020_baitarani-brahmani.png`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_august_2020_baitarani-brahmani.png)
   * SVG: [`reports/plots/event_timeline_august_2020_baitarani-brahmani.svg`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_august_2020_baitarani-brahmani.svg)

6. **October 2018 Cyclone Titli Floods (Ganjam District)**:
   * PNG: [`reports/plots/event_timeline_october_2018_cyclone_titli_flo.png`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_october_2018_cyclone_titli_flo.png)
   * SVG: [`reports/plots/event_timeline_october_2018_cyclone_titli_flo.svg`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_october_2018_cyclone_titli_flo.svg)

7. **September 2011 Historic Mahanadi Deluge (Cuttack District)**:
   * PNG: [`reports/plots/event_timeline_september_2011_historic_mahana.png`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_september_2011_historic_mahana.png)
   * SVG: [`reports/plots/event_timeline_september_2011_historic_mahana.svg`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/plots/event_timeline_september_2011_historic_mahana.svg)

### Data Artifacts
* Event Summary CSV: [`reports/flood_events_summary.csv`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/flood_events_summary.csv)
* Model Event Metrics CSV: [`reports/model_event_metrics.csv`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/reports/model_event_metrics.csv)
* Results Metrics CSV: [`results/metrics/flood_event_evaluation.csv`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/results/metrics/flood_event_evaluation.csv)

---

## 7. Conclusions & Recommendations

1. **Deploy Two-Tier Evaluation in Production**:
   Machine learning pipelines for hydrological early warning should report both Layer 1 (daily calibration/F1) and Layer 2 (event hit rate/lead time) to ensure both statistical rigor and disaster-management relevance.
2. **Adopt XGBoost as Operational Champion**:
   With a **63.88% event detection hit rate**, **1.22 days mean advance notice**, and the highest overall Event F1 (0.2971), XGBoost provides the most actionable and reliable early warnings among evaluated architectures.
3. **Incorporate Hydrodynamic Surge Variables for Cyclonic Events**:
   To address missed coastal surge inundations (such as Cyclone Yaas in May 2021), future iterations should ingest tidal gauge observations and upstream dam release discharge rates alongside precipitation telemetry.
