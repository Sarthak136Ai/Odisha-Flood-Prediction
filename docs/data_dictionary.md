# Odisha Flood Intelligence & Early Warning System: Master Data Dictionary

## Overview & Domain Context

The **Odisha Flood Intelligence & Early Warning System** ingests 24 years (2001–2024) of daily meteorological and hydrological observations across **30 standard districts** and **354 monitoring stations/blocks** in Odisha, India. The primary objective is to issue calibrated next-day flood risk alerts to assist the **Special Relief Commissioner (SRC)** and the **Odisha State Disaster Management Authority (OSDMA)** in early mobilization.

All features are engineered strictly using causal, chronological, and leak-free operations. For an observation on date $T$, only observations available up to date $T$ (current day rainfall and antecedent lags/rolling aggregates) are used to predict flood occurrences on day $T+1$ (`Flood_Next_Day`).

---

## 1. Statutory Rainfall Intensity Standards (IMD Criteria)

### Source & Authority
All rainfall classification thresholds strictly follow the official statutory standards established by the **India Meteorological Department (IMD)**, Ministry of Earth Sciences, Government of India:
- *Reference*: IMD Meteorological Glossary, National Weather Forecasting Centre (NWFC) Criteria for Daily (24-Hour) Precipitation Classification, and Odisha SRC Disaster Management Guidelines.
- *Measurement Window*: 24-hour cumulative rainfall recorded between 08:30 IST of day $T-1$ to 08:30 IST of day $T$.
- *Rainy Day Cutoff*: $\ge 2.5\text{ mm}$ (precipitation $< 2.5\text{ mm}$ is classified as Dry/Trace and does not qualify as a statutory Rainy Day).

### IMD Intensity Classification Matrix
| Category Name | IMD Rainfall Range ($R$) | Physical & Hydrological Significance | Early Warning Alert Level |
| :--- | :---: | :--- | :--- |
| **No Rain / Dry** | $R = 0.0\text{ mm}$ | Completely dry conditions; baseflow drainage. | Green (No Warning) |
| **Trace / Very Light** | $0.1\text{ mm} \le R \le 2.4\text{ mm}$ | Negligible precipitation; evaporated or absorbed by canopy. | Green (No Warning) |
| **Light Rainfall** | $2.5\text{ mm} \le R \le 15.5\text{ mm}$ | Minimal direct runoff; shallow infiltration. | Green (No Warning) |
| **Moderate Rainfall** | $15.6\text{ mm} \le R \le 64.4\text{ mm}$ | Steady precipitation; progressive topsoil saturation. | Green / Advisory |
| **Heavy Rainfall** | $64.5\text{ mm} \le R \le 115.5\text{ mm}$ | Significant surface runoff; localized waterlogging in coastal deltas. | Yellow / Orange Alert |
| **Very Heavy Rainfall** | $115.6\text{ mm} \le R \le 204.4\text{ mm}$ | Severe catchment runoff; river swelling; embankment stress. | Orange / Red Alert |
| **Extremely Heavy Rainfall** | $R \ge 204.5\text{ mm}$ | Catastrophic deluge; flash floods, major dam discharges, widespread breach. | Red Alert (Severe Hazard) |

---

## 2. IMD Rainfall Intensity Features

| Feature Name | Type | Unit | Formula / Condition | Hydrological & DRR Rationale |
| :--- | :---: | :---: | :--- | :--- |
| **`Light_Rainfall`** | Binary | $\{0, 1\}$ | $\mathbb{I}(2.5 \le R_T \le 15.5)$ | Identifies light monsoonal days that meet the rainy-day definition without severe surface runoff. |
| **`Moderate_Rainfall`** | Binary | $\{0, 1\}$ | $\mathbb{I}(15.6 \le R_T \le 64.4)$ | Identifies steady monsoonal showers that prime soil moisture saturation for subsequent storm peaks. |
| **`Heavy_Rainfall`** | Binary | $\{0, 1\}$ | $\mathbb{I}(64.5 \le R_T \le 115.5)$ | Captures severe rainfall events capable of overwhelming municipal stormwater drainage and low-lying deltaic depressions. |
| **`Very_Heavy_Rainfall`** | Binary | $\{0, 1\}$ | $\mathbb{I}(115.6 \le R_T \le 204.4)$ | Key precursor to riverine swelling in the Mahanadi, Baitarani, Brahmani, and Subarnarekha basins. |
| **`Extreme_Rainfall`** | Binary | $\{0, 1\}$ | $\mathbb{I}(R_T \ge 204.5)$ | Captures catastrophic cyclonic deluges (e.g. Cyclone Phailin, Titli, Fani, Amphan) that trigger emergency flood alerts. |
| **`Heavy_Rain_Indicator`** | Binary | $\{0, 1\}$ | $\mathbb{I}(R_T \ge 64.5)$ | Cumulative warning trigger: flags any day experiencing Heavy, Very Heavy, or Extremely Heavy rainfall ($R \ge 64.5\text{ mm}$). |
| **`Extreme_Rain_Indicator`** | Binary | $\{0, 1\}$ | $\mathbb{I}(R_T \ge 204.5)$ | Red Alert trigger: flags days crossing the highest statutory emergency classification. |

### Missing Data & Leakage Guarantees:
- **Missing Values**: If `Rainfall (mm)` is NaN or `Rainfall_Missing == 1`, all intensity indicators evaluate strictly to `0`. No positive alert indicator is ever fabricated for missing observations.
- **Corrupted Telemetry**: Any negative values ($R < 0$) are masked to `0`.
- **Temporal Causal Flow**: Features on observation day $T$ rely strictly on $R_T$. No future observations ($R_{T+1}, R_{T+2}$) are accessed.
- **Mutual Exclusivity**: The five discrete categories (`Light_Rainfall`, `Moderate_Rainfall`, `Heavy_Rainfall`, `Very_Heavy_Rainfall`, `Extreme_Rainfall`) are mutually exclusive. Their sum is $\le 1$ for every observation.

---

## 3. Core Spatio-Temporal & Baseline Features

### Identification & Spatial Attributes
| Feature Name | Type | Description | Values / Format |
| :--- | :---: | :--- | :--- |
| **`District`** | Categorical | Standard revenue district of Odisha. | 30 canonical uppercase names (e.g. `CUTTACK`, `BALASORE`). |
| **`Block/Station`** | Categorical | Revenue block or meteorological monitoring station. | 354 authentic administrative blocks in Odisha. |
| **`Date`** | String | Observation date. | `YYYY-MM-DD` formatted calendar string. |

### Temporal & Cyclical Attributes
| Feature Name | Type | Description | Range / Formula |
| :--- | :---: | :--- | :--- |
| **`Year`** | Integer | Calendar year of observation. | $2001 - 2025$ |
| **`Month_Number`** | Integer | Month of year. | $1 - 12$ |
| **`Day`** | Integer | Day of month. | $1 - 31$ |
| **`Day_of_Year`** | Integer | Ordinal day of the year. | $1 - 366$ |
| **`Day_of_Week`** | Integer | Day of week index (Monday = 0). | $0 - 6$ |
| **`Month_sin`** | Float | Sinusoidal cyclical encoding of month. | $\sin(2\pi \cdot \text{Month} / 12) \in [-1.0, 1.0]$ |
| **`Month_cos`** | Float | Cosine cyclical encoding of month. | $\cos(2\pi \cdot \text{Month} / 12) \in [-1.0, 1.0]$ |

### Current Day Precipitation
| Feature Name | Type | Description | Missing Value Treatment |
| :--- | :---: | :--- | :--- |
| **`Rainfall (mm)`** | Float | Total daily precipitation recorded on day $T$. | Imputed with 0.0; flagged by `Rainfall_Missing`. |
| **`Rainfall_Missing`** | Binary | Telemetry missingness indicator. | $1$ if raw observation was missing/null; else $0$. |

### Antecedent Lagged Precipitation
| Feature Name | Type | Description | Causal Implementation |
| :--- | :---: | :--- | :--- |
| **`Rainfall_Lag_1d`** | Float | Precipitation recorded on day $T-1$. | `grouped['Rainfall (mm)'].shift(1)` |
| **`Rainfall_Lag_2d`** | Float | Precipitation recorded on day $T-2$. | `grouped['Rainfall (mm)'].shift(2)` |
| **`Rainfall_Lag_3d`** | Float | Precipitation recorded on day $T-3$. | `grouped['Rainfall (mm)'].shift(3)` |
| **`Rainfall_Lag_7d`** | Float | Precipitation recorded on day $T-7$. | `grouped['Rainfall (mm)'].shift(7)` |

### Cumulative Antecedent Rolling Sums & Maxima
| Feature Name | Type | Window | Description | Mathematical Operation |
| :--- | :---: | :---: | :--- | :--- |
| **`Rainfall_Prev_3d_Sum`** | Float | 3 days | Sum of precipitation over $\{T-3, T-2, T-1\}$. | `shift(1).rolling(3).sum()` |
| **`Rainfall_Prev_7d_Sum`** | Float | 7 days | Sum of precipitation over $\{T-7, \dots, T-1\}$. | `shift(1).rolling(7).sum()` |
| **`Rainfall_Prev_15d_Sum`** | Float | 15 days | Sum of precipitation over $\{T-15, \dots, T-1\}$. | `shift(1).rolling(15).sum()` |
| **`Rainfall_Prev_30d_Sum`** | Float | 30 days | Sum of precipitation over $\{T-30, \dots, T-1\}$. | `shift(1).rolling(30).sum()` |
| **`Rainfall_Prev_3d_Max`** | Float | 3 days | Peak single-day precipitation over prior 3 days. | `shift(1).rolling(3).max()` |
| **`Rainfall_Prev_7d_Max`** | Float | 7 days | Peak single-day precipitation over prior 7 days. | `shift(1).rolling(7).max()` |
| **`Rainfall_Prev_15d_Max`** | Float | 15 days | Peak single-day precipitation over prior 15 days. | `shift(1).rolling(15).max()` |
| **`Rainfall_Prev_30d_Max`** | Float | 30 days | Peak single-day precipitation over prior 30 days. | `shift(1).rolling(30).max()` |

### Antecedent Wetness & Persistence Multipliers
| Feature Name | Type | Description | IMD Cutoff Applied |
| :--- | :---: | :--- | :--- |
| **`Rainy_Days_Prev_3d`** | Integer | Count of rainy days in prior 3-day window. | $\ge 2.5\text{ mm}$ on each day. |
| **`Rainy_Days_Prev_7d`** | Integer | Count of rainy days in prior 7-day window. | $\ge 2.5\text{ mm}$ on each day. |
| **`Rainy_Days_Prev_15d`** | Integer | Count of rainy days in prior 15-day window. | $\ge 2.5\text{ mm}$ on each day. |
| **`Rainy_Days_Prev_30d`** | Integer | Count of rainy days in prior 30-day window. | $\ge 2.5\text{ mm}$ on each day. |
| **`Consecutive_Rainy_Days_Before`** | Integer | Number of uninterrupted consecutive rainy days preceding day $T$. | Reset to 0 whenever daily rainfall falls below $2.5\text{ mm}$. |

---

## 4. Ground-Truth & Prediction Targets

| Target Name | Type | Role | Temporal Mechanics & Leakage Guardrail |
| :--- | :---: | :--- | :--- |
| **`Flood_Occurred`** | Binary | Same-day observed flood damage on day $T$. | Authoritative damage report from SRC. Used strictly as retrospective ground truth; set to 0 in unlabelled operational 2025 inference. |
| **`Flood_Next_Day`** | Binary | **Primary Prediction Target**: Flood inundation on day $T+1$. | Constructed via `shift(-1)` per station. **Strictly excluded from feature matrix.** Model uses observations at day $T$ to forecast $T+1$. |
