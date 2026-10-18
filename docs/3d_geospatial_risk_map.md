# 3D Geospatial Flood Risk Atlas & Vertical Extrusion Engine
## Technical Methodology, Architecture & Implementation Guide

**Project:** Odisha Flood Intelligence & Early Warning System  
**Module:** Interactive 3D Geospatial Flood Risk Map (`/3d-risk-map`)  
**Technology Stack:** Flask, Jinja2, MapLibre GL JS (v3.6.2), Chart.js (v4.4.0), GeoJSON (EPSG:4326)

---

## 1. Executive Summary

The **3D Geospatial Flood Risk Atlas** is a fully integrated, interactive 3D geospatial visualization module for the Odisha Flood Early Warning platform. It renders a 3D choropleth extrusion mesh of Odisha's 30 standard administrative districts. Vertical polygon heights and color tiers are dynamically driven by real model predictions ($T+1$ next-day flood probabilities), actual daily precipitation, antecedent multi-day soil saturation metrics, and 24-year historical flood benchmarks without synthetic data fabrication or lookahead leakage.

```
                    ┌────────────────────────────────────────────────────────┐
                    │               User Frontend Interface                  │
                    │   (/3d-risk-map with MapLibre GL JS 3D WebGL Canvas)   │
                    └───────────────────────────┬────────────────────────────┘
                                                │ REST API Query:
                                                │ GET /api/3d-risk-data?date=YYYY-MM-DD&layer=...
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                                Flask Backend Core (app.py)                                 │
│                                                                                            │
│   ┌─────────────────────────────┐                    ┌─────────────────────────────────┐   │
│   │ 30-District GeoJSON Boundary│                    │  Live Model Prediction Caches   │   │
│   │ (WGS-84, Census Dist Codes) │                    │  (2025 Operational & Historical) │   │
│   └──────────────┬──────────────┘                    └────────────────┬────────────────┘   │
│                  │                                                    │                    │
│                  └───────────────────────┬────────────────────────────┘                    │
│                                          ▼                                                 │
│                     ┌───────────────────────────────────────────┐                          │
│                     │       Geospatial Enrichment Engine        │                          │
│                     │    (src/geospatial/spatial_risk.py)       │                          │
│                     │  - District name canonical normalization  │                          │
│                     │  - Dynamic 3D extrusion height scaling    │                          │
│                     │  - Risk tier & hex color classification   │                          │
│                     │  - Sub-district block station integration │                          │
│                     └────────────────────┬──────────────────────┘                          │
└──────────────────────────────────────────┼─────────────────────────────────────────────────┘
                                           │ Enriched GeoJSON + Summary Metadata
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Interactive 3D WebGL Canvas                                │
│   • 3D Extrusion Mesh (0 - 60,000m)     • Hover Card & Glassmorphic Tooltip                │
│   • Dynamic View Controls (Pitch, Orbit) • Detail Inspector Panel & SHAP Attribution       │
│   • Layer Switcher & Risk Filter        • Top-8 Vulnerability Chart & League Comparator    │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Geographic Boundary Dataset & Licensing

### 2.1 Geographic Source & Format
- **Boundary Dataset:** Official 30-district administrative boundaries of Odisha, India.
- **Source:** DataMeet Open Maps Community / Survey of India / Census of India administrative GIS boundaries (`EPSG:4326`, WGS-84 coordinate reference system).
- **Licensing:** Creative Commons Attribution 2.5 India (CC-BY 2.5 IN) / Open Data Commons Open Database License (ODbL).
- **Storage Locations in Repository:**
  - Standardized GeoJSON: [`data/geospatial/odisha_districts.geojson`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/data/geospatial/odisha_districts.geojson)
  - Static Web Asset: [`static/data/odisha_districts.geojson`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/static/data/odisha_districts.geojson)
  - Dedicated Raw Archive: [`data/geospatial/odisha_districts_raw.geojson`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/data/geospatial/odisha_districts_raw.geojson)

### 2.2 Bounding Box & Geometry Verification
All 30 administrative district boundaries have been computationally validated:
- **Longitude Range:** $[81.3830^\circ\text{ E}, 87.4827^\circ\text{ E}]$
- **Latitude Range:** $[17.8026^\circ\text{ N}, 22.5636^\circ\text{ N}]$
- **Geometry Type:** Strict `Polygon` and `MultiPolygon` representations with valid closed linear rings.

---

## 3. Joining Methodology: Geographic Features & Model Predictions

To prevent joining errors caused by inconsistent spelling transliterations across Indian administrative databases, the engine joins geographic features to prediction data using a deterministic, canonical district identifier mapping.

### 3.1 Transliteration Normalization
In [`src/data/combine_data.py`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/src/data/combine_data.py) and [`src/geospatial/spatial_risk.py`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/src/geospatial/spatial_risk.py), the `standardize_district_name()` engine maps historical, census, and telemetry transliterations into standard uppercase canonical names:

| Raw Census / GeoJSON Name | Standardized District ID (`STANDARD_DISTRICTS`) | Census District Code | Headquarters | Centroid ($\text{Lat}, \text{Lon}$) |
| :--- | :--- | :---: | :--- | :---: |
| `Baleshwar` | `BALASORE` | 378 | Balasore | $21.4934^\circ\text{ N}, 86.9135^\circ\text{ E}$ |
| `Debagarh` | `DEOGARH` | 374 | Deogarh | $21.5333^\circ\text{ N}, 84.7333^\circ\text{ E}$ |
| `Jagatsinghapur` | `JAGATSINGHPUR` | 390 | Jagatsinghpur | $20.2667^\circ\text{ N}, 86.1667^\circ\text{ E}$ |
| `Jajapur` | `JAJPUR` | 382 | Panikoili / Jajpur | $20.8500^\circ\text{ N}, 86.3333^\circ\text{ E}$ |
| `Kendujhar` | `KEONJHAR` | 376 | Kendujhar | $21.6300^\circ\text{ N}, 85.5800^\circ\text{ E}$ |
| `Nabarangapur` | `NAWARANGPUR` | 394 | Nabarangpur | $19.2300^\circ\text{ N}, 82.5500^\circ\text{ E}$ |
| `Subarnapur` / `Sonepur` | `SUBARNAPUR` | 387 | Sonepur | $20.8300^\circ\text{ N}, 83.9100^\circ\text{ E}$ |
| `Bolangir` / `Baragarh` | `BALANGIR` / `BARGARH` | 386 / 371 | Balangir / Bargarh | Coordinate Table Verified |

### 3.2 Join Execution Protocol
1. For any target date, the engine queries the predictions repository (`data/predictions/2025_flood_risk_predictions.csv` or `data/geospatial/historical_preset_predictions.csv`).
2. Telemetry stations are grouped by standardized `District`.
3. District-level aggregate statistics are calculated:
   - **Mean Probability:** $\bar{P}_{\text{district}} = \frac{1}{N_{\text{blocks}}} \sum_{i=1}^{N_{\text{blocks}}} P_i$
   - **Peak Probability:** $P_{\text{max}} = \max_{i} P_i$ with identifying station name `peak_block`.
   - **Mean Daily Rainfall:** $\bar{R}_{\text{district}} = \frac{1}{N_{\text{blocks}}} \sum_{i=1}^{N_{\text{blocks}}} R_i$
   - **Antecedent Wetness Accumulations:** 3-day, 7-day, 15-day, and 30-day sums.
   - **SHAP Model Attributions:** Top 3 global/local risk drivers.
4. Each GeoJSON feature property is enriched in place, baking the computed metrics into the feature prior to rendering.

---

## 4. 3D WebGL Visualization & Lighting Architecture

### 4.1 Technology Choice: MapLibre GL JS
The module utilizes **MapLibre GL JS (v3.6.2)**, an open-source, non-commercial WebGL mapping engine:
- **Zero API Key Requirements:** Runs without proprietary tokens or usage limits.
- **Native 3D Fill Extrusion:** Uses hardware-accelerated GPU polygon extrusion (`type: 'fill-extrusion'`).
- **Free Vector/Raster Tiles:** Integrates CARTO Dark Matter raster tiles (high-contrast disaster response basemap displaying rivers, coastlines, state borders, and roads).
- **Directional 3D Shading:** Configured with a directional sun light anchor (`viewport`, intensity: $0.62$, position: $[1.15, 210, 30]$), generating photorealistic wall shading and depth gradients.

### 4.2 Multi-Layer 3D Extrusion Scale
The map supports 4 data layers selectable from the toolbar. Each layer calculates extrusion heights strictly in meters:

| Extrusion Layer | Data Driver | Formula / Scale Range | Height Range ($\text{meters}$) |
| :--- | :--- | :--- | :---: |
| **Predicted Flood Probability (Default)** | Model $P(\text{Flood})$ | $H = P \times 55,000\text{ m} + 1,000\text{ m}$ (tiered up to $62,000\text{ m}$) | $1,000\text{ m} - 62,000\text{ m}$ |
| **Daily Rainfall** | 24-hr Precipitation | $H = \min\left(50,000, \frac{R_{\text{daily}}}{200\text{ mm}} \times 49,000 + 1,000\right)$ | $1,000\text{ m} - 50,000\text{ m}$ |
| **7-Day Antecedent Rainfall** | Prev 7-Day Cumulative | $H = \min\left(50,000, \frac{R_{\text{7d}}}{400\text{ mm}} \times 49,000 + 1,000\right)$ | $1,000\text{ m} - 50,000\text{ m}$ |
| **24-Year Historical Flood Rate** | 2001–2024 Frequency | $H = \min\left(50,000, \frac{\text{Rate}_{\%}}{6\%} \times 49,000 + 1,000\right)$ | $1,000\text{ m} - 50,000\text{ m}$ |

### 4.3 Risk Classification Color Hierarchy
Color styling adheres strictly to disaster management communication standards:
- 🟢 **LOW RISK ($P < 0.30$):** `#10b981` (Emerald Green)
- 🟡 **MODERATE RISK ($0.30 \le P < 0.70$):** `#eab308` (Amber Yellow)
- 🟠 **HIGH RISK ($0.70 \le P < 0.85$):** `#f97316` (Vibrant Orange)
- 🔴 **VERY HIGH RISK ($P \ge 0.85$):** `#ef4444` (Crimson Red)

---

## 5. User Interaction & Operational Features

1. **Hover Tooltip:** Floating glassmorphic card displaying District Name, Headquarters, Risk Tier, Predicted Flood Probability %, Daily Rainfall, 7-Day Accumulation, and Peak Risk Block.
2. **Click & Inspect:** Smooth camera fly-to towards the district centroid; populates the right-hand Inspector Panel with:
   - Full precipitation telemetry ($1\text{d}, 3\text{d}, 7\text{d}, 15\text{d}, 30\text{d}$)
   - Top 3 SHAP feature attributions
   - Sub-district block stations list
   - Historical disaster statistics (total days and rate)
   - Direct button link to [`/drilldown?district=...`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/templates/drilldown.html)
3. **Camera Controls:**
   - **Reset View:** Re-centers camera on Odisha ($20.45^\circ\text{ N}, 84.45^\circ\text{ E}$, zoom: $6.85$, pitch: $55^\circ$, bearing: $-12^\circ$).
   - **2D/3D Mode Toggle:** Transitions smoothly between 3D extrusion ($55^\circ$ pitch) and planar 2D choropleth ($0^\circ$ pitch).
   - **Continuous Orbit (Auto-Rotate):** Automatically rotates camera around the state for disaster monitoring displays.
   - **Basemap Switcher:** CARTO Dark Canvas (default), CARTO Light, OpenStreetMap, and Esri Satellite.
4. **Historical Replay Presets:**
   - `2025-08-15`: Peak Monsoon 2025 (Operational Telemetry)
   - `2025-07-25`: High Inundation Risk Event (Operational Telemetry)
   - `2025-09-08`: Late Monsoon Surge (Operational Telemetry)
   - `2022-08-15`: Mahanadi River Basin Mega-Flood (Historical Replay)
   - `2024-09-16`: Subarnarekha / Balasore Extreme Event (Historical Replay)
   - `2020-08-27`: Bhadrak / Coastal Inundation Episode (Historical Replay)
   - `2019-08-08`: Statewide Monsoon Depression (Historical Replay)
5. **District Comparison Drawer:** Side-by-side metric comparison between any two districts and Chart.js Top-8 vulnerability rankings.

---

## 6. Sub-District Block Coverage & Known Limitations

### 6.1 Current Sub-District Integration
- The system integrates **354 block-level telemetry stations** across all 30 districts.
- For every date, block-level observations and predictions are ingested, identifying the single highest-risk station (`peak_block`) for every district.

### 6.2 Known Geographical Data Limitations
- **3D Polygon Scale:** Extrusion meshes are currently rendered at the **district level** because authenticated, topologically validated public-domain block-level boundary shapefiles (covering all 354 blocks with contiguous non-overlapping polygons) are not currently available in open government repositories.
- **Station Drilldown Continuity:** To maintain scientific rigor and avoid approximating boundaries, individual station records, rainfall trends, and local SHAP attributions are accessible via the Station Drilldown module ([`/drilldown`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/templates/drilldown.html)).
- **Future Expansion:** When statutory block boundary shapefiles are published by the State Remote Sensing Application Centre (ORSAC), the ingestion engine is architected to accept block GeoJSON without modifying the MapLibre visualization pipeline.

---

## 7. Setup & Usage Instructions

### 7.1 Running the Application Locally
```bash
# 1. Activate Python Environment
python -m venv venv
venv\Scripts\activate

# 2. Install Dependencies (if not already installed)
pip install -r requirements.txt

# 3. Start the Flask Disaster Intelligence Server
python app.py
```
Open browser at: `http://127.0.0.1:5000/3d-risk-map`

### 7.2 Running Automated Test Suites
```bash
# Run 3D Geospatial Test Suite
pytest tests/test_3d_risk_map.py -v

# Run Full Route Regression Suite
pytest tests/test_routes.py -v

# Run All Unit & Integration Tests (167 tests)
pytest -q
```
