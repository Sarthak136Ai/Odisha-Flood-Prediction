"""
Unit tests for climatological anomaly calculation and IMD rainfall-intensity indicators.
Exhaustively tests IMD threshold boundaries, mutual exclusivity, missing data safety,
and integration with data validation.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.rainfall_features import (
    ClimatologicalAnomalyEngine,
    RainfallIntensityEngine,
    compute_rainfall_intensity_features,
    enrich_rainfall_features,
    RAINFALL_INTENSITY_COLUMNS,
    IMD_LIGHT_MIN,
    IMD_LIGHT_MAX,
    IMD_MODERATE_MIN,
    IMD_MODERATE_MAX,
    IMD_HEAVY_MIN,
    IMD_HEAVY_MAX,
    IMD_VERY_HEAVY_MIN,
    IMD_VERY_HEAVY_MAX,
    IMD_EXTREME_MIN
)
from src.data.validate_data import validate_rainfall_intensity_features


def test_climatological_anomaly_engine():
    mock_train = pd.DataFrame({
        "District": ["CUTTACK", "CUTTACK", "PURI", "PURI"],
        "Month_Number": [8, 8, 8, 8],
        "Rainfall (mm)": [20.0, 40.0, 10.0, 30.0]
    })
    
    engine = ClimatologicalAnomalyEngine()
    engine.fit(mock_train)
    
    mock_test = pd.DataFrame({
        "District": ["CUTTACK", "PURI"],
        "Month_Number": [8, 8],
        "Rainfall (mm)": [50.0, 15.0]
    })
    
    anomalies = engine.transform(mock_test)
    assert len(anomalies) == 2
    # Cuttack expected mean = 30.0 -> anomaly = 50.0 - 30.0 = +20.0
    assert np.isclose(anomalies.iloc[0], 20.0)
    # Puri expected mean = 20.0 -> anomaly = 15.0 - 20.0 = -5.0
    assert np.isclose(anomalies.iloc[1], -5.0)


def test_enrich_rainfall_features():
    df = pd.DataFrame({
        "Rainfall (mm)": [10.0, 75.0, 150.0],
        "Rainfall_Prev_7d_Sum": [70.0, 140.0, 210.0],
        "Rainfall_Prev_30d_Sum": [300.0, 600.0, 900.0]
    })
    
    enriched = enrich_rainfall_features(df)
    assert "Rainfall_Prev_7d_Avg" in enriched.columns
    assert "Rainfall_Prev_30d_Avg" in enriched.columns
    assert "Is_Heavy_Rain" in enriched.columns
    assert "Is_Very_Heavy_Rain" in enriched.columns
    assert "Light_Rainfall" in enriched.columns
    assert "Heavy_Rainfall" in enriched.columns
    assert "Very_Heavy_Rainfall" in enriched.columns
    assert "Heavy_Rain_Indicator" in enriched.columns
    assert enriched["Is_Heavy_Rain"].tolist() == [0, 1, 1]
    assert enriched["Is_Very_Heavy_Rain"].tolist() == [0, 0, 1]
    assert enriched["Heavy_Rain_Indicator"].tolist() == [0, 1, 1]


def test_imd_intensity_threshold_boundaries():
    """
    Exhaustively verify all exact IMD statutory threshold boundary points:
    - 0.0 mm: Dry
    - 2.4 mm: Trace / Very Light (under rainy-day threshold)
    - 2.5 mm: Light Rainfall onset
    - 15.5 mm: Light Rainfall upper bound
    - 15.6 mm: Moderate Rainfall onset
    - 64.4 mm: Moderate Rainfall upper bound
    - 64.5 mm: Heavy Rainfall onset (Heavy Rain Warning trigger)
    - 115.5 mm: Heavy Rainfall upper bound
    - 115.6 mm: Very Heavy Rainfall onset
    - 204.4 mm: Very Heavy Rainfall upper bound
    - 204.5 mm: Extremely Heavy Rainfall onset (Red Alert trigger)
    - 320.0 mm: Extreme Cyclonic Deluge
    """
    test_values = [
        0.0, 1.2, 2.4, 2.5, 10.0, 15.5, 15.6, 40.0, 64.4,
        64.5, 90.0, 115.5, 115.6, 160.0, 204.4, 204.5, 320.0
    ]
    df = pd.DataFrame({"Rainfall (mm)": test_values})
    res = compute_rainfall_intensity_features(df)

    # 1. Dry & Trace (< 2.5 mm)
    for idx in [0, 1, 2]:
        row = res.iloc[idx]
        assert row["Light_Rainfall"] == 0
        assert row["Moderate_Rainfall"] == 0
        assert row["Heavy_Rainfall"] == 0
        assert row["Very_Heavy_Rainfall"] == 0
        assert row["Extreme_Rainfall"] == 0
        assert row["Heavy_Rain_Indicator"] == 0
        assert row["Extreme_Rain_Indicator"] == 0

    # 2. Light Rainfall (2.5 - 15.5 mm)
    for idx in [3, 4, 5]:
        row = res.iloc[idx]
        assert row["Light_Rainfall"] == 1
        assert row["Moderate_Rainfall"] == 0
        assert row["Heavy_Rainfall"] == 0
        assert row["Heavy_Rain_Indicator"] == 0

    # 3. Moderate Rainfall (15.6 - 64.4 mm)
    for idx in [6, 7, 8]:
        row = res.iloc[idx]
        assert row["Light_Rainfall"] == 0
        assert row["Moderate_Rainfall"] == 1
        assert row["Heavy_Rainfall"] == 0
        assert row["Heavy_Rain_Indicator"] == 0

    # 4. Heavy Rainfall (64.5 - 115.5 mm)
    for idx in [9, 10, 11]:
        row = res.iloc[idx]
        assert row["Light_Rainfall"] == 0
        assert row["Moderate_Rainfall"] == 0
        assert row["Heavy_Rainfall"] == 1
        assert row["Very_Heavy_Rainfall"] == 0
        assert row["Heavy_Rain_Indicator"] == 1
        assert row["Extreme_Rain_Indicator"] == 0

    # 5. Very Heavy Rainfall (115.6 - 204.4 mm)
    for idx in [12, 13, 14]:
        row = res.iloc[idx]
        assert row["Heavy_Rainfall"] == 0
        assert row["Very_Heavy_Rainfall"] == 1
        assert row["Extreme_Rainfall"] == 0
        assert row["Heavy_Rain_Indicator"] == 1
        assert row["Extreme_Rain_Indicator"] == 0
        assert row["Is_Very_Heavy_Rain"] == 1

    # 6. Extremely Heavy Rainfall (>= 204.5 mm)
    for idx in [15, 16]:
        row = res.iloc[idx]
        assert row["Very_Heavy_Rainfall"] == 0
        assert row["Extreme_Rainfall"] == 1
        assert row["Heavy_Rain_Indicator"] == 1
        assert row["Extreme_Rain_Indicator"] == 1


def test_intensity_mutual_exclusivity():
    """Assert discrete intensity categories sum to <= 1 for all continuous values."""
    rain_samples = np.linspace(0.0, 300.0, 601)
    df = pd.DataFrame({"Rainfall (mm)": rain_samples})
    res = compute_rainfall_intensity_features(df)
    
    categories = ["Light_Rainfall", "Moderate_Rainfall", "Heavy_Rainfall", "Very_Heavy_Rainfall", "Extreme_Rainfall"]
    category_sums = res[categories].sum(axis=1)
    
    assert (category_sums <= 1).all(), "Found multiple active mutually exclusive categories!"
    # Values >= 2.5 mm must activate exactly 1 category
    rainy_mask = df["Rainfall (mm)"] >= 2.5
    assert (category_sums[rainy_mask] == 1).all(), "Rainy observation (>= 2.5 mm) failed to activate any intensity category!"


def test_missing_and_negative_rainfall_handling():
    """Verify that missing (NaN), negative, and masked values result strictly in 0 across all flags."""
    df = pd.DataFrame({
        "Rainfall (mm)": [np.nan, None, -50.0, -999.0, 100.0],
        "Rainfall_Missing": [1, 1, 0, 0, 1]  # 100mm is flagged as missing
    })
    res = compute_rainfall_intensity_features(df)
    
    # All 5 rows should evaluate strictly to 0
    for col in RAINFALL_INTENSITY_COLUMNS:
        assert (res[col] == 0).all(), f"Found non-zero flags in '{col}' for invalid/missing rainfall!"


def test_single_scalar_classification():
    """Verify classify_single_rainfall works seamlessly for single scalars and edge cases."""
    engine = RainfallIntensityEngine()
    
    # Missing / None
    none_res = engine.classify_single_rainfall(None)
    assert none_res["Light_Rainfall"] == 0
    assert none_res["Heavy_Rain_Indicator"] == 0
    
    # Negative
    neg_res = engine.classify_single_rainfall(-10.0)
    assert neg_res["Moderate_Rainfall"] == 0
    
    # Moderate
    mod_res = engine.classify_single_rainfall(35.0)
    assert mod_res["Moderate_Rainfall"] == 1
    assert mod_res["Light_Rainfall"] == 0
    assert mod_res["Heavy_Rain_Indicator"] == 0
    
    # Extreme
    ext_res = engine.classify_single_rainfall(250.0)
    assert ext_res["Extreme_Rainfall"] == 1
    assert ext_res["Heavy_Rain_Indicator"] == 1
    assert ext_res["Extreme_Rain_Indicator"] == 1


def test_data_validation_suite_compliance():
    """Verify generated DataFrame passes the master validation suite."""
    df = pd.DataFrame({
        "Rainfall (mm)": [0.0, 5.0, 25.0, 80.0, 150.0, 220.0, np.nan],
        "Rainfall_Missing": [0, 0, 0, 0, 0, 0, 1]
    })
    enriched = compute_rainfall_intensity_features(df)
    
    val_report = validate_rainfall_intensity_features(enriched)
    assert val_report["is_valid"] is True
    assert len(val_report["errors"]) == 0
    assert len(val_report["passed_checks"]) > 0
