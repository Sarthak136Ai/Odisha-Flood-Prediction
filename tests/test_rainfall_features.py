"""
Unit tests for climatological anomaly calculation and IMD extreme rainfall indicators.
"""

import pandas as pd
import numpy as np
from src.features.rainfall_features import ClimatologicalAnomalyEngine, enrich_rainfall_features


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
    assert enriched["Is_Heavy_Rain"].tolist() == [0, 1, 1]
    assert enriched["Is_Very_Heavy_Rain"].tolist() == [0, 0, 1]
