"""
Automated Test Suite for 3D Geospatial Flood Risk Atlas.
Verifies:
1. /3d-risk-map page route and DOM components.
2. /api/3d-risk-data REST endpoint structure and metadata.
3. GeoJSON schema, CRS, and geometry validity (30 districts).
4. Standard district name and census code alignment.
5. Missing prediction handling and graceful defaults.
6. Invalid and empty date parameter resilience.
7. Empty prediction/historical DataFrame safety.
8. Risk tier classification boundaries and hex color mappings.
9. Multi-layer extrusion height scaling (probability, rainfall, antecedent, historical).
10. Historical disaster preset integration and raw GeoJSON endpoint.
"""

import os
import json
import importlib.util
import pytest
import pandas as pd
import numpy as np

# Load Flask app explicitly
_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
_spec = importlib.util.spec_from_file_location("root_flask_app", _app_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
app = _mod.app

from src.data.combine_data import STANDARD_DISTRICTS
from src.geospatial.spatial_risk import (
    load_odisha_districts_geojson,
    classify_flood_risk,
    build_3d_risk_geojson,
    ODISHA_DISTRICT_COORDINATES
)


@pytest.fixture
def client():
    """Create Flask test client in testing mode."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_3d_risk_map_page_route(client):
    """Verify /3d-risk-map route returns HTTP 200 and all essential 3D UI elements."""
    response = client.get("/3d-risk-map")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Odisha 3D Geospatial Flood Risk Atlas" in html
    assert "odisha-3d-map" in html
    assert "mapLibre" in html or "maplibre" in html
    assert "dataLayerSelect" in html
    assert "datePicker" in html
    assert "leagueChartCanvas" in html
    assert "leagueTable" in html
    assert "inspectorDistName" in html


def test_api_3d_risk_data_schema(client):
    """Verify /api/3d-risk-data returns a valid GeoJSON FeatureCollection and summary."""
    response = client.get("/api/3d-risk-data?date=2025-08-15&layer=probability")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    
    # GeoJSON validation
    geojson = data["geojson"]
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 30
    
    # Summary validation
    summary = data["summary"]
    assert summary["selected_date"] == "2025-08-15"
    assert summary["forecast_horizon"] == "T+1 (Next-Day Flood Inundation)"
    assert summary["total_districts"] == 30
    assert "highest_risk_district" in summary
    assert "risk_breakdown" in summary
    assert "league_table" in summary
    assert len(summary["league_table"]) == 30


def test_geojson_geometry_and_spatial_bounds():
    """Verify GeoJSON geometries are valid Polygons/MultiPolygons within Odisha's bounding box."""
    data = load_odisha_districts_geojson()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 30

    for feat in data["features"]:
        assert feat["type"] == "Feature"
        geom = feat["geometry"]
        assert geom["type"] in ["Polygon", "MultiPolygon"]
        coords = geom["coordinates"]
        assert len(coords) > 0

        # Extract all vertex points and verify bounding box
        def check_coords(c):
            if isinstance(c[0], (int, float)):
                lon, lat = c[0], c[1]
                assert 81.0 <= lon <= 88.0, f"Longitude {lon} out of Odisha bounds"
                assert 17.5 <= lat <= 23.0, f"Latitude {lat} out of Odisha bounds"
            else:
                for sub in c:
                    check_coords(sub)
        check_coords(coords)


def test_district_identifier_matching():
    """Verify all 30 district features match STANDARD_DISTRICTS and have centroid coordinates."""
    data = load_odisha_districts_geojson()
    dist_names = [f["properties"]["district_name"] for f in data["features"]]
    assert set(dist_names) == set(STANDARD_DISTRICTS)
    assert len(dist_names) == 30

    for feat in data["features"]:
        p = feat["properties"]
        d = p["district_name"]
        assert d in ODISHA_DISTRICT_COORDINATES
        assert "headquarters" in p
        assert "centroid_lat" in p
        assert "centroid_lon" in p
        assert 17.5 <= p["centroid_lat"] <= 23.0
        assert 81.0 <= p["centroid_lon"] <= 88.0


def test_risk_category_assignment_and_color_scale():
    """Verify risk classification tiers, color mappings, and height scales."""
    # Low Risk (< 0.30)
    tier, color, h = classify_flood_risk(0.15)
    assert tier == "LOW"
    assert color == "#10b981"
    assert h > 0

    # Moderate Risk (0.30 to < 0.70)
    tier, color, h = classify_flood_risk(0.50)
    assert tier == "MODERATE"
    assert color == "#eab308"
    assert h > 0

    # High Risk (0.70 to < 0.85)
    tier, color, h = classify_flood_risk(0.78)
    assert tier == "HIGH"
    assert color == "#f97316"
    assert h > 0

    # Very High Risk (>= 0.85)
    tier, color, h = classify_flood_risk(0.92)
    assert tier == "VERY HIGH"
    assert color == "#ef4444"
    assert h > 0

    # Bounds protection
    _, _, h_neg = classify_flood_risk(-0.5)
    _, _, h_huge = classify_flood_risk(1.5)
    assert h_neg >= 1000.0
    assert h_huge <= 62000.0


def test_missing_prediction_values_graceful_fallback():
    """Verify system handles missing prediction rows by falling back to historical baseline without NaN."""
    # Create empty predictions DataFrame
    empty_df = pd.DataFrame(columns=["District", "Date", "Predicted_Probability"])
    
    geojson, summary = build_3d_risk_geojson(
        date_str="2025-08-15",
        cached_2025_df=empty_df,
        district_risk_df=None,
        data_layer="probability"
    )
    
    assert len(geojson["features"]) == 30
    for feat in geojson["features"]:
        p = feat["properties"]
        assert not np.isnan(p["predicted_probability"])
        assert not np.isnan(p["extrusion_height"])
        assert p["risk_level"] in ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
        assert p["extrusion_height"] >= 1000.0


def test_invalid_date_parameter_handling():
    """Verify system handles invalid or missing date parameters safely."""
    # Test invalid string date
    geojson, summary = build_3d_risk_geojson("invalid-date-9999", cached_2025_df=None)
    assert len(geojson["features"]) == 30
    assert summary["total_districts"] == 30

    # Test empty string date
    geojson2, summary2 = build_3d_risk_geojson("", cached_2025_df=None)
    assert len(geojson2["features"]) == 30
    assert summary2["total_districts"] == 30


def test_multi_layer_extrusion_scaling():
    """Verify extrusion heights scale properly across different physical layers."""
    layers = ["probability", "rainfall", "antecedent_rainfall", "historical_risk"]
    for l in layers:
        geojson, summary = build_3d_risk_geojson("2025-08-15", data_layer=l)
        assert summary["data_layer"] == l
        assert len(geojson["features"]) == 30
        for f in geojson["features"]:
            h = f["properties"]["extrusion_height"]
            assert h >= 1000.0
            assert h <= 65000.0
            assert not np.isnan(h)


def test_historical_disaster_preset_integration():
    """Verify historical disaster replay dates return authentic historical context."""
    preset_dates = ["2022-08-15", "2024-09-16", "2020-08-27", "2019-08-08"]
    for d in preset_dates:
        geojson, summary = build_3d_risk_geojson(d, data_layer="probability")
        assert len(geojson["features"]) == 30
        assert summary["selected_date"] == d
        assert "Historical" in summary["data_source"]
        assert summary["highest_risk_district"]["name"] in STANDARD_DISTRICTS


def test_api_raw_geojson_endpoint(client):
    """Verify /api/geospatial/odisha-districts.geojson endpoint."""
    response = client.get("/api/geospatial/odisha-districts.geojson")
    assert response.status_code == 200
    data = response.get_json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 30
