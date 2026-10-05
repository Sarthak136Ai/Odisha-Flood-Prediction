import os
import sys
import importlib.util
import pytest

# Load root app.py explicitly to avoid collision with app/ directory
_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
_spec = importlib.util.spec_from_file_location("root_flask_app", _app_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
app = _mod.app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_predict_endpoint_high_risk(client):
    """Test /api/predict with heavy monsoon rainfall values."""
    payload = {
        "district": "CUTTACK",
        "block": "Banki",
        "date": "2025-08-15",
        "rainfall": 140.0,
        "lag_1d": 90.0,
        "lag_2d": 60.0,
        "rf_3d": 290.0,
        "rf_7d": 450.0,
        "rf_15d": 620.0,
        "rf_30d": 800.0,
        "rainy_7d": 6,
        "rainy_30d": 24,
        "consec_rainy": 5
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert "flood_probability" in json_data
    assert "risk_level" in json_data
    assert json_data["flood_probability"] > 0.50
    assert len(json_data["top_contributing_factors"]) > 0


def test_api_predict_endpoint_low_risk(client):
    """Test /api/predict with minimal dry weather rainfall values."""
    payload = {
        "district": "CUTTACK",
        "block": "Banki",
        "date": "2025-01-15",
        "rainfall": 0.0,
        "lag_1d": 0.0,
        "lag_2d": 0.0,
        "rf_3d": 0.0,
        "rf_7d": 0.0,
        "rf_15d": 0.0,
        "rf_30d": 0.0,
        "rainy_7d": 0,
        "rainy_30d": 0,
        "consec_rainy": 0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["flood_probability"] < 0.10
    assert json_data["risk_level"] == "LOW"


def test_api_simulate_endpoint(client):
    """Test /api/simulate endpoint."""
    payload = {
        "baseline_rainfall": 20.0,
        "simulated_rainfall": 120.0,
        "month": 8
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["simulated_probability"] > json_data["baseline_probability"]
    assert len(json_data["sensitivity_curve"]) > 0


def test_api_assistant_query_prediction(client):
    """Test /api/assistant/query with forecast question."""
    payload = {"query": "Will Cuttack have a flood tomorrow?"}
    response = client.post("/api/assistant/query", json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert "CUTTACK" in json_data["response"] or "Probability" in json_data["response"] or "Flood" in json_data["response"]


def test_api_assistant_query_historical(client):
    """Test /api/assistant/query with historical query."""
    payload = {"query": "Tell me about August 2020 in Bhadrak"}
    response = client.post("/api/assistant/query", json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert "Bhadrak" in json_data["response"] or "Rainfall" in json_data["response"]
