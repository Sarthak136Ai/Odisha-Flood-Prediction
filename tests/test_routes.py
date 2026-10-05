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
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_homepage_route(client):
    """Test dashboard / homepage route."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Odisha Flood Intel" in response.data or b"Command Center" in response.data


def test_dashboard_route(client):
    """Test /dashboard route."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Monitored Districts" in response.data


def test_risk_map_route(client):
    """Test /risk-map route."""
    response = client.get("/risk-map")
    assert response.status_code == 200
    assert b"Odisha Geospatial Risk" in response.data or b"Atlas" in response.data


def test_drilldown_route(client):
    """Test /drilldown route."""
    response = client.get("/drilldown")
    assert response.status_code == 200
    assert b"Station & Observation Selector" in response.data


def test_historical_route(client):
    """Test /historical route."""
    response = client.get("/historical")
    assert response.status_code == 200
    assert b"Historical Flood Timeline" in response.data


def test_forecaster_route(client):
    """Test /forecaster route."""
    response = client.get("/forecaster")
    assert response.status_code == 200
    assert b"Real-Time Flood Early Warning Forecaster" in response.data


def test_simulator_route(client):
    """Test /simulator route."""
    response = client.get("/simulator")
    assert response.status_code == 200
    assert b"What-If" in response.data


def test_unseen_2025_route(client):
    """Test /unseen-2025 route."""
    response = client.get("/unseen-2025")
    assert response.status_code == 200
    assert b"2025 Unseen Operational Flood Risk Monitor" in response.data


def test_benchmarks_route(client):
    """Test /benchmarks route."""
    response = client.get("/benchmarks")
    assert response.status_code == 200
    assert b"Model Benchmarks" in response.data


def test_explainability_route(client):
    """Test /explainability route."""
    response = client.get("/explainability")
    assert response.status_code == 200
    assert b"SHAP" in response.data


def test_downscaling_route(client):
    """Test /downscaling route."""
    response = client.get("/downscaling")
    assert response.status_code == 200
    assert b"Downscaling" in response.data


def test_assistant_route(client):
    """Test /assistant route."""
    response = client.get("/assistant")
    assert response.status_code == 200
    assert b"AI Flood Assistant" in response.data


def test_architecture_route(client):
    """Test /architecture route."""
    response = client.get("/architecture")
    assert response.status_code == 200
    assert b"System Architecture" in response.data


def test_api_district_risk(client):
    """Test /api/district-risk endpoint."""
    response = client.get("/api/district-risk")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert len(json_data["districts"]) == 30


def test_api_blocks(client):
    """Test /api/blocks/<district> endpoint."""
    response = client.get("/api/blocks/CUTTACK")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert "Banki" in json_data["blocks"] or len(json_data["blocks"]) > 0


def test_api_station_data(client):
    """Test /api/station-data endpoint."""
    response = client.get("/api/station-data?district=CUTTACK&block=Banki&date=2025-08-15")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert "data" in json_data
    assert "factors" in json_data


def test_api_system_status(client):
    """Test /api/system-status endpoint."""
    response = client.get("/api/system-status")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "operational"
    assert json_data["predictor_ready"] is True
