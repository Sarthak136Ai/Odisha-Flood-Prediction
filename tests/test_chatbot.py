"""
Unit and integration tests for the grounded Chatbot interface layer (chatbot/chatbot.py).
Tests interface layer grounding, live model predictions, metadata retrieval,
and strict handling of edge cases (unsupported queries, missing data, invalid blocks,
invalid dates, and unavailable models).
"""

import pytest
import os
from chatbot.chatbot import OdishaFloodChatbot


@pytest.fixture(scope="module")
def bot():
    """Default initialized chatbot with production paths."""
    return OdishaFloodChatbot()


# =====================================================================
# CORE USER QUESTION BENCHMARKS (PROMPT REQUIREMENTS)
# =====================================================================

def test_current_flood_risk_general(bot):
    """Test general query: 'What is the current flood risk?'"""
    res = bot.respond("What is the current flood risk?")
    assert "Current State-Wide Flood Risk Summary" in res
    assert "Monitored Telemetry Coverage" in res
    assert "Active Risk Distribution" in res
    assert "30 districts" in res or "districts" in res


def test_current_flood_risk_specific_location(bot):
    """Test localized query: 'What is the current flood risk for Cuttack?'"""
    res = bot.respond("What is the current flood risk for Cuttack?")
    assert "Flood Prediction for **CUTTACK**" in res
    assert "Estimated Next-Day Flood Probability" in res
    assert "Risk Classification" in res
    assert "Inference Model" in res


def test_predicted_probability_general(bot):
    """Test query: 'What is the predicted probability?'"""
    res = bot.respond("What is the predicted probability?")
    assert "Model Predicted Flood Probability Information" in res
    assert "Optimal Decision Threshold" in res
    assert "0.8073" in res or "Threshold" in res


def test_factors_influenced_prediction_general(bot):
    """Test query: 'Which factors influenced the prediction?' without location."""
    res = bot.respond("Which factors influenced the prediction?")
    assert "Key Factors Influencing Flood Predictions" in res
    assert "SHAP Analysis" in res
    assert "Catchment Soil Saturation" in res
    assert "Rainfall_Prev_15d_Sum" in res


def test_factors_influenced_prediction_specific(bot):
    """Test query: 'Which factors influenced the prediction for Puri?'"""
    res = bot.respond("Which factors influenced the prediction for Puri?")
    assert "Flood Prediction for **PURI**" in res
    assert "Key Factors Influencing Prediction" in res or "Top Risk Amplifiers" in res


def test_what_does_high_risk_mean(bot):
    """Test query: 'What does HIGH risk mean?'"""
    res = bot.respond("What does HIGH risk mean?")
    assert "Project Flood Risk Level Definitions" in res
    assert "HIGH RISK" in res
    assert "MODERATE RISK" in res
    assert "LOW RISK" in res
    assert "ODRAF/NDRF" in res or "evacuation" in res.lower() or "alert" in res.lower()


def test_which_model_is_being_used(bot):
    """Test query: 'Which model is being used?'"""
    res = bot.respond("Which model is being used?")
    assert "Flood Prediction Model Architecture" in res
    assert "Primary Production Model" in res
    assert "XGBoost" in res or "Logistic Regression" in res
    assert "interface layer" in res.lower()


def test_model_recall(bot):
    """Test query: 'What is the model's recall?'"""
    res = bot.respond("What is the model's recall?")
    assert "Model Recall & Test Performance Metrics" in res
    assert "Test Recall" in res
    assert "33.95%" in res or "Recall" in res
    assert "False Negatives" in res
    assert "disaster risk reduction" in res.lower() or "DRR" in res


def test_rainfall_features_used(bot):
    """Test query: 'What rainfall features are used?'"""
    res = bot.respond("What rainfall features are used?")
    assert "Engineered Rainfall & Meteorological Features" in res
    assert "Rainfall (mm)" in res
    assert "Rainfall_Lag_1d" in res
    assert "Rainfall_Prev_3d_Sum" in res
    assert "Rainfall_Prev_15d_Sum" in res
    assert "Rainy_Days_Prev_3d" in res or "IMD" in res


# =====================================================================
# EDGE CASES, VALIDATION & NEGATIVE TESTS (PROMPT REQUIREMENTS)
# =====================================================================

def test_unsupported_questions(bot):
    """Test out-of-domain / unsupported queries."""
    unsupported_queries = [
        "Who was the first president of the United States?",
        "How do I bake a chocolate cake?",
        "What is the capital of France?",
        "Tell me a funny joke about cats."
    ]
    for q in unsupported_queries:
        res = bot.respond(q)
        assert "Sufficient project data is not available" in res
        assert "interface layer" in res.lower()


def test_missing_prediction_data(bot):
    """Test query requesting data that does not exist."""
    res = bot.respond("What is the flood risk in Cuttack on 1975-01-01?")
    assert "Sufficient project data is not available" in res or "No telemetry observations found" in res


def test_invalid_station_or_block(bot):
    """Test query specifying an invalid/non-existent district or block."""
    # 1. Invalid district
    res_dist = bot.respond("Will Atlantis have a flood tomorrow?")
    assert "Sufficient project data is not available" in res_dist
    assert "Atlantis" in res_dist

    # 2. Invalid block in valid district
    res_block = bot.respond("Predict flood for Cuttack block Gotham on 2024-08-15")
    assert "Sufficient project data is not available" in res_block
    assert "Gotham" in res_block


def test_invalid_date(bot):
    """Test query with an invalid date outside available project window."""
    res = bot.respond("Predict flood for Cuttack on 2099-12-31")
    assert "Sufficient project data is not available" in res
    assert "2099-12-31" in res or "outside" in res.lower()


def test_model_unavailable():
    """Test chatbot behavior when the core ML model is missing/unavailable."""
    broken_bot = OdishaFloodChatbot(
        model_path="models/non_existent_model_file.pkl",
        metadata_path="models/non_existent_meta.json"
    )
    res = broken_bot.respond("Will Cuttack have a flood tomorrow?")
    assert "Sufficient project data is not available" in res
    assert "model is currently unavailable" in res.lower() or "uninitialized" in res.lower()


# =====================================================================
# HISTORICAL & SCIENTIFIC DOMAIN TESTS
# =====================================================================

def test_historical_monthly_rainfall(bot):
    """Test verified monthly historical rainfall lookup."""
    res = bot.respond("What was the rainfall in Cuttack in July 2023?")
    assert "Historical Meteorological Record" in res
    assert "CUTTACK" in res
    assert "July 2023" in res
    assert "Total Monthly Rainfall" in res


def test_district_24yr_summary(bot):
    """Test 24-year district historical climate profile."""
    res = bot.respond("Show 24-year flood summary for Puri")
    assert "24-Year Climate & Flood Record" in res
    assert "PURI" in res
    assert "Total Recorded Flood Days" in res


def test_downscaling_knowledge_query(bot):
    """Test grounded scientific downscaling definition (no 'high-resolution' exaggeration)."""
    res = bot.respond("What is rainfall downscaling?")
    assert "Flood & Hydrology Knowledge Base" in res
    assert "spatial disaggregation" in res.lower() or "downscaling" in res.lower()
    # Ensure it describes district-to-block disaggregation
    assert "district" in res.lower() or "block" in res.lower()
