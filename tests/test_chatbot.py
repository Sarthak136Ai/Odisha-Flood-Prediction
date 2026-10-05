"""
Unit tests for Odisha Flood Chatbot routing, queries, and live model prediction interface.
"""

import pytest
from chatbot.chatbot import OdishaFloodChatbot


@pytest.fixture(scope="module")
def bot():
    return OdishaFloodChatbot()


def test_chatbot_prediction_intent(bot):
    res = bot.respond("Will Cuttack have a flood tomorrow?")
    assert "Flood Prediction for" in res
    assert "Estimated Next-Day Flood Probability" in res
    assert "Risk Classification" in res


def test_chatbot_historical_query(bot):
    res = bot.respond("What was the rainfall in Cuttack in July 2023?")
    assert "Historical Meteorological Query" in res
    assert "Total Monthly Rainfall" in res
    assert "CUTTACK" in res


def test_chatbot_district_summary(bot):
    res = bot.respond("Show 24-year flood summary for Puri")
    assert "24-Year Climate & Flood Summary for" in res
    assert "PURI" in res


def test_chatbot_knowledge_downscaling(bot):
    res = bot.respond("What is rainfall downscaling?")
    assert "Flood & Hydrology Knowledge Base" in res
    assert "fine-scale" in res.lower()


def test_chatbot_dataset_overview(bot):
    res = bot.respond("How many years of data are in the dataset?")
    assert "24-Year Dataset Architecture" in res
    assert "2,752,252" in res
