"""Streamlit UI components package."""

from app.components.prediction import render_prediction_component
from app.components.map import render_odisha_map
from app.components.charts import render_climate_analytics, render_model_comparison_chart
from app.components.chatbot_ui import render_chatbot_component
from app.components.timeline import render_historical_timeline_component
from app.components.unseen_2025 import render_unseen_2025_component
from app.components.drilldown import render_drilldown_component
from app.components.what_if import render_what_if_simulator

__all__ = [
    "render_prediction_component",
    "render_odisha_map",
    "render_climate_analytics",
    "render_model_comparison_chart",
    "render_chatbot_component",
    "render_historical_timeline_component",
    "render_unseen_2025_component",
    "render_drilldown_component",
    "render_what_if_simulator"
]
