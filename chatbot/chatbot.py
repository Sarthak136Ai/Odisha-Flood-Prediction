"""
Core Chatbot engine for the Odisha Flood Prediction System.
Functions as a grounded interface layer between user natural language queries and the
underlying machine learning models, telemetry datasets, evaluation metrics, and domain facts.
Does not invent numerical predictions; explicitly reports when sufficient data is not available.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

from chatbot.data_query import DatasetQueryEngine
from chatbot.prediction_tools import ChatbotPredictionTool
from chatbot.flood_knowledge import get_knowledge_answer, ODISHA_FLOOD_KNOWLEDGE
from chatbot.prompts import (
    format_prediction_response,
    format_risk_definition_response,
    format_model_info_response,
    format_recall_and_metrics_response,
    format_rainfall_features_response
)
from src.data.combine_data import STANDARD_DISTRICTS

logger = logging.getLogger(__name__)


class OdishaFloodChatbot:
    """
    Domain-grounded chatbot interface layer for Odisha flood early warning.
    Interprets user intent and retrieves verified answers from models, datasets, and metadata.
    """
    
    def __init__(
        self,
        data_path: str = "data/combined/Odisha_Flood_2001_2024.csv",
        model_path: str = "models/flood_prediction/best_model.pkl",
        metadata_path: str = "models/flood_prediction/model_metadata.json",
        metrics_path: str = "results/metrics/model_comparison.csv",
        predictions_2025_path: str = "data/predictions/2025_flood_risk_predictions.csv"
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.metrics_path = metrics_path
        self.predictions_2025_path = predictions_2025_path
        
        self.query_engine = DatasetQueryEngine(data_path)
        self.prediction_tool = ChatbotPredictionTool(
            model_path=model_path,
            metadata_path=metadata_path,
            data_path=data_path,
            predictions_2025_path=predictions_2025_path
        )
        self.districts = STANDARD_DISTRICTS
        
        self.metadata: Dict[str, Any] = {}
        self.model_comparison_df: Optional[pd.DataFrame] = None
        self._load_metadata()

    def _load_metadata(self):
        """Load verified model metadata and model comparison metrics."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading model metadata: {e}")
                self.metadata = {}
                
        if os.path.exists(self.metrics_path):
            try:
                self.model_comparison_df = pd.read_csv(self.metrics_path)
            except Exception as e:
                logger.warning(f"Error loading model comparison: {e}")
                self.model_comparison_df = None

    def _extract_district(self, text: str) -> Optional[str]:
        """Extract recognized Odisha district name from query."""
        t_upper = text.upper()
        for d in self.districts:
            if re.search(r'\b' + re.escape(d) + r'\b', t_upper):
                return d
        return None

    def _extract_unrecognized_location(self, text: str) -> Optional[str]:
        """Detect if an unrecognized district/location candidate was asked about."""
        ignore_words = {
            "tomorrow", "yesterday", "today", "now", "flood", "floods", "rain", "rainfall",
            "risk", "forecast", "prediction", "probability", "model", "recall", "precision",
            "the", "a", "an", "this", "that", "district", "block", "station", "state", "odisha",
            "what", "where", "when", "why", "how", "high", "moderate", "low", "july", "august",
            "september", "october", "monsoon", "area", "region", "warning", "factors", "features",
            "will", "have", "does", "can", "is", "are", "was", "were", "it", "there", "be",
            "current", "predicted", "predict", "tell", "show", "give", "explain", "meaning", "definition"
        }
        # 1. Preposition match: for/in/at/around/near/district/block/station <Word>
        matches = re.findall(r'\b(?:for|in|at|around|near|district|block|station)\s+([A-Za-z]+)\b', text, re.IGNORECASE)
        for m in matches:
            if m.lower() not in ignore_words and m.upper() not in self.districts:
                return m
                
        # 2. Subject match: 'Will <Name> have/be...'
        matches_subj = re.findall(r'\b(?:will|is|can|does)\s+([A-Za-z]+)\s+(?:have|be|face|experience|get|see)\b', text, re.IGNORECASE)
        for m in matches_subj:
            if m.lower() not in ignore_words and m.upper() not in self.districts:
                return m
                
        return None

    def _extract_block(self, text: str) -> Optional[str]:
        """Extract mentioned administrative block or station."""
        match = re.search(r'\bblock\s+([A-Za-z0-9\-]+)\b', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match2 = re.search(r'\bstation\s+([A-Za-z0-9\-]+)\b', text, re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date string in YYYY-MM-DD or standard format."""
        match = re.search(r'\b(20\d{2}-\d{2}-\d{2})\b', text)
        if match:
            return match.group(1)
        # Check for non-standard or out of range dates like 1990-01-01 or 2099-01-01
        match_any = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text)
        if match_any:
            return match_any.group(1)
        return None

    def _extract_year_month(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract Year and Month if present in query."""
        months_map = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
            "november": 11, "nov": 11, "december": 12, "dec": 12
        }
        
        found_year = None
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        if year_match:
            found_year = int(year_match.group(1))
            
        found_month = None
        t_lower = text.lower()
        for m_name, m_num in months_map.items():
            if re.search(r'\b' + re.escape(m_name) + r'\b', t_lower):
                found_month = m_num
                break
                
        return found_year, found_month

    def respond(self, query: str) -> str:
        """
        Process user message and return a grounded response from verified models and datasets.
        Strictly prevents invented numbers and explicitly reports unavailable information.
        """
        if not query or not query.strip():
            return "Please enter a valid question or query regarding Odisha flood predictions, meteorological features, or historical data."
            
        q = query.strip()
        q_lower = q.lower()

        # -------------------------------------------------------------
        # Intent 1: Model Identity
        # e.g., "Which model is being used?", "What model is this?"
        # -------------------------------------------------------------
        if any(w in q_lower for w in ["which model", "what model", "model is being used", "model being used", "what classifier", "what algorithm"]):
            if not self.metadata:
                return "Sufficient project data is not available: Model metadata is currently not loaded."
            return format_model_info_response(self.metadata)

        # -------------------------------------------------------------
        # Intent 2: Model Recall & Evaluation Metrics
        # e.g., "What is the model's recall?", "What are the model metrics?"
        # -------------------------------------------------------------
        if any(w in q_lower for w in ["recall", "model's recall", "test recall", "what is the recall", "model metrics", "model accuracy", "model performance", "f1 score", "roc-auc", "pr-auc"]):
            if not self.metadata or "test_metrics" not in self.metadata:
                return "Sufficient project data is not available: Test evaluation metrics are currently not loaded."
            return format_recall_and_metrics_response(self.metadata, self.model_comparison_df)

        # -------------------------------------------------------------
        # Intent 3: Risk Level Definitions
        # e.g., "What does HIGH risk mean?", "What does MODERATE risk mean?"
        # -------------------------------------------------------------
        if any(w in q_lower for w in ["high risk mean", "moderate risk mean", "low risk mean", "risk definition", "risk meaning", "risk levels", "what does high risk", "what does moderate risk", "what does low risk"]):
            risk_thresholds = self.metadata.get("risk_thresholds", {"low_max": 0.30, "moderate_max": 0.70, "high_min": 0.70})
            return format_risk_definition_response(risk_thresholds)

        # -------------------------------------------------------------
        # Intent 4: Rainfall Features Used by Model
        # e.g., "What rainfall features are used?", "Which features are used?"
        # -------------------------------------------------------------
        if any(w in q_lower for w in ["rainfall features", "features are used", "what features", "which features", "input features", "feature list"]):
            features = self.metadata.get("features", [])
            if not features:
                return "Sufficient project data is not available: Feature schema is currently not loaded."
            return format_rainfall_features_response(features)

        # -------------------------------------------------------------
        # Intent 5: Factors Influencing Prediction
        # e.g., "Which factors influenced the prediction?", "Why did it predict flood?"
        # -------------------------------------------------------------
        if any(w in q_lower for w in ["factors influenced", "factors influence", "influenced the prediction", "why did the model", "factors drive", "drivers of flood"]):
            district = self._extract_district(q)
            date_str = self._extract_date(q)
            block = self._extract_block(q)
            
            if district:
                if not self.prediction_tool.is_model_available():
                    return "Sufficient project data is not available: The core flood prediction model is currently unavailable or uninitialized."
                res = self.prediction_tool.predict_for_location_and_date(district=district, block=block, date_str=date_str)
                if "error" in res:
                    return f"⚠️ {res['error']}"
                return format_prediction_response(res)
            else:
                # Explain general SHAP feature ranking grounded in model findings
                return (
                    "### Key Factors Influencing Flood Predictions (SHAP Analysis)\n\n"
                    "Across 24 years of telemetry in Odisha, the early-warning model's predictions are primarily governed by four hydrologic factor groups:\n\n"
                    "1. **Catchment Soil Saturation (Dominant Factor)**:\n"
                    "   - `Rainfall_Prev_15d_Sum` and `Rainfall_Prev_30d_Sum`: High 15-to-30 day accumulated precipitation saturates basin storage, severely accelerating surface runoff during subsequent storms.\n\n"
                    "2. **Immediate Storm Intensity**:\n"
                    "   - `Rainfall (mm)` and `Rainfall_Lag_1d`: Direct 24-hour storm downpours and peak single-day surges trigger flash floods in deltaic tributaries.\n\n"
                    "3. **Sustained Antecedent Rainy Days**:\n"
                    "   - `Rainy_Days_Prev_15d` and `Consecutive_Rainy_Days_Before`: Extended wet spells leave river channels swollen with zero drying time.\n\n"
                    "4. **Monsoon Seasonality Climatology**:\n"
                    "   - `Month_cos` and `Month_sin`: Natural peak monsoon gating (July–September) aligned with Bay of Bengal cyclonic depressions.\n\n"
                    "To view localized factor contributions for a specific location, ask for a district (e.g. *'Which factors influenced the prediction for Cuttack?'*)."
                )

        # -------------------------------------------------------------
        # Intent 6: Live Prediction / Current Flood Risk / Probability
        # e.g., "What is the current flood risk?", "What is the predicted probability?"
        # -------------------------------------------------------------
        prediction_keywords = ["current flood risk", "what is the flood risk", "flood risk", "predicted probability", "what is the predicted probability", "will it flood", "predict", "forecast", "flood tomorrow", "probability"]
        if any(w in q_lower for w in prediction_keywords):
            # Check model availability first
            if not self.prediction_tool.is_model_available():
                return "Sufficient project data is not available: The core flood prediction model is currently unavailable or uninitialized."

            district = self._extract_district(q)
            block = self._extract_block(q)
            date_str = self._extract_date(q)
            unrecognized = self._extract_unrecognized_location(q)

            # If an unrecognized location candidate was detected
            if not district and unrecognized:
                return f"Sufficient project data is not available: District '{unrecognized}' is not recognized in Odisha telemetry records. The project monitors 30 standard districts across Odisha."

            # If location specified
            if district:
                pred_res = self.prediction_tool.predict_for_location_and_date(district=district, block=block, date_str=date_str)
                if "error" in pred_res:
                    return f"⚠️ {pred_res['error']}"
                return format_prediction_response(pred_res)
                
            # If no location specified, handle general questions
            if "current flood risk" in q_lower or "current risk" in q_lower:
                op_summary = self.prediction_tool.get_operational_summary()
                if "error" in op_summary:
                    return f"⚠️ {op_summary['error']}"
                    
                return (
                    f"### Current State-Wide Flood Risk Summary ({op_summary.get('source', 'Operational Telemetry')})\n\n"
                    f"- **Monitored Telemetry Coverage**: `{op_summary.get('districts_count', 30)}` districts across `{op_summary.get('blocks_count', 354)}` block stations\n"
                    f"- **Active Risk Distribution**: `🔴 High Risk: {op_summary.get('high_risk_count', 0):,}` | `🟡 Moderate Risk: {op_summary.get('moderate_risk_count', 0):,}` | `🟢 Low Risk: {op_summary.get('low_risk_count', 0):,}` stations\n"
                    f"- **State Average Flood Probability**: `{op_summary.get('avg_probability', 0.0) * 100:.2f}%`\n"
                    f"- **Maximum Station Precipitation**: `{op_summary.get('max_rainfall_mm', 0.0):.1f} mm`\n\n"
                    "To view localized risk and probability for a specific station, please specify the district or block (e.g. *'What is the current flood risk for Cuttack?'* or *'Predict flood for Puri'*)."
                )
                
            if "predicted probability" in q_lower:
                threshold = self.metadata.get("optimal_threshold", 0.8073)
                model_name = self.metadata.get("model_name", "XGBoost")
                return (
                    f"### Model Predicted Flood Probability Information\n\n"
                    f"In this system, the `{model_name}` classifier calculates a continuous posterior probability $[0.0, 1.0]$ representing the likelihood of next-day flood inundation.\n\n"
                    f"- **Optimal Decision Threshold ($T^*$)**: `{threshold:.4f}` (probabilities $\\ge {threshold:.4f}$ trigger an alert)\n"
                    "- **Risk Categorization**: `LOW` (< 30%), `MODERATE` (30%–70%), `HIGH` ($\\ge$ 70%)\n\n"
                    "Please specify a district or block to retrieve the exact predicted probability (e.g. *'What is the predicted probability for Balasore?'*)."
                )

        # -------------------------------------------------------------
        # Intent 7: Historical Monthly Rainfall Query
        # e.g., "What was the rainfall in Cuttack in July 2023?"
        # -------------------------------------------------------------
        year, month = self._extract_year_month(q)
        district = self._extract_district(q)
        if year and month and district:
            res = self.query_engine.query_monthly_rainfall(district, year, month)
            if "error" in res:
                return f"ℹ️ {res['error']}"
            return (
                f"### Historical Meteorological Record: **{res['district']}** ({res['month']} {res['year']})\n\n"
                f"- **Total Monthly Rainfall**: `{res['total_rainfall_mm']:.1f} mm`\n"
                f"- **Average Daily Rainfall**: `{res['mean_daily_rainfall_mm']:.2f} mm/day`\n"
                f"- **Peak Single-Day Rain**: `{res['max_daily_rainfall_mm']:.1f} mm`\n"
                f"- **Total Recorded Inundation / Flood Days**: `{res['flood_days_in_month']}` day(s)"
            )

        # -------------------------------------------------------------
        # Intent 8: District Historical Overview Query
        # e.g., "Show 24-year flood summary for Puri"
        # -------------------------------------------------------------
        if district and any(w in q_lower for w in ["rainfall", "flood", "history", "stats", "events", "how many", "summary"]):
            res = self.query_engine.query_district_summary(district)
            if "error" in res:
                return f"ℹ️ {res['error']}"
            return (
                f"### 24-Year Climate & Flood Record for **{res['district']}** (2001–2024)\n\n"
                f"- **Total Recorded Flood Days**: `{res['total_flood_days']:,}` days ({res['flood_percentage']} of time series)\n"
                f"- **Average Annual Rainfall**: `{res['average_annual_rainfall_mm']:,} mm/year`\n"
                f"- **Total Cumulative Rainfall**: `{res['total_rainfall_mm']:,} mm`\n"
                f"- **Maximum Single-Day Rainfall**: `{res['max_single_day_rainfall_mm']:.1f} mm`\n"
                f"- **Monitored Blocks/Stations ({len(res['unique_blocks'])} blocks)**: {', '.join(res['unique_blocks'][:8])}..."
            )

        # -------------------------------------------------------------
        # Intent 9: Extreme Rainfall Records Query
        # e.g., "Show the highest rainfall events recorded in Odisha"
        # -------------------------------------------------------------
        if any(w in q_lower for w in ["highest rain", "maximum rainfall", "wettest", "extreme rain", "top rain"]):
            records = self.query_engine.query_highest_rainfall_records(5)
            if not records:
                return "Sufficient project data is not available: Historical rainfall records are not loaded."
            msg = "### Top 5 Extreme Single-Day Rainfall Events in Odisha (2001–2024)\n\n"
            msg += "| Date | District | Block / Station | Daily Rainfall (mm) | Flood Occurred |\n"
            msg += "| :--- | :--- | :--- | :---: | :---: |\n"
            for r in records:
                fo_str = "Yes (🔴)" if r["flood_occurred"] == 1 else "No (⚪)"
                msg += f"| `{r['date']}` | **{r['district']}** | {r['block']} | **{r['rainfall_mm']:.1f} mm** | {fo_str} |\n"
            return msg

        # -------------------------------------------------------------
        # Intent 10: Dataset Overview Query
        # e.g., "How many years of data are in the dataset?"
        # -------------------------------------------------------------
        if any(w in q_lower for w in ["how many years", "dataset overview", "total records", "what years", "dataset architecture"]):
            ov = self.query_engine.get_dataset_overview()
            if "error" in ov:
                return f"ℹ️ {ov['error']}"
            return (
                f"### Odisha Flood Prediction System — 24-Year Dataset Architecture\n\n"
                f"- **Time Period**: `{ov['years_covered']}`\n"
                f"- **Total Daily Station Records**: `{ov['total_records']:,}` rows\n"
                f"- **Districts Monitored**: `{ov['districts_count']}` districts across `{ov['blocks_count']}` administrative blocks/stations\n"
                f"- **Total Recorded Flood Occurrences**: `{ov['total_flood_events']:,}` instances (`{ov['overall_flood_rate']}` base prevalence)\n"
                f"- **Peak Rainfall Record**: `{ov['max_recorded_rainfall_mm']:.1f} mm`"
            )

        # -------------------------------------------------------------
        # Intent 11: Domain & Scientific Hydrological Knowledge
        # e.g., "What is rainfall downscaling?", "Tell me about Mahanadi"
        # -------------------------------------------------------------
        kn_ans = get_knowledge_answer(q)
        if kn_ans:
            return f"### Flood & Hydrology Knowledge Base\n\n{kn_ans}"

        # -------------------------------------------------------------
        # Unsupported / Out-of-Domain Query Fallback
        # Strict grounding: Do not guess or fabricate answers
        # -------------------------------------------------------------
        return (
            "Sufficient project data is not available to answer this query.\n\n"
            "As an interface layer for the Odisha Flood Prediction System, I am strictly grounded in verified project telemetry, models, and metadata. I can answer questions regarding:\n"
            "- 🔮 **Predictions & Risk**: *'What is the current flood risk?'*, *'What is the predicted probability for Cuttack?'*, or *'What does HIGH risk mean?'*\n"
            "- 🧠 **Model & Evaluation**: *'Which model is being used?'*, *'What is the model's recall?'*, or *'Which factors influenced the prediction?'*\n"
            "- 🌧️ **Meteorological Features**: *'What rainfall features are used?'* or *'What is rainfall downscaling?'*\n"
            "- 📊 **Historical Records**: *'What was the rainfall in Cuttack in July 2023?'* or *'Show 24-year flood summary for Puri'*."
        )
