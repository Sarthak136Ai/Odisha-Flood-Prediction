"""
Core Chatbot engine for Odisha flood prediction, historical querying, and explainability.
Routes incoming user queries to specialized tools without hallucinations.
"""

import os
import re
from typing import Dict, Any, Optional, Tuple, List

from chatbot.data_query import DatasetQueryEngine
from chatbot.prediction_tools import ChatbotPredictionTool
from chatbot.flood_knowledge import get_knowledge_answer, ODISHA_FLOOD_KNOWLEDGE
from chatbot.prompts import format_prediction_response
from src.data.combine_data import STANDARD_DISTRICTS


class OdishaFloodChatbot:
    """Intelligent Flood Chatbot combining ML predictions, dataset queries, and domain knowledge."""
    
    def __init__(
        self,
        data_path: str = "data/combined/Odisha_Flood_2001_2024.csv",
        model_path: str = "models/flood_prediction/best_model.pkl",
        metadata_path: str = "models/flood_prediction/model_metadata.json"
    ):
        self.query_engine = DatasetQueryEngine(data_path)
        self.prediction_tool = ChatbotPredictionTool(model_path, metadata_path, data_path)
        self.districts = STANDARD_DISTRICTS

    def _extract_district(self, text: str) -> Optional[str]:
        """Extract mentioned Odisha district name from text."""
        t_upper = text.upper()
        for d in self.districts:
            if re.search(r'\b' + re.escape(d) + r'\b', t_upper):
                return d
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract YYYY-MM-DD date if present."""
        match = re.search(r'\b(20\d{2}-\d{2}-\d{2})\b', text)
        if match:
            return match.group(1)
        return None

    def _extract_year_month(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract Year and Month if present."""
        months_map = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
            "november": 11, "nov": 11, "december": 12, "dec": 12
        }
        
        found_year = None
        year_match = re.search(r'\b(20\d{2})\b', text)
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
        """Process user message and return structured response."""
        q = query.strip()
        q_lower = q.lower()
        
        # 1. Prediction / Early Warning Intent
        if any(w in q_lower for w in ["will", "flood tomorrow", "predict", "forecast", "risk of flood", "warning", "probability"]):
            district = self._extract_district(q)
            date_str = self._extract_date(q)
            
            if not district:
                district = "CUTTACK" # Default demo representative district if none specified
                
            pred_res = self.prediction_tool.predict_for_location_and_date(district=district, date_str=date_str)
            if "error" in pred_res:
                return f"⚠️ {pred_res['error']}"
            return format_prediction_response(pred_res)
            
        # 2. Historical Monthly Rainfall Query
        year, month = self._extract_year_month(q)
        district = self._extract_district(q)
        if year and month and district:
            res = self.query_engine.query_monthly_rainfall(district, year, month)
            if "error" in res:
                return f"ℹ️ {res['error']}"
            return (
                f"### Historical Meteorological Query: **{res['district']}** ({res['month']} {res['year']})\n\n"
                f"- **Total Monthly Rainfall**: `{res['total_rainfall_mm']:.1f} mm`\n"
                f"- **Average Daily Rainfall**: `{res['mean_daily_rainfall_mm']:.2f} mm/day`\n"
                f"- **Peak Single-Day Rain**: `{res['max_daily_rainfall_mm']:.1f} mm`\n"
                f"- **Total Inundation / Flood Days**: `{res['flood_days_in_month']}` day(s)"
            )
            
        # 3. District Historical Overview Query
        if district and any(w in q_lower for w in ["rainfall", "flood", "history", "stats", "events", "how many"]):
            res = self.query_engine.query_district_summary(district)
            if "error" in res:
                return f"ℹ️ {res['error']}"
            return (
                f"### 24-Year Climate & Flood Summary for **{res['district']}** (2001–2024)\n\n"
                f"- **Total Recorded Flood Days**: `{res['total_flood_days']:,}` days ({res['flood_percentage']} of time series)\n"
                f"- **Average Annual Rainfall**: `{res['average_annual_rainfall_mm']:,} mm/year`\n"
                f"- **Total Cumulative Rainfall**: `{res['total_rainfall_mm']:,} mm`\n"
                f"- **Maximum Single-Day Rainfall**: `{res['max_single_day_rainfall_mm']:.1f} mm`\n"
                f"- **Monitored Blocks/Stations ({len(res['unique_blocks'])} blocks)**: {', '.join(res['unique_blocks'][:8])}..."
            )
            
        # 4. Highest Rainfall Records Query
        if any(w in q_lower for w in ["highest rain", "maximum rainfall", "wettest", "extreme rain", "top rain"]):
            records = self.query_engine.query_highest_rainfall_records(5)
            msg = "### Top 5 Extreme Single-Day Rainfall Events in Odisha (2001–2024)\n\n"
            msg += "| Date | District | Block / Station | Daily Rainfall (mm) | Flood Occurred |\n"
            msg += "| :--- | :--- | :--- | :---: | :---: |\n"
            for r in records:
                fo_str = "Yes (🔴)" if r["flood_occurred"] == 1 else "No (⚪)"
                msg += f"| `{r['date']}` | **{r['district']}** | {r['block']} | **{r['rainfall_mm']:.1f} mm** | {fo_str} |\n"
            return msg
            
        # 5. Dataset Overview Query
        if any(w in q_lower for w in ["how many years", "dataset overview", "total records", "rows", "what years"]):
            ov = self.query_engine.get_dataset_overview()
            return (
                f"### Odisha Flood Prediction System — 24-Year Dataset Architecture\n\n"
                f"- **Time Period**: `{ov['years_covered']}`\n"
                f"- **Total Daily Station Records**: `{ov['total_records']:,}` rows\n"
                f"- **Districts Monitored**: `{ov['districts_count']}` districts across `{ov['blocks_count']}` administrative blocks/stations\n"
                f"- **Total Recorded Flood Occurrences**: `{ov['total_flood_events']:,}` instances (`{ov['overall_flood_rate']}` base frequency)\n"
                f"- **Peak Rainfall Record**: `{ov['max_recorded_rainfall_mm']:.1f} mm`"
            )
            
        # 6. Domain / Scientific Knowledge
        kn_ans = get_knowledge_answer(q)
        if kn_ans:
            return f"### Flood & Hydrology Knowledge Base\n\n{kn_ans}"
            
        # Default helpful menu
        return (
            "Hello! I am the **Odisha Flood Early Warning & Climate Assistant**. How can I assist you today?\n\n"
            "You can ask me questions such as:\n"
            "- 🔮 **Live Prediction**: *'Will Cuttack have a flood tomorrow?'* or *'Predict flood for Puri on 2024-08-15'*\n"
            "- 📊 **Historical Climate**: *'What was the rainfall in Cuttack in July 2023?'* or *'Show 24-year flood summary for Balasore'*\n"
            "- ⚡ **Extreme Events**: *'Show the highest rainfall events recorded in Odisha'*\n"
            "- 🧠 **AI & Downscaling**: *'What is rainfall downscaling?'* or *'How does SHAP explain flood risk?'*\n"
            "- 📁 **Dataset Stats**: *'How many years and records are in the dataset?'*"
        )
