"""
Historical data query module for Odisha flood dataset (2001-2024).
Provides exact, verified answers from the combined dataset without hallucinations.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from src.data.load_data import load_combined_data


class DatasetQueryEngine:
    """Engine to perform fast aggregate and granular queries on 24-year dataset."""
    
    def __init__(self, data_path: str = "data/combined/Odisha_Flood_2001_2024.csv"):
        self.data_path = data_path
        self.df = None
        self._load()

    def _load(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            self.df["Date"] = self.df["Date"].astype(str)

    def get_dataset_overview(self) -> Dict[str, Any]:
        """Return high-level dataset statistics."""
        if self.df is None:
            return {"error": "Dataset not loaded"}
        return {
            "total_records": len(self.df),
            "years_covered": f"{self.df['Year'].min()} - {self.df['Year'].max()} (24 years)",
            "districts_count": int(self.df["District"].nunique()),
            "blocks_count": int(self.df["Block/Station"].nunique()),
            "total_flood_events": int(self.df["Flood_Occurred"].sum()),
            "overall_flood_rate": f"{(self.df['Flood_Occurred'].mean() * 100):.2f}%",
            "max_recorded_rainfall_mm": float(self.df["Rainfall (mm)"].max())
        }

    def query_district_summary(self, district: str) -> Dict[str, Any]:
        """Query 24-year totals for a given district."""
        d_clean = district.strip().upper()
        sub = self.df[self.df["District"] == d_clean]
        if len(sub) == 0:
            return {"error": f"District '{district}' not found in dataset."}
            
        return {
            "district": d_clean,
            "total_observations": len(sub),
            "total_flood_days": int(sub["Flood_Occurred"].sum()),
            "flood_percentage": f"{(sub['Flood_Occurred'].mean() * 100):.2f}%",
            "total_rainfall_mm": round(float(sub["Rainfall (mm)"].sum()), 1),
            "average_annual_rainfall_mm": round(float(sub["Rainfall (mm)"].sum() / 24.0), 1),
            "max_single_day_rainfall_mm": float(sub["Rainfall (mm)"].max()),
            "unique_blocks": sorted(list(sub["Block/Station"].unique()))
        }

    def query_monthly_rainfall(self, district: str, year: int, month: int) -> Dict[str, Any]:
        """Query total and mean rainfall for a specific month and year."""
        d_clean = district.strip().upper()
        sub = self.df[
            (self.df["District"] == d_clean) &
            (self.df["Year"] == year) &
            (self.df["Month_Number"] == month)
        ]
        if len(sub) == 0:
            return {"error": f"No records found for {district} in {month}/{year}."}
            
        month_names = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        month_name = month_names[month] if 1 <= month <= 12 else str(month)
        
        return {
            "district": d_clean,
            "year": year,
            "month": month_name,
            "total_rainfall_mm": round(float(sub["Rainfall (mm)"].sum()), 1),
            "mean_daily_rainfall_mm": round(float(sub["Rainfall (mm)"].mean()), 2),
            "max_daily_rainfall_mm": float(sub["Rainfall (mm)"].max()),
            "flood_days_in_month": int(sub["Flood_Occurred"].sum())
        }

    def query_highest_rainfall_records(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top N highest single-day rainfall events across Odisha (2001-2024)."""
        top_rows = self.df.sort_values(by="Rainfall (mm)", ascending=False).head(top_n)
        records = []
        for _, row in top_rows.iterrows():
            records.append({
                "district": row["District"],
                "block": row["Block/Station"],
                "date": row["Date"],
                "rainfall_mm": float(row["Rainfall (mm)"]),
                "flood_occurred": int(row["Flood_Occurred"])
            })
        return records
