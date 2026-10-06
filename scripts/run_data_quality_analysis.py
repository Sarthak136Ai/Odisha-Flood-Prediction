"""
CLI Runner for automated Data Quality Analysis of Odisha Flood Prediction datasets.
Generates:
- reports/data_quality_report.html
- reports/data_quality_summary.csv
- reports/plots/*.png
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_quality import DataQualityAnalyzer


def main():
    print("=" * 80)
    print("ODISHA FLOOD PREDICTION: AUTOMATED DATA QUALITY ANALYSIS")
    print("=" * 80)
    
    analyzer = DataQualityAnalyzer(
        raw_historical_dir="data/raw/historical",
        raw_operational_path="data/raw/operational/2025.csv",
        processed_historical_path="data/processed/historical_2001_2024.csv",
        processed_operational_path="data/processed/operational_2025.csv",
        output_dir="reports"
    )
    
    results = analyzer.run_full_analysis()
    
    print("\n" + "=" * 80)
    print("DATA QUALITY AUDIT COMPLETE")
    print("=" * 80)
    print(f"Total Processed Records: {results['processed']['total_records']:,}")
    print(f"Districts: {results['processed']['num_districts']}")
    print(f"Stations: {results['processed']['num_stations']}")
    print(f"Date Range: {results['processed']['date_range'][0]} to {results['processed']['date_range'][1]}")
    print(f"Duplicate [District, Block, Date] Records in Processed: {results['processed']['location_date_duplicates']}")
    print(f"Negative Rainfall Count: {results['processed']['rainfall_stats']['negative_rainfall_count']}")
    print(f"Missing Rainfall Count: {results['processed']['rainfall_stats']['missing_rainfall_count']}")
    print(f"Historical Flood Rate: {results['processed']['class_distribution']['Flood_Occurred_rate_pct']}%")
    print(f"\nHTML Report: {results['html_path']}")
    print(f"CSV Summary: {results['csv_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
