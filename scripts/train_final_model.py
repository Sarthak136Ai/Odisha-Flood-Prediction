"""
CLI runner script for training and evaluating all 4 flood prediction models
across 24-year temporal splits (Train 2001-2018, Val 2019-2021, Test 2022-2024).
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.train_models import train_and_evaluate_all_models


def main():
    print("=" * 70)
    print("PHASE 5 & 6: FLOOD PREDICTION MODEL TRAINING & TEMPORAL EVALUATION")
    print("=" * 70)
    
    results = train_and_evaluate_all_models("config.yaml")
    
    df_comp = results["comparison_df"]
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS (CHRONOLOGICAL TEST SET 2022-2024):")
    print("=" * 70)
    
    display_cols = [
        "Model", "Optimal_Threshold", "Test_ROC_AUC", "Test_PR_AUC",
        "Test_F1", "Test_Recall", "Test_Precision", "Test_Accuracy", "Test_Specificity"
    ]
    print(df_comp[display_cols].to_string(index=False))
    print("\n" + "=" * 70)
    print(f"BEST MODEL SELECTED: {results['best_model_name']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
