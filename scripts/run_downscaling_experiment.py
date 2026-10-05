"""
Script to execute Phase 9 (Rainfall Downscaling) & Phase 10 (Downscaled vs Baseline Prediction Benchmark).
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_combined_data, load_config
from src.downscaling.prepare_downscaling_data import create_coarse_rainfall_representation
from src.downscaling.train_downscaling import train_downscaling_model
from src.downscaling.apply_downscaling import generate_downscaled_flood_features
from src.downscaling.evaluate_downscaling import (
    evaluate_downscaling_quality,
    compare_flood_prediction_impact,
    plot_downscaling_comparison
)


def main():
    print("=" * 70)
    print("PHASE 9 & 10: RAINFALL DOWNSCALING & PREDICTION IMPACT EXPERIMENT")
    print("=" * 70)
    
    config = load_config("config.yaml")
    results_downscaling_dir = config["paths"]["results_downscaling_dir"]
    os.makedirs(results_downscaling_dir, exist_ok=True)
    
    # 1. Load data & create coarse spatial signals
    print("\n1. Loading combined dataset and computing coarse regional rainfall representations...")
    df = load_combined_data(config["paths"]["combined_data_path"], parse_dates=False)
    df_coarse = create_coarse_rainfall_representation(df)
    
    # Split train and test
    train_mask = df_coarse["Year"].isin(config["split"]["train_years"])
    test_mask = df_coarse["Year"].isin(config["split"]["test_years"])
    
    train_df = df_coarse[train_mask].dropna(subset=[config["target"]["name"]]).copy()
    test_df = df_coarse[test_mask].dropna(subset=[config["target"]["name"]]).copy()
    
    # 2. Train Downscaling Model
    print("\n2. Training statistical downscaling model (HistGradientBoostingRegressor on 2001-2018)...")
    downscaling_features = [
        "coarse_rainfall_mean",
        "coarse_rainfall_std",
        "Month_sin",
        "Month_cos",
        "Day_of_Year"
    ]
    
    down_model, train_metrics = train_downscaling_model(
        train_df=train_df,
        features=downscaling_features,
        save_path="models/downscaling/downscaling_model.pkl"
    )
    print(f"Downscaling Training Metrics: RMSE={train_metrics['train_rmse']:.2f} mm, MAE={train_metrics['train_mae']:.2f} mm, R2={train_metrics['train_r2']:.4f}")
    
    # 3. Apply Downscaling on Test Set (2022-2024)
    print("\n3. Downscaling test set rainfall (2022-2024) and reconstructing rolling flood features...")
    df_test_downscaled = generate_downscaled_flood_features(
        test_df,
        down_model,
        downscaling_features=downscaling_features
    )
    
    # 4. Evaluate Downscaling Quality
    y_test_true_rf = test_df["Rainfall (mm)"].values
    y_test_down_rf = df_test_downscaled["Rainfall (mm)"].values
    
    rf_eval = evaluate_downscaling_quality(y_test_true_rf, y_test_down_rf)
    print("\n--- DOWNSCALING QUALITY ON UNSEEN TEST SET (2022-2024) ---")
    for k, v in rf_eval.items():
        print(f"  {k}: {v}")
        
    # Plot rainfall comparison
    plot_downscaling_comparison(
        y_test_true_rf,
        y_test_down_rf,
        test_df["Date"],
        save_path=os.path.join(results_downscaling_dir, "rainfall_comparison.png")
    )
    print("Rainfall comparison plot saved.")
    
    # 5. Compare Flood Prediction Performance: Original vs Downscaled
    print("\n5. Benchmarking Flood Early Warning Model: Original vs Downscaled Rainfall...")
    flood_model_path = "models/flood_prediction/best_model.pkl"
    flood_metadata_path = "models/flood_prediction/model_metadata.json"
    
    flood_model = joblib.load(flood_model_path)
    with open(flood_metadata_path, "r") as f:
        meta = json.load(f)
        
    feature_cols = meta["features"]
    threshold = float(meta.get("optimal_threshold", 0.5))
    target_col = config["target"]["name"]
    
    X_test_orig = test_df[feature_cols]
    X_test_down = df_test_downscaled[feature_cols]
    y_test_flood = test_df[target_col].values.astype(int)
    
    df_impact = compare_flood_prediction_impact(
        flood_model,
        X_test_orig,
        X_test_down,
        y_test_flood,
        threshold=threshold
    )
    
    impact_csv_path = os.path.join(results_downscaling_dir, "downscaling_metrics.csv")
    df_impact.to_csv(impact_csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("FLOOD PREDICTION COMPARISON (CHRONOLOGICAL TEST SET 2022-2024):")
    print("=" * 70)
    cols_to_show = ["Rainfall_Source", "roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy"]
    print(df_impact[cols_to_show].to_string(index=False))
    
    # Scientific Conclusion
    delta_f1 = df_impact.loc[1, "f1"] - df_impact.loc[0, "f1"]
    delta_pr_auc = df_impact.loc[1, "pr_auc"] - df_impact.loc[0, "pr_auc"]
    
    print("\n--- SCIENTIFIC CONCLUSION ---")
    if abs(delta_f1) < 0.02 and abs(delta_pr_auc) < 0.02:
        conclusion = "Downscaling preserved flood prediction performance with negligible variance."
    elif delta_f1 > 0:
        conclusion = f"Downscaling improved flood prediction F1 by +{delta_f1:.4f}."
    else:
        conclusion = f"Downscaling produced a slight reduction in prediction F1 by {delta_f1:.4f} due to spatial smoothing."
    print(f"Outcome: {conclusion}")
    print("=" * 70)


if __name__ == "__main__":
    main()
