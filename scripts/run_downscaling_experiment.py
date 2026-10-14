"""
Script to execute Phase 9 (Rainfall Downscaling Scientific Audit) & Phase 10 (Downscaled vs Baseline Prediction Benchmark).
Evaluates Observed vs Original/Coarse vs Downscaled rainfall across Train (2001-2018), Val (2019-2021), and Test (2022-2024).
Computes MAE, RMSE, R², Bias, and Pearson correlation.
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
    compare_downscaling_benchmarks,
    evaluate_intensity_tiers,
    compare_flood_prediction_impact,
    plot_downscaling_comparison,
    plot_comprehensive_downscaling_audit
)


def main():
    print("=" * 80)
    print("PHASE 9 & 10: RAINFALL DOWNSCALING SCIENTIFIC AUDIT & PREDICTION IMPACT")
    print("=" * 80)
    
    config = load_config("config.yaml")
    results_downscaling_dir = config["paths"]["results_downscaling_dir"]
    reports_plots_dir = "reports/plots"
    static_plots_dir = "static/images/plots"
    
    os.makedirs(results_downscaling_dir, exist_ok=True)
    os.makedirs(reports_plots_dir, exist_ok=True)
    os.makedirs(static_plots_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load data & create coarse spatial signals
    print("\n1. Loading combined dataset and computing coarse regional rainfall representations...")
    df = load_combined_data(config["paths"]["combined_data_path"], parse_dates=False)
    df_coarse = create_coarse_rainfall_representation(df)
    
    # Define strictly isolated temporal splits
    train_mask = df_coarse["Year"].isin(config["split"]["train_years"])
    val_mask = df_coarse["Year"].isin(config["split"]["val_years"])
    test_mask = df_coarse["Year"].isin(config["split"]["test_years"])
    
    train_df = df_coarse[train_mask].dropna(subset=[config["target"]["name"]]).copy()
    val_df = df_coarse[val_mask].dropna(subset=[config["target"]["name"]]).copy()
    test_df = df_coarse[test_mask].dropna(subset=[config["target"]["name"]]).copy()
    
    print(f"Dataset split sizes: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
    
    # 2. Train Downscaling Model (Strictly on 2001-2018)
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
    print(f"Downscaling Train Fit: MAE={train_metrics['train_mae']:.4f} mm, RMSE={train_metrics['train_rmse']:.4f} mm, R²={train_metrics['train_r2']:.4f}, Bias={train_metrics['train_bias']:.4f} mm, r={train_metrics['train_pearson_r']:.4f}")
    
    # 3. Evaluate Across Train, Val, and Test Splits (Observed vs Coarse vs Downscaled)
    print("\n3. Evaluating Downscaling Quality across Train, Val, and Test partitions...")
    eval_dfs = []
    
    split_datasets = [
        ("Train (2001-2018)", train_df),
        ("Validation (2019-2021)", val_df),
        ("Test (2022-2024)", test_df)
    ]
    
    test_down_preds = None
    for split_label, split_df in split_datasets:
        y_obs = split_df["Rainfall (mm)"].values
        y_crs = split_df["coarse_rainfall_mean"].values
        y_dwn = np.clip(down_model.predict(split_df[downscaling_features]), 0.0, None)
        
        if split_label.startswith("Test"):
            test_down_preds = y_dwn
            
        df_bench = compare_downscaling_benchmarks(
            y_observed=y_obs,
            y_coarse=y_crs,
            y_downscaled=y_dwn,
            split_name=split_label
        )
        eval_dfs.append(df_bench)
        
    full_eval_df = pd.concat(eval_dfs, ignore_index=True)
    
    eval_csv_path = os.path.join(results_downscaling_dir, "downscaling_evaluation_results.csv")
    full_eval_df.to_csv(eval_csv_path, index=False)
    full_eval_df.to_csv("reports/downscaling_evaluation_results.csv", index=False)
    
    print("\n" + "=" * 90)
    print("DOWNSCALING BENCHMARK: OBSERVED vs ORIGINAL COARSE vs DOWNSCALED MODEL")
    print("=" * 90)
    print(full_eval_df.to_string(index=False))
    
    # 4. Evaluate across IMD Rainfall Intensity Categories on Test Set (2022-2024)
    print("\n4. Evaluating performance across IMD rainfall intensity tiers on Test Set (2022-2024)...")
    y_test_obs = test_df["Rainfall (mm)"].values
    y_test_crs = test_df["coarse_rainfall_mean"].values
    y_test_dwn = test_down_preds
    
    df_tiers = evaluate_intensity_tiers(y_test_obs, y_test_crs, y_test_dwn)
    tier_csv_path = os.path.join(results_downscaling_dir, "downscaling_intensity_results.csv")
    df_tiers.to_csv(tier_csv_path, index=False)
    print(df_tiers.to_string(index=False))
    
    # 5. Apply Downscaling and Recompute Rolling Flood Features on Test Set
    print("\n5. Downscaling test set rainfall (2022-2024) and reconstructing rolling flood features...")
    df_test_downscaled = generate_downscaled_flood_features(
        test_df,
        down_model,
        downscaling_features=downscaling_features
    )
    
    # 6. Benchmark Downstream Flood Early Warning Model Impact
    print("\n6. Benchmarking Flood Early Warning Model: Original vs Downscaled Rainfall Inputs...")
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
    
    print("\n" + "=" * 90)
    print("DOWNSTREAM FLOOD PREDICTION SKILL COMPARISON (CHRONOLOGICAL TEST SET 2022-2024):")
    print("=" * 90)
    cols_to_show = ["Rainfall_Source", "roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy", "brier_score"]
    print(df_impact[cols_to_show].to_string(index=False))
    
    # 7. Generate Visualizations
    print("\n7. Generating publication-grade comparison plots...")
    # Comprehensive 4-panel audit plot
    audit_plot_path = os.path.join(results_downscaling_dir, "downscaling_audit_comparison.png")
    extra_paths = [
        os.path.join(results_downscaling_dir, "rainfall_comparison.png"),
        os.path.join(static_plots_dir, "rainfall_comparison.png"),
        os.path.join(reports_plots_dir, "downscaling_comparison.png"),
        os.path.join(reports_plots_dir, "downscaling_comparison.svg")
    ]
    plot_comprehensive_downscaling_audit(
        y_observed=y_test_obs,
        y_coarse=y_test_crs,
        y_downscaled=y_test_dwn,
        save_path=audit_plot_path,
        extra_save_paths=extra_paths
    )
    print("Saved audit comparison plots to results, reports, and static directories.")
    
    # Backward compatible 2-panel plot
    plot_downscaling_comparison(
        y_test_obs,
        y_test_dwn,
        test_df["Date"],
        save_path=os.path.join(results_downscaling_dir, "downscaling_2panel_comparison.png")
    )
    
    # Scientific Conclusion
    delta_f1 = df_impact.loc[1, "f1"] - df_impact.loc[0, "f1"]
    delta_pr_auc = df_impact.loc[1, "pr_auc"] - df_impact.loc[0, "pr_auc"]
    
    print("\n--- SCIENTIFIC AUDIT SUMMARY ---")
    print(f"1. Test Coarse Baseline:  MAE={full_eval_df.loc[4, 'MAE (mm)']:.4f} mm, RMSE={full_eval_df.loc[4, 'RMSE (mm)']:.4f} mm, R²={full_eval_df.loc[4, 'R2']:.4f}, Bias={full_eval_df.loc[4, 'Bias (mm)']:.4f} mm, r={full_eval_df.loc[4, 'Pearson_r']:.4f}")
    print(f"2. Test Downscaled Model: MAE={full_eval_df.loc[5, 'MAE (mm)']:.4f} mm, RMSE={full_eval_df.loc[5, 'RMSE (mm)']:.4f} mm, R²={full_eval_df.loc[5, 'R2']:.4f}, Bias={full_eval_df.loc[5, 'Bias (mm)']:.4f} mm, r={full_eval_df.loc[5, 'Pearson_r']:.4f}")
    print(f"3. Downstream Flood Impact: F1 Delta = {delta_f1:+.4f}, PR-AUC Delta = {delta_pr_auc:+.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
