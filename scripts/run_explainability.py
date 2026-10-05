"""
Script to execute Phase 8: Explainable AI and SHAP analysis on the trained flood prediction model.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_combined_data, load_config
from src.explainability.feature_importance import extract_feature_importance, plot_feature_importance
from src.explainability.shap_analysis import compute_shap_analysis, explain_instance


def main():
    print("=" * 70)
    print("PHASE 8: EXPLAINABLE AI & SHAP ATTRIBUTION ANALYSIS")
    print("=" * 70)
    
    config = load_config("config.yaml")
    model_path = "models/flood_prediction/best_model.pkl"
    metadata_path = "models/flood_prediction/model_metadata.json"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
        
    model = joblib.load(model_path)
    with open(metadata_path, "r") as f:
        import json
        metadata = json.load(f)
        
    feature_cols = metadata["features"]
    print(f"Loaded trained model: {metadata['model_name']} ({len(feature_cols)} features)")
    
    # 1. Feature Importance Extraction
    df_imp = extract_feature_importance(model, feature_cols)
    imp_csv_path = "results/metrics/feature_importance.csv"
    imp_plot_path = "results/plots/feature_importance.png"
    
    df_imp.to_csv(imp_csv_path, index=False)
    plot_feature_importance(df_imp, save_path=imp_plot_path)
    
    print("\n--- TOP 10 MOST INFLUENTIAL FEATURES ---")
    print(df_imp.head(10).to_string(index=False))
    print(f"\nFeature importance plot saved to: {imp_plot_path}")
    
    # 2. SHAP Analysis on Sample
    print("\n" * 2 + "-" * 70)
    print("COMPUTING GLOBAL SHAP ATTRIBUTIONS...")
    print("-" * 70)
    
    df = load_combined_data(config["paths"]["combined_data_path"], parse_dates=False)
    train_df = df[df["Year"].isin(config["split"]["train_years"])].dropna(subset=[config["target"]["name"]])
    test_df = df[df["Year"].isin(config["split"]["test_years"])].dropna(subset=[config["target"]["name"]])
    
    np.random.seed(42)
    train_sample = train_df[feature_cols].sample(n=min(2500, len(train_df)), random_state=42)
    test_sample = test_df[feature_cols].sample(n=min(2500, len(test_df)), random_state=42)
    
    shap_plot_path = "results/plots/shap_summary.png"
    shap_csv_path = "results/metrics/shap_feature_importance.csv"
    
    _, _, df_shap = compute_shap_analysis(
        model,
        train_sample,
        test_sample,
        save_plot_path=shap_plot_path,
        save_csv_path=shap_csv_path
    )
    
    print("\n--- TOP 10 SHAP FEATURES BY MEAN ABSOLUTE VALUE ---")
    print(df_shap.head(10).to_string(index=False))
    print(f"\nSHAP summary plot saved to: {shap_plot_path}")
    
    # 3. Test Local Explanation
    print("\n" * 2 + "-" * 70)
    print("SAMPLE LOCAL PREDICTION EXPLANATION (Extreme vs Dry)")
    print("-" * 70)
    sample_row = test_sample.iloc[0].to_dict()
    explanation = explain_instance(model, sample_row, feature_cols)
    print("Top Positive Risk Drivers:")
    for d in explanation["top_risk_drivers"]:
        print(f"  - {d['feature']} = {d['value']:.2f} (Contribution: +{d['contribution']:.4f})")
    print("Top Risk Mitigators:")
    for m in explanation["top_mitigators"]:
        print(f"  - {m['feature']} = {m['value']:.2f} (Contribution: {m['contribution']:.4f})")
        
    print("\n" + "=" * 70)
    print("PHASE 8 EXPLAINABLE AI COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
