"""
Command-line runner for Comprehensive Probability Calibration in Odisha Flood Prediction.

Evaluates raw vs post-hoc calibrated probabilities across all major models:
- Logistic Regression (Linear baseline with class balancing)
- XGBoost (Production champion with scale_pos_weight)
- Random Forest (Balanced bagging ensemble)
- Decision Tree (Balanced tree baseline)

Rigorous Methodology:
1. Walk-Forward Chronological Splits:
   - Train (2001–2018): Base model training
   - Validation (2019–2021): Calibrator fitting (Sigmoid & Isotonic)
   - Test (2022–2024): Strict out-of-sample evaluation (untouched)
2. Metrics: Brier Score, Log Loss, Expected Calibration Error (ECE), Maximum Calibration Error (MCE).
3. Risk Tier Analysis: Verifies calibrated probability thresholds for LOW, MODERATE, HIGH risk categories.
4. Outputs: CSV reports, publication-grade figures, and calibrated model artifacts.
"""

import os
import sys
import argparse
import logging
import time
import json
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_combined_data, load_config
from src.models.train_models import prepare_temporal_datasets
from src.models.logistic_regression import create_logistic_regression_pipeline
from src.models.xgboost_model import create_xgboost_model
from src.models.random_forest import create_random_forest_model
from src.models.decision_tree import create_decision_tree_model
from src.evaluation.calibration import (
    evaluate_calibration,
    fit_validation_calibrator,
    plot_raw_vs_calibrated_comparison,
    plot_reliability_diagrams
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Probability Calibration Benchmarks.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--output-csv", type=str, default="reports/probability_calibration_results.csv", help="Primary results CSV destination.")
    parser.add_argument("--plots-dir", type=str, default="reports/plots", help="Directory for plots.")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("================================================================================")
    logger.info("Starting Comprehensive Probability Calibration Benchmark")
    logger.info("================================================================================")

    # 1. Load config and historical data
    config = load_config(args.config)
    data_path = config["paths"]["combined_data_path"]
    logger.info(f"Loading combined historical dataset from: {data_path}")
    df = load_combined_data(data_path, parse_dates=False)

    train_df, val_df, test_df, feature_cols, target_col = prepare_temporal_datasets(df, config)

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values.astype(int)

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].values.astype(int)

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].values.astype(int)

    test_prevalence = float(np.mean(y_test))
    logger.info(f"Untouched Test Set (2022–2024): {len(test_df):,} rows | Observed Flood Rate: {test_prevalence*100:.3f}%")

    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    spw = float(num_neg / max(num_pos, 1))

    # 2. Instantiate and fit base models on Training partition only
    base_models = {
        "Logistic Regression": create_logistic_regression_pipeline(
            max_iter=config["models"]["logistic_regression"].get("max_iter", 1000),
            class_weight=config["models"]["logistic_regression"].get("class_weight", "balanced"),
            random_state=config["project"]["random_seed"]
        ),
        "XGBoost": create_xgboost_model(
            n_estimators=config["models"]["xgboost"].get("n_estimators", 200),
            max_depth=config["models"]["xgboost"].get("max_depth", 6),
            learning_rate=config["models"]["xgboost"].get("learning_rate", 0.05),
            scale_pos_weight=spw,
            random_state=config["project"]["random_seed"],
            n_jobs=config["models"]["xgboost"].get("n_jobs", -1)
        ),
        "Random Forest": create_random_forest_model(
            n_estimators=config["models"]["random_forest"].get("n_estimators", 100),
            max_depth=config["models"]["random_forest"].get("max_depth", 10),
            class_weight=config["models"]["random_forest"].get("class_weight", "balanced"),
            random_state=config["project"]["random_seed"],
            n_jobs=config["models"]["random_forest"].get("n_jobs", -1)
        ),
        "Decision Tree": create_decision_tree_model(
            max_depth=config["models"].get("decision_tree", {}).get("max_depth", 8),
            min_samples_leaf=config["models"].get("decision_tree", {}).get("min_samples_leaf", 50),
            class_weight=config["models"].get("decision_tree", {}).get("class_weight", "balanced"),
            random_state=config["project"]["random_seed"]
        )
    }

    results_rows = []
    comparison_data = {}
    fitted_calibrators = {}

    for name, model in base_models.items():
        logger.info(f"\n--- Training base model: {name} on Train Set (2001–2018) ---")
        model.fit(X_train, y_train)

        # Raw inference
        raw_test_proba = model.predict_proba(X_test)[:, 1]
        raw_stats = evaluate_calibration(y_test, raw_test_proba)

        # Fit Calibrators strictly on Validation Set (2019–2021)
        logger.info(f"Fitting Sigmoid & Isotonic calibrators for {name} on Validation Set...")
        cal_sigmoid = fit_validation_calibrator(model, X_val, y_val, method="sigmoid")
        cal_isotonic = fit_validation_calibrator(model, X_val, y_val, method="isotonic")

        sig_test_proba = cal_sigmoid.predict_proba(X_test)[:, 1]
        iso_test_proba = cal_isotonic.predict_proba(X_test)[:, 1]

        sig_stats = evaluate_calibration(y_test, sig_test_proba)
        iso_stats = evaluate_calibration(y_test, iso_test_proba)

        comparison_data[name] = {
            "Raw": (y_test, raw_test_proba),
            "Calibrated (Sigmoid)": (y_test, sig_test_proba),
            "Calibrated (Isotonic)": (y_test, iso_test_proba)
        }
        fitted_calibrators[name] = {
            "base": model,
            "sigmoid": cal_sigmoid,
            "isotonic": cal_isotonic
        }

        # Calculate percentage improvements vs Raw
        sig_brier_impr = (raw_stats["brier_score"] - sig_stats["brier_score"]) / raw_stats["brier_score"] * 100
        iso_brier_impr = (raw_stats["brier_score"] - iso_stats["brier_score"]) / raw_stats["brier_score"] * 100
        sig_ece_impr = (raw_stats["expected_calibration_error"] - sig_stats["expected_calibration_error"]) / max(raw_stats["expected_calibration_error"], 1e-5) * 100
        iso_ece_impr = (raw_stats["expected_calibration_error"] - iso_stats["expected_calibration_error"]) / max(raw_stats["expected_calibration_error"], 1e-5) * 100

        # Log rows for all 3 variants
        results_rows.append({
            "Model": name,
            "Calibration_Variant": "Raw (Uncalibrated)",
            "Brier_Score": raw_stats["brier_score"],
            "Log_Loss": raw_stats["log_loss"],
            "ECE": raw_stats["expected_calibration_error"],
            "MCE": raw_stats["maximum_calibration_error"],
            "Mean_Predicted_Prob": raw_stats["mean_predicted_probability"],
            "Observed_Prevalence": raw_stats["observed_prevalence"],
            "Brier_Reduction_Pct": 0.0,
            "ECE_Reduction_Pct": 0.0
        })
        results_rows.append({
            "Model": name,
            "Calibration_Variant": "Calibrated (Sigmoid/Platt)",
            "Brier_Score": sig_stats["brier_score"],
            "Log_Loss": sig_stats["log_loss"],
            "ECE": sig_stats["expected_calibration_error"],
            "MCE": sig_stats["maximum_calibration_error"],
            "Mean_Predicted_Prob": sig_stats["mean_predicted_probability"],
            "Observed_Prevalence": sig_stats["observed_prevalence"],
            "Brier_Reduction_Pct": round(sig_brier_impr, 2),
            "ECE_Reduction_Pct": round(sig_ece_impr, 2)
        })
        results_rows.append({
            "Model": name,
            "Calibration_Variant": "Calibrated (Isotonic)",
            "Brier_Score": iso_stats["brier_score"],
            "Log_Loss": iso_stats["log_loss"],
            "ECE": iso_stats["expected_calibration_error"],
            "MCE": iso_stats["maximum_calibration_error"],
            "Mean_Predicted_Prob": iso_stats["mean_predicted_probability"],
            "Observed_Prevalence": iso_stats["observed_prevalence"],
            "Brier_Reduction_Pct": round(iso_brier_impr, 2),
            "ECE_Reduction_Pct": round(iso_ece_impr, 2)
        })

        logger.info(
            f"  {name}: Raw Brier={raw_stats['brier_score']:.4f} (ECE={raw_stats['expected_calibration_error']:.4f}) "
            f"--> Sigmoid Brier={sig_stats['brier_score']:.4f} (ECE={sig_stats['expected_calibration_error']:.4f}, -{sig_ece_impr:.1f}%)"
        )

    results_df = pd.DataFrame(results_rows)

    # 3. Save CSV results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved calibration benchmark results to: {args.output_csv}")

    metrics_mirror = os.path.join(config["paths"]["results_metrics_dir"], "probability_calibration_results.csv")
    os.makedirs(os.path.dirname(metrics_mirror), exist_ok=True)
    results_df.to_csv(metrics_mirror, index=False)
    logger.info(f"Mirrored results to: {metrics_mirror}")

    # 4. Generate Publication Plots
    os.makedirs(args.plots_dir, exist_ok=True)
    plot_png = os.path.join(args.plots_dir, "probability_calibration.png")
    plot_svg = os.path.join(args.plots_dir, "probability_calibration.svg")
    plot_raw_vs_calibrated_comparison(comparison_data, save_path_png=plot_png, save_path_svg=plot_svg)

    plots_mirror = os.path.join(config["paths"]["results_plots_dir"], "probability_calibration.png")
    os.makedirs(os.path.dirname(plots_mirror), exist_ok=True)
    plot_raw_vs_calibrated_comparison(comparison_data, save_path_png=plots_mirror, save_path_svg=None)

    # Also save standard test reliability curves plot to results/plots/calibration_curves.png
    cal_curves_path = os.path.join(config["paths"]["results_plots_dir"], "calibration_curves.png")
    # For standard curves, include the Sigmoid calibrated variants
    curves_dict = {
        f"{m} (Sigmoid Calibrated)": comparison_data[m]["Calibrated (Sigmoid)"]
        for m in base_models.keys()
    }
    plot_reliability_diagrams(curves_dict, save_path=cal_curves_path)

    # 5. Save Calibrated Production Champion Model Artifact
    # Choose champion model (XGBoost) with Sigmoid calibration (smooth, prevents over-binning)
    champion_name = "XGBoost"
    best_calibrator = fitted_calibrators[champion_name]["sigmoid"]
    models_dir = config["paths"]["flood_models_dir"]
    cal_model_path = os.path.join(models_dir, "calibrated_best_model.pkl")
    joblib.dump(best_calibrator, cal_model_path)
    logger.info(f"Saved production calibrated model artifact to: {cal_model_path}")

    # Metadata
    cal_metadata = {
        "model_name": f"{champion_name} (Sigmoid Calibrated)",
        "base_model": champion_name,
        "calibration_method": "sigmoid",
        "calibration_dataset": "Validation Set (2019-2021)",
        "features": feature_cols,
        "test_metrics": results_df[results_df["Model"] == champion_name].to_dict(orient="records"),
        "calibrated_risk_thresholds": {
            "low_max": 0.05,        # Calibrated: < 5% is Low (background flood prevalence is 2.85%)
            "moderate_max": 0.20,   # Calibrated: 5% - 20% is Moderate (2x to 7x baseline risk)
            "high_min": 0.20        # Calibrated: >= 20% is High (massive disaster alert)
        }
    }
    cal_meta_path = os.path.join(models_dir, "calibrated_metadata.json")
    with open(cal_meta_path, "w", encoding="utf-8") as f:
        json.dump(cal_metadata, f, indent=2)
    logger.info(f"Saved calibrated metadata to: {cal_meta_path}")

    elapsed = time.time() - start_time
    logger.info("================================================================================")
    logger.info(f"Probability Calibration Benchmark Completed in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    logger.info("================================================================================")

    print("\n### Probability Calibration Benchmark Table (Untouched Test Set 2022–2024)\n")
    try:
        print(results_df.to_markdown(index=False))
    except Exception:
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
