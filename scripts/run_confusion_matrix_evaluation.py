"""
Comprehensive Confusion Matrix Evaluation Runner for Odisha Flood Prediction Models.

Discovers all major models in the repository, evaluates predictions on the untouched
2022–2024 test set, calculates full confusion matrix breakdowns (TN, FP, FN, TP, Acc,
Prec, Recall, F1, F2, FNR, FPR), and generates individual/multi-model plots (PNG & SVG)
plus structured CSV/JSON results.
"""

import os
import sys
import json
import time
import logging
import argparse
import importlib
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_combined_data, load_config
from src.evaluation.confusion_matrix import (
    calculate_confusion_matrix_metrics,
    plot_styled_confusion_matrix,
    plot_multi_model_confusion_matrix_grid
)
from src.models.train_models import prepare_temporal_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def discover_available_models(config: dict, scale_pos_weight: float = 38.96) -> Dict[str, Any]:
    """
    Dynamically discover and instantiate all available model architectures in the repository.
    Scans src/models/ and instantiates models using config hyperparameters.
    """
    discovered_models = {}
    
    # Model module registry mappings
    model_factories = [
        ("Logistic Regression", "src.models.logistic_regression", "create_logistic_regression_pipeline", {
            "max_iter": config.get("models", {}).get("logistic_regression", {}).get("max_iter", 1000),
            "class_weight": config.get("models", {}).get("logistic_regression", {}).get("class_weight", "balanced"),
            "random_state": config.get("project", {}).get("random_seed", 42)
        }),
        ("Decision Tree", "src.models.decision_tree", "create_decision_tree_model", {
            "max_depth": config.get("models", {}).get("decision_tree", {}).get("max_depth", 8),
            "min_samples_leaf": config.get("models", {}).get("decision_tree", {}).get("min_samples_leaf", 50),
            "class_weight": config.get("models", {}).get("decision_tree", {}).get("class_weight", "balanced"),
            "random_state": config.get("project", {}).get("random_seed", 42)
        }),
        ("Random Forest", "src.models.random_forest", "create_random_forest_model", {
            "n_estimators": config.get("models", {}).get("random_forest", {}).get("n_estimators", 100),
            "max_depth": config.get("models", {}).get("random_forest", {}).get("max_depth", 14),
            "class_weight": "balanced",
            "random_state": config.get("project", {}).get("random_seed", 42),
            "n_jobs": config.get("models", {}).get("random_forest", {}).get("n_jobs", -1)
        }),
        ("XGBoost", "src.models.xgboost_model", "create_xgboost_model", {
            "n_estimators": config.get("models", {}).get("xgboost", {}).get("n_estimators", 150),
            "max_depth": config.get("models", {}).get("xgboost", {}).get("max_depth", 6),
            "learning_rate": config.get("models", {}).get("xgboost", {}).get("learning_rate", 0.05),
            "scale_pos_weight": scale_pos_weight,
            "random_state": config.get("project", {}).get("random_seed", 42),
            "n_jobs": config.get("models", {}).get("xgboost", {}).get("n_jobs", -1)
        }),
        ("ANN (MLP)", "src.models.ann_model", "create_ann_model", {
            "hidden_layer_sizes": tuple(config.get("models", {}).get("ann", {}).get("hidden_layer_sizes", [64, 32])),
            "activation": config.get("models", {}).get("ann", {}).get("activation", "relu"),
            "batch_size": 4096,
            "max_iter": 30,
            "random_state": config.get("project", {}).get("random_seed", 42)
        })
    ]
    
    for name, mod_path, func_name, kwargs in model_factories:
        try:
            mod = importlib.import_module(mod_path)
            factory_fn = getattr(mod, func_name)
            model_inst = factory_fn(**kwargs)
            discovered_models[name] = model_inst
            logger.info(f"Discovered model architecture: {name} (from {mod_path}.{func_name})")
        except Exception as e:
            logger.warning(f"Could not load model {name} from {mod_path}: {e}")
            
    return discovered_models


def load_model_thresholds(thresholds_path: str = "models/flood_prediction/model_thresholds.json") -> Dict[str, Dict[str, float]]:
    """Load pre-computed validation-optimal thresholds if available."""
    if os.path.exists(thresholds_path):
        try:
            with open(thresholds_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            parsed = {}
            for model_name, info in data.items():
                f1_t = info.get("balanced_f1_strategy", {}).get("selected_threshold", 0.5)
                f2_t = info.get("disaster_averse_f2_strategy", {}).get("selected_threshold", 0.5)
                parsed[model_name] = {
                    "f1_threshold": float(f1_t),
                    "f2_threshold": float(f2_t)
                }
            return parsed
        except Exception as e:
            logger.warning(f"Could not parse thresholds from {thresholds_path}: {e}")
            
    # Default fallback thresholds
    return {
        "Logistic Regression": {"f1_threshold": 0.8503, "f2_threshold": 0.5200},
        "Decision Tree": {"f1_threshold": 0.8265, "f2_threshold": 0.6184},
        "Random Forest": {"f1_threshold": 0.7607, "f2_threshold": 0.4363},
        "XGBoost": {"f1_threshold": 0.8073, "f2_threshold": 0.5083},
        "ANN (MLP)": {"f1_threshold": 0.1075, "f2_threshold": 0.0133}
    }


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive confusion matrix evaluation on untouched test set.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports directory")
    parser.add_argument("--results-dir", type=str, default="results/metrics", help="Results metrics directory")
    parser.add_argument("--plots-dir", type=str, default="reports/plots", help="Plots directory")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("ODISHA FLOOD PREDICTION: COMPREHENSIVE CONFUSION MATRIX EVALUATION")
    logger.info("=" * 80)

    config = load_config(args.config)
    dataset_path = config["paths"]["combined_data_path"]
    
    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    logger.info(f"Loading historical dataset from {dataset_path}...")
    df = load_combined_data(dataset_path, parse_dates=False)

    train_df, val_df, test_df, feature_cols, target_col = prepare_temporal_datasets(df, config)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values.astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values.astype(int)

    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    logger.info(f"Test Set (Untouched 2022–2024): {len(X_test):,} total observations ({np.sum(y_test==1):,} actual floods, {np.sum(y_test==0):,} actual non-floods)")

    # 1. Discover models
    models = discover_available_models(config, scale_pos_weight=scale_pos_weight)
    thresholds_lookup = load_model_thresholds()

    grid_predictions = {}
    summary_records = []

    for name, model in models.items():
        logger.info(f"\n--- Training {name} on Train Set (2001–2018, {len(X_train):,} rows) ---")
        model.fit(X_train, y_train)

        # Predict on UNTOUCHED Test Set (2022–2024)
        test_proba = model.predict_proba(X_test)[:, 1]

        # Get thresholds
        m_thresh = thresholds_lookup.get(name, {"f1_threshold": 0.5, "f2_threshold": 0.5})
        t_f1 = m_thresh["f1_threshold"]
        t_f2 = m_thresh["f2_threshold"]

        eval_scenarios = [
            ("Validation-Optimal F1 (Balanced)", t_f1),
            ("Disaster-Averse F2 (Recall-Weighted)", t_f2),
            ("Standard Baseline (0.50)", 0.50)
        ]

        for strat_label, thresh in eval_scenarios:
            y_pred_t = (test_proba >= thresh).astype(int)
            metrics = calculate_confusion_matrix_metrics(
                y_true=y_test,
                y_pred=y_pred_t,
                model_name=name,
                threshold=thresh,
                split_name="Test Set (2022–2024)"
            )
            metrics["strategy"] = strat_label
            summary_records.append(metrics)

        # Primary optimal strategy for individual and grid plotting
        y_pred_f1 = (test_proba >= t_f1).astype(int)
        grid_predictions[name] = (y_test, y_pred_f1, t_f1)

        # Generate individual model plot in PNG & SVG
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        ind_png_path = os.path.join(args.plots_dir, f"confusion_matrix_{slug}.png")
        ind_svg_path = os.path.join(args.plots_dir, f"confusion_matrix_{slug}.svg")
        res_png_path = os.path.join("results/plots", f"confusion_matrix_{slug}.png")
        
        plot_styled_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred_f1,
            model_name=f"{name} (Balanced F1)",
            threshold=t_f1,
            save_path_png=ind_png_path,
            save_path_svg=ind_svg_path
        )
        # Also copy/save to results/plots
        plot_styled_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred_f1,
            model_name=f"{name} (Balanced F1)",
            threshold=t_f1,
            save_path_png=res_png_path
        )

    # 2. Generate multi-model comparison grid in PNG & SVG
    grid_png_path = os.path.join(args.plots_dir, "confusion_matrices_all_models.png")
    grid_svg_path = os.path.join(args.plots_dir, "confusion_matrices_all_models.svg")
    res_grid_png = os.path.join("results/plots", "confusion_matrices_all_models.png")

    plot_multi_model_confusion_matrix_grid(
        models_predictions=grid_predictions,
        save_path_png=grid_png_path,
        save_path_svg=grid_svg_path
    )
    plot_multi_model_confusion_matrix_grid(
        models_predictions=grid_predictions,
        save_path_png=res_grid_png
    )

    # 3. Save Summary Tables (CSV and JSON)
    summary_df = pd.DataFrame(summary_records)
    
    csv_report_path = os.path.join(args.reports_dir, "confusion_matrix_summary.csv")
    json_report_path = os.path.join(args.reports_dir, "confusion_matrix_summary.json")
    csv_results_path = os.path.join(args.results_dir, "confusion_matrix_evaluation.csv")
    json_results_path = os.path.join(args.results_dir, "confusion_matrix_evaluation.json")

    summary_df.to_csv(csv_report_path, index=False)
    summary_df.to_csv(csv_results_path, index=False)
    
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
        
    with open(json_results_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    logger.info(f"Saved confusion matrix summary CSV to {csv_report_path}")
    logger.info(f"Saved confusion matrix summary JSON to {json_report_path}")

    # 4. Display Formatted Table
    print("\n" + "=" * 125)
    print("UNTOUCHED TEST SET (2022–2024, 343,758 OBS) — CONFUSION MATRIX EVALUATION BREAKDOWN")
    print("=" * 125)
    display_cols = [
        "model_name", "strategy", "threshold",
        "true_negatives_tn", "false_positives_fp", "false_negatives_fn", "true_positives_tp",
        "accuracy", "precision", "recall_sensitivity", "f1_score", "f2_score",
        "false_negative_rate_fnr", "false_positive_rate_fpr"
    ]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 125 + "\n")

    elapsed = time.time() - start_time
    logger.info(f"Confusion matrix evaluation completed successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
