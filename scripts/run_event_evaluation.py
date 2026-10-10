"""
Command-line runner for Event-Based Flood Evaluation in Odisha.

This script:
1. Loads the historical dataset (2001–2024).
2. Performs walk-forward chronological splitting (Train: 2001–2018, Val: 2019–2021, Test: 2022–2024).
3. Evaluates all models on the untouched Test Set (2022–2024) across all stations/blocks.
4. Performs detailed event evaluation on key historical disaster episodes:
   - August 2022 Mahanadi Basin Floods (Cuttack)
   - September 2024 Subarnarekha River Inundations (Balasore)
   - May 2021 Cyclone Yaas Deluge (Balasore)
   - August 2020 Baitarani-Brahmani Delta Spate (Bhadrak)
   - October 2018 Cyclone Titli Floods (Ganjam)
   - September 2011 Historic Mahanadi Deltaic Inundations (Cuttack)
5. Generates high-resolution timeline visualizations and multi-model benchmark plots.
6. Outputs reports to reports/flood_events_summary.csv and reports/model_event_metrics.csv.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_combined_data, load_config
from src.models.train_models import prepare_temporal_datasets
from src.models.logistic_regression import create_logistic_regression_pipeline
from src.models.decision_tree import create_decision_tree_model
from src.models.random_forest import create_random_forest_model
from src.models.xgboost_model import create_xgboost_model
from src.models.ann_model import create_ann_model
from src.evaluation.threshold_optimization import ThresholdOptimizer
from src.evaluation.event_evaluation import (
    FloodEventDetector,
    plot_flood_event_timeline,
    plot_multi_model_event_metrics_comparison
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_major_historical_events(
    df: pd.DataFrame,
    trained_models: Dict[str, Any],
    thresholds: Dict[str, float],
    feature_cols: List[str],
    target_col: str,
    detector: FloodEventDetector,
    plots_dir: str = "reports/plots"
) -> pd.DataFrame:
    """
    Evaluate specific prominent real historical disaster episodes across Odisha districts.
    """
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean["Date_Parsed"] = pd.to_datetime(df_clean["Date"])

    historical_episodes = [
        {
            "event_name": "August 2022 Mahanadi Basin Floods",
            "year": 2022,
            "split": "Test (2022-2024)",
            "start_window": "2022-08-01",
            "end_window": "2022-08-31",
            "focal_district": "CUTTACK",
            "display_district": "Cuttack",
            "description": "Severe swelling of Mahanadi river discharging >12 lakh cusecs through Mundali barrage."
        },
        {
            "event_name": "September 2024 Subarnarekha Inundations",
            "year": 2024,
            "split": "Test (2022-2024)",
            "start_window": "2024-09-01",
            "end_window": "2024-10-15",
            "focal_district": "BALASORE",
            "display_district": "Balasore",
            "description": "Subarnarekha & Budhabalanga river spate following deep Bay of Bengal depression."
        },
        {
            "event_name": "May 2021 Cyclone Yaas & Monsoon Deluge",
            "year": 2021,
            "split": "Validation (2019-2021)",
            "start_window": "2021-05-20",
            "end_window": "2021-06-15",
            "focal_district": "BALASORE",
            "display_district": "Balasore",
            "description": "Very severe cyclonic storm Yaas coastal surge and heavy rainfall."
        },
        {
            "event_name": "August 2020 Baitarani-Brahmani Delta Spate",
            "year": 2020,
            "split": "Validation (2019-2021)",
            "start_window": "2020-07-20",
            "end_window": "2020-08-31",
            "focal_district": "BHADRAK",
            "display_district": "Bhadrak",
            "description": "Baitarani and Brahmani deltaic inundations across coastal blocks."
        },
        {
            "event_name": "October 2018 Cyclone Titli Floods",
            "year": 2018,
            "split": "Train (2001-2018)",
            "start_window": "2018-10-01",
            "end_window": "2018-10-25",
            "focal_district": "GANJAM",
            "display_district": "Ganjam",
            "description": "Severe inundations in Rushikulya basin and southern coastal blocks."
        },
        {
            "event_name": "September 2011 Historic Mahanadi Deluge",
            "year": 2011,
            "split": "Train (2001-2018)",
            "start_window": "2011-08-25",
            "end_window": "2011-09-30",
            "focal_district": "CUTTACK",
            "display_district": "Cuttack",
            "description": "Historic Mahanadi deltaic inundation affecting over 19 districts in Odisha."
        }
    ]

    event_rows = []
    
    # Use Production Champion (XGBoost) or Best Model for timeline plotting
    primary_model_name = "XGBoost" if "XGBoost" in trained_models else list(trained_models.keys())[0]
    primary_model = trained_models[primary_model_name]
    primary_thresh = thresholds.get(primary_model_name, 0.5)

    for ep in historical_episodes:
        w_start = pd.to_datetime(ep["start_window"])
        w_end = pd.to_datetime(ep["end_window"])
        dist_code = ep["focal_district"]
        dist_display = ep["display_district"]

        # Filter telemetry for focal district during event window
        mask = (
            (df_clean["District"].str.upper() == dist_code.upper()) &
            (df_clean["Date_Parsed"] >= w_start) &
            (df_clean["Date_Parsed"] <= w_end)
        )
        ep_df = df_clean[mask].sort_values("Date_Parsed").copy()
        if len(ep_df) == 0:
            continue

        # Aggregate district daily series (mean precipitation, max flood occurrence)
        daily_dist = ep_df.groupby("Date_Parsed").agg({
            "Rainfall (mm)": "mean",
            target_col: "max"
        }).reset_index()

        # Generate model prediction probabilities for all records in window
        X_ep = ep_df[feature_cols]
        ep_df["Predicted_Proba"] = primary_model.predict_proba(X_ep)[:, 1]
        daily_probas = ep_df.groupby("Date_Parsed")["Predicted_Proba"].mean().values

        y_true_daily = daily_dist[target_col].values.astype(int)
        y_pred_daily = (daily_probas >= primary_thresh).astype(int)
        dates_daily = daily_dist["Date_Parsed"]
        rainfall_daily = daily_dist["Rainfall (mm)"]

        # Segment into discrete events
        actual_evts = detector.extract_events_from_series(dates_daily, y_true_daily, probability_series=daily_probas, event_prefix="ACT")
        pred_evts = detector.extract_events_from_series(dates_daily, y_pred_daily, probability_series=daily_probas, event_prefix="PRD")

        matched, fa_list, metrics = detector.match_events(
            actual_events=actual_evts,
            predicted_events=pred_evts,
            lead_tolerance_days=2,
            daily_y_true=pd.Series(y_true_daily),
            daily_y_pred=pd.Series(y_pred_daily),
            daily_dates=dates_daily
        )

        for m in matched:
            event_rows.append({
                "Event_Name": ep["event_name"],
                "Temporal_Split": ep["split"],
                "Focal_District": dist_display,
                "Actual_Start": m["actual_start"],
                "Actual_End": m["actual_end"],
                "Actual_Duration_Days": m["actual_duration_days"],
                "Detected": m["detected"],
                "Lead_Time_Days": m["lead_time_days"],
                "Missed_Days": m["missed_days"],
                "Coverage_Pct": m["coverage_pct"],
                "Predicted_Start": m["predicted_start"],
                "Predicted_End": m["predicted_end"],
                "Predicted_Duration_Days": m["predicted_duration_days"],
                "Peak_Inundation_Date": m["peak_date"],
                "Evaluating_Model": primary_model_name
            })

        # Generate Timeline Plots for major events
        slug = ep["event_name"].lower().replace(" ", "_").replace("–", "-").replace("&", "and")
        png_path = os.path.join(plots_dir, f"event_timeline_{slug[:30]}.png")
        svg_path = os.path.join(plots_dir, f"event_timeline_{slug[:30]}.svg")
        
        plot_flood_event_timeline(
            dates=dates_daily,
            rainfall_mm=rainfall_daily,
            y_true=pd.Series(y_true_daily),
            y_proba=pd.Series(daily_probas),
            threshold=primary_thresh,
            event_name=ep["event_name"],
            district_name=dist_display,
            save_path_png=png_path,
            save_path_svg=svg_path
        )

    expected_cols = [
        "Event_Name", "Temporal_Split", "Focal_District", "Actual_Start", "Actual_End",
        "Actual_Duration_Days", "Detected", "Lead_Time_Days", "Missed_Days", "Coverage_Pct",
        "Predicted_Start", "Predicted_End", "Predicted_Duration_Days", "Peak_Inundation_Date", "Evaluating_Model"
    ]
    if len(event_rows) == 0:
        events_summary_df = pd.DataFrame(columns=expected_cols)
    else:
        events_summary_df = pd.DataFrame(event_rows)
    return events_summary_df


def main():
    parser = argparse.ArgumentParser(description="Run complete event-based flood prediction evaluation.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports directory")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("ODISHA FLOOD PREDICTION: EVENT-BASED EVALUATION & TIMELINE BENCHMARK")
    logger.info("=" * 80)

    config = load_config(args.config)
    dataset_path = config["paths"]["combined_data_path"]
    plots_dir = os.path.join(args.reports_dir, "plots")
    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    logger.info(f"Loading historical dataset from {dataset_path}...")
    df = load_combined_data(dataset_path, parse_dates=False)

    train_df, val_df, test_df, feature_cols, target_col = prepare_temporal_datasets(df, config)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values.astype(int)

    X_val = val_df[feature_cols]
    y_val = val_df[target_col].values.astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values.astype(int)

    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    # Instantiate models
    models = {
        "Logistic Regression": create_logistic_regression_pipeline(
            max_iter=config["models"]["logistic_regression"]["max_iter"],
            class_weight="balanced",
            random_state=config["project"]["random_seed"]
        ),
        "Decision Tree": create_decision_tree_model(
            max_depth=config["models"].get("decision_tree", {}).get("max_depth", 8),
            min_samples_leaf=config["models"].get("decision_tree", {}).get("min_samples_leaf", 50),
            class_weight="balanced",
            random_state=config["project"]["random_seed"]
        ),
        "Random Forest": create_random_forest_model(
            n_estimators=config["models"]["random_forest"]["n_estimators"],
            max_depth=config["models"]["random_forest"]["max_depth"],
            class_weight="balanced",
            random_state=config["project"]["random_seed"],
            n_jobs=config["models"]["random_forest"]["n_jobs"]
        ),
        "XGBoost": create_xgboost_model(
            n_estimators=config["models"]["xgboost"]["n_estimators"],
            max_depth=config["models"]["xgboost"]["max_depth"],
            learning_rate=config["models"]["xgboost"]["learning_rate"],
            scale_pos_weight=scale_pos_weight,
            random_state=config["project"]["random_seed"],
            n_jobs=config["models"]["xgboost"]["n_jobs"]
        ),
        "ANN (MLP)": create_ann_model(
            hidden_layer_sizes=tuple(config["models"]["ann"]["hidden_layer_sizes"]),
            activation=config["models"]["ann"]["activation"],
            batch_size=4096,
            max_iter=30,
            random_state=config["project"]["random_seed"]
        )
    }

    optimizer = ThresholdOptimizer(random_state=config["project"]["random_seed"])
    detector = FloodEventDetector(max_gap_days=1, min_event_duration=1)

    trained_models = {}
    optimal_thresholds = {}
    models_event_benchmarks = {}

    for name, model in models.items():
        logger.info(f"\n--- Training {name} for Event-Based Evaluation ---")
        model.fit(X_train, y_train)
        trained_models[name] = model

        val_proba = model.predict_proba(X_val)[:, 1]
        f1_opt = optimizer.optimize_threshold(y_val, val_proba, strategy="f1")
        t_opt = f1_opt["threshold"]
        optimal_thresholds[name] = t_opt

        # Predict on Test Set (2022–2024)
        test_proba = model.predict_proba(X_test)[:, 1]
        y_pred_test = (test_proba >= t_opt).astype(int)

        # Spatiotemporal event evaluation across all stations/blocks
        _, _, event_metrics = detector.evaluate_dataframe_events(
            df=test_df,
            y_true_col=target_col,
            y_pred=y_pred_test,
            y_proba=test_proba,
            date_col="Date",
            station_col="Block/Station",
            lead_tolerance_days=2
        )
        models_event_benchmarks[name] = event_metrics
        logger.info(f"[{name}] Event Hit Rate: {event_metrics['event_recall_hit_rate']*100:.2f}% | Event Precision: {event_metrics['event_precision']*100:.2f}% | Mean Lead Time: {event_metrics['mean_lead_time_days']:.2f} days | False Alarms: {event_metrics['false_alarm_events']:,}")

    # 1. Historical Disaster Event Table
    logger.info("\n--- Evaluating Major Historical Disaster Episodes ---")
    history_events_df = evaluate_major_historical_events(
        df=df,
        trained_models=trained_models,
        thresholds=optimal_thresholds,
        feature_cols=feature_cols,
        target_col=target_col,
        detector=detector,
        plots_dir=plots_dir
    )

    # 2. Multi-Model Event Comparison Plot
    plot_multi_model_event_metrics_comparison(
        models_event_data=models_event_benchmarks,
        save_path_png=os.path.join(plots_dir, "event_evaluation_comparison_all_models.png"),
        save_path_svg=os.path.join(plots_dir, "event_evaluation_comparison_all_models.svg")
    )

    # 3. Model Event Benchmark Table
    m_rows = []
    for m_name, m_stats in models_event_benchmarks.items():
        row = {"Model": m_name, "Threshold": optimal_thresholds[m_name]}
        row.update(m_stats)
        m_rows.append(row)
    model_event_df = pd.DataFrame(m_rows)

    # Save CSV artifacts
    csv_events_summary = os.path.join(args.reports_dir, "flood_events_summary.csv")
    csv_model_event = os.path.join(args.reports_dir, "model_event_metrics.csv")
    csv_results_event = os.path.join("results/metrics", "flood_event_evaluation.csv")

    history_events_df.to_csv(csv_events_summary, index=False)
    model_event_df.to_csv(csv_model_event, index=False)
    model_event_df.to_csv(csv_results_event, index=False)

    logger.info(f"Saved historical flood events summary to {csv_events_summary}")
    logger.info(f"Saved model event metrics to {csv_model_event}")

    # Display Terminal Output Tables
    print("\n" + "=" * 125)
    print("HISTORICAL FLOOD DISASTER EVENT-BASED EVALUATION TABLE (REAL GROUND-TRUTH TELEMETRY)")
    print("=" * 125)
    display_cols = [
        "Event_Name", "Temporal_Split", "Focal_District", "Actual_Start", "Actual_End",
        "Actual_Duration_Days", "Detected", "Lead_Time_Days", "Missed_Days", "Coverage_Pct"
    ]
    if len(history_events_df) > 0 and all(c in history_events_df.columns for c in display_cols):
        print(history_events_df[display_cols].to_string(index=False))
    else:
        print(history_events_df.to_string(index=False))
    print("=" * 125 + "\n")

    print("\n" + "=" * 115)
    print("TEST SET (2022-2024) MACRO EVENT-LEVEL PERFORMANCE BENCHMARK ACROSS MODELS")
    print("=" * 115)
    m_disp_cols = [
        "Model", "Threshold", "total_actual_events", "detected_events_hits", "missed_events",
        "event_recall_hit_rate", "event_precision", "event_f1_score", "mean_lead_time_days", "false_alarm_events"
    ]
    print(model_event_df[m_disp_cols].to_string(index=False))
    print("=" * 115 + "\n")

    elapsed = time.time() - start_time
    logger.info(f"Event-based evaluation completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
