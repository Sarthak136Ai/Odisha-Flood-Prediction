"""
Diagnostic script for scientific investigation of potential leakage in Odisha Flood Prediction.
Analyzes:
1. Target leakage & correlation analysis
2. Temporal leakage & train/val/test overlap
3. Feature distributions across splits
4. Model coefficients for trained Logistic Regression
5. Ablation study: evaluating impact of Flood_Occurred, Year, and meteorological features
6. Duplicate records & station/date duplication
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    brier_score_loss,
    confusion_matrix
)

# Insert project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_combined_data, load_config
from src.evaluation.metrics import calculate_metrics, find_optimal_threshold


def run_diagnostics():
    print("=" * 80)
    print("SCIENTIFIC LEAKAGE & MODEL METHODOLOGY AUDIT")
    print("=" * 80)

    # 1. Load Data
    config = load_config("config.yaml")
    data_path = config["paths"]["combined_data_path"]
    print(f"Loading data from {data_path}...")
    df = load_combined_data(data_path, parse_dates=False)
    print(f"Total rows: {len(df):,}, Total columns: {len(df.columns)}")

    # 2. Check Target & Clean
    target_col = "Flood_Next_Day"
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean[target_col] = df_clean[target_col].astype(int)

    # 3. Check Duplicates & Overlaps
    print("\n" + "-" * 50)
    print("1. DUPLICATE & OVERLAP CHECKS")
    print("-" * 50)
    dups = df.duplicated(subset=["District", "Block/Station", "Date"]).sum()
    print(f"Duplicate [District, Block/Station, Date] records: {dups}")

    train_years = config["split"]["train_years"]
    val_years = config["split"]["val_years"]
    test_years = config["split"]["test_years"]

    train_df = df_clean[df_clean["Year"].isin(train_years)].copy()
    val_df = df_clean[df_clean["Year"].isin(val_years)].copy()
    test_df = df_clean[df_clean["Year"].isin(test_years)].copy()

    print(f"Train dates: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df):,} rows)")
    print(f"Val dates:   {val_df['Date'].min()} to {val_df['Date'].max()} ({len(val_df):,} rows)")
    print(f"Test dates:  {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df):,} rows)")

    train_dates = set(train_df["Date"].unique())
    val_dates = set(val_df["Date"].unique())
    test_dates = set(test_df["Date"].unique())

    print(f"Overlap Train & Val dates: {len(train_dates.intersection(val_dates))}")
    print(f"Overlap Val & Test dates:  {len(val_dates.intersection(test_dates))}")
    print(f"Overlap Train & Test dates: {len(train_dates.intersection(test_dates))}")

    train_stations = set(train_df["District"] + "_" + train_df["Block/Station"])
    val_stations = set(val_df["District"] + "_" + val_df["Block/Station"])
    test_stations = set(test_df["District"] + "_" + test_df["Block/Station"])
    print(f"Unique Stations - Train: {len(train_stations)}, Val: {len(val_stations)}, Test: {len(test_stations)}")

    # 4. Target & Base Rate
    print("\n" + "-" * 50)
    print("2. BASE RATES & FLOOD_OCCURRED ANALYSIS")
    print("-" * 50)
    print(f"Train Flood_Next_Day Rate: {train_df[target_col].mean()*100:.3f}% ({train_df[target_col].sum():,} floods)")
    print(f"Val Flood_Next_Day Rate:   {val_df[target_col].mean()*100:.3f}% ({val_df[target_col].sum():,} floods)")
    print(f"Test Flood_Next_Day Rate:  {test_df[target_col].mean()*100:.3f}% ({test_df[target_col].sum():,} floods)")

    if "Flood_Occurred" in df_clean.columns:
        print(f"Train Flood_Occurred Rate: {train_df['Flood_Occurred'].mean()*100:.3f}%")
        print(f"Val Flood_Occurred Rate:   {val_df['Flood_Occurred'].mean()*100:.3f}%")
        print(f"Test Flood_Occurred Rate:  {test_df['Flood_Occurred'].mean()*100:.3f}%")

        # Multi-day flood persistence check: P(Flood_Next_Day=1 | Flood_Occurred=1) vs P(Flood_Next_Day=1 | Flood_Occurred=0)
        p_f1_given_f0 = train_df[train_df["Flood_Occurred"] == 0][target_col].mean()
        p_f1_given_f1 = train_df[train_df["Flood_Occurred"] == 1][target_col].mean()
        print(f"\n[Persistence Analysis in Training Data]:")
        print(f"P(Flood_Next_Day=1 | Flood_Occurred=0) = {p_f1_given_f1*100:.2f}%")
        print(f"P(Flood_Next_Day=1 | Flood_Occurred=1) = {p_f1_given_f1*100:.2f}% (Persistence probability)")
        print(f"P(Flood_Next_Day=1 | Flood_Occurred=0) = {p_f1_given_f0*100:.2f}% (Onset probability)")

    # 5. Feature Correlation with Target
    print("\n" + "-" * 50)
    print("3. FEATURE-TARGET CORRELATION ANALYSIS")
    print("-" * 50)
    num_cols = [
        c for c in train_df.columns 
        if c not in ["District", "Block/Station", "Date", "Date_Parsed", target_col]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    corrs = []
    for c in num_cols:
        r = train_df[c].corr(train_df[target_col])
        corrs.append({"Feature": c, "Pearson_Correlation": r, "Abs_Correlation": abs(r)})
    df_corrs = pd.DataFrame(corrs).sort_values(by="Abs_Correlation", ascending=False)
    print(df_corrs.to_string(index=False))

    # 6. Current Model Coefficient Inspection
    print("\n" + "-" * 50)
    print("4. CURRENT LOGISTIC REGRESSION COEFFICIENTS")
    print("-" * 50)
    model_path = "models/flood_prediction/best_model.pkl"
    meta_path = "models/flood_prediction/model_metadata.json"
    if os.path.exists(model_path) and os.path.exists(meta_path):
        cur_model = joblib.load(model_path)
        with open(meta_path, "r") as f:
            cur_meta = json.load(f)
        feat_names = cur_meta["features"]
        if hasattr(cur_model, "named_steps") and "classifier" in cur_model.named_steps:
            clf = cur_model.named_steps["classifier"]
            scaler = cur_model.named_steps["scaler"]
            coefs = clf.coef_[0]
            intercept = clf.intercept_[0]
            df_coefs = pd.DataFrame({
                "Feature": feat_names,
                "Coefficient": coefs,
                "Abs_Coefficient": np.abs(coefs),
                "Scaler_Mean": scaler.mean_,
                "Scaler_Scale": scaler.scale_
            }).sort_values(by="Abs_Coefficient", ascending=False)
            print(f"Intercept: {intercept:.4f}")
            print(df_coefs.to_string(index=False))

    # 7. Controlled Ablation Experiments
    print("\n" + "-" * 50)
    print("5. CONTROLLED ABLATION EXPERIMENTS")
    print("-" * 50)

    feature_sets = {
        "1. Full Baseline (Current config with Flood_Occurred & Year)": [
            "Year", "Month_Number", "Day", "Day_of_Year", "Day_of_Week", "Month_sin", "Month_cos",
            "Rainfall (mm)", "Rainfall_Missing", "Rainfall_Lag_1d", "Rainfall_Lag_2d", "Rainfall_Lag_3d", "Rainfall_Lag_7d",
            "Rainfall_Prev_3d_Sum", "Rainfall_Prev_7d_Sum", "Rainfall_Prev_15d_Sum", "Rainfall_Prev_30d_Sum",
            "Rainfall_Prev_3d_Max", "Rainfall_Prev_7d_Max", "Rainfall_Prev_15d_Max", "Rainfall_Prev_30d_Max",
            "Rainy_Days_Prev_3d", "Rainy_Days_Prev_7d", "Rainy_Days_Prev_15d", "Rainy_Days_Prev_30d",
            "Consecutive_Rainy_Days_Before", "Flood_Occurred"
        ],
        "2. Without Flood_Occurred (With Year)": [
            "Year", "Month_Number", "Day", "Day_of_Year", "Day_of_Week", "Month_sin", "Month_cos",
            "Rainfall (mm)", "Rainfall_Missing", "Rainfall_Lag_1d", "Rainfall_Lag_2d", "Rainfall_Lag_3d", "Rainfall_Lag_7d",
            "Rainfall_Prev_3d_Sum", "Rainfall_Prev_7d_Sum", "Rainfall_Prev_15d_Sum", "Rainfall_Prev_30d_Sum",
            "Rainfall_Prev_3d_Max", "Rainfall_Prev_7d_Max", "Rainfall_Prev_15d_Max", "Rainfall_Prev_30d_Max",
            "Rainy_Days_Prev_3d", "Rainy_Days_Prev_7d", "Rainy_Days_Prev_15d", "Rainy_Days_Prev_30d",
            "Consecutive_Rainy_Days_Before"
        ],
        "3. Pure Meteorological & Seasonal (No Flood_Occurred, No Year)": [
            "Month_Number", "Day", "Day_of_Year", "Day_of_Week", "Month_sin", "Month_cos",
            "Rainfall (mm)", "Rainfall_Missing", "Rainfall_Lag_1d", "Rainfall_Lag_2d", "Rainfall_Lag_3d", "Rainfall_Lag_7d",
            "Rainfall_Prev_3d_Sum", "Rainfall_Prev_7d_Sum", "Rainfall_Prev_15d_Sum", "Rainfall_Prev_30d_Sum",
            "Rainfall_Prev_3d_Max", "Rainfall_Prev_7d_Max", "Rainfall_Prev_15d_Max", "Rainfall_Prev_30d_Max",
            "Rainy_Days_Prev_3d", "Rainy_Days_Prev_7d", "Rainy_Days_Prev_15d", "Rainy_Days_Prev_30d",
            "Consecutive_Rainy_Days_Before"
        ],
        "4. Pure Rainfall Accumulation Only (No calendar, no persistence)": [
            "Rainfall (mm)", "Rainfall_Lag_1d", "Rainfall_Lag_2d", "Rainfall_Lag_3d", "Rainfall_Lag_7d",
            "Rainfall_Prev_3d_Sum", "Rainfall_Prev_7d_Sum", "Rainfall_Prev_15d_Sum", "Rainfall_Prev_30d_Sum",
            "Rainfall_Prev_3d_Max", "Rainfall_Prev_7d_Max", "Rainfall_Prev_15d_Max", "Rainfall_Prev_30d_Max",
            "Rainy_Days_Prev_3d", "Rainy_Days_Prev_7d", "Rainy_Days_Prev_15d", "Rainy_Days_Prev_30d",
            "Consecutive_Rainy_Days_Before"
        ],
        "5. Persistence Only (Flood_Occurred only)": [
            "Flood_Occurred"
        ]
    }

    ablation_results = []

    for exp_name, feats in feature_sets.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ])

        X_tr = train_df[feats]
        y_tr = train_df[target_col].values
        X_v = val_df[feats]
        y_v = val_df[target_col].values
        X_te = test_df[feats]
        y_te = test_df[target_col].values

        pipe.fit(X_tr, y_tr)

        val_prob = pipe.predict_proba(X_v)[:, 1]
        opt_thresh, opt_f1 = find_optimal_threshold(y_v, val_prob, metric="f1")

        val_roc = roc_auc_score(y_v, val_prob)
        val_pr = average_precision_score(y_v, val_prob)

        test_prob = pipe.predict_proba(X_te)[:, 1]
        test_roc = roc_auc_score(y_te, test_prob)
        test_pr = average_precision_score(y_te, test_prob)
        test_metrics = calculate_metrics(y_te, test_prob, threshold=opt_thresh, prefix="test_")

        ablation_results.append({
            "Experiment": exp_name,
            "Num_Features": len(feats),
            "Opt_Threshold": round(opt_thresh, 4),
            "Val_ROC_AUC": round(val_roc, 4),
            "Val_PR_AUC": round(val_pr, 4),
            "Val_F1": round(opt_f1, 4),
            "Test_ROC_AUC": round(test_roc, 4),
            "Test_PR_AUC": round(test_pr, 4),
            "Test_F1": round(test_metrics["test_f1"], 4),
            "Test_Recall": round(test_metrics["test_recall"], 4),
            "Test_Precision": round(test_metrics["test_precision"], 4),
            "Test_Brier": round(test_metrics["test_brier_score"], 4)
        })

    df_abl = pd.DataFrame(ablation_results)
    print("\nABLATION EXPERIMENT RESULTS (Logistic Regression):")
    print(df_abl.to_string(index=False))

    # Also test Random Forest and XGBoost under Pure Meteorological (No Flood_Occurred, No Year)
    print("\n" + "-" * 50)
    print("6. MULTI-MODEL BENCHMARK UNDER PURE METEOROLOGICAL SETTING (No Flood_Occurred, No Year)")
    print("-" * 50)
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    pure_feats = feature_sets["3. Pure Meteorological & Seasonal (No Flood_Occurred, No Year)"]
    num_neg = np.sum(train_df[target_col] == 0)
    num_pos = np.sum(train_df[target_col] == 1)
    spw = float(num_neg / (num_pos + 1e-5))

    pure_models = {
        "Logistic Regression (Pure Met)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "Random Forest (Pure Met)": RandomForestClassifier(
            n_estimators=100, max_depth=14, class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "XGBoost (Pure Met)": XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=spw, random_state=42, n_jobs=-1, eval_metric="logloss"
        )
    }

    pure_results = []
    X_tr = train_df[pure_feats]
    y_tr = train_df[target_col].values
    X_v = val_df[pure_feats]
    y_v = val_df[target_col].values
    X_te = test_df[pure_feats]
    y_te = test_df[target_col].values

    for m_name, model in pure_models.items():
        print(f"Training {m_name}...")
        model.fit(X_tr, y_tr)
        val_prob = model.predict_proba(X_v)[:, 1]
        opt_thresh, opt_f1 = find_optimal_threshold(y_v, val_prob, metric="f1")
        val_roc = roc_auc_score(y_v, val_prob)
        val_pr = average_precision_score(y_v, val_prob)

        test_prob = model.predict_proba(X_te)[:, 1]
        test_roc = roc_auc_score(y_te, test_prob)
        test_pr = average_precision_score(y_te, test_prob)
        test_metrics = calculate_metrics(y_te, test_prob, threshold=opt_thresh, prefix="test_")

        pure_results.append({
            "Model": m_name,
            "Opt_Threshold": round(opt_thresh, 4),
            "Val_ROC_AUC": round(val_roc, 4),
            "Val_PR_AUC": round(val_pr, 4),
            "Val_F1": round(opt_f1, 4),
            "Test_ROC_AUC": round(test_roc, 4),
            "Test_PR_AUC": round(test_pr, 4),
            "Test_F1": round(test_metrics["test_f1"], 4),
            "Test_Recall": round(test_metrics["test_recall"], 4),
            "Test_Precision": round(test_metrics["test_precision"], 4),
            "Test_Brier": round(test_metrics["test_brier_score"], 4)
        })

    df_pure = pd.DataFrame(pure_results)
    print("\nPURE METEOROLOGICAL MULTI-MODEL BENCHMARK:")
    print(df_pure.to_string(index=False))

    # Save diagnostic results to results/metrics/leakage_diagnostics.json
    diag_summary = {
        "duplicates": int(dups),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "feature_correlations": df_corrs.to_dict(orient="records"),
        "ablation_results": df_abl.to_dict(orient="records"),
        "pure_meteorological_benchmark": df_pure.to_dict(orient="records")
    }

    diag_out_path = "results/metrics/leakage_diagnostics.json"
    os.makedirs(os.path.dirname(diag_out_path), exist_ok=True)
    with open(diag_out_path, "w") as f:
        json.dump(diag_summary, f, indent=2)
    print(f"\nSaved diagnostic data to {diag_out_path}")


if __name__ == "__main__":
    run_diagnostics()
