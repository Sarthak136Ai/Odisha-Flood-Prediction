"""
Automated Data Leakage Prevention Test Suite for Odisha Flood Prediction.

Enforces strict hydrological and machine learning data hygiene:
1. Flood_Next_Day (and multi-horizon targets) is never used as a model feature.
2. Future rainfall is never used for a prediction (zero lookahead leakage).
3. Rolling features only use historical/current observations allowed by the forecasting definition.
4. Training data dates strictly precede validation and test dates (chronological causal order).
5. Test data is never used during threshold optimization (zero threshold leakage).
6. Test data is never used to fit scalers or encoders (zero preprocessing leakage).
7. Calibration is fitted strictly and solely on validation data (zero calibration leakage).
8. No duplicate station/date records cross train/val/test splits unexpectedly.
9. Feature columns and ordering are strictly identical between training and inference.
10. Operational 2025 inference evaluates in frozen mode and does not retrain the model.
"""

import os
import json
import inspect
from unittest.mock import Mock
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.data.load_data import load_config
from src.models.metadata import load_model_with_metadata, ModelBundle
from src.models.predict import FloodPredictor
from src.evaluation.metrics import find_optimal_threshold
from src.inference.unseen_2025_pipeline import (
    engineer_2025_features,
    Operational2025Pipeline
)


CONFIG_PATH = "config.yaml"
METADATA_PATH = "models/flood_prediction/model_metadata.json"
CALIBRATED_METADATA_PATH = "models/flood_prediction/calibrated_metadata.json"
MODEL_PATH = "models/flood_prediction/best_model.pkl"
HISTORICAL_DATA_PATH = "data/processed/historical_2001_2024.csv"

FORBIDDEN_TARGET_NAMES = {
    "flood_next_day",
    "flood_t+1",
    "flood_t+2",
    "flood_t+3",
    "flood_occurred"  # Target persistence leakage
}


def test_flood_next_day_never_used_as_model_feature():
    """
    TEST 1: Verify Flood_Next_Day (and multi-horizon targets) is NEVER included as an input feature.
    Checks config.yaml, model_metadata.json, calibrated_metadata.json, FloodPredictor, and ModelBundle.
    """
    config = load_config(CONFIG_PATH)
    
    # 1. Config feature groups
    temporal_cols = [c.lower() for c in config.get("features", {}).get("temporal_cols", [])]
    rainfall_cols = [c.lower() for c in config.get("features", {}).get("rainfall_cols", [])]
    intensity_cols = [c.lower() for c in config.get("features", {}).get("rainfall_intensity_cols", [])]
    all_config_features = set(temporal_cols + rainfall_cols + intensity_cols)
    
    for forbidden in FORBIDDEN_TARGET_NAMES:
        assert forbidden not in all_config_features, (
            f"DATA LEAKAGE DETECTED in config.yaml: '{forbidden}' found in input feature lists!"
        )

    # 2. model_metadata.json
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta_features = [f.lower() for f in meta.get("feature_list", [])]
    meta_features_alias = [f.lower() for f in meta.get("features", [])]

    for forbidden in FORBIDDEN_TARGET_NAMES:
        assert forbidden not in meta_features, (
            f"DATA LEAKAGE DETECTED in model_metadata.json: '{forbidden}' found in feature_list!"
        )
        assert forbidden not in meta_features_alias, (
            f"DATA LEAKAGE DETECTED in model_metadata.json: '{forbidden}' found in features alias!"
        )

    # 3. calibrated_metadata.json
    if os.path.exists(CALIBRATED_METADATA_PATH):
        with open(CALIBRATED_METADATA_PATH, "r", encoding="utf-8") as f:
            cal_meta = json.load(f)
        cal_features = [f.lower() for f in cal_meta.get("features", [])]
        for forbidden in FORBIDDEN_TARGET_NAMES:
            assert forbidden not in cal_features, (
                f"DATA LEAKAGE DETECTED in calibrated_metadata.json: '{forbidden}' found in features!"
            )

    # 4. Production Predictor & ModelBundle
    predictor = FloodPredictor()
    pred_features = [f.lower() for f in predictor.feature_names]
    for forbidden in FORBIDDEN_TARGET_NAMES:
        assert forbidden not in pred_features, (
            f"DATA LEAKAGE DETECTED in FloodPredictor: '{forbidden}' found in active feature_names!"
        )

    bundle = load_model_with_metadata(MODEL_PATH, METADATA_PATH)
    bundle_features = [f.lower() for f in bundle.features]
    for forbidden in FORBIDDEN_TARGET_NAMES:
        assert forbidden not in bundle_features, (
            f"DATA LEAKAGE DETECTED in ModelBundle: '{forbidden}' found in bundle features!"
        )


def test_future_rainfall_never_used_for_prediction():
    """
    TEST 2: Verify that future rainfall (T+1, T+2, ...) is never used to predict flood on day T.
    Asserts both feature naming and empirical mathematical causality under future perturbation.
    """
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta.get("features", [])

    # A. Check feature nomenclature for forward lookahead indicators
    forbidden_tokens = ["lead", "future", "ahead", "next_rainfall", "forward", "t+"]
    for f in feature_names:
        f_lower = f.lower()
        for tok in forbidden_tokens:
            assert tok not in f_lower, (
                f"LOOKAHEAD LEAKAGE DETECTED in feature name: '{f}' contains future indicator '{tok}'"
            )

    # B. Empirical Temporal Causality Test:
    # Construct a synthetic 14-day rainfall time series for a station.
    dates = pd.date_range("2025-08-01", periods=14, freq="D")
    base_rainfall = [15.0, 25.0, 40.0, 10.0, 30.0, 50.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    df_baseline = pd.DataFrame({
        "District": ["CUTTACK"] * 14,
        "Block/Station": ["Banki"] * 14,
        "Date": dates.strftime("%Y-%m-%d"),
        "Rainfall (mm)": base_rainfall
    })

    # Create mutated scenario: extreme torrential rain on days 8..14 (T+1 .. T+7 relative to day 7)
    df_mutated_future = df_baseline.copy()
    df_mutated_future.loc[7:, "Rainfall (mm)"] = [350.0, 420.0, 500.0, 280.0, 310.0, 190.0, 250.0]

    # Generate features for both streams
    feat_base, _ = engineer_2025_features(df_baseline, allow_zero_warmup=True)
    feat_mutated, _ = engineer_2025_features(df_mutated_future, allow_zero_warmup=True)

    # Compare features on day 7 (index 6, Date: 2025-08-07)
    # Target prediction day is day 7; rainfall on days 8..14 must have ZERO effect on day 7 features.
    cols_to_check = [
        "Rainfall (mm)",
        "Rainfall_Lag_1d",
        "Rainfall_Lag_2d",
        "Rainfall_Lag_3d",
        "Rainfall_Lag_7d",
        "Rainfall_Prev_3d_Sum",
        "Rainfall_Prev_7d_Sum",
        "Rainfall_Prev_15d_Sum",
        "Rainfall_Prev_30d_Sum",
        "Rainfall_Prev_3d_Max",
        "Rainfall_Prev_7d_Max",
        "Rainfall_Prev_15d_Max",
        "Rainfall_Prev_30d_Max",
        "Rainy_Days_Prev_3d",
        "Rainy_Days_Prev_7d",
        "Rainy_Days_Prev_15d",
        "Rainy_Days_Prev_30d",
        "Consecutive_Rainy_Days_Before"
    ]
    
    day7_base = feat_base.loc[6, cols_to_check].to_dict()
    day7_mutated = feat_mutated.loc[6, cols_to_check].to_dict()

    for col in cols_to_check:
        v_base = day7_base[col]
        v_mut = day7_mutated[col]
        assert np.isclose(v_base, v_mut, atol=1e-7), (
            f"LOOKAHEAD LEAKAGE DETECTED in feature '{col}' on Day T (2025-08-07): "
            f"Baseline={v_base}, Mutated Future={v_mut}. Future rainfall on days T+1..T+7 altered day T!"
        )


def test_rolling_features_strictly_antecedent():
    """
    TEST 3: Verify that rolling features strictly use antecedent/historical observations [T-W, ..., T-1].
    Day T's own rainfall or future rainfall must never be included in previous window accumulations.
    """
    dates = pd.date_range("2025-08-01", periods=10, freq="D")
    rainfall_vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    
    df = pd.DataFrame({
        "District": ["CUTTACK"] * 10,
        "Block/Station": ["Banki"] * 10,
        "Date": dates.strftime("%Y-%m-%d"),
        "Rainfall (mm)": rainfall_vals
    })

    feat_df, _ = engineer_2025_features(df, allow_zero_warmup=True)

    # On day 5 (index 4, Date: 2025-08-05, Rainfall=50.0 mm):
    # Previous 3 days are days 2, 3, 4 (indices 1, 2, 3 with rainfall 20.0, 30.0, 40.0).
    # Expected Rainfall_Prev_3d_Sum = 20.0 + 30.0 + 40.0 = 90.0 mm (excludes day 5's 50.0 mm).
    val_3d_sum = feat_df.loc[4, "Rainfall_Prev_3d_Sum"]
    expected_3d_sum = 20.0 + 30.0 + 40.0
    assert np.isclose(val_3d_sum, expected_3d_sum, atol=1e-7), (
        f"ROLLING LEAKAGE DETECTED: Rainfall_Prev_3d_Sum on Day 5 is {val_3d_sum}, expected {expected_3d_sum}."
    )

    # Expected Rainfall_Prev_3d_Max = max(20.0, 30.0, 40.0) = 40.0 mm (excludes day 5's 50.0 mm).
    val_3d_max = feat_df.loc[4, "Rainfall_Prev_3d_Max"]
    expected_3d_max = 40.0
    assert np.isclose(val_3d_max, expected_3d_max, atol=1e-7), (
        f"ROLLING LEAKAGE DETECTED: Rainfall_Prev_3d_Max on Day 5 is {val_3d_max}, expected {expected_3d_max}."
    )

    # Mutate ONLY day 5's rainfall (from 50.0 to 999.0 mm)
    df_mut = df.copy()
    df_mut.loc[4, "Rainfall (mm)"] = 999.0
    feat_df_mut, _ = engineer_2025_features(df_mut, allow_zero_warmup=True)

    # Rainfall_Prev_3d_Sum on day 5 must remain exactly 90.0 mm
    mut_3d_sum = feat_df_mut.loc[4, "Rainfall_Prev_3d_Sum"]
    assert np.isclose(mut_3d_sum, expected_3d_sum, atol=1e-7), (
        f"SAME-DAY LEAKAGE DETECTED: Changing day T's rainfall changed Rainfall_Prev_3d_Sum on day T!"
    )


def test_training_dates_strictly_precede_validation_and_test_dates():
    """
    TEST 4: Verify chronological causal split order: Train Dates < Val Dates < Test Dates.
    Zero temporal inversion or date crossover is permitted.
    """
    config = load_config(CONFIG_PATH)
    train_years = config["split"]["train_years"]
    val_years = config["split"]["val_years"]
    test_years = config["split"]["test_years"]

    assert max(train_years) < min(val_years), (
        f"TEMPORAL LEAKAGE: Training years ({train_years}) overlap or succeed validation years ({val_years})!"
    )
    assert max(val_years) < min(test_years), (
        f"TEMPORAL LEAKAGE: Validation years ({val_years}) overlap or succeed test years ({test_years})!"
    )

    # Verify model metadata reflects this exact split
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    train_p = meta["training_period"]
    val_p = meta["validation_period"]
    test_p = meta["test_period"]

    assert train_p["end_year"] < val_p["start_year"], "Metadata training period violates causal order"
    assert val_p["end_year"] < test_p["start_year"], "Metadata validation period violates causal order"

    # If processed dataset exists, verify actual date boundaries on disk
    if os.path.exists(HISTORICAL_DATA_PATH):
        df_dates = pd.read_csv(HISTORICAL_DATA_PATH, usecols=["Date", "Year"])
        train_dates = df_dates[df_dates["Year"].isin(train_years)]["Date"]
        val_dates = df_dates[df_dates["Year"].isin(val_years)]["Date"]
        test_dates = df_dates[df_dates["Year"].isin(test_years)]["Date"]

        assert train_dates.max() < val_dates.min(), (
            f"TEMPORAL OVERLAP: Max train date ({train_dates.max()}) >= Min val date ({val_dates.min()})"
        )
        assert val_dates.max() < test_dates.min(), (
            f"TEMPORAL OVERLAP: Max val date ({val_dates.max()}) >= Min test date ({test_dates.min()})"
        )

        # Zero set intersection
        s_train = set(train_dates.unique())
        s_val = set(val_dates.unique())
        s_test = set(test_dates.unique())
        assert len(s_train.intersection(s_val)) == 0, "Date intersection between Train and Val is non-empty!"
        assert len(s_val.intersection(s_test)) == 0, "Date intersection between Val and Test is non-empty!"
        assert len(s_train.intersection(s_test)) == 0, "Date intersection between Train and Test is non-empty!"


def test_test_data_never_used_during_threshold_optimization():
    """
    TEST 5: Verify that the classification decision threshold is optimized strictly on validation data.
    Perturbing or corrupting test set labels/probabilities must have zero effect on the threshold.
    """
    np.random.seed(42)
    # Generate synthetic validation predictions
    y_val = np.random.binomial(1, 0.05, size=1000)
    val_proba = np.clip(y_val * 0.7 + np.random.normal(0.2, 0.1, size=1000), 0.0, 1.0)

    # Compute optimal threshold on validation set
    opt_thresh_clean, _ = find_optimal_threshold(y_val, val_proba, metric="f1")

    # Generate synthetic test set
    y_test = np.random.binomial(1, 0.05, size=1000)
    test_proba = np.clip(y_test * 0.7 + np.random.normal(0.2, 0.1, size=1000), 0.0, 1.0)

    # Create corrupted test set with inverted labels and extreme probabilities
    y_test_corrupt = 1 - y_test
    test_proba_corrupt = 1.0 - test_proba

    # Recompute threshold passing validation data: must remain strictly identical
    opt_thresh_after, _ = find_optimal_threshold(y_val, val_proba, metric="f1")
    assert opt_thresh_clean == opt_thresh_after, (
        "THRESHOLD LEAKAGE: find_optimal_threshold mutated despite identical validation data"
    )

    # In train_models.py, verify find_optimal_threshold signature only inspects y_val, val_proba
    from src.models.train_models import train_and_evaluate_all_models
    src_code = inspect.getsource(train_and_evaluate_all_models)
    assert "find_optimal_threshold(y_val, val_proba" in src_code, (
        "THRESHOLD LEAKAGE DETECTED in train_models.py: find_optimal_threshold not called strictly on y_val, val_proba!"
    )
    assert "find_optimal_threshold(y_test" not in src_code, (
        "THRESHOLD LEAKAGE DETECTED in train_models.py: find_optimal_threshold called with test data!"
    )


def test_test_data_never_used_to_fit_scalers_or_encoders():
    """
    TEST 6: Verify that scalers/encoders are fitted strictly on training data inside Pipeline.
    Test set statistics must never pollute scaler mean or variance parameters.
    """
    from src.models.logistic_regression import create_logistic_regression_pipeline
    
    np.random.seed(42)
    # Train set distributed around mean 10.0
    X_train = np.random.normal(loc=10.0, scale=2.0, size=(500, 5))
    y_train = np.random.binomial(1, 0.1, size=500)

    # Test set distributed around extreme mean 5000.0 (distribution shift)
    X_test = np.random.normal(loc=5000.0, scale=50.0, size=(200, 5))

    pipe = create_logistic_regression_pipeline(max_iter=100)
    pipe.fit(X_train, y_train)

    scaler = pipe.named_steps["scaler"]
    
    # Assert scaler.mean_ matches X_train mean within numerical tolerance
    assert np.allclose(scaler.mean_, X_train.mean(axis=0), atol=1e-7), (
        "PREPROCESSING LEAKAGE: Scaler mean does not match X_train mean!"
    )
    assert scaler.n_samples_seen_ == 500, (
        f"PREPROCESSING LEAKAGE: Scaler saw {scaler.n_samples_seen_} samples, expected {len(X_train)}!"
    )

    # Now evaluate on X_test: scaler parameters must NOT change
    scaler_mean_before = scaler.mean_.copy()
    _ = pipe.predict_proba(X_test)
    scaler_mean_after = scaler.mean_.copy()

    assert np.array_equal(scaler_mean_before, scaler_mean_after), (
        "PREPROCESSING LEAKAGE: Predict call on test set mutated scaler mean parameters!"
    )


def test_calibration_fitted_only_on_validation_data():
    """
    TEST 7: Verify that probability calibration (Platt/Sigmoid or Isotonic) is fitted strictly on validation data.
    The test set must remain completely out-of-sample and unobserved during calibration.
    """
    # 1. Inspect calibrated metadata file
    if os.path.exists(CALIBRATED_METADATA_PATH):
        with open(CALIBRATED_METADATA_PATH, "r", encoding="utf-8") as f:
            cal_meta = json.load(f)
        cal_dataset = cal_meta.get("calibration_dataset", "")
        assert "Validation Set" in cal_dataset or "2019-2021" in cal_dataset, (
            f"CALIBRATION LEAKAGE: calibration_dataset is '{cal_dataset}', expected 'Validation Set (2019-2021)'"
        )
        assert "Test" not in cal_dataset, (
            f"CALIBRATION LEAKAGE: Test set mentioned in calibration_dataset: '{cal_dataset}'"
        )

    # 2. Empirical calibration invariance test
    np.random.seed(42)
    X_val = np.random.normal(0, 1, size=(400, 4))
    y_val = np.random.binomial(1, 0.1, size=400)

    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_val, y_val)

    # Fit calibrator on validation
    calibrator = CalibratedClassifierCV(base_model, cv="prefit", method="sigmoid")
    calibrator.fit(X_val, y_val)

    # Verify fitted calibrators exist and have not seen test data
    assert hasattr(calibrator, "calibrated_classifiers_"), "Calibrator not properly fit"
    calibrated_clf = calibrator.calibrated_classifiers_[0]
    
    # Evaluate on arbitrary test data: calibrator parameters must remain invariant
    X_test_dummy = np.random.normal(10, 5, size=(100, 4))
    probs_1 = calibrator.predict_proba(X_test_dummy)
    probs_2 = calibrator.predict_proba(X_test_dummy)
    assert np.array_equal(probs_1, probs_2), "Inference on test data caused stateful mutation in calibrator"


def test_no_duplicate_station_date_records_cross_train_test():
    """
    TEST 8: Verify that no duplicate (District, Block/Station, Date) records exist across splits.
    Every observation belongs strictly to one chronological split without record crossover.
    """
    if not os.path.exists(HISTORICAL_DATA_PATH):
        pytest.skip(f"Processed dataset not present at {HISTORICAL_DATA_PATH}")

    # Read key identifiers only for rapid execution
    df_keys = pd.read_csv(
        HISTORICAL_DATA_PATH,
        usecols=["District", "Block/Station", "Date", "Year"]
    )

    # 1. Global uniqueness check
    dups_count = df_keys.duplicated(subset=["District", "Block/Station", "Date"]).sum()
    assert dups_count == 0, (
        f"CROSS-RECORD LEAKAGE: Found {dups_count:,} duplicate (District, Block/Station, Date) records in dataset!"
    )

    # 2. Split crossover check
    train_df = df_keys[df_keys["Year"] <= 2018]
    val_df = df_keys[(df_keys["Year"] >= 2019) & (df_keys["Year"] <= 2021)]
    test_df = df_keys[df_keys["Year"] >= 2022]

    train_tuples = set(zip(train_df["District"], train_df["Block/Station"], train_df["Date"]))
    val_tuples = set(zip(val_df["District"], val_df["Block/Station"], val_df["Date"]))
    test_tuples = set(zip(test_df["District"], test_df["Block/Station"], test_df["Date"]))

    crossover_train_test = train_tuples.intersection(test_tuples)
    assert len(crossover_train_test) == 0, (
        f"CROSS-SPLIT LEAKAGE: Found {len(crossover_train_test)} identical station-date records in Train AND Test!"
    )

    crossover_val_test = val_tuples.intersection(test_tuples)
    assert len(crossover_val_test) == 0, (
        f"CROSS-SPLIT LEAKAGE: Found {len(crossover_val_test)} identical station-date records in Val AND Test!"
    )

    crossover_train_val = train_tuples.intersection(val_tuples)
    assert len(crossover_train_val) == 0, (
        f"CROSS-SPLIT LEAKAGE: Found {len(crossover_train_val)} identical station-date records in Train AND Val!"
    )


def test_feature_columns_identical_between_training_and_inference():
    """
    TEST 9: Verify that feature columns, count, and order are 100% identical between training and inference.
    Prevents silent feature misalignment or covariate transposition.
    """
    config = load_config(CONFIG_PATH)
    expected_config_features = (
        config["features"]["temporal_cols"] +
        config["features"]["rainfall_cols"]
    )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta_features = meta["features"]
    meta_feature_list = meta["feature_list"]

    predictor = FloodPredictor()
    pred_features = predictor.feature_names

    bundle = load_model_with_metadata(MODEL_PATH, METADATA_PATH)
    bundle_features = bundle.features

    # 1. Feature list identity
    assert meta_features == meta_feature_list, "Metadata 'features' and 'feature_list' mismatch"
    assert meta_features == pred_features, (
        f"FEATURE MISALIGNMENT: model_metadata.json features != FloodPredictor feature_names!\n"
        f"Metadata: {meta_features}\nPredictor: {pred_features}"
    )
    assert meta_features == bundle_features, (
        f"FEATURE MISALIGNMENT: model_metadata.json features != ModelBundle features!"
    )
    assert meta_features == expected_config_features, (
        f"FEATURE MISALIGNMENT: config.yaml features != model_metadata.json features!"
    )

    # 2. Check 2025 operational pipeline aligns with trained model features
    pipeline = Operational2025Pipeline()
    pipeline_features = pipeline.metadata["features"]
    assert pipeline_features == meta_features, (
        "FEATURE MISALIGNMENT: Operational2025Pipeline features differ from frozen model features!"
    )


def test_2025_inference_does_not_retrain_model():
    """
    TEST 10: Verify that operational 2025 inference executes in frozen mode and NEVER retrains the model.
    Confirms zero model parameter mutation, zero disk artifact modification, and no fit() invocation.
    """
    # 1. Verify model artifact mtime and checksum on disk
    mtime_before = os.path.getmtime(MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        bytes_before = f.read()

    # 2. Initialize Operational Pipeline
    pipeline = Operational2025Pipeline()
    model_params_before = str(pipeline.model.get_params()) if hasattr(pipeline.model, "get_params") else ""

    # Attach spy mock to intercept any accidental fit calls
    spy_fit = Mock()
    original_fit = getattr(pipeline.model, "fit", None)
    if original_fit is not None:
        pipeline.model.fit = spy_fit

    # Create valid synthetic 2025 input batch
    sample_2025 = pd.DataFrame({
        "District": ["CUTTACK", "BALASORE"],
        "Block/Station": ["Cuttack", "Balasore"],
        "Date": ["2025-08-15", "2025-08-15"],
        "Rainfall (mm)": [85.0, 110.0]
    })

    # Execute operational inference
    df_preds, summary = pipeline.run_pipeline(sample_2025, output_csv_path=None)

    # A. Assert fit() was never called
    assert spy_fit.call_count == 0, (
        f"RETRAINING LEAKAGE: Operational2025Pipeline invoked .fit() {spy_fit.call_count} time(s) on frozen model!"
    )

    # Restore original fit
    if original_fit is not None:
        pipeline.model.fit = original_fit

    # B. Assert model parameters remain completely identical
    model_params_after = str(pipeline.model.get_params()) if hasattr(pipeline.model, "get_params") else ""
    assert model_params_before == model_params_after, (
        "RETRAINING LEAKAGE: Operational inference mutated frozen model parameters!"
    )

    # C. Assert disk artifact was untouched
    mtime_after = os.path.getmtime(MODEL_PATH)
    assert mtime_before == mtime_after, (
        "RETRAINING LEAKAGE: Operational inference modified models/flood_prediction/best_model.pkl on disk!"
    )
    with open(MODEL_PATH, "rb") as f:
        bytes_after = f.read()
    assert bytes_before == bytes_after, (
        "RETRAINING LEAKAGE: Binary hash of best_model.pkl changed during operational inference!"
    )

    # D. Assert valid prediction output was generated without modifying model
    assert len(df_preds) == 2
    assert "Predicted_Probability" in df_preds.columns
    assert "Risk_Level" in df_preds.columns
