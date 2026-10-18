"""
Unit and integration tests for model metadata, experiment reproducibility,
atomic model bundle loading, and strict pre-prediction schema validation.
"""

import os
import json
import re
import platform
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.metadata import (
    SchemaValidationError,
    ModelBundle,
    load_model_with_metadata,
    generate_model_metadata,
    save_model_metadata,
    load_model_metadata,
    validate_feature_schema,
    compute_dataset_hash,
    get_environment_versions,
    extract_model_hyperparameters
)
from src.models.predict import FloodPredictor


METADATA_PATH = "models/flood_prediction/model_metadata.json"
MODEL_PATH = "models/flood_prediction/best_model.pkl"
DATASET_PATH = "data/processed/historical_2001_2024.csv"
CALIBRATED_METADATA_PATH = "models/flood_prediction/calibrated_metadata.json"
CALIBRATED_MODEL_PATH = "models/flood_prediction/calibrated_best_model.pkl"

REQUIRED_METADATA_KEYS = [
    "model_name",
    "training_date",
    "dataset_version",
    "dataset_hash",
    "training_period",
    "validation_period",
    "test_period",
    "feature_list",
    "preprocessing_version",
    "random_seed",
    "hyperparameters",
    "threshold",
    "calibration_method",
    "python_version",
    "library_versions"
]


def test_metadata_file_contains_all_required_keys():
    """Verify model_metadata.json contains all 15 required reproducibility fields."""
    assert os.path.exists(METADATA_PATH), f"Missing metadata file at {METADATA_PATH}"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for key in REQUIRED_METADATA_KEYS:
        assert key in meta, f"Missing required key '{key}' in {METADATA_PATH}"
        assert meta[key] is not None, f"Key '{key}' must not be null in {METADATA_PATH}"

    # Verify types and non-emptiness
    assert isinstance(meta["model_name"], str) and len(meta["model_name"]) > 0
    assert isinstance(meta["training_date"], str) and len(meta["training_date"]) > 0
    assert isinstance(meta["dataset_version"], str) and len(meta["dataset_version"]) > 0
    assert isinstance(meta["dataset_hash"], str) and len(meta["dataset_hash"]) == 64
    assert isinstance(meta["training_period"], dict) and "years" in meta["training_period"]
    assert isinstance(meta["validation_period"], dict) and "years" in meta["validation_period"]
    assert isinstance(meta["test_period"], dict) and "years" in meta["test_period"]
    assert isinstance(meta["feature_list"], list) and len(meta["feature_list"]) > 0
    assert isinstance(meta["preprocessing_version"], str)
    assert isinstance(meta["random_seed"], int)
    assert isinstance(meta["hyperparameters"], dict) and len(meta["hyperparameters"]) > 0
    assert isinstance(meta["threshold"], (float, int))
    assert isinstance(meta["calibration_method"], str)
    assert isinstance(meta["python_version"], str) and len(meta["python_version"]) > 0
    assert isinstance(meta["library_versions"], dict) and len(meta["library_versions"]) >= 5


def test_dataset_hash_is_valid_sha256():
    """Verify dataset_hash is a valid 64-character SHA-256 hexadecimal string matching data."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ds_hash = meta["dataset_hash"]
    assert re.match(r"^[0-9a-fA-F]{64}$", ds_hash), f"Invalid SHA-256 hex string: {ds_hash}"

    if os.path.exists(DATASET_PATH):
        computed_hash = compute_dataset_hash(DATASET_PATH)
        assert ds_hash.lower() == computed_hash.lower(), (
            f"Stored dataset hash ({ds_hash}) does not match freshly computed SHA-256 ({computed_hash})"
        )


def test_environment_versions_are_dynamic():
    """Verify python_version and library_versions reflect dynamic runtime discovery."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["python_version"] == platform.python_version()

    discovered_libs = get_environment_versions()
    for lib, ver in discovered_libs.items():
        assert lib in meta["library_versions"]
        if ver != "unavailable":
            assert meta["library_versions"][lib] == ver


def test_model_and_metadata_load_together():
    """Verify the model is loaded together with its metadata via load_model_with_metadata."""
    bundle = load_model_with_metadata(MODEL_PATH, METADATA_PATH)
    assert isinstance(bundle, ModelBundle)
    assert bundle.model is not None
    assert isinstance(bundle.metadata, dict)
    assert bundle.model_name == "XGBoost"
    assert len(bundle.features) == 22
    assert bundle.threshold == pytest.approx(0.8073, rel=1e-3)
    assert bundle.calibration_method == "none"
    assert len(bundle.hyperparameters) > 0


def test_flood_predictor_load_with_metadata():
    """Verify FloodPredictor can be instantiated directly via load_with_metadata helper."""
    predictor = FloodPredictor.load_with_metadata(MODEL_PATH, METADATA_PATH)
    assert predictor.bundle is not None
    assert predictor.model is not None
    assert predictor.feature_names == predictor.bundle.features
    assert predictor.threshold == predictor.bundle.threshold


def test_schema_validation_missing_feature_raises():
    """Verify that an input missing any required feature column raises SchemaValidationError."""
    bundle = load_model_with_metadata(MODEL_PATH, METADATA_PATH)
    
    # Create input with one missing feature ('Rainfall (mm)' omitted)
    features_subset = [f for f in bundle.features if f != "Rainfall (mm)"]
    df_missing = pd.DataFrame([{f: 1.0 for f in features_subset}])

    with pytest.raises(SchemaValidationError) as exc_info:
        bundle.validate_schema(df_missing)
    assert "missing" in str(exc_info.value).lower()
    assert "Rainfall (mm)" in str(exc_info.value)


def test_schema_validation_numpy_dimension_mismatch_raises():
    """Verify that a NumPy array with incorrect column dimension raises SchemaValidationError."""
    bundle = load_model_with_metadata(MODEL_PATH, METADATA_PATH)
    expected_dim = len(bundle.features)

    # 2D array with 10 features instead of 22
    X_bad_dim = np.zeros((5, expected_dim - 5))
    with pytest.raises(SchemaValidationError) as exc_info:
        bundle.validate_schema(X_bad_dim)
    assert "dimension mismatch" in str(exc_info.value).lower()


def test_schema_validation_non_numeric_dtype_raises():
    """Verify that an input containing non-numeric data types raises SchemaValidationError."""
    bundle = load_model_with_metadata(MODEL_PATH, METADATA_PATH)
    
    data = {f: 1.0 for f in bundle.features}
    data["Rainfall (mm)"] = "EXTREME_RAIN"  # string instead of numeric float
    df_bad_type = pd.DataFrame([data])

    with pytest.raises(SchemaValidationError) as exc_info:
        bundle.validate_schema(df_bad_type)
    assert "data type mismatch" in str(exc_info.value).lower()
    assert "Rainfall (mm)" in str(exc_info.value)


def test_schema_validation_nan_values_raise():
    """Verify that an input containing NaN values raises SchemaValidationError."""
    bundle = load_model_with_metadata(MODEL_PATH, METADATA_PATH)
    
    data = {f: 1.0 for f in bundle.features}
    data["Rainfall_Lag_1d"] = np.nan
    df_nan = pd.DataFrame([data])

    with pytest.raises(SchemaValidationError) as exc_info:
        bundle.validate_schema(df_nan)
    assert "nan" in str(exc_info.value).lower()


def test_schema_validation_empty_input_raises():
    """Verify that an empty DataFrame (0 rows) raises SchemaValidationError."""
    bundle = load_model_with_metadata(MODEL_PATH, METADATA_PATH)
    df_empty = pd.DataFrame(columns=bundle.features)

    with pytest.raises(SchemaValidationError) as exc_info:
        bundle.validate_schema(df_empty)
    assert "empty" in str(exc_info.value).lower()


def test_schema_validation_valid_input_passes():
    """Verify that valid input passes schema validation and preserves strict column ordering."""
    bundle = load_model_with_metadata(MODEL_PATH, METADATA_PATH)
    
    # Create input with columns intentionally shuffled and extra metadata column
    shuffled_features = list(reversed(bundle.features))
    data = {f: 5.0 for f in shuffled_features}
    data["District"] = "CUTTACK"  # Extra metadata column
    data["Year"] = 2024
    df_input = pd.DataFrame([data])

    validated_df = bundle.validate_schema(df_input)
    assert isinstance(validated_df, pd.DataFrame)
    assert list(validated_df.columns) == bundle.features
    assert validated_df.shape == (1, len(bundle.features))


def test_flood_predictor_enforces_schema_validation():
    """Verify FloodPredictor actively blocks predictions when schema validation fails."""
    predictor = FloodPredictor()
    
    # Incomplete dictionary missing rainfall features
    bad_sample = {
        "Month_sin": -0.5,
        "Month_cos": 0.866,
        "Day_of_Week": 2
    }
    with pytest.raises(SchemaValidationError):
        predictor.predict_single(bad_sample)

    with pytest.raises(SchemaValidationError):
        predictor.predict_probability(pd.DataFrame([bad_sample]))


def test_generate_model_metadata_utility_automatic(tmp_path):
    """Verify generate_model_metadata automatically gathers hyperparameters and environment without hardcoding."""
    clf = LogisticRegression(C=0.5, solver="liblinear", random_state=123)
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, 20)
    clf.fit(X, y)

    dummy_csv = tmp_path / "dummy_data.csv"
    dummy_csv.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    meta = generate_model_metadata(
        model=clf,
        model_name="CustomLogReg",
        feature_cols=["feat_a", "feat_b", "feat_c"],
        threshold=0.65,
        dataset_path=str(dummy_csv),
        dataset_version="1.5.0",
        preprocessing_version="1.0.0",
        calibration_method="none",
        random_seed=123
    )

    assert meta["model_name"] == "CustomLogReg"
    assert meta["threshold"] == 0.65
    assert meta["dataset_version"] == "1.5.0"
    assert len(meta["dataset_hash"]) == 64
    assert meta["hyperparameters"]["C"] == 0.5
    assert meta["hyperparameters"]["solver"] == "liblinear"
    assert meta["python_version"] == platform.python_version()
    assert "numpy" in meta["library_versions"]


def test_calibrated_metadata_completeness():
    """Verify calibrated_metadata.json contains all required reproducibility keys."""
    assert os.path.exists(CALIBRATED_METADATA_PATH), f"Missing {CALIBRATED_METADATA_PATH}"
    with open(CALIBRATED_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for key in REQUIRED_METADATA_KEYS:
        assert key in meta, f"Missing required key '{key}' in {CALIBRATED_METADATA_PATH}"

    assert meta["calibration_method"] == "sigmoid"
    assert len(meta["feature_list"]) == 22
    assert meta["dataset_version"] == "2.0.0"
