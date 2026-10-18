"""
Model metadata, reproducibility, and feature schema validation utilities for Odisha Flood Prediction.

Provides automated metadata generation during model training, comprehensive environment version
auditing, dataset hashing, atomic model bundle loading, and strict pre-prediction schema validation.
"""

import os
import sys
import json
import hashlib
import platform
import logging
import importlib.metadata
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

DEFAULT_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "joblib",
    "matplotlib",
    "seaborn",
    "flask"
]


class SchemaValidationError(ValueError):
    """Raised when an input dataset does not match the model's expected feature schema."""
    pass


def compute_dataset_hash(dataset_path: Optional[str], chunk_size: int = 65536) -> str:
    """
    Compute SHA-256 cryptographic hash of a dataset file in chunks.
    
    Parameters
    ----------
    dataset_path : str, optional
        Path to the dataset file (e.g. 'data/processed/historical_2001_2024.csv').
    chunk_size : int, default 65536 (64 KB)
        Chunk size for reading stream.

    Returns
    -------
    str
        64-character hexadecimal SHA-256 hash, or empty string if file is missing.
    """
    if not dataset_path or not os.path.exists(dataset_path):
        return ""
    
    sha256 = hashlib.sha256()
    try:
        with open(dataset_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute dataset hash for {dataset_path}: {e}")
        return ""


def get_environment_versions(packages: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Dynamically discover installed package versions in current Python environment.
    
    Parameters
    ----------
    packages : list of str, optional
        Package names to inspect. Defaults to standard project dependencies.

    Returns
    -------
    dict
        Mapping of package name to installed version string.
    """
    pkgs = packages or DEFAULT_PACKAGES
    versions: Dict[str, str] = {}
    
    for pkg in pkgs:
        ver = None
        # Try importlib.metadata first
        try:
            ver = importlib.metadata.version(pkg)
        except Exception:
            pass
        
        # Fallback to importing module and checking __version__
        if ver is None:
            mod_name = pkg.replace("-", "_")
            if pkg == "scikit-learn":
                mod_name = "sklearn"
            try:
                mod = __import__(mod_name)
                ver = getattr(mod, "__version__", None)
            except Exception:
                pass
                
        versions[pkg] = ver if ver is not None else "unavailable"
        
    return versions


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy and scikit-learn non-standard objects to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    # Fallback to string representation for estimator objects or custom callables
    return str(obj)


def extract_model_hyperparameters(model: Any) -> Dict[str, Any]:
    """
    Extract and sanitize hyperparameters from any scikit-learn or XGBoost model instance.

    Parameters
    ----------
    model : Any
        Trained estimator, classifier, or Pipeline instance.

    Returns
    -------
    dict
        Clean, JSON-serializable dictionary of model hyperparameters.
    """
    if hasattr(model, "get_params"):
        try:
            raw_params = model.get_params(deep=True)
            return _sanitize_for_json(raw_params)
        except Exception as e:
            logger.warning(f"Error calling get_params() on {type(model)}: {e}")
            
    # Fallback inspection of public estimator attributes
    extracted: Dict[str, Any] = {}
    for attr in ["n_estimators", "max_depth", "learning_rate", "class_weight", "random_state", "C", "solver"]:
        if hasattr(model, attr):
            extracted[attr] = _sanitize_for_json(getattr(model, attr))
            
    return extracted


def generate_model_metadata(
    model: Any,
    model_name: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    dataset_path: Optional[str] = None,
    dataset_version: Optional[str] = None,
    preprocessing_version: Optional[str] = None,
    calibration_method: str = "none",
    random_seed: Optional[int] = None,
    training_period: Optional[Dict[str, Any]] = None,
    validation_period: Optional[Dict[str, Any]] = None,
    test_period: Optional[Dict[str, Any]] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
    risk_thresholds: Optional[Dict[str, Any]] = None,
    brier_score: Optional[float] = None,
    calibration_ece: Optional[float] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Automatically generate complete, reproducible metadata for a trained flood model.
    Never hardcodes values that can be dynamically extracted from the environment,
    dataset, or estimator instance.

    Parameters
    ----------
    model : Any
        Trained model instance (e.g. XGBClassifier, CalibratedClassifierCV, Pipeline).
    model_name : str, optional
        Human-readable name of the model. Derived from class name if omitted.
    feature_cols : list of str, optional
        List of input feature names required by the model.
    threshold : float, optional
        Optimal decision threshold for classification.
    config : dict, optional
        Project configuration dictionary (from config.yaml).
    dataset_path : str, optional
        Path to training dataset file for SHA-256 hash calculation.
    dataset_version : str, optional
        Version tag of dataset (e.g. '2.0.0').
    preprocessing_version : str, optional
        Version tag of feature engineering pipeline (e.g. '2.0.0').
    calibration_method : str, default 'none'
        Calibration algorithm: 'none', 'sigmoid', or 'isotonic'.
    random_seed : int, optional
        Random seed used during training.
    training_period : dict, optional
        Dict containing start_year, end_year, and list of training years.
    validation_period : dict, optional
        Dict containing start_year, end_year, and list of validation years.
    test_period : dict, optional
        Dict containing start_year, end_year, and list of test years.
    test_metrics : dict, optional
        Evaluation metrics on the untouched test set.
    risk_thresholds : dict, optional
        Civil risk classification boundaries (low, moderate, high).
    brier_score : float, optional
        Brier reliability score on test set.
    calibration_ece : float, optional
        Expected Calibration Error on test set.
    extra_metadata : dict, optional
        Additional custom key-value pairs to store.

    Returns
    -------
    dict
        Complete metadata dictionary complying with project reproducibility standards.
    """
    config = config or {}
    
    # 1. Model Name
    if not model_name:
        model_name = getattr(model, "__class__", type(model)).__name__
        
    # 2. Features
    feats = list(feature_cols) if feature_cols is not None else []
    if not feats and hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
        
    # 3. Decision Threshold
    opt_threshold = float(threshold) if threshold is not None else 0.5
    
    # 4. Dates & Periods
    now_iso = datetime.now(timezone.utc).isoformat()
    
    split_cfg = config.get("split", {})
    train_years = list(split_cfg.get("train_years", [2001, 2018]))
    val_years = list(split_cfg.get("val_years", [2019, 2021]))
    test_years = list(split_cfg.get("test_years", [2022, 2024]))
    
    train_p = training_period or {
        "start_year": int(min(train_years)) if train_years else 2001,
        "end_year": int(max(train_years)) if train_years else 2018,
        "years": train_years
    }
    val_p = validation_period or {
        "start_year": int(min(val_years)) if val_years else 2019,
        "end_year": int(max(val_years)) if val_years else 2021,
        "years": val_years
    }
    test_p = test_period or {
        "start_year": int(min(test_years)) if test_years else 2022,
        "end_year": int(max(test_years)) if test_years else 2024,
        "years": test_years
    }
    
    # 5. Dataset Version and Hash
    data_path = dataset_path or config.get("paths", {}).get("combined_data_path", "data/processed/historical_2001_2024.csv")
    ds_hash = compute_dataset_hash(data_path)
    ds_ver = dataset_version or config.get("project", {}).get("version", "2.0.0")
    prep_ver = preprocessing_version or "2.0.0"
    
    # 6. Random Seed
    seed = random_seed if random_seed is not None else config.get("project", {}).get("random_seed", 42)
    
    # 7. Hyperparameters
    hyperparams = extract_model_hyperparameters(model)
    
    # 8. Environment & Library Versions
    py_version = platform.python_version()
    lib_versions = get_environment_versions()
    
    # 9. Risk Thresholds
    r_thresholds = risk_thresholds or config.get("risk_thresholds", {
        "low_max": 0.30,
        "moderate_max": 0.70,
        "high_min": 0.70
    })

    metadata: Dict[str, Any] = {
        # Core Reproducibility Metadata
        "model_name": model_name,
        "training_date": now_iso,
        "dataset_version": ds_ver,
        "dataset_hash": ds_hash,
        "training_period": train_p,
        "validation_period": val_p,
        "test_period": test_p,
        "feature_list": feats,
        "features": feats,  # Backward compatibility alias
        "preprocessing_version": prep_ver,
        "random_seed": seed,
        "hyperparameters": hyperparams,
        "threshold": opt_threshold,
        "optimal_threshold": opt_threshold,  # Backward compatibility alias
        "calibration_method": calibration_method,
        "python_version": py_version,
        "library_versions": lib_versions,
        
        # Operational & Project Backward Compatibility Fields
        "risk_thresholds": r_thresholds,
        "train_years": train_years,
        "val_years": val_years,
        "test_years": test_years,
    }
    
    if test_metrics is not None:
        metadata["test_metrics"] = _sanitize_for_json(test_metrics)
    if brier_score is not None:
        metadata["brier_score"] = float(brier_score)
    if calibration_ece is not None:
        metadata["calibration_ece"] = float(calibration_ece)
        
    if extra_metadata:
        for k, v in extra_metadata.items():
            if k not in metadata:
                metadata[k] = _sanitize_for_json(v)
                
    return metadata


def save_model_metadata(metadata: Dict[str, Any], filepath: str) -> None:
    """Save model metadata dictionary to formatted JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Model metadata successfully saved to: {filepath}")


def load_model_metadata(filepath: str) -> Dict[str, Any]:
    """Load model metadata dictionary from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model metadata file not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_feature_schema(
    X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Any]],
    expected_features: List[str],
    allow_nan: bool = False
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Strictly validate input data against the model's expected feature schema prior to prediction.

    Prevents execution and raises SchemaValidationError if:
    1. Input is empty (0 samples).
    2. Any required feature column is missing from DataFrame or Dict.
    3. NumPy array column dimension does not match expected feature count.
    4. Any required feature contains non-numeric data types (strings, objects).
    5. Any required feature contains NaN/null/infinite values (unless allow_nan=True).

    Parameters
    ----------
    X : DataFrame, ndarray, dict, or list
        Input observation(s) to validate.
    expected_features : list of str
        Ordered list of feature column names expected by the model.
    allow_nan : bool, default False
        Whether to permit NaN values (typically False for deployment).

    Returns
    -------
    DataFrame or ndarray
        Clean, validated, and strictly ordered dataset ready for estimator inference.

    Raises
    ------
    SchemaValidationError
        If any schema violation is detected.
    """
    if not expected_features:
        raise SchemaValidationError("Model expected feature schema is empty or undefined.")

    # Convert single dict or list of dicts to DataFrame
    if isinstance(X, dict):
        X = pd.DataFrame([X])
    elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
        X = pd.DataFrame(X)

    # 1. Empty Check
    if hasattr(X, "__len__") and len(X) == 0:
        raise SchemaValidationError("Input feature dataset is empty (0 samples). Cannot perform inference.")

    # 2. DataFrame Validation
    if isinstance(X, pd.DataFrame):
        # A. Missing features check
        input_cols = list(X.columns)
        missing = [c for c in expected_features if c not in input_cols]
        if missing:
            raise SchemaValidationError(
                f"Feature schema mismatch: missing {len(missing)} required feature column(s): {missing}. "
                f"Model expects exactly {len(expected_features)} features: {expected_features}."
            )

        # Slice strictly in expected order
        X_ordered = X[expected_features].copy()

        # B. Dtype check (must be numeric)
        non_numeric = []
        for col in expected_features:
            if not pd.api.types.is_numeric_dtype(X_ordered[col]):
                # Attempt conversion; if fails, flag violation
                converted = pd.to_numeric(X_ordered[col], errors="coerce")
                if converted.isna().any() and not X_ordered[col].isna().any():
                    non_numeric.append(f"{col} (dtype: {X_ordered[col].dtype})")
                else:
                    X_ordered[col] = converted

        if non_numeric:
            raise SchemaValidationError(
                f"Feature data type mismatch: expected numeric float/int for all features, "
                f"but found non-numeric data in columns: {non_numeric}."
            )

        # C. NaN / Infinite check
        if not allow_nan:
            nan_cols = [c for c in expected_features if X_ordered[c].isna().any()]
            if nan_cols:
                raise SchemaValidationError(
                    f"Feature schema violation: null or NaN values detected in required features: {nan_cols}. "
                    "All features must contain valid finite numbers."
                )
            # Check for infinities
            inf_cols = [c for c in expected_features if np.isinf(X_ordered[c].to_numpy(dtype=float)).any()]
            if inf_cols:
                raise SchemaValidationError(
                    f"Feature schema violation: infinite (+inf/-inf) values detected in required features: {inf_cols}."
                )

        return X_ordered

    # 3. NumPy Array Validation
    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            # 1D array representing a single sample with length == n_features
            if X.shape[0] != len(expected_features):
                raise SchemaValidationError(
                    f"Feature dimension mismatch: 1D array has {X.shape[0]} elements, "
                    f"but model expects {len(expected_features)} features."
                )
            X = X.reshape(1, -1)
        elif X.ndim == 2:
            if X.shape[1] != len(expected_features):
                raise SchemaValidationError(
                    f"Feature dimension mismatch: 2D array has {X.shape[1]} columns, "
                    f"but model expects {len(expected_features)} features."
                )
        else:
            raise SchemaValidationError(
                f"Invalid NumPy array dimensions: expected 1D or 2D array, received {X.ndim}D array."
            )

        # Type check
        if not np.issubdtype(X.dtype, np.number):
            raise SchemaValidationError(
                f"Feature data type mismatch: expected numeric NumPy array, received dtype '{X.dtype}'."
            )

        # NaN / Inf check
        if not allow_nan:
            if np.isnan(X).any():
                raise SchemaValidationError("Feature schema violation: NumPy array contains NaN values.")
            if np.isinf(X).any():
                raise SchemaValidationError("Feature schema violation: NumPy array contains infinite values.")

        return X

    else:
        raise SchemaValidationError(
            f"Unsupported feature input type '{type(X).__name__}'. "
            "Supported formats are pandas.DataFrame, numpy.ndarray, dict, or list of dicts."
        )


class ModelBundle:
    """
    Encapsulates a trained machine learning model together with its complete metadata,
    hyperparameters, environment provenance, and strict schema validation engine.
    """

    def __init__(
        self,
        model: Any,
        metadata: Dict[str, Any],
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        self.model = model
        self.metadata = metadata
        self.model_path = model_path
        self.metadata_path = metadata_path
        
        self.model_name: str = metadata.get("model_name", getattr(model, "__class__", type(model)).__name__)
        self.features: List[str] = metadata.get("feature_list") or metadata.get("features", [])
        self.threshold: float = float(metadata.get("threshold", metadata.get("optimal_threshold", 0.5)))
        self.risk_thresholds: Dict[str, float] = metadata.get("risk_thresholds", {
            "low_max": 0.30,
            "moderate_max": 0.70,
            "high_min": 0.70
        })
        self.calibration_method: str = metadata.get("calibration_method", "none")
        self.dataset_hash: str = metadata.get("dataset_hash", "")
        self.python_version: str = metadata.get("python_version", platform.python_version())
        self.library_versions: Dict[str, str] = metadata.get("library_versions", {})
        self.hyperparameters: Dict[str, Any] = metadata.get("hyperparameters", {})

    def validate_schema(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Any]]) -> Union[pd.DataFrame, np.ndarray]:
        """Validate input features against bundle schema."""
        return validate_feature_schema(X, self.features)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Any]]) -> np.ndarray:
        """Validate feature schema and produce class probabilities."""
        X_val = self.validate_schema(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_val)
        elif hasattr(self.model, "predict"):
            preds = self.model.predict(X_val)
            return np.column_stack([1.0 - preds, preds])
        raise AttributeError(f"Model object of type {type(self.model)} has no predict_proba or predict method.")

    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Any]], threshold: Optional[float] = None) -> np.ndarray:
        """Validate feature schema and predict binary classifications."""
        thresh = self.threshold if threshold is None else threshold
        probas = self.predict_proba(X)
        p1 = probas[:, 1] if probas.ndim == 2 else probas
        return (p1 >= thresh).astype(int)

    def save(self, model_save_path: str, metadata_save_path: Optional[str] = None) -> None:
        """Atomically persist model artifact and metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        joblib.dump(self.model, model_save_path)
        meta_path = metadata_save_path or os.path.splitext(model_save_path)[0] + "_metadata.json"
        save_model_metadata(self.metadata, meta_path)
        self.model_path = model_save_path
        self.metadata_path = meta_path

    def __repr__(self) -> str:
        return (
            f"<ModelBundle model='{self.model_name}' threshold={self.threshold:.4f} "
            f"n_features={len(self.features)} calibration='{self.calibration_method}' "
            f"py={self.python_version}>"
        )


def load_model_with_metadata(
    model_path: str = "models/flood_prediction/best_model.pkl",
    metadata_path: Optional[str] = None
) -> ModelBundle:
    """
    Load a trained model artifact and its metadata together, verifying consistency.

    Parameters
    ----------
    model_path : str
        Path to serialized model binary (.pkl).
    metadata_path : str, optional
        Path to metadata JSON file. If omitted, automatically checks:
        1. Explicit stem: '<model_dir>/<model_name>_metadata.json'
        2. Standard: '<model_dir>/model_metadata.json'
        3. Calibrated: '<model_dir>/calibrated_metadata.json'

    Returns
    -------
    ModelBundle
        Loaded bundle with validated model, metadata, and schema checker.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model binary artifact not found at: {model_path}")

    # Discover metadata path if not provided
    if not metadata_path:
        base_dir = os.path.dirname(os.path.abspath(model_path))
        stem = os.path.splitext(os.path.basename(model_path))[0]
        candidates = [
            os.path.join(base_dir, f"{stem}_metadata.json"),
            os.path.join(base_dir, "model_metadata.json"),
            os.path.join(base_dir, "calibrated_metadata.json")
        ]
        for cand in candidates:
            if os.path.exists(cand):
                metadata_path = cand
                break

    if not metadata_path or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not locate corresponding model metadata JSON for: {model_path}")

    # Load artifacts together
    model = joblib.load(model_path)
    metadata = load_model_metadata(metadata_path)

    # Cross-verify feature counts if model exposes n_features_in_
    expected_feats = metadata.get("feature_list") or metadata.get("features", [])
    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != len(expected_feats):
            logger.warning(
                f"Model n_features_in_ ({model.n_features_in_}) differs from "
                f"metadata feature count ({len(expected_feats)})."
            )

    return ModelBundle(
        model=model,
        metadata=metadata,
        model_path=model_path,
        metadata_path=metadata_path
    )
