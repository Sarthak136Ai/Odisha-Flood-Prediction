# Model Metadata, Experiment Reproducibility & Schema Validation

## 1. Overview & Objective

The **Odisha Flood Intelligence & Early Warning System** implements a zero-leakage, scientifically reproducible machine learning pipeline. In mission-critical disaster management, operational decisions (such as district evacuation alerts and emergency rescue deployments) rely directly on the veracity and auditability of the predictive models.

To guarantee that any trained model can be precisely traced back to its training code, runtime environment, hyperparameter state, and exact dataset snapshot:
1. Every trained model is paired with a comprehensive, dynamically extracted `model_metadata.json`.
2. Values that can be determined programmatically (Python runtime, package versions, cryptographic dataset hash, model hyperparameters) are **never hardcoded**.
3. A centralized utility (`src/models/metadata.py`) generates and serializes metadata during the training process.
4. Models and metadata are loaded atomically as a unified `ModelBundle`.
5. Strict pre-prediction schema validation prevents inference if input features exhibit missing columns, dimensional mismatch, non-numeric types, or null/infinite values.

---

## 2. Comprehensive Model Metadata Schema

Every model artifact saved under `models/flood_prediction/` (such as `best_model.pkl` and `calibrated_best_model.pkl`) is governed by an accompanying metadata JSON file.

### Required Reproducibility Fields:

| Key | Type | Description | Extraction Mechanism |
| :--- | :---: | :--- | :--- |
| `model_name` | `string` | Architecture / model name (e.g., `"XGBoost"`, `"Logistic Regression"`). | Parameter or `estimator.__class__.__name__` |
| `training_date` | `string` | ISO 8601 timestamp with UTC timezone (e.g. `2026-10-18T03:25:28Z`). | `datetime.now(timezone.utc).isoformat()` |
| `dataset_version` | `string` | Semantic version of the training dataset (e.g., `"2.0.0"`). | Extracted from `config.yaml` (`project.version`) |
| `dataset_hash` | `string` | 64-character SHA-256 cryptographic checksum of training data file. | `compute_dataset_hash(dataset_path)` (chunked stream) |
| `training_period` | `object` | Explicit training date range (`start_year`, `end_year`, `years` list). | Extracted from chronological split definition |
| `validation_period`| `object` | Explicit validation date range for threshold tuning and calibration. | Extracted from chronological split definition |
| `test_period` | `object` | Untouched out-of-sample test date range (`2022–2024`). | Extracted from chronological split definition |
| `feature_list` | `list[str]` | Complete ordered list of input feature names (22 features). | Model pipeline schema or `feature_names_in_` |
| `preprocessing_version` | `string` | Version of the feature engineering pipeline (`"2.0.0"`). | Project pipeline specification |
| `random_seed` | `integer` | Seed governing stochastic initialization and train/test splits (`42`). | Extracted from `config.yaml` (`project.random_seed`) |
| `hyperparameters` | `object` | Full dictionary of model parameters. | `model.get_params(deep=True)` with JSON sanitization |
| `threshold` | `float` | Optimal classification decision threshold ($T^* = 0.8073$). | Validation-tuned optimization metric |
| `calibration_method` | `string` | Probability calibration applied (`"none"`, `"sigmoid"`, or `"isotonic"`). | Specified during model fitting |
| `python_version` | `string` | Full Python runtime version (e.g. `"3.10.0"`). | `platform.python_version()` |
| `library_versions` | `object` | Mapping of core dependencies to installed versions. | `importlib.metadata.version()` |

### Backward Compatibility Fields:
To ensure existing web dashboards, REST endpoints, and legacy tools do not break, the metadata file preserves:
- `features`: Exact alias for `feature_list`
- `optimal_threshold`: Exact alias for `threshold`
- `risk_thresholds`: Standard disaster warning risk categories (`LOW` < 30%, `MODERATE` 30–70%, `HIGH` $\ge 70\%$)
- `train_years`, `val_years`, `test_years`: List of years in each split
- `test_metrics`: Complete dictionary of test set performance (ROC-AUC, PR-AUC, F1, Recall, Precision, Confusion Matrix)
- `brier_score`, `calibration_ece`: Reliability calibration metrics

---

## 3. Automated Metadata Generation Utility

The module [`src/models/metadata.py`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/src/models/metadata.py) provides the factory function `generate_model_metadata()`:

```python
from src.models.metadata import generate_model_metadata, save_model_metadata

metadata = generate_model_metadata(
    model=best_model,
    model_name=best_model_name,
    feature_cols=feature_cols,
    threshold=best_threshold,
    config=config,
    dataset_path=combined_path,
    test_metrics=best_row.to_dict(),
    brier_score=float(best_row["Test_Brier_Score"]),
    calibration_ece=float(best_row["Test_Calibration_ECE"]),
    calibration_method="none"
)

save_model_metadata(metadata, "models/flood_prediction/model_metadata.json")
```

### Key Technical Mechanisms:
1. **Dynamic Environment Introspection**: Calls `importlib.metadata.version()` across `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `joblib`, `matplotlib`, `seaborn`, and `flask` at runtime.
2. **Cryptographic Dataset Hashing**: Reads `data/processed/historical_2001_2024.csv` in 64 KB memory-safe chunks and computes its SHA-256 hash (`1145a2ae6ac1bdaf798ae4f44c86c5f057c0a11cca9cba369dba49450df8f5f9`).
3. **Parameter Sanitization**: Handles special float values (e.g., `np.nan` $\to$ `None`, `np.inf` $\to$ `"Infinity"`) and non-primitive objects so that all hyperparameters serialize cleanly into valid JSON.

---

## 4. Atomic Model Bundle Loading

Models should never be loaded in isolation from their metadata. The `ModelBundle` class encapsulates both the model object and its metadata dictionary:

```python
from src.models.metadata import load_model_with_metadata

bundle = load_model_with_metadata(
    model_path="models/flood_prediction/best_model.pkl",
    metadata_path="models/flood_prediction/model_metadata.json"
)

print(bundle)
# <ModelBundle model='XGBoost' threshold=0.8073 n_features=22 calibration='none' py=3.10.0>
```

### Integration with `FloodPredictor`:
The production class [`FloodPredictor`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/src/models/predict.py) directly integrates `ModelBundle`:
- Initializes via `load()` which calls `load_model_with_metadata()`.
- Provides class constructor `FloodPredictor.from_bundle(bundle)`.
- Exposes `FloodPredictor.load_with_metadata(model_path, metadata_path)`.

---

## 5. Strict Pre-Prediction Feature Schema Validation

To prevent silent failures, bad predictions, or model crashes due to missing or misaligned data, `validate_feature_schema()` is executed on every input prior to prediction:

```python
from src.models.metadata import validate_feature_schema, SchemaValidationError

try:
    X_clean = validate_feature_schema(X_input, expected_features=bundle.features)
except SchemaValidationError as e:
    logger.error(f"Prediction rejected due to schema error: {e}")
```

### Validation Failure Modes & Enforcement Matrix:

| Violation | Test Condition | Result |
| :--- | :--- | :--- |
| **Missing Columns** | One or more required features omitted from DataFrame or Dict. | Raises `SchemaValidationError` with list of missing feature names. |
| **Dimension Mismatch** | NumPy array has $M \ne 22$ columns. | Raises `SchemaValidationError` stating expected vs received feature counts. |
| **Non-Numeric Dtypes** | Column contains string, object, or categorical types that cannot be coerced. | Raises `SchemaValidationError` identifying specific invalid columns and their dtypes. |
| **Null / NaN Values** | Required feature contains `NaN` or `None`. | Raises `SchemaValidationError` identifying columns with missing values. |
| **Infinite Values** | Required feature contains `+inf` or `-inf`. | Raises `SchemaValidationError` identifying columns with infinite values. |
| **Empty Input** | DataFrame or array contains 0 rows. | Raises `SchemaValidationError` rejecting empty inputs. |
| **Column Ordering** | Input has columns in random order with extra metadata (e.g. `District`, `Date`). | Validates presence, strips unneeded columns, and reorders strictly to model's expected feature order. |

Both `FloodPredictor.predict_probability()` and `FloodPredictor.predict_single()` execute this validation, guaranteeing that model inference is never invoked on invalid schemas.

---

## 6. Automated Test Verification

The test suite in [`tests/test_model_metadata.py`](file:///c:/Users/hp/OneDrive/Desktop/Odisha_Flood_Prediction/tests/test_model_metadata.py) contains 14 unit and integration tests:

1. `test_metadata_file_contains_all_required_keys`: Verifies all 15 required fields exist, are non-null, and have correct data types.
2. `test_dataset_hash_is_valid_sha256`: Verifies dataset hash is a 64-character SHA-256 string matching fresh computation.
3. `test_environment_versions_are_dynamic`: Confirms Python runtime and library versions match current environment.
4. `test_model_and_metadata_load_together`: Confirms atomic loading via `load_model_with_metadata` and `ModelBundle`.
5. `test_flood_predictor_load_with_metadata`: Confirms `FloodPredictor.load_with_metadata` instantiates properly.
6. `test_schema_validation_missing_feature_raises`: Rejects inputs missing required features.
7. `test_schema_validation_numpy_dimension_mismatch_raises`: Rejects array with wrong column dimension.
8. `test_schema_validation_non_numeric_dtype_raises`: Rejects non-numeric types.
9. `test_schema_validation_nan_values_raise`: Rejects null/NaN values.
10. `test_schema_validation_empty_input_raises`: Rejects 0-row inputs.
11. `test_schema_validation_valid_input_passes`: Allows valid inputs and ensures schema ordering.
12. `test_flood_predictor_enforces_schema_validation`: Confirms `FloodPredictor` blocks prediction on invalid schemas.
13. `test_generate_model_metadata_utility_automatic`: Tests dynamic extraction on a fresh scikit-learn model without hardcoding.
14. `test_calibrated_metadata_completeness`: Confirms `calibrated_metadata.json` conforms to reproducibility standards.
