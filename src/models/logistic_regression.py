"""
Logistic Regression model pipeline with standard scaling and class weighting.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def create_logistic_regression_pipeline(
    max_iter: int = 1000,
    class_weight: str = "balanced",
    C: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Create a Logistic Regression pipeline with StandardScaler.
    Scaler is fit only on training data inside the pipeline.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            C=C,
            solver="lbfgs",
            random_state=random_state
        ))
    ])
    return model
