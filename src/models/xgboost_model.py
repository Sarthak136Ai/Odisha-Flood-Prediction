"""
XGBoost flood prediction model with scale_pos_weight optimization.
"""

from xgboost import XGBClassifier
from typing import Optional


def create_xgboost_model(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    scale_pos_weight: float = 1.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1
) -> XGBClassifier:
    """
    Create an XGBoost classifier configured for flood early warning.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss"
    )
    return model
