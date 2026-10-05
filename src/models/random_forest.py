"""
Random Forest flood prediction model with balanced class weighting.
"""

from sklearn.ensemble import RandomForestClassifier
from typing import Optional


def create_random_forest_model(
    n_estimators: int = 200,
    max_depth: Optional[int] = 16,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    class_weight: str = "balanced_subsample",
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomForestClassifier:
    """
    Create a Random Forest classifier configured for imbalanced flood prediction.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
    return model
