"""
Unit tests for Decision Tree baseline classifier.
"""

import numpy as np
from src.models.decision_tree import create_decision_tree_model


def test_decision_tree_instantiation_and_fit():
    dt = create_decision_tree_model(max_depth=4, random_state=42)
    X = np.random.randn(100, 10)
    y = np.random.binomial(1, 0.2, 100)
    
    dt.fit(X, y)
    preds = dt.predict(X)
    probas = dt.predict_proba(X)
    
    assert preds.shape == (100,)
    assert probas.shape == (100, 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
