"""
Artificial Neural Network (ANN/MLP) flood prediction pipeline.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Tuple


def create_ann_model(
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    activation: str = "relu",
    alpha: float = 0.0001,
    batch_size: int = 2048,
    max_iter: int = 50,
    learning_rate_init: float = 0.001,
    random_state: int = 42
) -> Pipeline:
    """
    Create a Multi-Layer Perceptron (ANN) pipeline with StandardScaler and mini-batching.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=random_state
        ))
    ])
    return model
