"""
Decision Tree baseline model with class weighting and depth regularization.
"""

from sklearn.tree import DecisionTreeClassifier


def create_decision_tree_model(
    max_depth: int = 8,
    min_samples_leaf: int = 50,
    class_weight: str = "balanced",
    random_state: int = 42
) -> DecisionTreeClassifier:
    """
    Create a Decision Tree baseline classifier.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state
    )
    return model
