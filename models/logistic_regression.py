"""Logistic Regression baseline factory."""

from sklearn.linear_model import LogisticRegression


def build_model(random_state: int = 42) -> LogisticRegression:
    """Create the regularized multinomial baseline classifier."""
    return LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=random_state
    )
