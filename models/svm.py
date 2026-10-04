"""Support Vector Machine baseline factory."""

from sklearn.svm import SVC


def build_model(random_state: int = 42) -> SVC:
    """Create an RBF SVM with probabilities for multiclass ROC-AUC."""
    return SVC(
        C=1.0,
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )
