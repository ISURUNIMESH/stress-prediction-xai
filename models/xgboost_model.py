"""XGBoost baseline factory."""

from xgboost import XGBClassifier


def build_model(random_state: int = 42) -> XGBClassifier:
    """Create a modest multiclass XGBoost classifier."""
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
