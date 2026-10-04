"""Train and export the final selected SVM inference pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("Use the project's Python 3.12 environment to train the final model.")

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train_evaluate import DATA_FILE, EXCLUDED, RANDOM_STATE, TARGET, make_pipeline
from models.svm import build_model as build_svm

ARTIFACTS = ROOT / "application" / "artifacts"
PREDICTORS = [
    "Age",
    "Gender",
    "University",
    "Degree_Program",
    "Academic_Year",
    "Sleep_Hours",
    "Exercise_Frequency",
    "Screen_Time_Hours",
    "Social_Media_Hours",
    "Study_Hours",
    "Academic_Pressure",
    "Financial_Stress",
    "Family_Support",
]
NESTED_CV = {
    "macro_f1_mean": 0.3425265445837976,
    "macro_f1_std": 0.02151743133261998,
    "balanced_accuracy_mean": 0.34510755626159956,
    "balanced_accuracy_std": 0.02415686741735075,
}


def package_versions():
    versions = {"python": sys.version.split()[0]}
    for package in [
        "Flask",
        "joblib",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "shap",
        "matplotlib",
        "seaborn",
        "scipy",
    ]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def main():
    data = pd.read_csv(DATA_FILE)
    if TARGET not in data:
        raise ValueError(f"Processed dataset is missing target {TARGET}")
    forbidden = {f"PSS_Q{i}" for i in range(1, 11)} | {"PSS10_Total", TARGET}
    observed_predictors = [column for column in data.columns if column not in EXCLUDED]
    if observed_predictors != PREDICTORS:
        raise ValueError(
            f"Dataset predictor schema changed. Expected {PREDICTORS}, found {observed_predictors}"
        )
    if forbidden.intersection(observed_predictors):
        raise ValueError("PSS or target leakage detected in predictor list")

    X = data[PREDICTORS].copy()
    y_text = data[TARGET].astype(str)
    classes = sorted(y_text.unique().tolist())
    if classes != ["High Stress", "Low Stress", "Moderate Stress"]:
        raise ValueError(f"Unexpected target classes: {classes}")
    target_lookup = {name: index for index, name in enumerate(classes)}
    y = y_text.map(target_lookup).astype(int)

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    model = build_svm(RANDOM_STATE)
    # C=2.0 was selected in all five outer folds of the nested CV search;
    # use that selected RBF SVM specification when fitting the deployable model.
    model.set_params(C=2.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced")
    pipeline = make_pipeline(model, numeric, categorical, scale_numeric=True)
    pipeline.fit(X, y)

    preprocessor = pipeline.named_steps["preprocess"]
    transformed_names = preprocessor.get_feature_names_out().tolist()
    X_background_train, _ = train_test_split(
        X, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    background_raw = X_background_train.sample(
        n=min(30, len(X_background_train)), random_state=RANDOM_STATE
    )
    background_transformed = preprocessor.transform(background_raw)
    if hasattr(background_transformed, "toarray"):
        background_transformed = background_transformed.toarray()
    background = {
        "data": np.asarray(background_transformed, dtype=np.float64),
        "feature_names": transformed_names,
        "sample_size": int(len(background_raw)),
        "random_state": RANDOM_STATE,
        "source": "random sample from seed-42 stratified 80% training partition; held-out 20% excluded",
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS / "final_svm_pipeline.joblib", compress=3)
    joblib.dump(background, ARTIFACTS / "shap_background.joblib", compress=3)

    categorical_values = {
        column: sorted(X[column].astype(str).unique().tolist()) for column in categorical
    }
    numeric_ranges = {}
    numeric_distributions = {}
    categorical_frequencies = {}
    for column in numeric:
        values = pd.to_numeric(X[column], errors="raise")
        numeric_ranges[column] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "integer_only": bool(np.all(np.equal(values, np.floor(values)))),
        }
        numeric_distributions[column] = {
            "median": float(values.median()),
            "mean": float(values.mean()),
            "q1": float(values.quantile(0.25)),
            "q3": float(values.quantile(0.75)),
            "iqr_fence_minimum": float(values.quantile(0.25) - 1.5 * (values.quantile(0.75) - values.quantile(0.25))),
            "iqr_fence_maximum": float(values.quantile(0.75) + 1.5 * (values.quantile(0.75) - values.quantile(0.25))),
            "value_counts": {str(key): int(count) for key, count in values.value_counts().sort_index().items()},
        }
    for column in categorical:
        categorical_frequencies[column] = {
            str(key): int(count)
            for key, count in X[column].astype(str).value_counts().sort_index().items()
        }
    metadata = {
        "model": "SVM",
        "model_type": "sklearn.svm.SVC (RBF, probability=True) in sklearn Pipeline",
        "selected_parameters": {"C": 2.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
        "feature_names": PREDICTORS,
        "transformed_feature_names": transformed_names,
        "target_name": TARGET,
        "classes": classes,
        "training_dataset_size": int(len(data)),
        "shap_background_training_rows": int(len(X_background_train)),
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_STATE,
        "model_selection_criterion": "Highest nested 5x5 CV outer mean Macro F1 among candidates; Balanced Accuracy as tie-breaker.",
        "research_evaluation_metrics": {
            "protocol": "Nested Stratified Cross-Validation; internal evaluation estimates, not training metrics",
            "outer_folds": 5,
            "macro_f1_mean": NESTED_CV["macro_f1_mean"],
            "macro_f1_std": NESTED_CV["macro_f1_std"],
            "balanced_accuracy_mean": NESTED_CV["balanced_accuracy_mean"],
            "balanced_accuracy_std": NESTED_CV["balanced_accuracy_std"],
        },
        "preprocessing": "Same models.train_evaluate.make_pipeline: numeric median imputation and StandardScaler; categorical most-frequent imputation and OneHotEncoder(handle_unknown='ignore').",
        "excluded_predictors": sorted(forbidden),
        "numeric_ranges": numeric_ranges,
        "numeric_distributions": numeric_distributions,
        "categorical_values": categorical_values,
        "categorical_frequencies": categorical_frequencies,
        "packages": package_versions(),
        "research_notice": "This system is a research prototype and does not provide a medical diagnosis.",
        "training_metrics_claimed": False,
    }
    (ARTIFACTS / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved pipeline: {ARTIFACTS / 'final_svm_pipeline.joblib'}")
    print(f"Saved metadata: {ARTIFACTS / 'model_metadata.json'}")
    print(f"Saved SHAP background: {ARTIFACTS / 'shap_background.joblib'}")


if __name__ == "__main__":
    main()
