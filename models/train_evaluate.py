"""Train, compare, and cross-validate the four baseline classifiers."""

from pathlib import Path
import json
import sys
import warnings

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(
        "This modeling project is pinned to CPython 3.12. Create the environment "
        "with 'py -3.12 -m venv .venv' and activate it before running models."
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib.metadata import version as package_version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
import seaborn as sns
from models.logistic_regression import build_model as build_logistic_regression
from models.random_forest import build_model as build_random_forest
from models.svm import build_model as build_svm
from models.xgboost_model import build_model as build_xgboost


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "dataset" / "processed" / "cleaned_stress_dataset.csv"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
TARGET = "Stress_Level"
EXCLUDED = ["Timestamp", "Consent", "PSS10_Total", TARGET] + [
    f"PSS_Q{i}" for i in range(1, 11)
]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_models() -> dict:
    """Return fixed, reproducible baseline estimators."""
    return {
        "Logistic Regression": build_logistic_regression(RANDOM_STATE),
        "SVM": build_svm(RANDOM_STATE),
        "Random Forest": build_random_forest(RANDOM_STATE),
        "XGBoost": build_xgboost(RANDOM_STATE),
    }


def make_pipeline(model, numeric_columns, categorical_columns, scale_numeric=True):
    """Create preprocessing inside each estimator to prevent split leakage."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("classifier", model),
        ]
    )


def aligned_probabilities(estimator, features, class_labels):
    """Align probability columns to the global class order."""
    raw = estimator.predict_proba(features)
    output = np.zeros((len(features), len(class_labels)))
    for index, label in enumerate(estimator.classes_):
        output[:, list(class_labels).index(label)] = raw[:, index]
    return output


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError("Run preprocessing/preprocess.py first")
    RESULTS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    figures = RESULTS / "figures"
    figures.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_FILE)
    if TARGET not in data:
        raise ValueError(f"Processed data must contain {TARGET}")
    y_text = data[TARGET].astype(str)
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_text), index=data.index, name=TARGET)
    X = data.drop(columns=[column for column in EXCLUDED if column in data.columns])
    if X.empty:
        raise ValueError("No predictor features remain after excluding leakage/admin columns")

    class_counts = y.value_counts()
    if len(class_counts) != 3 or class_counts.min() < 10:
        raise ValueError("Stratified 10-fold CV requires at least 10 samples in every class")
    labels = sorted(y.unique())
    class_names = label_encoder.classes_.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_macro": "precision_macro",
        "precision_weighted": "precision_weighted",
        "recall_macro": "recall_macro",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
        "roc_auc_ovr_macro": make_scorer(
            roc_auc_score,
            response_method="predict_proba",
            multi_class="ovr",
            average="macro",
        ),
    }

    model_rows, cv_rows = [], []
    fitted = {}
    for name, estimator in build_models().items():
        scale = name in {"Logistic Regression", "SVM"}
        pipeline = make_pipeline(estimator, numeric, categorical, scale)
        print(f"Fitting {name} ...", flush=True)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        prediction = pipeline.predict(X_test)
        probabilities = aligned_probabilities(pipeline, X_test, labels)
        auc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted", labels=labels)
        model_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                "precision_weighted": precision_score(y_test, prediction, average="weighted", zero_division=0),
                "precision_macro": precision_score(y_test, prediction, average="macro", zero_division=0),
                "recall_weighted": recall_score(y_test, prediction, average="weighted", zero_division=0),
                "recall_macro": recall_score(y_test, prediction, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_test, prediction, average="weighted", zero_division=0),
                "f1_macro": f1_score(y_test, prediction, average="macro", zero_division=0),
                "roc_auc_ovr_weighted": auc,
            }
        )
        report = classification_report(
            y_test,
            prediction,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(METRICS / f"{slug(name)}_classification_report.csv")

        matrix = confusion_matrix(y_test, prediction, labels=labels)
        pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
            METRICS / f"{slug(name)}_confusion_matrix.csv"
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
        )
        plt.xlabel("Predicted stress level")
        plt.ylabel("Actual stress level")
        plt.title(f"{name} confusion matrix")
        plt.tight_layout()
        plt.savefig(figures / f"{slug(name)}_confusion_matrix.png", dpi=160)
        plt.close()

        cv_result = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
        for metric in scoring:
            values = cv_result[f"test_{metric}"]
            cv_rows.append(
                {"model": name, "metric": metric, "mean": values.mean(), "std": values.std(ddof=1)}
            )
        print(f"Completed {name}", flush=True)

    comparison = pd.DataFrame(model_rows).sort_values(
        ["f1_macro", "f1_weighted"], ascending=False
    )
    comparison.to_csv(METRICS / "model_comparison.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(METRICS / "cross_validation_10fold.csv", index=False)
    comparison.set_index("model").plot(kind="bar", figsize=(11, 6))
    plt.ylabel("Test score")
    plt.ylim(0, 1)
    plt.title("Held-out test performance")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "model_performance_comparison.png", dpi=160)
    plt.close()

    # One-vs-rest ROC plots use the same fixed held-out test partition.
    y_binary = label_binarize(y_test, classes=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, pipeline in fitted.items():
        probabilities = aligned_probabilities(pipeline, X_test, labels)
        RocCurveDisplay.from_predictions(
            y_binary.ravel(), probabilities.ravel(), name=name, ax=ax
        )
    ax.set_title("Micro-average multiclass ROC curves (one-vs-rest)")
    fig.tight_layout()
    fig.savefig(figures / "roc_curves.png", dpi=160)
    plt.close(fig)

    metadata = {
        "features_used": X.columns.tolist(),
        "excluded_columns": {column: exclusion_reason(column) for column in EXCLUDED if column in data},
        "target": TARGET,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "python_version": sys.version,
        "package_versions": {
            package: package_version(distribution)
            for package, distribution in {
                "pandas": "pandas",
                "numpy": "numpy",
                "scikit-learn": "scikit-learn",
                "xgboost": "xgboost",
                "shap": "shap",
                "matplotlib": "matplotlib",
                "seaborn": "seaborn",
            }.items()
        },
        "classes": class_names,
        "class_encoding": {name: index for index, name in enumerate(class_names)},
        "class_counts": {name: int(class_counts[index]) for index, name in enumerate(class_names)},
        "selected_for_shap_by": "highest held-out macro F1, weighted F1 tie-breaker",
        "selected_for_shap": comparison.iloc[0]["model"],
    }
    (METRICS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nHeld-out comparison:")
    print(comparison.to_string(index=False))
    print(f"Results saved in {RESULTS}")


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def exclusion_reason(column: str) -> str:
    if column.startswith("PSS_Q") or column == "PSS10_Total":
        return "Excluded to prevent target leakage from PSS-derived Stress_Level"
    if column in {"Timestamp", "Consent"}:
        return "Administrative field, not a predictive response"
    return "Target column"


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
