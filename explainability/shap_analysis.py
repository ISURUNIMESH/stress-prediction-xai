"""Explain the model ranked highest by held-out macro F1 using SHAP."""

import json
from pathlib import Path
import sys

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("Run SHAP analysis in the project's CPython 3.12 environment.")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models.train_evaluate import (
    DATA_FILE,
    EXCLUDED,
    RANDOM_STATE,
    TEST_SIZE,
    build_models,
    make_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
SHAP_DIR = RESULTS / "shap_plots"
TARGET = "Stress_Level"


def main() -> None:
    metadata_path = METRICS / "run_metadata.json"
    comparison_path = METRICS / "model_comparison.csv"
    if not metadata_path.exists() or not comparison_path.exists():
        raise FileNotFoundError("Run python -m models.train_evaluate before SHAP analysis")
    data = pd.read_csv(DATA_FILE)
    y = pd.Series(LabelEncoder().fit_transform(data[TARGET].astype(str)), index=data.index)
    X = data.drop(columns=[column for column in EXCLUDED if column in data.columns])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = str(pd.read_csv(comparison_path).iloc[0]["model"])

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    pipeline = make_pipeline(
        build_models()[model_name], numeric, categorical,
        scale_numeric=model_name in {"Logistic Regression", "SVM"},
    )
    pipeline.fit(X_train, y_train)
    transformer = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    transformed_names = transformer.get_feature_names_out()
    sample = X_test.sample(n=min(100, len(X_test)), random_state=RANDOM_STATE)
    transformed = transformer.transform(sample)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    background_source = X_train.sample(n=min(50, len(X_train)), random_state=RANDOM_STATE)
    background_transformed = transformer.transform(background_source)
    background = (
        background_transformed.toarray()
        if hasattr(background_transformed, "toarray")
        else np.asarray(background_transformed)
    )

    if model_name in {"Random Forest", "XGBoost"}:
        explainer = shap.TreeExplainer(classifier)
        values = explainer.shap_values(dense, check_additivity=False)
    elif model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(classifier, background)
        values = explainer.shap_values(dense)
    else:
        # Kernel SHAP is model-agnostic and does not impose a tree explainer on SVM.
        prediction_function = lambda rows: classifier.predict_proba(rows)
        explainer = shap.KernelExplainer(prediction_function, background)
        values = explainer.shap_values(dense, nsamples=100)

    class_values = normalize_shap_values(values, len(y.unique()), dense.shape[0], dense.shape[1])
    mean_abs = np.mean(np.abs(class_values), axis=(0, 1))
    importance = pd.DataFrame(
        {"feature": transformed_names, "mean_absolute_shap_value": mean_abs}
    ).sort_values("mean_absolute_shap_value", ascending=False)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(SHAP_DIR / "global_feature_importance.csv", index=False)

    top = importance.head(20).sort_values("mean_absolute_shap_value")
    ax = top.plot.barh(x="feature", y="mean_absolute_shap_value", legend=False, figsize=(9, 7))
    ax.set_xlabel("Mean absolute SHAP value across classes")
    ax.set_ylabel("Transformed feature")
    ax.set_title(f"Global SHAP importance: {model_name}")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_feature_importance.png", dpi=160)
    plt.close()

    summary_values = class_values.reshape(-1, dense.shape[1])
    summary_features = np.tile(dense, (class_values.shape[0], 1))
    shap.summary_plot(
        summary_values,
        summary_features,
        feature_names=transformed_names,
        show=False,
    )
    plt.title(f"SHAP summary: {model_name} (class-specific contributions)")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    local_index = 0
    local_class = int(np.argmax(classifier.predict_proba(dense[local_index : local_index + 1])[0]))
    local_effect = class_values[local_class, local_index]
    local = pd.DataFrame(
        {
            "feature": transformed_names,
            "shap_value": local_effect,
            "explained_class": metadata["classes"][local_class],
        }
    )
    local["absolute_shap_value"] = local["shap_value"].abs()
    local.sort_values("absolute_shap_value", ascending=False).to_csv(
        SHAP_DIR / "local_prediction_contributions.csv", index=False
    )
    metadata["shap_model"] = model_name
    metadata["shap_explainer"] = {
        "Random Forest/XGBoost": "TreeExplainer",
        "Logistic Regression": "LinearExplainer",
        "SVM": "KernelExplainer",
    }[model_name]
    (METRICS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"SHAP model: {model_name} (ranked highest by held-out macro F1)")
    print(f"SHAP outputs saved in {SHAP_DIR}")


def normalize_shap_values(values, class_count, sample_count, feature_count):
    """Normalize SHAP versions' list/array outputs to (class, sample, feature)."""
    if isinstance(values, list):
        array = np.stack([np.asarray(item) for item in values], axis=0)
    else:
        array = np.asarray(values)
        if array.ndim == 3 and array.shape == (sample_count, feature_count, class_count):
            array = np.moveaxis(array, -1, 0)
        elif array.ndim == 2:
            array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"Unexpected SHAP value shape: {array.shape}")
    if array.shape[0] == 1 and class_count > 1:
        array = np.repeat(array, class_count, axis=0)
    if array.shape[0] != class_count:
        raise ValueError(f"Expected SHAP values for {class_count} classes; got {array.shape}")
    return array


if __name__ == "__main__":
    main()
