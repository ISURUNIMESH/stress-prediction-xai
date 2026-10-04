"""Nested CV experiment for three-class stress prediction improvement.

SMOTENC is applied to an ordinal-coded mixed feature matrix inside each
training partition; one-hot encoding follows resampling. Decision thresholds
are selected only from inner out-of-fold probabilities and never alter the
probabilities returned by the fitted SVM.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("Run this experiment with the project's CPython 3.12 environment.")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "dataset" / "processed" / "cleaned_stress_dataset.csv"
OUTPUT = ROOT / "results" / "three_class_improvement"
REPORTS = OUTPUT / "classification_reports"
MATRICES = OUTPUT / "confusion_matrices"
DISTRIBUTIONS = OUTPUT / "prediction_distributions"
THRESHOLD_DIR = OUTPUT / "threshold_analysis"
SHAP_DIR = OUTPUT / "shap"
CANDIDATE_ARTIFACT_DIR = ROOT / "application" / "artifacts" / "three_class_candidate"
TARGET = "Stress_Level"
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
CLASS_NAMES = ["High Stress", "Low Stress", "Moderate Stress"]
PSS_FIELDS = [*[f"PSS_Q{i}" for i in range(1, 11)], "PSS10_Total"]
EXCLUDED = ["Timestamp", "Consent", TARGET, *PSS_FIELDS]
CURRENT_SVM_PARAMS = {"C": 2.0, "gamma": "scale", "class_weight": "balanced"}
WEIGHT_CANDIDATES = [
    "balanced",
    {0: 2.0, 1: 2.0, 2: 1.0},
    {0: 4.0, 1: 4.0, 2: 1.0},
]
WEIGHTED_GRID = [
    {"C": c, "gamma": gamma, "class_weight": weight}
    for c in (1.0, 2.0)
    for gamma in ("scale", 0.1)
    for weight in WEIGHT_CANDIDATES
]
SMOTENC_TUNED_GRID = [
    {"C": c, "gamma": gamma, "class_weight": "balanced", "sampling_ratio": ratio}
    for c in (1.0, 2.0)
    for gamma in ("scale", 0.1)
    for ratio in (0.25, 0.50)
]
THRESHOLDS = [
    {"name": "argmax", "values": [1 / 3, 1 / 3, 1 / 3]},
    *[
        {"name": f"ovr_H{high:.2f}_L{low:.2f}_M0.33", "values": [high, low, 1 / 3]}
        for high in (0.20, 0.25, 0.30, 1 / 3)
        for low in (0.20, 0.25, 0.30, 1 / 3)
    ],
]
BASE_FAMILIES = {
    "Current SVM": {
        "sampler": None,
        "grid": [CURRENT_SVM_PARAMS],
        "threshold_variant": False,
    },
    "SMOTENC + SVM": {
        "sampler": 0.25,
        "grid": [{"C": 2.0, "gamma": "scale", "class_weight": None, "sampling_ratio": 0.25}],
        "threshold_variant": True,
    },
    "SMOTENC + weighted SVM": {
        "sampler": 0.25,
        "grid": [
            {"C": 2.0, "gamma": "scale", "class_weight": weight, "sampling_ratio": 0.25}
            for weight in WEIGHT_CANDIDATES
        ],
        "threshold_variant": True,
    },
    "Tuned weighted SVM": {
        "sampler": None,
        "grid": WEIGHTED_GRID,
        "threshold_variant": True,
    },
    "SMOTENC + tuned SVM": {
        "sampler": "tuned",
        "grid": SMOTENC_TUNED_GRID,
        "threshold_variant": True,
    },
}


@dataclass
class FittedStressSVM:
    """Pickleable candidate model preserving probability outputs."""

    preprocessor: object
    postprocessor: object | None
    classifier: SVC
    feature_names: list[str]
    class_names: list[str]
    thresholds: list[float]

    def transform(self, X):
        matrix = self.preprocessor.transform(X)
        if self.postprocessor is not None:
            matrix = self.postprocessor.transform(matrix)
        return matrix

    def predict_proba(self, X):
        raw = self.classifier.predict_proba(self.transform(X))
        aligned = np.zeros((len(X), len(self.class_names)), dtype=float)
        for position, class_id in enumerate(self.classifier.classes_):
            aligned[:, int(class_id)] = raw[:, position]
        return aligned

    def predict_hard(self, X):
        """Use SVC.predict, matching the existing application's label path."""
        return self.classifier.predict(self.transform(X))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities - np.asarray(self.thresholds)[None, :], axis=1)


def load_predictors(data: pd.DataFrame | None = None):
    frame = pd.read_csv(DATA_FILE) if data is None else data.copy()
    if TARGET not in frame:
        raise ValueError(f"Processed data is missing {TARGET}")
    X = frame.drop(columns=[column for column in EXCLUDED if column in frame.columns]).copy()
    if X.empty or set(PSS_FIELDS + [TARGET]).intersection(X.columns):
        raise ValueError("Predictor leakage or empty feature set detected")
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(frame[TARGET].astype(str)), index=frame.index, name=TARGET)
    if encoder.classes_.tolist() != CLASS_NAMES:
        raise ValueError(f"Unexpected target classes: {encoder.classes_.tolist()}")
    return X, y, encoder


def numeric_and_categorical(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    if not numeric or not categorical:
        raise ValueError("SMOTENC experiment requires numeric and categorical predictors")
    return numeric, categorical


def smotenc_strategy(y_train, ratio):
    """Create conservative per-minority targets using this training fold only."""
    counts = Counter(np.asarray(y_train, dtype=int).tolist())
    majority = max(counts.values())
    desired = int(math.ceil(majority * float(ratio)))
    strategy = {class_id: desired for class_id, count in counts.items() if count < desired}
    if not strategy:
        raise ValueError("SMOTENC strategy does not request an increase for this training fold")
    return strategy


def build_smotenc_preprocessors(X_train):
    numeric, categorical = numeric_and_categorical(X_train)
    pre = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)),
            ]), categorical),
        ],
        sparse_threshold=0.0,
    )
    numeric_count = len(numeric)
    feature_count = len(numeric) + len(categorical)
    post = ColumnTransformer(
        [
            ("numeric", "passthrough", list(range(numeric_count))),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(range(numeric_count, feature_count))),
        ],
        sparse_threshold=0.3,
    )
    return pre, post, list(range(numeric_count, feature_count))


def build_standard_preprocessor(X_train):
    numeric, categorical = numeric_and_categorical(X_train)
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        sparse_threshold=0.3,
    )


def fit_model(X_train, y_train, params, sampler_ratio=None, seed=SEED):
    """Fit preprocessing, optional SMOTENC, and SVM on this training split only."""
    y_values = np.asarray(y_train, dtype=int)
    sampler = None
    post = None
    if sampler_ratio is not None:
        pre, post, categorical_indices = build_smotenc_preprocessors(X_train)
        prepared = pre.fit_transform(X_train)
        ratio = sampler_ratio
        if isinstance(params, dict) and "sampling_ratio" in params:
            ratio = params["sampling_ratio"]
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=smotenc_strategy(y_values, ratio),
            k_neighbors=5,
            random_state=seed,
        )
        prepared, y_values = sampler.fit_resample(prepared, y_values)
        prepared = post.fit_transform(prepared)
    else:
        pre = build_standard_preprocessor(X_train)
        prepared = pre.fit_transform(X_train)

    classifier = SVC(
        kernel="rbf",
        probability=True,
        random_state=seed,
        C=float(params["C"]),
        gamma=params["gamma"],
        class_weight=params.get("class_weight"),
    )
    classifier.fit(prepared, y_values)
    transformed_names = list(pre.get_feature_names_out())
    if post is not None:
        transformed_names = list(post.get_feature_names_out())
    return FittedStressSVM(pre, post, classifier, transformed_names, CLASS_NAMES, [1 / 3] * 3)


def predict_probabilities(fitted, X):
    probabilities = fitted.predict_proba(X)
    return probabilities


def apply_thresholds(probabilities, thresholds):
    """Select a class by maximum probability minus its locked OVR cutoff."""
    return np.argmax(np.asarray(probabilities) - np.asarray(thresholds)[None, :], axis=1)


def compute_metrics(y_true, predicted, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    per_precision = precision_score(y_true, predicted, labels=np.arange(3), average=None, zero_division=0)
    per_recall = recall_score(y_true, predicted, labels=np.arange(3), average=None, zero_division=0)
    per_f1 = f1_score(y_true, predicted, labels=np.arange(3), average=None, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "roc_auc_macro_ovr": float(roc_auc_score(y_true, probabilities, labels=np.arange(3), multi_class="ovr", average="macro")),
        "roc_auc_weighted_ovr": float(roc_auc_score(y_true, probabilities, labels=np.arange(3), multi_class="ovr", average="weighted")),
    }
    for class_id, class_name in enumerate(("high", "low", "moderate")):
        result[f"{class_name}_precision"] = float(per_precision[class_id])
        result[f"{class_name}_recall"] = float(per_recall[class_id])
        result[f"{class_name}_f1"] = float(per_f1[class_id])
    return result


def fold_scores(y_true, probabilities, threshold, hard_predictions=None):
    if hard_predictions is not None:
        predicted = np.asarray(hard_predictions, dtype=int)
    else:
        predicted = apply_thresholds(probabilities, threshold)
    return compute_metrics(y_true, predicted, probabilities), predicted


def parameter_grid(family):
    return list(family["grid"])


def select_inner_configurations(model_name, family, X_train, y_train, outer_fold, trace_rows):
    """Choose parameters/thresholds from five inner validation folds only."""
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    candidate_scores = []
    for parameter_index, params in enumerate(parameter_grid(family)):
        inner_outputs = []
        for inner_fold, (train_indices, valid_indices) in enumerate(inner.split(X_train, y_train), start=1):
            ratio = family["sampler"]
            if ratio == "tuned":
                ratio = params["sampling_ratio"]
            fitted = fit_model(
                X_train.iloc[train_indices], y_train.iloc[train_indices], params,
                sampler_ratio=ratio, seed=SEED + outer_fold * 100 + inner_fold,
            )
            X_inner_valid = X_train.iloc[valid_indices]
            probabilities = predict_probabilities(fitted, X_inner_valid)
            hard_predictions = fitted.predict_hard(X_inner_valid)
            inner_outputs.append((np.asarray(y_train.iloc[valid_indices]), probabilities, hard_predictions))

        thresholds_to_evaluate = THRESHOLDS if family["threshold_variant"] else THRESHOLDS[:1]
        for threshold in thresholds_to_evaluate:
            metrics_by_fold = [
                fold_scores(
                    y_values,
                    probs,
                    threshold["values"],
                    hard_predictions=hard if threshold["name"] == "argmax" else None,
                )[0]
                for y_values, probs, hard in inner_outputs
            ]
            row = {
                "family": model_name,
                "parameter_index": parameter_index,
                "parameters": params,
                "threshold_name": threshold["name"],
                "threshold_values": threshold["values"],
                "inner_macro_f1_mean": float(np.mean([item["macro_f1"] for item in metrics_by_fold])),
                "inner_balanced_accuracy_mean": float(np.mean([item["balanced_accuracy"] for item in metrics_by_fold])),
                "inner_low_recall_mean": float(np.mean([item["low_recall"] for item in metrics_by_fold])),
                "inner_high_recall_mean": float(np.mean([item["high_recall"] for item in metrics_by_fold])),
                "outer_fold": outer_fold,
            }
            trace_rows.append({**row, "parameters": json.dumps(params, sort_keys=True)})
            candidate_scores.append(row)

    candidate_scores.sort(
        key=lambda row: (row["inner_macro_f1_mean"], row["inner_balanced_accuracy_mean"]),
        reverse=True,
    )
    best_default = max(
        (row for row in candidate_scores if row["threshold_name"] == "argmax"),
        key=lambda row: (row["inner_macro_f1_mean"], row["inner_balanced_accuracy_mean"]),
    )
    best_threshold = candidate_scores[0]
    return best_default, best_threshold, candidate_scores


def metrics_for_outer_fold(model_name, variant, family, best_row, X_train, y_train, X_test, y_test, outer_fold):
    params = best_row["parameters"]
    ratio = family["sampler"]
    if ratio == "tuned":
        ratio = params["sampling_ratio"]
    fitted = fit_model(
        X_train, y_train, params, sampler_ratio=ratio,
        seed=SEED + 1000 + outer_fold,
    )
    probabilities = predict_probabilities(fitted, X_test)
    threshold_values = best_row["threshold_values"]
    hard_predictions = fitted.predict_hard(X_test) if best_row["threshold_name"] == "argmax" else None
    metrics, predicted = fold_scores(y_test, probabilities, threshold_values, hard_predictions=hard_predictions)
    result = {
        "model": model_name if variant == "argmax" else f"Threshold-tuned {model_name}",
        "family": model_name,
        "decision_variant": variant,
        "outer_fold": outer_fold,
        "selected_parameters": json.dumps(params, sort_keys=True),
        "selected_threshold": best_row["threshold_name"],
        "threshold_values": json.dumps(threshold_values),
        **metrics,
    }
    return result, predicted, probabilities, fitted


def write_tables(fold_rows, prediction_rows, threshold_rows, selected_rows, metadata):
    for directory in (OUTPUT, REPORTS, MATRICES, DISTRIBUTIONS, THRESHOLD_DIR, SHAP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUTPUT / "nested_cv_results.csv", index=False)
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(OUTPUT / "outer_fold_predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(THRESHOLD_DIR / "inner_threshold_candidates.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(THRESHOLD_DIR / "selected_by_outer_fold.csv", index=False)

    numeric_metrics = [column for column in folds.select_dtypes(include=np.number).columns if column != "outer_fold"]
    summary_rows = []
    for model, group in folds.groupby("model", sort=False):
        for metric in numeric_metrics:
            summary_rows.append({
                "model": model,
                "metric": metric,
                "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=1)),
                "outer_folds": int(len(group)),
                "uncertainty_note": "Descriptive SD across overlapping outer CV partitions; no inferential claim.",
            })
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "nested_cv_summary.csv", index=False)

    distribution_rows = []
    for model, group in predictions.groupby("model", sort=False):
        for fold, fold_group in group.groupby("outer_fold", sort=True):
            counts = fold_group["predicted"].value_counts()
            distribution_rows.append({
                "model": model,
                "outer_fold": int(fold),
                "predicted_low_count": int(counts.get("Low Stress", 0)),
                "predicted_moderate_count": int(counts.get("Moderate Stress", 0)),
                "predicted_high_count": int(counts.get("High Stress", 0)),
                "low_predicted": bool(counts.get("Low Stress", 0)),
                "moderate_predicted": bool(counts.get("Moderate Stress", 0)),
                "high_predicted": bool(counts.get("High Stress", 0)),
            })
    distributions = pd.DataFrame(distribution_rows)
    distributions.to_csv(DISTRIBUTIONS / "predicted_counts_by_outer_fold.csv", index=False)
    distribution_summary = distributions.groupby("model", sort=False).agg(
        folds_low_predicted=("low_predicted", "sum"),
        folds_moderate_predicted=("moderate_predicted", "sum"),
        folds_high_predicted=("high_predicted", "sum"),
    ).reset_index()
    distribution_summary.to_csv(DISTRIBUTIONS / "class_presence_across_folds.csv", index=False)

    pooled_confusions = []
    pooled_reports = []
    for model, group in predictions.groupby("model", sort=False):
        actual = [CLASS_NAMES.index(value) for value in group["actual"]]
        predicted = [CLASS_NAMES.index(value) for value in group["predicted"]]
        matrix = confusion_matrix(actual, predicted, labels=np.arange(3))
        for actual_id, actual_name in enumerate(CLASS_NAMES):
            for predicted_id, predicted_name in enumerate(CLASS_NAMES):
                pooled_confusions.append({"model": model, "actual": actual_name, "predicted": predicted_name, "count": int(matrix[actual_id, predicted_id])})
        report = classification_report(actual, predicted, labels=np.arange(3), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        for label, metrics in report.items():
            pooled_reports.append({"model": model, "class_or_average": label, **(metrics if isinstance(metrics, dict) else {"value": metrics})})
        slug = model.lower().replace("+", "plus").replace(" ", "_")
        pd.DataFrame(pooled_confusions[-9:]).pivot(index="actual", columns="predicted", values="count").reindex(index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(MATRICES / f"{slug}.csv")
        pd.DataFrame([row for row in pooled_reports if row["model"] == model]).drop(columns="model").to_csv(REPORTS / f"{slug}.csv", index=False)
    pd.DataFrame(pooled_confusions).to_csv(OUTPUT / "confusion_matrices.csv", index=False)
    pd.DataFrame(pooled_reports).to_csv(OUTPUT / "classification_reports.csv", index=False)

    comparison = make_model_comparison(folds, distributions)
    comparison.to_csv(OUTPUT / "model_comparison.csv", index=False)
    (OUTPUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return folds, predictions, comparison, distributions


def make_model_comparison(folds, distributions):
    metrics = ["macro_f1", "balanced_accuracy", "low_recall", "moderate_recall", "high_recall", "low_f1", "moderate_f1", "high_f1"]
    rows = []
    for model, group in folds.groupby("model", sort=False):
        row = {"Model": model}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["outer_folds"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows).merge(distributions.groupby("model", sort=False).agg(
        folds_low_predicted=("low_predicted", "sum"),
        folds_moderate_predicted=("moderate_predicted", "sum"),
        folds_high_predicted=("high_predicted", "sum"),
    ).reset_index(), left_on="Model", right_on="model", how="left").drop(columns="model")


def select_final_candidate(comparison, folds):
    baseline = comparison.loc[comparison["Model"] == "Current SVM"].iloc[0]
    candidates = comparison[comparison["Model"] != "Current SVM"].sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean"], ascending=False, kind="mergesort"
    )
    best = candidates.iloc[0]
    baseline_folds = folds[folds.model == "Current SVM"].set_index("outer_fold")
    candidate_folds = folds[folds.model == best["Model"]].set_index("outer_fold")
    macro_wins = int((candidate_folds.macro_f1 > baseline_folds.macro_f1).sum())
    minority_recall = (best.low_recall_mean + best.high_recall_mean) / 2
    baseline_minority_recall = (baseline.low_recall_mean + baseline.high_recall_mean) / 2
    gate = {
        "macro_f1_improved": bool(best.macro_f1_mean > baseline.macro_f1_mean),
        "balanced_accuracy_not_lower": bool(best.balanced_accuracy_mean >= baseline.balanced_accuracy_mean),
        "mean_minority_recall_improved": bool(minority_recall > baseline_minority_recall),
        "moderate_recall_drop_at_most_0_15": bool(best.moderate_recall_mean >= baseline.moderate_recall_mean - 0.15),
        "all_classes_predicted_in_3_of_5_or_more_folds": bool(
            best.folds_low_predicted >= 3 and best.folds_moderate_predicted >= 3 and best.folds_high_predicted >= 3
        ),
        "macro_f1_higher_in_at_least_3_outer_folds": bool(macro_wins >= 3),
    }
    selected = bool(all(gate.values()))
    return best, baseline, gate, macro_wins, selected


def train_candidate_artifact(X, y, encoder, selected_row, selected_fold_rows, gate, metadata):
    model_name = selected_row["Model"]
    family_name = str(selected_fold_rows.iloc[0]["family"])
    family = BASE_FAMILIES[family_name]
    final_inner_trace = []
    default_config, threshold_config, _ = select_inner_configurations(
        family_name, family, X, y, 0, final_inner_trace,
    )
    use_thresholds = str(model_name).startswith("Threshold-tuned ")
    final_config = threshold_config if use_thresholds else default_config
    params = final_config["parameters"]
    thresholds = final_config["threshold_values"]
    ratio = family["sampler"]
    if ratio == "tuned":
        ratio = params["sampling_ratio"]
    fitted = fit_model(X, y, params, sampler_ratio=ratio, seed=SEED)
    fitted.thresholds = thresholds
    CANDIDATE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    joblib.dump(fitted, CANDIDATE_ARTIFACT_DIR / "final_model.joblib")
    background = X.sample(n=min(50, len(X)), random_state=SEED)
    joblib.dump(background, CANDIDATE_ARTIFACT_DIR / "shap_background.joblib")
    class_distribution = {name: int((y == index).sum()) for index, name in enumerate(encoder.classes_)}
    artifact_metadata = {
        "model": model_name,
        "model_type": "RBF SVC with probability=True",
        "training_rows": len(X),
        "feature_names": X.columns.tolist(),
        "classes": encoder.classes_.tolist(),
        "class_distribution": class_distribution,
        "selected_parameters": params,
        "thresholds_probability_cutoffs_class_order_high_low_moderate": thresholds,
        "decision_method": "argmax(p_class - locked inner-CV probability cutoff); predict_proba returns unmodified SVC probabilities",
        "smotenc_strategy": ratio,
        "nested_cv_macro_f1_mean": float(selected_row.macro_f1_mean),
        "nested_cv_balanced_accuracy_mean": float(selected_row.balanced_accuracy_mean),
        "selection_gate": gate,
        "final_training_parameters_selected_by": "Stratified 5-fold inner CV on the complete dataset after nested-CV evaluation; selection uses Macro F1, then Balanced Accuracy.",
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
    }
    (CANDIDATE_ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(artifact_metadata, indent=2), encoding="utf-8")
    generate_candidate_shap(fitted, X, model_name)
    return artifact_metadata


def generate_candidate_shap(fitted, X, model_name):
    """Kernel SHAP on the model's probability outputs using transformed predictors."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    background_raw = X.sample(n=min(25, len(X)), random_state=SEED)
    explain_raw = X.sample(n=min(40, len(X)), random_state=SEED + 1)
    background = fitted.transform(background_raw)
    explain = fitted.transform(explain_raw)
    if hasattr(background, "toarray"):
        background = background.toarray()
    if hasattr(explain, "toarray"):
        explain = explain.toarray()
    prediction_function = lambda rows: fitted.classifier.predict_proba(rows)
    explainer = shap.KernelExplainer(prediction_function, background)
    values = explainer.shap_values(explain, nsamples=100)
    if isinstance(values, list):
        values = np.stack([np.asarray(item) for item in values], axis=0)
    else:
        values = np.asarray(values)
        if values.ndim == 3 and values.shape == (len(explain), explain.shape[1], 3):
            values = np.moveaxis(values, -1, 0)
    if values.ndim != 3:
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}")
    names = fitted.feature_names
    importance = pd.DataFrame({"feature": names, "mean_absolute_shap_value": np.mean(np.abs(values), axis=(0, 1))}).sort_values("mean_absolute_shap_value", ascending=False)
    importance.to_csv(SHAP_DIR / "global_feature_importance.csv", index=False)
    shap.summary_plot(values.reshape(-1, explain.shape[1]), np.tile(explain, (values.shape[0], 1)), feature_names=names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    pd.DataFrame({"feature": names, "shap_value": values[0, 0], "class": CLASS_NAMES[0]}).assign(abs_shap=lambda frame: frame.shap_value.abs()).sort_values("abs_shap", ascending=False).to_csv(SHAP_DIR / "local_high_stress_probability_explanation.csv", index=False)
    (SHAP_DIR / "metadata.json").write_text(json.dumps({"explainer": "KernelExplainer", "explained_output": "SVC class probabilities; thresholds affect class decisions only", "background_rows": len(background), "explained_rows": len(explain), "model": model_name, "causal_claim": False}, indent=2), encoding="utf-8")


def write_summary(comparison, best, baseline, gate, macro_wins, selected, metadata):
    lines = [
        "# Three-class improvement experiment", "",
        "This experiment is separate from and does not overwrite baseline, prior imbalance, nested-CV, SHAP, dataset, or application results.", "",
        "## Protocol", "",
        "Nested stratified 5-fold outer and 5-fold inner CV, shuffled with random_state=42. All imputation, scaling, ordinal coding, one-hot encoding, SMOTENC, class-weight selection, SVM fitting, hyperparameter selection, and decision-threshold selection are confined to training/inner folds. Outer folds are not used for tuning.", "",
        "SMOTENC operates on numeric-scaled plus ordinal-coded mixed training features, with categorical positions explicitly supplied; one-hot encoding occurs after resampling. SVM probabilities remain unchanged. Threshold vectors are selected from inner-validation probabilities and affect only the predicted label via argmax(p - threshold).", "",
        "## Outer nested-CV results", "",
        "Mean +/- SD across five outer folds (descriptive; folds are overlapping partitions, not independent populations):", "",
        "| Model | Macro F1 | Balanced Accuracy | Low Recall | Moderate Recall | High Recall | Low F1 | Moderate F1 | High F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        model = getattr(row, "Model")
        def fmt(metric):
            return f"{getattr(row, metric + '_mean'):.4f} +/- {getattr(row, metric + '_std'):.4f}"
        lines.append(f"| {model} | {fmt('macro_f1')} | {fmt('balanced_accuracy')} | {fmt('low_recall')} | {fmt('moderate_recall')} | {fmt('high_recall')} | {fmt('low_f1')} | {fmt('moderate_f1')} | {fmt('high_f1')} |")
    lines.extend([
        "", "## Selection and interpretation", "",
        f"The highest-Macro-F1 non-reference candidate was **{best['Model']}** (macro F1 {best.macro_f1_mean:.4f}; balanced accuracy {best.balanced_accuracy_mean:.4f}). Current SVM was {baseline.macro_f1_mean:.4f} Macro F1 and {baseline.balanced_accuracy_mean:.4f} balanced accuracy.",
        f"Candidate Macro F1 exceeded the reference in {macro_wins}/5 paired outer folds. Predeclared candidate gate: `{json.dumps(gate, sort_keys=True)}`.",
        f"**Candidate selected: {'Yes' if selected else 'No'}. Application model changed: No.** {'A separate candidate artifact and SHAP results were generated; the current application artifact remains unchanged.' if selected else 'The candidate did not satisfy the full improvement gate; no candidate artifact was created.'}",
        "", "No p-values or confidence intervals are calculated from five outer folds. This is internal validation on one research dataset and does not establish external generalization or causality.", "",
        "SMOTENC sampling ratios and all class weights were predefined in experiment metadata. Threshold selection optimizes inner-fold Macro F1, with Balanced Accuracy as tie-breaker; outer labels were used only for evaluation.",
    ])
    (OUTPUT / "research_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment outputs: {OUTPUT}")
    if CANDIDATE_ARTIFACT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate artifact directory: {CANDIDATE_ARTIFACT_DIR}")
    X, y, encoder = load_predictors()
    counts = {name: int((y == index).sum()) for index, name in enumerate(encoder.classes_)}
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    fold_rows, prediction_rows, threshold_rows, selected_rows = [], [], [], []

    for family_name, family in BASE_FAMILIES.items():
        print(f"Tuning {family_name}", flush=True)
        for outer_fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            best_default, best_threshold, candidates = select_inner_configurations(
                family_name, family, X_train, y_train, outer_fold, threshold_rows,
            )
            for variant, selected_config in (("argmax", best_default), ("threshold_tuned", best_threshold)):
                # The current SVM is the fixed reference, with no threshold-tuned duplicate.
                if family_name == "Current SVM" and variant == "threshold_tuned":
                    continue
                row, predicted, probabilities, _ = metrics_for_outer_fold(
                    family_name, variant, family, selected_config,
                    X_train, y_train, X_test, y_test, outer_fold,
                )
                fold_rows.append(row)
                selected_rows.append({
                    "model": row["model"], "family": family_name, "outer_fold": outer_fold,
                    "selected_parameters": row["selected_parameters"],
                    "selected_threshold": row["selected_threshold"],
                    "threshold_values": row["threshold_values"],
                    "inner_macro_f1_mean": selected_config["inner_macro_f1_mean"],
                    "inner_balanced_accuracy_mean": selected_config["inner_balanced_accuracy_mean"],
                })
                for position, sample_index in enumerate(test_idx):
                    prediction_rows.append({
                        "model": row["model"], "outer_fold": outer_fold, "sample_index": int(sample_index),
                        "actual": encoder.classes_[int(y.iloc[sample_index])],
                        "predicted": encoder.classes_[int(predicted[position])],
                        **{f"probability_{encoder.classes_[class_id]}": float(probabilities[position, class_id]) for class_id in range(3)},
                    })
                print(f"  fold {outer_fold} {row['model']}: Macro F1={row['macro_f1']:.4f}; BA={row['balanced_accuracy']:.4f}; Low/High recall={row['low_recall']:.3f}/{row['high_recall']:.3f}", flush=True)

    metadata = {
        "experiment": "Three-class improvement: SMOTENC, class-weight, and threshold comparison",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "random_state": SEED,
        "number_of_samples": len(X),
        "class_distribution": counts,
        "outer_cv": {"type": "StratifiedKFold", "n_splits": OUTER_FOLDS, "shuffle": True, "random_state": SEED},
        "inner_cv": {"type": "StratifiedKFold", "n_splits": INNER_FOLDS, "shuffle": True, "random_state": SEED},
        "feature_list": X.columns.tolist(),
        "excluded_predictors": PSS_FIELDS + [TARGET],
        "numeric_features": numeric_and_categorical(X)[0],
        "categorical_features": numeric_and_categorical(X)[1],
        "class_weight_candidates": WEIGHT_CANDIDATES,
        "smotenc_ratios": [0.25, 0.50],
        "smotenc": "Training-fold only. Impute/scale numeric columns and impute/ordinal-code categorical columns on training data; SMOTENC receives explicit categorical column indices; one-hot encode after SMOTENC. k_neighbors=5.",
        "smotenc_sampling_rule": "For each training partition and ratio r, compute target_count=ceil(training_partition_majority_count*r); request SMOTENC synthesis only for classes whose current training count is below target_count. The ratio is selected from 0.25 or 0.50 using inner validation only; outer-validation rows are never used to compute targets.",
        "threshold_method": "For each class, subtract a predefined one-vs-rest probability cutoff and predict argmax(adjusted scores). Probabilities are not modified. Cutoffs selected by inner-fold mean Macro F1, Balanced Accuracy tie-breaker.",
        "threshold_grid": THRESHOLDS,
        "model_grids": {name: family["grid"] for name, family in BASE_FAMILIES.items()},
        "primary_metric": "Macro F1",
        "secondary_metric": "Balanced Accuracy",
        "candidate_gate": {
            "macro_f1_higher_than_current": True,
            "balanced_accuracy_at_least_current": True,
            "mean_low_high_recall_higher_than_current": True,
            "moderate_recall_drop_maximum": 0.15,
            "all_classes_predicted_in_at_least_three_outer_folds": True,
            "macro_f1_higher_in_at_least_three_paired_outer_folds": True,
        },
        "uncertainty_note": "Five outer folds are overlapping evaluation partitions, not independent populations; no inferential test or confidence interval.",
        "application_model_changed": False,
        "candidate_artifact_created": False,
        "shap_status": "Deferred unless a candidate passes the predeclared selection gate.",
        "packages": {
            "imbalanced-learn": __import__("imblearn").__version__,
            "scikit-learn": __import__("sklearn").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "shap": shap.__version__,
        },
    }
    folds, predictions, comparison, distributions = write_tables(
        fold_rows, prediction_rows, threshold_rows, selected_rows, metadata,
    )
    best, baseline, gate, macro_wins, selected = select_final_candidate(comparison, folds)
    metadata["best_non_reference_model"] = str(best["Model"])
    metadata["candidate_gate_results"] = {key: bool(value) for key, value in gate.items()}
    metadata["candidate_selected"] = bool(selected)
    if selected:
        chosen_rows = pd.DataFrame(selected_rows)
        chosen_rows = chosen_rows[chosen_rows.model == best["Model"]]
        artifact_metadata = train_candidate_artifact(X, y, encoder, best, chosen_rows, gate, metadata)
        metadata["candidate_artifact_created"] = True
        metadata["candidate_artifact_metadata"] = artifact_metadata
        metadata["shap_status"] = "Kernel SHAP generated for the separate selected candidate."
    else:
        metadata["shap_status"] = "Not generated because no candidate passed the predeclared selection gate."
    metadata["application_model_changed"] = False
    (OUTPUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(comparison, best, baseline, gate, macro_wins, selected, metadata)
    print("\nOuter-fold comparison (means):", flush=True)
    print(comparison[["Model", "macro_f1_mean", "balanced_accuracy_mean", "low_recall_mean", "moderate_recall_mean", "high_recall_mean"]].to_string(index=False), flush=True)
    print(f"Candidate selected: {selected}; application model changed: False", flush=True)


if __name__ == "__main__":
    main()
