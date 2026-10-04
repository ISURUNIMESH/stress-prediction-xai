"""Leakage-safe nested CV comparison isolated under results/imbalance_improved/."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("Run this experiment with the project CPython 3.12 environment.")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "dataset" / "processed" / "cleaned_stress_dataset.csv"
OUTPUT = ROOT / "results" / "imbalance_improved"
REPORTS = OUTPUT / "classification_reports"
MATRICES = OUTPUT / "confusion_matrices"
TARGET = "Stress_Level"
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
CLASS_NAMES = ["High Stress", "Low Stress", "Moderate Stress"]
PSS_FIELDS = [*[f"PSS_Q{i}" for i in range(1, 11)], "PSS10_Total"]
EXCLUDED = ["Timestamp", "Consent", TARGET, *PSS_FIELDS]

WEIGHTED_SVM_GRID = [
    {"C": [1.0], "gamma": ["scale"], "class_weight": ["balanced"]},
    {"C": [2.0], "gamma": ["scale"], "class_weight": ["balanced"]},
    {"C": [2.0], "gamma": [0.1], "class_weight": ["balanced"]},
    {"C": [2.0], "gamma": ["scale"], "class_weight": [{0: 4.0, 1: 4.0, 2: 1.0}]},
    {"C": [2.0], "gamma": ["scale"], "class_weight": [{0: 6.0, 1: 6.0, 2: 1.0}]},
]
SVM_FIXED = {"C": [2.0], "gamma": ["scale"], "class_weight": ["balanced"]}
MODEL_SPECS = {
    "Current SVM": {"kind": "svm", "grid": [SVM_FIXED], "resample": None},
    "RandomOverSampler + SVM": {"kind": "svm", "grid": [SVM_FIXED], "resample": "random_over_sampler"},
    "Tuned Class-Weighted SVM": {"kind": "svm", "grid": WEIGHTED_SVM_GRID, "resample": None},
    "Logistic Regression Balanced": {"kind": "logistic", "grid": [{"C": [1.0]}], "resample": None},
    "Random Forest Balanced": {"kind": "forest", "grid": [{}], "resample": None},
    "XGBoost Training Weights": {"kind": "xgboost", "grid": [{}], "resample": None},
}
SMOTE_CANDIDATES = ["SMOTE + SVM", "SMOTE + Tuned SVM"]


def load_predictors(data: pd.DataFrame | None = None):
    """Return the unchanged research predictors and encoded target, rejecting leakage."""
    frame = pd.read_csv(DATA_FILE) if data is None else data.copy()
    if TARGET not in frame:
        raise ValueError(f"Processed dataset is missing {TARGET}")
    X = frame.drop(columns=[column for column in EXCLUDED if column in frame.columns]).copy()
    forbidden = set(PSS_FIELDS + [TARGET])
    if forbidden.intersection(X.columns):
        raise ValueError("PSS/target leakage detected in predictors")
    if not len(X.columns):
        raise ValueError("No predictors remain")
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(frame[TARGET].astype(str)), index=frame.index, name=TARGET)
    if encoder.classes_.tolist() != CLASS_NAMES:
        raise ValueError(f"Unexpected class labels: {encoder.classes_.tolist()}")
    return X, y, encoder


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers=transformers)


def build_classifier(kind: str, params: dict, seed: int = SEED):
    if kind == "svm":
        return SVC(kernel="rbf", probability=True, random_state=seed, **params)
    if kind == "logistic":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, **params)
    if kind == "forest":
        return RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=1,
        )
    if kind == "xgboost":
        return XGBClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
            random_state=seed, n_jobs=1,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def random_oversample_training(X_train, y_train, seed=SEED):
    """Duplicate only transformed training rows to the majority-class count."""
    y_array = np.asarray(y_train)
    counts = np.bincount(y_array)
    if np.any(counts == 0):
        raise ValueError("Every training fold must contain each class")
    rng = np.random.default_rng(seed)
    extra = []
    for class_id, count in enumerate(counts):
        if count < counts.max():
            members = np.flatnonzero(y_array == class_id)
            extra.extend(rng.choice(members, size=int(counts.max() - count), replace=True).tolist())
    if not extra:
        return X_train, y_array
    indices = np.concatenate([np.arange(len(y_array)), np.asarray(extra, dtype=int)])
    rng.shuffle(indices)
    return X_train[indices], y_array[indices]


def fit_candidate(name, X_train, y_train, params, seed=SEED):
    """Fit preprocessing and model on this fold's train partition only."""
    spec = MODEL_SPECS[name]
    kind = spec["kind"]
    preprocessor = build_preprocessor(X_train, scale_numeric=kind in {"svm", "logistic"})
    transformed = preprocessor.fit_transform(X_train)
    labels = np.asarray(y_train)
    if spec["resample"] == "random_over_sampler":
        transformed, labels = random_oversample_training(transformed, labels, seed)
    classifier = build_classifier(kind, params, seed)
    if kind == "xgboost":
        weights = compute_sample_weight(class_weight="balanced", y=labels)
        classifier.fit(transformed, labels, sample_weight=weights)
    else:
        classifier.fit(transformed, labels)
    return {"preprocessor": preprocessor, "classifier": classifier}


def predict_candidate(fitted, X):
    transformed = fitted["preprocessor"].transform(X)
    classifier = fitted["classifier"]
    predicted = classifier.predict(transformed)
    raw = classifier.predict_proba(transformed)
    probabilities = np.zeros((len(X), len(CLASS_NAMES)), dtype=float)
    for source_index, class_id in enumerate(classifier.classes_):
        probabilities[:, int(class_id)] = raw[:, source_index]
    return predicted, probabilities


def calculate_metrics(y_true, predicted, probabilities):
    y_true = np.asarray(y_true)
    predicted = np.asarray(predicted)
    recall = recall_score(y_true, predicted, labels=np.arange(3), average=None, zero_division=0)
    return {
        "accuracy": float(np.mean(y_true == predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "low_recall": float(recall[1]),
        "moderate_recall": float(recall[2]),
        "high_recall": float(recall[0]),
        "roc_auc": float(roc_auc_score(y_true, probabilities, labels=np.arange(3), multi_class="ovr", average="macro")),
    }


def select_inner_parameters(model_name, X_train, y_train):
    """Select mean inner Macro F1, Balanced Accuracy as tie-breaker."""
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    candidates = list(ParameterGrid(MODEL_SPECS[model_name]["grid"]))
    scored = []
    for candidate_index, params in enumerate(candidates):
        fold_metrics = []
        for fold, (train_index, validation_index) in enumerate(inner.split(X_train, y_train), start=1):
            fit = fit_candidate(
                model_name, X_train.iloc[train_index], y_train.iloc[train_index], params,
                seed=SEED + fold,
            )
            prediction, _ = predict_candidate(fit, X_train.iloc[validation_index])
            truth = y_train.iloc[validation_index]
            fold_metrics.append({
                "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
            })
        scored.append({
            "candidate_index": candidate_index,
            "parameters": params,
            "inner_macro_f1_mean": float(np.mean([item["macro_f1"] for item in fold_metrics])),
            "inner_balanced_accuracy_mean": float(np.mean([item["balanced_accuracy"] for item in fold_metrics])),
        })
    scored.sort(key=lambda item: (item["inner_macro_f1_mean"], item["inner_balanced_accuracy_mean"]), reverse=True)
    return scored[0], scored


def metric_summary(folds):
    rows = []
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1", "low_recall", "moderate_recall", "high_recall", "roc_auc"]
    valid = folds[folds["macro_f1"].notna()]
    for model, group in valid.groupby("model", sort=False):
        for metric in metrics:
            rows.append({
                "model": model, "metric": metric,
                "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=1)),
                "outer_folds": int(len(group)),
                "interval": "Not calculated; five CV folds are not independent participants.",
            })
    return pd.DataFrame(rows)


def write_outputs(fold_rows, parameter_rows, prediction_rows, confusion_rows, report_rows, metadata):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MATRICES.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUTPUT / "nested_cv_results.csv", index=False)
    metric_summary(folds).to_csv(OUTPUT / "nested_cv_summary.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(OUTPUT / "inner_cv_selected_parameters.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT / "outer_fold_predictions.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(OUTPUT / "confusion_matrices.csv", index=False)
    pd.DataFrame(report_rows).to_csv(OUTPUT / "classification_reports.csv", index=False)
    for model, group in pd.DataFrame(report_rows).groupby("model", sort=False):
        slug = model.lower().replace(" ", "_").replace("+", "plus")
        group.drop(columns="model").to_csv(REPORTS / f"{slug}.csv", index=False)
    for model, group in pd.DataFrame(confusion_rows).groupby("model", sort=False):
        slug = model.lower().replace(" ", "_").replace("+", "plus")
        matrix = group.pivot(index="actual", columns="predicted", values="count").reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)
        matrix.to_csv(MATRICES / f"{slug}.csv")
    (OUTPUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return folds


def main():
    X, y, encoder = load_predictors()
    counts = {name: int((y == index).sum()) for index, name in enumerate(encoder.classes_)}
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    fold_rows, parameter_rows, prediction_rows, confusion_rows, report_rows = [], [], [], [], []

    for model_name, spec in MODEL_SPECS.items():
        print(f"Running {model_name}", flush=True)
        pooled_true, pooled_pred = [], []
        for outer_fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
            X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
            y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]
            selected, all_inner = select_inner_parameters(model_name, X_outer_train, y_outer_train)
            final_fit = fit_candidate(
                model_name, X_outer_train, y_outer_train, selected["parameters"], seed=SEED + outer_fold,
            )
            prediction, probabilities = predict_candidate(final_fit, X_outer_test)
            scores = calculate_metrics(y_outer_test, prediction, probabilities)
            fold_rows.append({"model": model_name, "outer_fold": outer_fold, **scores})
            parameter_rows.append({
                "model": model_name, "outer_fold": outer_fold,
                "selected_parameters": json.dumps(selected["parameters"], sort_keys=True),
                "inner_macro_f1_mean": selected["inner_macro_f1_mean"],
                "inner_balanced_accuracy_mean": selected["inner_balanced_accuracy_mean"],
                "candidate_scores": json.dumps(all_inner, sort_keys=True),
            })
            for position, sample_index in enumerate(test_idx):
                prediction_rows.append({
                    "model": model_name, "outer_fold": outer_fold, "sample_index": int(sample_index),
                    "actual": encoder.classes_[int(y.iloc[sample_index])],
                    "predicted": encoder.classes_[int(prediction[position])],
                    **{f"probability_{encoder.classes_[class_id]}": float(probabilities[position, class_id]) for class_id in range(3)},
                })
            pooled_true.extend(y_outer_test.tolist())
            pooled_pred.extend(prediction.tolist())
            print(f"  outer fold {outer_fold}: Macro F1={scores['macro_f1']:.4f}; Balanced Accuracy={scores['balanced_accuracy']:.4f}", flush=True)

        matrix = confusion_matrix(pooled_true, pooled_pred, labels=np.arange(3))
        for actual_id, actual_name in enumerate(CLASS_NAMES):
            for predicted_id, predicted_name in enumerate(CLASS_NAMES):
                confusion_rows.append({"model": model_name, "actual": actual_name, "predicted": predicted_name, "count": int(matrix[actual_id, predicted_id])})
        report = classification_report(pooled_true, pooled_pred, labels=np.arange(3), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        for label, values in report.items():
            report_rows.append({"model": model_name, "class_or_average": label, **(values if isinstance(values, dict) else {"value": values})})

    folds = pd.DataFrame(fold_rows)
    # SMOTE candidates are explicitly recorded as skipped if imbalanced-learn is unavailable.
    skipped = []
    try:
        import imblearn  # noqa: F401
    except ImportError:
        skipped = SMOTE_CANDIDATES
        for name in skipped:
            for fold in range(1, OUTER_FOLDS + 1):
                fold_rows.append({"model": name, "outer_fold": fold, "status": "skipped: imbalanced-learn/SMOTENC unavailable"})

    metadata = {
        "experiment": "Leakage-safe improved imbalance comparison",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED, "number_of_samples": int(len(X)),
        "class_distribution": counts,
        "outer_cv": {"type": "StratifiedKFold", "n_splits": OUTER_FOLDS, "shuffle": True, "random_state": SEED},
        "inner_cv": {"type": "StratifiedKFold", "n_splits": INNER_FOLDS, "shuffle": True, "random_state": SEED},
        "feature_list": X.columns.tolist(), "excluded_pss_fields": PSS_FIELDS,
        "excluded_target": TARGET,
        "preprocessing": "Numeric median imputation and StandardScaler for SVM/LR; categorical most-frequent imputation and OneHotEncoder(handle_unknown='ignore'); fit separately within each inner/outer training fold.",
        "imbalance_methods": {
            "current_svm": "SVC RBF with application metadata parameters C=2, gamma=scale, class_weight=balanced",
            "random_oversampler_svm": "Training-only random oversampling on the transformed inner/outer training partition; validation/test rows are never resampled.",
            "tuned_weighted_svm": WEIGHTED_SVM_GRID,
            "logistic_regression": "class_weight=balanced",
            "random_forest": "class_weight=balanced",
            "xgboost": "compute_sample_weight('balanced') from each current training fold only; no binary scale_pos_weight",
            "smotenc": "Skipped because imbalanced-learn is not installed in the project Python 3.12 environment; no ordinary SMOTE on one-hot features.",
        },
        "skipped_candidates": skipped,
        "selection_metric": "Mean inner-CV Macro F1; mean inner-CV Balanced Accuracy as tie-breaker.",
        "deployment_gate": "A candidate is eligible to replace the current SVM only if outer mean Macro F1 and Balanced Accuracy both improve and Low/High Stress recall are not lower; any final selection must be documented and separately exported.",
        "confidence_intervals": "Not calculated from five folds; fold SD is descriptive because CV training partitions overlap.",
        "no_research_data_or_existing_artifacts_modified": True,
    }
    folds = write_outputs(fold_rows, parameter_rows, prediction_rows, confusion_rows, report_rows, metadata)
    comparison = make_comparison(folds)
    comparison.to_csv(OUTPUT / "model_comparison.csv", index=False)
    deployed = write_research_summary(folds, comparison, metadata)
    metadata["application_model_changed"] = deployed
    (OUTPUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)


def make_comparison(folds):
    metric_columns = ["macro_f1", "balanced_accuracy", "low_recall", "high_recall", "moderate_recall", "weighted_f1"]
    valid = folds[folds["macro_f1"].notna()]
    rows = []
    for model, group in valid.groupby("model", sort=False):
        row = {"Model": model}
        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        row["outer_folds"] = len(group)
        rows.append(row)
    for model in SMOTE_CANDIDATES:
        if model not in valid.model.unique():
            rows.append({"Model": model, "status": "Skipped: imbalanced-learn/SMOTENC unavailable"})
    return pd.DataFrame(rows)


def write_research_summary(folds, comparison, metadata):
    valid = comparison[comparison["macro_f1_mean"].notna()].copy()
    valid = valid.sort_values(["macro_f1_mean", "balanced_accuracy_mean"], ascending=False, kind="mergesort")
    baseline = valid[valid.Model == "Current SVM"].iloc[0]
    best = valid.iloc[0]
    alternatives = valid[valid.Model != "Current SVM"]
    best_alternative = alternatives.iloc[0]
    deployed = (
        best_alternative.macro_f1_mean > baseline.macro_f1_mean
        and best_alternative.balanced_accuracy_mean > baseline.balanced_accuracy_mean
        and best_alternative.low_recall_mean >= baseline.low_recall_mean
        and best_alternative.high_recall_mean >= baseline.high_recall_mean
    )
    fold_comp = folds[folds["macro_f1"].notna()].pivot(index="outer_fold", columns="model", values="macro_f1")
    if best_alternative.Model in fold_comp:
        diffs = fold_comp[best_alternative.Model] - fold_comp["Current SVM"]
        fold_consistency = f"Macro F1 was higher in {int((diffs > 0).sum())} of five corresponding outer folds; this is descriptive, not an independent-fold significance test."
        paired_lines = "\n".join(f"| {fold} | {fold_comp.loc[fold, 'Current SVM']:.4f} | {fold_comp.loc[fold, best_alternative.Model]:.4f} | {diffs.loc[fold]:+.4f} |" for fold in fold_comp.index)
    else:
        fold_consistency = "Fold-level comparison unavailable."
        paired_lines = ""
    lines = [
        "# Improved imbalance experiment summary", "",
        "This separate experiment leaves the baseline, imbalance-aware, nested-CV, SHAP, dataset, and application artifacts unchanged.", "",
        "## Method", "",
        f"Outer and inner validation use stratified 5-fold CV (seed {SEED}, shuffled). Selection uses inner mean Macro F1, then Balanced Accuracy as tie-breaker. Preprocessing and all model fits occur within each training fold. Validation folds are never resampled. {metadata['imbalance_methods']['smotenc']}", "",
        "## Nested CV results", "",
        "Outer-fold mean +/- SD (five folds; descriptive only):", "",
        "| Model | Macro F1 | Balanced Accuracy | Low recall | Moderate recall | High recall | Weighted F1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in valid.itertuples(index=False):
        def fmt(metric):
            return f"{getattr(row, metric + '_mean'):.4f} +/- {getattr(row, metric + '_std'):.4f}"
        lines.append(f"| {row.Model} | {fmt('macro_f1')} | {fmt('balanced_accuracy')} | {fmt('low_recall')} | {fmt('moderate_recall')} | {fmt('high_recall')} | {fmt('weighted_f1')} |")
    lines.extend([
        "", "## Comparison against current SVM", "",
        f"By mean outer Macro F1, the highest measured configuration overall is **{best.Model}**. The highest-Macro-F1 alternative is **{best_alternative.Model}**. Relative to Current SVM, that alternative changes Macro F1 by {best_alternative.macro_f1_mean - baseline.macro_f1_mean:+.4f}, Balanced Accuracy by {best_alternative.balanced_accuracy_mean - baseline.balanced_accuracy_mean:+.4f}, Low Stress recall by {best_alternative.low_recall_mean - baseline.low_recall_mean:+.4f}, and High Stress recall by {best_alternative.high_recall_mean - baseline.high_recall_mean:+.4f}.",
        fold_consistency, "",
        "| Outer fold | Current SVM Macro F1 | Candidate Macro F1 | Difference |", "|---:|---:|---:|---:|", paired_lines,
        "", "Five folds are not treated as independent participants. No p-value or confidence interval is calculated.", "",
        f"**Application model changed: {'Yes' if deployed else 'No'}.** The separate nested comparison {'meets' if deployed else 'does not meet'} the conservative deployment gate (Macro F1 and Balanced Accuracy both higher, with Low and High Stress recall no worse). No model artifact has been replaced or exported in this experiment run.",
        "", "## Limitations", "",
        "This is internal validation on the same research sample, not external validation. Five outer folds give limited evidence about variability and no independent-fold uncertainty interval. The model-family winner is selected on these outer means and should be treated as exploratory. SMOTENC candidates were skipped because imbalanced-learn was unavailable; no invalid one-hot SMOTE was used.",
    ])
    (OUTPUT / "research_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return deployed


if __name__ == "__main__":
    main()
