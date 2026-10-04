"""Run imbalance-aware analyses without modifying baseline artifacts.

All outputs are isolated under results/imbalance_aware/. Class weights for
XGBoost are recomputed from the current training partition only, including
inside each CV fold. Preprocessing is fitted within every training partition.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("Run this experiment with the project's CPython 3.12 environment.")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import friedmanchisquare, t, wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from models.train_evaluate import (
    DATA_FILE,
    EXCLUDED,
    RANDOM_STATE,
    TEST_SIZE,
    build_models,
    make_pipeline,
)
from models.xgboost_model import build_model as build_xgboost


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results"
BASELINE_METRICS = BASELINE / "metrics"
OUTPUT = BASELINE / "imbalance_aware"
METRICS_DIR = OUTPUT / "metrics"
FIGURES_DIR = OUTPUT / "figures"
CV_DIR = OUTPUT / "cv"
ERRORS_DIR = OUTPUT / "errors"
ABLATION_DIR = OUTPUT / "ablation"
SUBGROUP_DIR = OUTPUT / "subgroup"
ROBUSTNESS_DIR = OUTPUT / "robustness"
STATISTICS_DIR = OUTPUT / "statistics"
CALIBRATION_DIR = OUTPUT / "calibration"
SHAP_DIR = OUTPUT / "shap"

TARGET = "Stress_Level"
CLASSES_IN_REPORT = ["Low Stress", "Moderate Stress", "High Stress"]
SEEDS = [42, 123, 2024, 2025, 2026]
N_SPLITS = 10
MIN_SUBGROUP_SIZE = 30
MIN_CLASS_SUPPORT_FOR_SUBGROUP = 5
BOOTSTRAP_REPLICATES = 2000
SHAP_BACKGROUND_SIZE = 30
SHAP_FINAL_SAMPLE_SIZE = 30
SHAP_STABILITY_ROWS_PER_FOLD = 8
SHAP_NSAMPLES = 100

FEATURE_GROUPS = {
    "Demographic": ["Age", "Gender", "University"],
    "Academic": ["Degree_Program", "Academic_Year", "Study_Hours", "Academic_Pressure"],
    "Lifestyle": ["Sleep_Hours", "Exercise_Frequency", "Financial_Stress", "Family_Support"],
    "Social_Behavioural": ["Screen_Time_Hours", "Social_Media_Hours"],
}
ABLATIONS = {
    "A_Demographic": ["Demographic"],
    "B_Demographic_Academic": ["Demographic", "Academic"],
    "C_Demographic_Lifestyle": ["Demographic", "Lifestyle"],
    "D_Demographic_Social_Behavioural": ["Demographic", "Social_Behavioural"],
    "E_Academic_Lifestyle": ["Academic", "Lifestyle"],
    "F_Academic_Social_Behavioural": ["Academic", "Social_Behavioural"],
    "G_All_Predictors": list(FEATURE_GROUPS),
}


def ensure_output_directories() -> None:
    for directory in (
        METRICS_DIR,
        FIGURES_DIR,
        CV_DIR,
        ERRORS_DIR,
        ABLATION_DIR,
        SUBGROUP_DIR,
        ROBUSTNESS_DIR,
        STATISTICS_DIR,
        CALIBRATION_DIR,
        SHAP_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load processed data, validate observed feature-group mappings, encode y."""
    if not DATA_FILE.exists():
        raise FileNotFoundError("Run preprocessing/preprocess.py before this experiment")
    data = pd.read_csv(DATA_FILE)
    if TARGET not in data:
        raise ValueError(f"Processed dataset is missing target {TARGET}")
    missing_group_columns = sorted(
        {column for values in FEATURE_GROUPS.values() for column in values} - set(data.columns)
    )
    if missing_group_columns:
        raise ValueError(f"Feature groups contain columns absent from the actual dataset: {missing_group_columns}")
    excluded = [column for column in EXCLUDED if column in data.columns]
    X = data.drop(columns=excluded)
    if set(X.columns) != {column for values in FEATURE_GROUPS.values() for column in values}:
        raise ValueError("Feature groups do not cover exactly the non-PSS predictors in this dataset")
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(data[TARGET].astype(str)), index=data.index, name=TARGET)
    label_ids = np.arange(len(encoder.classes_))
    if len(encoder.classes_) != 3:
        raise ValueError(f"Expected 3 stress classes, found {encoder.classes_.tolist()}")
    return data, X, y, encoder, label_ids


def feature_columns(X: pd.DataFrame):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return numeric, categorical


def fit_estimator(name, X_train, y_train, numeric, categorical, sample_weighting=True):
    """Fit a class-weighted pipeline; XGBoost receives training-only sample weights."""
    model = build_models()[name]
    if "n_jobs" in model.get_params(deep=False):
        # Single-process fitting avoids restricted Windows worker-pool creation.
        model.set_params(n_jobs=1)
    pipeline = make_pipeline(
        model,
        numeric,
        categorical,
        scale_numeric=name in {"Logistic Regression", "SVM"},
    )
    weights = None
    if name == "XGBoost" and sample_weighting:
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=weights)
    else:
        pipeline.fit(X_train, y_train)
    return pipeline, weights


def aligned_probabilities(estimator, X, classes):
    raw = estimator.predict_proba(X)
    probabilities = np.zeros((len(X), len(classes)), dtype=float)
    for source_index, label in enumerate(estimator.classes_):
        probabilities[:, int(label)] = raw[:, source_index]
    return probabilities


def calculate_metrics(y_true, prediction, probabilities, labels):
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision_macro": precision_score(y_true, prediction, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, prediction, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, prediction, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, prediction, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, prediction, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, prediction, average="weighted", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(
            y_true, probabilities, labels=labels, multi_class="ovr", average="macro"
        ),
        "roc_auc_ovr_weighted": roc_auc_score(
            y_true, probabilities, labels=labels, multi_class="ovr", average="weighted"
        ),
    }


def class_metric_rows(y_true, prediction, encoder, model, split_name):
    report = classification_report(
        y_true,
        prediction,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )
    return [
        {
            "model": model,
            "split": split_name,
            "class": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1": report[class_name]["f1-score"],
            "support": int(report[class_name]["support"]),
        }
        for class_name in encoder.classes_
    ]


def write_classification_report(y_true, prediction, encoder, model):
    report = classification_report(
        y_true,
        prediction,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().rename_axis("class_or_average").reset_index().assign(model=model)


def run_main_models(X, y, encoder, labels):
    """Run same held-out split and ten matched folds for all weighted models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    numeric, categorical = feature_columns(X)
    test_rows, report_rows, confusion_rows, per_class = [], [], [], []
    test_predictions, oof_probability = {}, {}
    all_cv_fold_rows, cv_class_rows, weight_audit, baseline_xgb_fold_rows = [], [], [], []
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for name in build_models():
        print(f"Imbalance-aware held-out fit: {name}", flush=True)
        fitted, train_weights = fit_estimator(name, X_train, y_train, numeric, categorical)
        if name == "XGBoost":
            heldout_audit = {
                f"train_n_{encoder.classes_[cls]}": int((y_train == cls).sum()) for cls in labels
            }
            heldout_audit.update(
                {
                    "model": name,
                    "fold": "held_out_train",
                    "weight_min": float(np.min(train_weights)),
                    "weight_max": float(np.max(train_weights)),
                    "weight_mean": float(np.mean(train_weights)),
                    "weights_calculated_from": "seed-42 training partition only",
                }
            )
            weight_audit.append(heldout_audit)
        prediction = fitted.predict(X_test)
        probabilities = aligned_probabilities(fitted, X_test, labels)
        test_predictions[name] = prediction
        test_rows.append({"model": name, **calculate_metrics(y_test, prediction, probabilities, labels)})
        report_rows.append(write_classification_report(y_test, prediction, encoder, name))
        per_class.extend(class_metric_rows(y_test, prediction, encoder, name, "test"))
        matrix = confusion_matrix(y_test, prediction, labels=labels)
        for actual_id, actual_name in enumerate(encoder.classes_):
            for predicted_id, predicted_name in enumerate(encoder.classes_):
                confusion_rows.append(
                    {
                        "model": name,
                        "actual_class": actual_name,
                        "predicted_class": predicted_name,
                        "count": int(matrix[actual_id, predicted_id]),
                    }
                )

        oof_probability[name] = np.zeros((len(X), len(labels)), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            X_fold_train, X_fold_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_fold_train, y_fold_valid = y.iloc[train_idx], y.iloc[valid_idx]
            fold_model, fold_weights = fit_estimator(
                name, X_fold_train, y_fold_train, numeric, categorical
            )
            fold_prediction = fold_model.predict(X_fold_valid)
            fold_probabilities = aligned_probabilities(fold_model, X_fold_valid, labels)
            oof_probability[name][valid_idx] = fold_probabilities
            fold_metric = calculate_metrics(
                y_fold_valid, fold_prediction, fold_probabilities, labels
            )
            all_cv_fold_rows.append({"model": name, "fold": fold, **fold_metric})
            cv_class_rows.extend(
                {
                    "model": name,
                    "fold": fold,
                    "class": row["class"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "support": row["support"],
                }
                for row in class_metric_rows(
                    y_fold_valid, fold_prediction, encoder, name, "cv"
                )
            )
            if name == "XGBoost":
                audit = {f"train_n_{encoder.classes_[cls]}": int((y_fold_train == cls).sum()) for cls in labels}
                audit.update(
                    {
                        "model": name,
                        "fold": fold,
                        "weight_min": float(np.min(fold_weights)),
                        "weight_max": float(np.max(fold_weights)),
                        "weight_mean": float(np.mean(fold_weights)),
                        "weights_calculated_from": "this fold training partition only",
                    }
                )
                weight_audit.append(audit)
                # Refit the unweighted baseline XGBoost on the identical CV fold.
                baseline_xgb = make_pipeline(
                    build_xgboost(RANDOM_STATE), numeric, categorical, scale_numeric=False
                )
                baseline_xgb.named_steps["classifier"].set_params(n_jobs=1)
                baseline_xgb.fit(X_fold_train, y_fold_train)
                base_prediction = baseline_xgb.predict(X_fold_valid)
                base_probability = aligned_probabilities(baseline_xgb, X_fold_valid, labels)
                baseline_xgb_fold_rows.append(
                    {
                        "model": "XGBoost",
                        "fold": fold,
                        **calculate_metrics(y_fold_valid, base_prediction, base_probability, labels),
                    }
                )
        print(f"Completed 10-fold CV: {name}", flush=True)

    test_df = pd.DataFrame(test_rows).sort_values(["f1_macro", "balanced_accuracy"], ascending=False)
    test_df.to_csv(METRICS_DIR / "imbalance_aware_model_comparison.csv", index=False)
    pd.concat(report_rows, ignore_index=True).to_csv(
        METRICS_DIR / "imbalance_aware_classification_reports.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        METRICS_DIR / "imbalance_aware_confusion_matrices.csv", index=False
    )
    pd.DataFrame(per_class).to_csv(METRICS_DIR / "per_class_test_metrics.csv", index=False)
    fold_df = pd.DataFrame(all_cv_fold_rows)
    fold_df.to_csv(CV_DIR / "fold_level_scores.csv", index=False)
    pd.DataFrame(cv_class_rows).to_csv(CV_DIR / "fold_level_per_class_metrics.csv", index=False)
    cv_summary = (
        fold_df.melt(id_vars=["model", "fold"], var_name="metric", value_name="score")
        .groupby(["model", "metric"])["score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    cv_summary.to_csv(METRICS_DIR / "imbalance_aware_cv_results.csv", index=False)
    pd.DataFrame(weight_audit).to_csv(CV_DIR / "xgboost_training_weight_audit.csv", index=False)
    pd.DataFrame(baseline_xgb_fold_rows).to_csv(
        CV_DIR / "baseline_xgboost_fold_level_scores.csv", index=False
    )
    plot_confusions(confusion_rows, encoder)
    plot_model_comparison(test_df)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_df": test_df,
        "test_predictions": test_predictions,
        "oof_probability": oof_probability,
        "fold_df": fold_df,
        "cv_class_df": pd.DataFrame(cv_class_rows),
        "cv_summary": cv_summary,
        "numeric": numeric,
        "categorical": categorical,
    }


def plot_confusions(confusion_rows, encoder):
    for model in build_models():
        frame = pd.DataFrame([row for row in confusion_rows if row["model"] == model])
        matrix = frame.pivot(index="actual_class", columns="predicted_class", values="count")
        matrix = matrix.reindex(index=encoder.classes_, columns=encoder.classes_)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Imbalance-aware {model}: confusion matrix")
        plt.xlabel("Predicted class")
        plt.ylabel("Actual class")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{slug(model)}_confusion_matrix.png", dpi=160)
        plt.close()


def plot_model_comparison(test_df):
    columns = ["balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc_ovr_macro"]
    test_df.set_index("model")[columns].plot(kind="bar", figsize=(11, 6))
    plt.ylim(0, 1)
    plt.ylabel("Test score")
    plt.title("Imbalance-aware model comparison (macro metrics emphasized)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "imbalance_aware_model_comparison.png", dpi=160)
    plt.close()


def baseline_comparison(aware, encoder):
    """Read immutable baseline metrics and compare on the identical seed-42 split."""
    baseline_test = pd.read_csv(BASELINE_METRICS / "model_comparison.csv")
    baseline_cv = pd.read_csv(BASELINE_METRICS / "cross_validation_10fold.csv")
    aware_test = aware["test_df"]
    aware_cv = aware["cv_summary"]
    baseline_reports = {
        name: pd.read_csv(BASELINE_METRICS / f"{slug(name)}_classification_report.csv").set_index("Unnamed: 0")
        for name in baseline_test["model"]
    }
    aware_reports = pd.read_csv(METRICS_DIR / "imbalance_aware_classification_reports.csv")
    rows = []
    for _, base_row in baseline_test.iterrows():
        name = base_row["model"]
        aware_row = aware_test.loc[aware_test.model == name].iloc[0]
        base_fold = baseline_cv[baseline_cv.model == name].set_index("metric")
        aware_fold = aware_cv[aware_cv.model == name].set_index("metric")
        base_report = baseline_reports[name]
        aware_report = aware_reports[aware_reports.model == name].set_index("class_or_average")
        record = {
            "model": name,
            "baseline_test_accuracy": float(base_row["accuracy"]),
            "imbalance_aware_test_accuracy": float(aware_row["accuracy"]),
            "delta_test_accuracy": float(aware_row["accuracy"] - base_row["accuracy"]),
            "baseline_test_macro_f1": float(base_row["f1_macro"]),
            "imbalance_aware_test_macro_f1": float(aware_row["f1_macro"]),
            "delta_test_macro_f1": float(aware_row["f1_macro"] - base_row["f1_macro"]),
            "baseline_test_balanced_accuracy": float(base_row["balanced_accuracy"]),
            "imbalance_aware_test_balanced_accuracy": float(aware_row["balanced_accuracy"]),
            "delta_test_balanced_accuracy": float(aware_row["balanced_accuracy"] - base_row["balanced_accuracy"]),
            "baseline_test_precision_macro": float(base_row["precision_macro"]),
            "imbalance_aware_test_precision_macro": float(aware_row["precision_macro"]),
            "baseline_test_precision_weighted": float(base_row["precision_weighted"]),
            "imbalance_aware_test_precision_weighted": float(aware_row["precision_weighted"]),
            "baseline_test_recall_macro": float(base_row["recall_macro"]),
            "imbalance_aware_test_recall_macro": float(aware_row["recall_macro"]),
            "baseline_test_recall_weighted": float(base_row["recall_weighted"]),
            "imbalance_aware_test_recall_weighted": float(aware_row["recall_weighted"]),
            "baseline_test_f1_weighted": float(base_row["f1_weighted"]),
            "imbalance_aware_test_f1_weighted": float(aware_row["f1_weighted"]),
            "baseline_test_roc_auc_ovr_weighted": float(base_row["roc_auc_ovr_weighted"]),
            "imbalance_aware_test_roc_auc_ovr_weighted": float(aware_row["roc_auc_ovr_weighted"]),
            "baseline_cv_macro_f1_mean": float(base_fold.loc["f1_macro", "mean"]),
            "baseline_cv_macro_f1_std": float(base_fold.loc["f1_macro", "std"]),
            "imbalance_aware_cv_macro_f1_mean": float(aware_fold.loc["f1_macro", "mean"]),
            "imbalance_aware_cv_macro_f1_std": float(aware_fold.loc["f1_macro", "std"]),
            "baseline_cv_balanced_accuracy_mean": float(base_fold.loc["balanced_accuracy", "mean"]),
            "imbalance_aware_cv_balanced_accuracy_mean": float(aware_fold.loc["balanced_accuracy", "mean"]),
            "same_class_weight_setting_as_baseline": name != "XGBoost",
        }
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
            "roc_auc_ovr_weighted",
        ]:
            record[f"baseline_cv_{metric}_mean"] = float(base_fold.loc[metric, "mean"])
            record[f"imbalance_aware_cv_{metric}_mean"] = float(aware_fold.loc[metric, "mean"])
            record[f"delta_cv_{metric}_mean"] = float(
                aware_fold.loc[metric, "mean"] - base_fold.loc[metric, "mean"]
            )
        for class_name in CLASSES_IN_REPORT:
            base_class = base_report.loc[class_name]
            aware_class = aware_report.loc[class_name]
            key = slug(class_name).replace("_", "_")
            record[f"baseline_{key}_recall"] = float(base_class["recall"])
            record[f"imbalance_aware_{key}_recall"] = float(aware_class["recall"])
            record[f"delta_{key}_recall"] = float(aware_class["recall"] - base_class["recall"])
            record[f"baseline_{key}_precision"] = float(base_class["precision"])
            record[f"imbalance_aware_{key}_precision"] = float(aware_class["precision"])
            record[f"delta_{key}_precision"] = float(aware_class["precision"] - base_class["precision"])
            record[f"baseline_{key}_f1"] = float(base_class["f1-score"])
            record[f"imbalance_aware_{key}_f1"] = float(aware_class["f1-score"])
            record[f"delta_{key}_f1"] = float(aware_class["f1-score"] - base_class["f1-score"])
        for key, value in record.items():
            if key.startswith("delta_") and isinstance(value, (int, float)) and abs(value) < 1e-12:
                record[key] = 0.0
        rows.append(record)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT / "baseline_vs_imbalance_comparison.csv", index=False)
    return comparison


def majority_class_baseline(y, encoder, labels):
    """Evaluate the fixed always-Moderate comparator on holdout and CV folds."""
    data = pd.read_csv(DATA_FILE)
    X = data.drop(columns=[column for column in EXCLUDED if column in data.columns])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    moderate_id = int(np.flatnonzero(encoder.classes_ == "Moderate Stress")[0])
    test_prediction = np.full(len(y_test), moderate_id)
    test_metrics = {
        "split": "test",
        "accuracy": accuracy_score(y_test, test_prediction),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_prediction),
        "f1_macro": f1_score(y_test, test_prediction, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, test_prediction, average="weighted", zero_division=0),
    }
    for class_id, class_name in enumerate(encoder.classes_):
        idx = y_test.to_numpy() == class_id
        test_metrics[f"recall_{slug(class_name)}"] = float(np.mean(test_prediction[idx] == class_id))
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    for fold, (_, valid_idx) in enumerate(cv.split(X, y), start=1):
        truth = y.iloc[valid_idx]
        pred = np.full(len(valid_idx), moderate_id)
        fold_rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(truth, pred),
                "balanced_accuracy": balanced_accuracy_score(truth, pred),
                "f1_macro": f1_score(truth, pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(truth, pred, average="weighted", zero_division=0),
            }
        )
    cv_df = pd.DataFrame(fold_rows)
    for metric in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]:
        test_metrics[f"cv_{metric}_mean"] = cv_df[metric].mean()
        test_metrics[f"cv_{metric}_std"] = cv_df[metric].std(ddof=1)
    pd.DataFrame([test_metrics]).to_csv(METRICS_DIR / "majority_class_baseline.csv", index=False)
    cv_df.to_csv(CV_DIR / "majority_class_baseline_fold_scores.csv", index=False)


def calculate_statistics(fold_df, comparison):
    """Friedman model comparison and conditional Holm-adjusted Wilcoxon tests."""
    rows = []
    posthoc = []
    from itertools import combinations

    for metric in ["f1_macro", "balanced_accuracy"]:
        wide = fold_df.pivot(index="fold", columns="model", values=metric)
        model_names = list(wide.columns)
        statistic, p_value = friedmanchisquare(*(wide[name].to_numpy() for name in model_names))
        kendall_w = statistic / (N_SPLITS * (len(model_names) - 1))
        rows.append(
            {
                "analysis": "Friedman across four imbalance-aware models",
                "metric": metric,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "effect_kendall_w": float(kendall_w),
                "alpha": 0.05,
                "significant": bool(p_value < 0.05),
                "note": "Matched stratified folds; folds share overlapping training data.",
            }
        )
        if p_value < 0.05:
            tests = []
            for left, right in combinations(model_names, 2):
                delta = wide[left].to_numpy() - wide[right].to_numpy()
                if np.allclose(delta, 0.0, atol=1e-12, rtol=0.0):
                    raw_p, test_stat = 1.0, 0.0
                else:
                    try:
                        result = wilcoxon(wide[left], wide[right], zero_method="wilcox", alternative="two-sided")
                        raw_p = float(result.pvalue)
                        test_stat = float(result.statistic)
                    except ValueError:
                        raw_p, test_stat = 1.0, 0.0
                tests.append(
                    {
                        "metric": metric,
                        "model_a": left,
                        "model_b": right,
                        "wilcoxon_statistic": test_stat,
                        "p_value_raw": raw_p,
                        "median_paired_difference_a_minus_b": float(np.median(delta)),
                    }
                )
            adjust_holm(tests)
            posthoc.extend(tests)

    # Fold-level comparisons against the immutable baseline summaries.
    base_cv = pd.read_csv(BASELINE_METRICS / "cross_validation_10fold.csv")
    baseline_xgb_folds = pd.read_csv(CV_DIR / "baseline_xgboost_fold_level_scores.csv")
    aware_summary = fold_df.groupby("model")["f1_macro"].agg(["mean", "std"])
    for model in aware_summary.index:
        baseline_mean = float(
            base_cv[(base_cv.model == model) & (base_cv.metric == "f1_macro")]["mean"].iloc[0]
        )
        # For LR/SVM/RF the baseline already uses the requested balanced class weights.
        if model != "XGBoost":
            base_values = fold_df.loc[fold_df.model == model].sort_values("fold")["f1_macro"].to_numpy()
            aware_values = fold_df.loc[fold_df.model == model].sort_values("fold")["f1_macro"].to_numpy()
            if np.allclose(base_values - aware_values, 0.0, atol=1e-12, rtol=0.0):
                statistic, p_value = 0.0, 1.0
            else:
                try:
                    test_result = wilcoxon(base_values, aware_values, zero_method="wilcox", alternative="two-sided")
                    statistic, p_value = float(test_result.statistic), float(test_result.pvalue)
                except ValueError:
                    statistic, p_value = 0.0, 1.0
            note = "Both baseline and secondary specification use class_weight='balanced'; models refit on identical folds."
        else:
            base_values = baseline_xgb_folds.sort_values("fold")["f1_macro"].to_numpy()
            aware_values = fold_df.loc[fold_df.model == model].sort_values("fold")["f1_macro"].to_numpy()
            if np.allclose(base_values - aware_values, 0.0, atol=1e-12, rtol=0.0):
                statistic, p_value = 0.0, 1.0
            else:
                try:
                    test_result = wilcoxon(base_values, aware_values, zero_method="wilcox", alternative="two-sided")
                    statistic, p_value = float(test_result.statistic), float(test_result.pvalue)
                except ValueError:
                    statistic, p_value = 0.0, 1.0
            note = "Paired Wilcoxon test on baseline-unweighted and training-fold-weighted XGBoost Macro F1."
        rows.append(
            {
                "analysis": "Baseline vs imbalance-aware CV Macro F1",
                "metric": f"{model} CV Macro F1",
                "statistic": statistic,
                "p_value": p_value,
                "baseline_mean": baseline_mean,
                "imbalance_aware_mean": float(aware_summary.loc[model, "mean"]),
                "delta_mean": float(aware_summary.loc[model, "mean"] - baseline_mean),
                "effect_kendall_w": np.nan,
                "alpha": 0.05,
                "significant": False,
                "note": note,
            }
        )
    pd.DataFrame(rows).to_csv(STATISTICS_DIR / "statistical_tests.csv", index=False)
    pd.DataFrame(posthoc).to_csv(STATISTICS_DIR / "posthoc_pairwise_tests.csv", index=False)


def adjust_holm(tests):
    order = sorted(range(len(tests)), key=lambda idx: tests[idx]["p_value_raw"])
    running = 0.0
    count = len(tests)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (count - rank) * tests[idx]["p_value_raw"])
        running = max(running, adjusted)
        tests[idx]["p_value_holm"] = running
        tests[idx]["significant_holm_0_05"] = running < 0.05


def bootstrap_confidence_intervals(y_test, predictions, probabilities, labels):
    """Stratified percentile CIs for held-out Macro F1 and balanced accuracy."""
    rng = np.random.default_rng(RANDOM_STATE)
    y_array = np.asarray(y_test)
    class_indices = {label: np.flatnonzero(y_array == label) for label in labels}
    rows = []
    for model, pred in predictions.items():
        pred = np.asarray(pred)
        values = {"f1_macro": [], "balanced_accuracy": []}
        for _ in range(BOOTSTRAP_REPLICATES):
            sample_idx = np.concatenate(
                [rng.choice(class_indices[label], size=len(class_indices[label]), replace=True) for label in labels]
            )
            truth, estimate = y_array[sample_idx], pred[sample_idx]
            values["f1_macro"].append(f1_score(truth, estimate, average="macro", zero_division=0))
            values["balanced_accuracy"].append(balanced_accuracy_score(truth, estimate))
        point = {
            "f1_macro": f1_score(y_array, pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_array, pred),
        }
        for metric in values:
            low, high = np.quantile(values[metric], [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_95_lower": low,
                    "ci_95_upper": high,
                    "method": "stratified percentile bootstrap, 2000 replicates",
                    "random_state": RANDOM_STATE,
                }
            )
    pd.DataFrame(rows).to_csv(METRICS_DIR / "confidence_intervals.csv", index=False)


def run_ablation(selected_model, X, y, encoder, labels):
    """Ten-fold feature group ablation using the selected weighted model."""
    rows, fold_rows = [], []
    for experiment, groups in ABLATIONS.items():
        columns = [column for group in groups for column in FEATURE_GROUPS[group]]
        X_subset = X[columns]
        numeric, categorical = feature_columns(X_subset)
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_subset, y), start=1):
            model, weights = fit_estimator(
                selected_model,
                X_subset.iloc[train_idx],
                y.iloc[train_idx],
                numeric,
                categorical,
            )
            truth = y.iloc[valid_idx]
            prediction = model.predict(X_subset.iloc[valid_idx])
            probabilities = aligned_probabilities(model, X_subset.iloc[valid_idx], labels)
            metric = calculate_metrics(truth, prediction, probabilities, labels)
            class_rows = class_metric_rows(truth, prediction, encoder, selected_model, "ablation")
            by_class = {row["class"]: row for row in class_rows}
            fold_rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "features": json.dumps(columns),
                    "f1_macro": metric["f1_macro"],
                    "balanced_accuracy": metric["balanced_accuracy"],
                    "low_recall": by_class["Low Stress"]["recall"],
                    "low_f1": by_class["Low Stress"]["f1"],
                    "high_recall": by_class["High Stress"]["recall"],
                    "high_f1": by_class["High Stress"]["f1"],
                }
            )
        current = pd.DataFrame([row for row in fold_rows if row["experiment"] == experiment])
        rows.append(
            {
                "experiment": experiment,
                "feature_groups": "+".join(groups),
                "features": json.dumps(columns),
                "n_features": len(columns),
                **{
                    f"{metric}_{stat}": current[metric].agg(stat)
                    for metric in ["f1_macro", "balanced_accuracy", "low_recall", "low_f1", "high_recall", "high_f1"]
                    for stat in ["mean", "std"]
                },
            }
        )
    pd.DataFrame(rows).to_csv(ABLATION_DIR / "ablation_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(ABLATION_DIR / "ablation_fold_level_results.csv", index=False)


def run_robustness(candidate_models, X, y, labels):
    rows = []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        numeric, categorical = feature_columns(X)
        for name in candidate_models:
            model, _ = fit_estimator(name, X_train, y_train, numeric, categorical)
            prediction = model.predict(X_test)
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "f1_macro": f1_score(y_test, prediction, average="macro", zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                }
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(ROBUSTNESS_DIR / "robustness_fold_seed_results.csv", index=False)
    summary = raw.groupby("model").agg(
        macro_f1_mean=("f1_macro", "mean"),
        macro_f1_std=("f1_macro", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    ).reset_index()
    summary.to_csv(ROBUSTNESS_DIR / "robustness_results.csv", index=False)


def run_error_analysis(candidate_models, aware, encoder):
    y_true = aware["y_test"].to_numpy()
    X_test = aware["X_test"].reset_index(drop=True)
    rows = []
    for model in candidate_models:
        prediction = np.asarray(aware["test_predictions"][model])
        for actual_id, actual_name in enumerate(encoder.classes_):
            actual_idx = np.flatnonzero(y_true == actual_id)
            for predicted_id, predicted_name in enumerate(encoder.classes_):
                flow_idx = np.flatnonzero((y_true == actual_id) & (prediction == predicted_id))
                profile = {}
                if len(flow_idx):
                    group = X_test.iloc[flow_idx]
                    for column in group:
                        if pd.api.types.is_numeric_dtype(group[column]):
                            profile[column] = {"mean": float(group[column].mean()), "median": float(group[column].median())}
                        else:
                            profile[column] = {"mode": str(group[column].mode().iloc[0])}
                rows.append(
                    {
                        "model": model,
                        "actual_class": actual_name,
                        "predicted_class": predicted_name,
                        "count": len(flow_idx),
                        "proportion_of_actual_class": len(flow_idx) / len(actual_idx) if len(actual_idx) else np.nan,
                        "error": actual_id != predicted_id,
                        "predictor_profile": json.dumps(profile),
                    }
                )
    pd.DataFrame(rows).to_csv(ERRORS_DIR / "error_analysis.csv", index=False)


def run_subgroups(candidate_model, aware, encoder):
    y_true = np.asarray(aware["y_test"])
    prediction = np.asarray(aware["test_predictions"][candidate_model])
    X_test = aware["X_test"].reset_index(drop=True)
    rows = []
    for variable in ["Gender", "University", "Academic_Year", "Degree_Program"]:
        for group in sorted(X_test[variable].dropna().astype(str).unique()):
            idx = np.flatnonzero(X_test[variable].astype(str).to_numpy() == group)
            truth, predicted = y_true[idx], prediction[idx]
            counts = {class_name: int(np.sum(truth == class_id)) for class_id, class_name in enumerate(encoder.classes_)}
            adequate = len(idx) >= MIN_SUBGROUP_SIZE and all(
                counts[class_name] >= MIN_CLASS_SUPPORT_FOR_SUBGROUP for class_name in encoder.classes_
            )
            record = {
                "model": candidate_model,
                "subgroup_variable": variable,
                "subgroup": group,
                "test_n": len(idx),
                "adequate_for_macro_metrics": adequate,
                "status": "reported" if adequate else "insufficient subgroup/class support; metrics withheld",
                "class_counts": json.dumps(counts),
                "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0) if adequate else np.nan,
                "balanced_accuracy": balanced_accuracy_score(truth, predicted) if adequate else np.nan,
            }
            for class_id, class_name in enumerate(encoder.classes_):
                support = counts[class_name]
                record[f"recall_{slug(class_name)}"] = (
                    float(np.mean(predicted[truth == class_id] == class_id))
                    if support >= MIN_CLASS_SUPPORT_FOR_SUBGROUP
                    else np.nan
                )
            rows.append(record)
    pd.DataFrame(rows).to_csv(SUBGROUP_DIR / "subgroup_results.csv", index=False)


def run_calibration(candidate_models, y, encoder, oof_probability):
    one_hot = label_binarize(y, classes=np.arange(len(encoder.classes_)))
    rows = []
    fig, axes = plt.subplots(len(candidate_models), len(encoder.classes_), figsize=(15, 8), squeeze=False)
    for model_idx, model in enumerate(candidate_models):
        probability = oof_probability[model]
        brier = float(np.mean(np.sum((one_hot - probability) ** 2, axis=1)))
        rows.append({"model": model, "measure": "multiclass_brier_score", "class": "all", "value": brier,
                     "method": "mean row-wise sum of squared one-hot probability errors"})
        for class_id, class_name in enumerate(encoder.classes_):
            fraction_positive, mean_predicted = calibration_curve(
                one_hot[:, class_id], probability[:, class_id], n_bins=10, strategy="uniform"
            )
            axes[model_idx, class_id].plot(mean_predicted, fraction_positive, marker="o")
            axes[model_idx, class_id].plot([0, 1], [0, 1], linestyle="--", color="gray")
            axes[model_idx, class_id].set_title(f"{model}: {class_name}")
            axes[model_idx, class_id].set_xlabel("Mean predicted probability")
            axes[model_idx, class_id].set_ylabel("Observed fraction positive")
            rows.append({"model": model, "measure": "calibration_curve_bins", "class": class_name,
                         "value": json.dumps({"mean_predicted": mean_predicted.tolist(),
                                               "fraction_positive": fraction_positive.tolist()}),
                         "method": "one-vs-rest, 10 uniform bins on out-of-fold predictions"})
    fig.tight_layout()
    fig.savefig(CALIBRATION_DIR / "calibration_curves.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(CALIBRATION_DIR / "calibration_results.csv", index=False)


def explain_selected_model(selected_model, aware, X, y, encoder, labels):
    """Run model-appropriate SHAP after the Macro-F1/CV selection is fixed."""
    numeric, categorical = feature_columns(X)
    model, _ = fit_estimator(
        selected_model, aware["X_train"], aware["y_train"], numeric, categorical
    )
    transformer = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    feature_names = transformer.get_feature_names_out()
    test_sample = aware["X_test"].sample(
        n=min(SHAP_FINAL_SAMPLE_SIZE, len(aware["X_test"])), random_state=RANDOM_STATE
    )
    transformed = transformer.transform(test_sample)
    dense_test = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    background_source = aware["X_train"].sample(
        n=min(SHAP_BACKGROUND_SIZE, len(aware["X_train"])), random_state=RANDOM_STATE
    )
    transformed_background = transformer.transform(background_source)
    background = transformed_background.toarray() if hasattr(transformed_background, "toarray") else np.asarray(transformed_background)
    explainer = shap.KernelExplainer(classifier.predict_proba, background)
    values = explainer.shap_values(dense_test, nsamples=SHAP_NSAMPLES)
    class_values = normalize_shap(values, len(labels), len(dense_test), dense_test.shape[1])
    global_importance = np.mean(np.abs(class_values), axis=(0, 1))
    importance = pd.DataFrame({"feature": feature_names, "mean_absolute_shap_value": global_importance})
    importance = importance.sort_values("mean_absolute_shap_value", ascending=False)
    importance.to_csv(SHAP_DIR / "global_feature_importance.csv", index=False)
    importance.head(20).sort_values("mean_absolute_shap_value").plot.barh(
        x="feature", y="mean_absolute_shap_value", legend=False, figsize=(10, 7)
    )
    plt.xlabel("Mean absolute SHAP value across classes")
    plt.title(f"Imbalance-aware SHAP global importance: {selected_model}")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_feature_importance.png", dpi=160)
    plt.close()
    shap.summary_plot(
        class_values.reshape(-1, dense_test.shape[1]),
        np.tile(dense_test, (len(labels), 1)),
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"Imbalance-aware SHAP summary: {selected_model}")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    sample_idx = 0
    predicted_class = int(np.argmax(classifier.predict_proba(dense_test[sample_idx:sample_idx + 1])[0]))
    local = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": class_values[predicted_class, sample_idx],
            "explained_class": encoder.classes_[predicted_class],
        }
    )
    local["absolute_shap_value"] = local.shap_value.abs()
    local.sort_values("absolute_shap_value", ascending=False).to_csv(
        SHAP_DIR / "local_prediction_contributions.csv", index=False
    )
    info = {
        "selected_model": selected_model,
        "explainer": "KernelExplainer",
        "background_strategy": "random sample from seed-42 training split after fold-safe preprocessing fit",
        "background_sample_size": len(background_source),
        "explained_sample_size": len(test_sample),
        "nsamples": SHAP_NSAMPLES,
        "random_state": RANDOM_STATE,
        "prediction_explained": str(encoder.classes_[predicted_class]),
        "selection_criterion": "highest 10-fold CV macro F1, CV balanced accuracy as tie-breaker",
        "causal_interpretation": False,
    }
    (SHAP_DIR / "shap_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    run_shap_stability(selected_model, X, y, encoder, labels, numeric, categorical)


def normalize_shap(values, classes, samples, features):
    if isinstance(values, list):
        array = np.stack([np.asarray(value) for value in values], axis=0)
    else:
        array = np.asarray(values)
        if array.ndim == 3 and array.shape == (samples, features, classes):
            array = np.moveaxis(array, -1, 0)
        elif array.ndim == 2:
            array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"Unexpected multiclass SHAP value shape {array.shape}")
    if array.shape == (samples, features, classes):
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] == 1 and classes > 1:
        array = np.repeat(array, classes, axis=0)
    if array.shape != (classes, samples, features):
        raise ValueError(f"Expected {(classes, samples, features)} SHAP values, got {array.shape}")
    return array


def run_shap_stability(selected_model, X, y, encoder, labels, numeric, categorical):
    """Estimate feature rank stability across all held-out CV folds."""
    fold_importances = []
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        fold_model, _ = fit_estimator(
            selected_model, X.iloc[train_idx], y.iloc[train_idx], numeric, categorical
        )
        transformer = fold_model.named_steps["preprocess"]
        classifier = fold_model.named_steps["classifier"]
        feature_names = transformer.get_feature_names_out()
        valid_frame = X.iloc[valid_idx]
        sample = valid_frame.sample(
            n=min(SHAP_STABILITY_ROWS_PER_FOLD, len(valid_frame)),
            random_state=RANDOM_STATE + fold,
        )
        background_frame = X.iloc[train_idx].sample(
            n=min(SHAP_BACKGROUND_SIZE, len(train_idx)), random_state=RANDOM_STATE + fold
        )
        sample_encoded = transformer.transform(sample)
        background_encoded = transformer.transform(background_frame)
        sample_dense = sample_encoded.toarray() if hasattr(sample_encoded, "toarray") else np.asarray(sample_encoded)
        background_dense = background_encoded.toarray() if hasattr(background_encoded, "toarray") else np.asarray(background_encoded)
        explainer = shap.KernelExplainer(classifier.predict_proba, background_dense)
        values = explainer.shap_values(sample_dense, nsamples=SHAP_NSAMPLES)
        class_values = normalize_shap(values, len(labels), len(sample_dense), sample_dense.shape[1])
        fold_importances.append(
            pd.Series(np.mean(np.abs(class_values), axis=(0, 1)), index=feature_names, name=fold)
        )
        print(f"SHAP stability fold {fold}/{N_SPLITS}", flush=True)
    aligned = pd.concat(fold_importances, axis=1).fillna(0)
    ranks = aligned.rank(axis=0, ascending=False, method="average")
    top_k = min(10, len(aligned))
    stability = pd.DataFrame(
        {
            "feature": aligned.index,
            "mean_absolute_shap_importance": aligned.mean(axis=1).to_numpy(),
            "mean_rank": ranks.mean(axis=1).to_numpy(),
            "rank_sd": ranks.std(axis=1, ddof=1).to_numpy(),
            "top_10_frequency": (ranks <= top_k).mean(axis=1).to_numpy(),
            "folds_included": N_SPLITS,
        }
    ).sort_values("mean_absolute_shap_importance", ascending=False)
    stability.to_csv(SHAP_DIR / "shap_stability.csv", index=False)
    top = stability.head(20).sort_values("top_10_frequency")
    top.plot.barh(x="feature", y="top_10_frequency", legend=False, figsize=(10, 7))
    plt.xlabel("Fraction of folds in top 10")
    plt.title("SHAP feature rank stability across 10 CV folds")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_stability_top10_frequency.png", dpi=160)
    plt.close()


def run_final_selection(aware, comparison, encoder):
    """Select solely using fold-level Macro F1, then fold balanced accuracy."""
    summaries = aware["cv_summary"]
    model_summary = summaries.pivot(index="model", columns="metric", values="mean")
    # Explicit lexicographic ordering: CV Macro F1 first, CV balanced accuracy second.
    selected = model_summary.sort_values(
        ["f1_macro", "balanced_accuracy"], ascending=False, kind="mergesort"
    ).index[0]
    selected_test = aware["test_df"].loc[aware["test_df"].model == selected].iloc[0]
    selected_cv = model_summary.loc[selected]
    report = pd.read_csv(METRICS_DIR / "imbalance_aware_classification_reports.csv")
    selected_report = report[report.model == selected].set_index("class_or_average")
    baseline = pd.read_csv(BASELINE_METRICS / "model_comparison.csv")
    base_selected = baseline[baseline.model == selected].iloc[0]
    recalls = {class_name: float(selected_report.loc[class_name, "recall"]) for class_name in encoder.classes_}
    f1s = {class_name: float(selected_report.loc[class_name, "f1-score"]) for class_name in encoder.classes_}
    result = {
                "selected_model": selected,
        "selection_criterion": "Highest mean 10-fold CV Macro F1; CV Balanced Accuracy is the tie-breaker.",
        "selection_values": {
            "cv_macro_f1_mean": float(selected_cv["f1_macro"]),
            "cv_macro_f1_std": float(summaries.query("model == @selected and metric == 'f1_macro'")["std"].iloc[0]),
            "cv_balanced_accuracy_mean": float(selected_cv["balanced_accuracy"]),
            "cv_balanced_accuracy_std": float(summaries.query("model == @selected and metric == 'balanced_accuracy'")["std"].iloc[0]),
        },
        "held_out_test": {
            "macro_f1": float(selected_test["f1_macro"]),
            "balanced_accuracy": float(selected_test["balanced_accuracy"]),
            "minority_class_recall": {c: recalls[c] for c in ["Low Stress", "High Stress"]},
            "minority_class_f1": {c: f1s[c] for c in ["Low Stress", "High Stress"]},
            "macro_roc_auc_ovr": float(selected_test["roc_auc_ovr_macro"]),
            "weighted_roc_auc_ovr": float(selected_test["roc_auc_ovr_weighted"]),
            "accuracy": float(selected_test["accuracy"]),
            "f1_weighted": float(selected_test["f1_weighted"]),
        },
        "baseline_comparison": {
            "baseline_test_macro_f1": float(base_selected["f1_macro"]),
            "imbalance_aware_test_macro_f1": float(selected_test["f1_macro"]),
            "test_macro_f1_improved": bool(selected_test["f1_macro"] > base_selected["f1_macro"] + 1e-12),
            "minority_class_recall_improved": {
                "Low Stress": bool(recalls["Low Stress"] > baseline_reports_for_selected(selected)["Low Stress"]["recall"] + 1e-12),
                "High Stress": bool(recalls["High Stress"] > baseline_reports_for_selected(selected)["High Stress"]["recall"] + 1e-12),
            },
            "baseline_test_balanced_accuracy": float(base_selected["balanced_accuracy"]),
            "imbalance_aware_test_balanced_accuracy": float(selected_test["balanced_accuracy"]),
            "test_balanced_accuracy_improved": bool(selected_test["balanced_accuracy"] > base_selected["balanced_accuracy"] + 1e-12),
            "baseline_cv_macro_f1_mean": float(
                pd.read_csv(BASELINE_METRICS / "cross_validation_10fold.csv").query(
                    "model == @selected and metric == 'f1_macro'"
                )["mean"].iloc[0]
            ),
            "imbalance_aware_cv_macro_f1_mean": float(selected_cv["f1_macro"]),
            "baseline_cv_macro_f1_std": float(
                pd.read_csv(BASELINE_METRICS / "cross_validation_10fold.csv").query(
                    "model == @selected and metric == 'f1_macro'"
                )["std"].iloc[0]
            ),
            "imbalance_aware_cv_macro_f1_std": float(
                summaries.query("model == @selected and metric == 'f1_macro'")["std"].iloc[0]
            ),
        },
        "final_shap_not_run": True,
    }
    (METRICS_DIR / "final_model_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return selected


def baseline_reports_for_selected(model):
    report = pd.read_csv(BASELINE_METRICS / f"{slug(model)}_classification_report.csv")
    return report.set_index("Unnamed: 0").to_dict(orient="index")


def main():
    ensure_output_directories()
    data, X, y, encoder, labels = load_data()
    print(f"Observed dataset: {data.shape[0]} rows; {data.shape[1]} processed columns")
    print("Verified feature groups:", json.dumps(FEATURE_GROUPS))
    print("Class counts:", json.dumps(data[TARGET].value_counts().to_dict()))

    aware = run_main_models(X, y, encoder, labels)
    comparison = baseline_comparison(aware, encoder)
    majority_class_baseline(y, encoder, labels)
    bootstrap_confidence_intervals(
        aware["y_test"], aware["test_predictions"], aware["oof_probability"], labels
    )
    calculate_statistics(aware["fold_df"], comparison)

    selected = run_final_selection(aware, comparison, encoder)
    # Final candidate set is chosen by CV Macro F1 ranks only, before test-based analysis.
    cv_rank = aware["cv_summary"].query("metric == 'f1_macro'").sort_values("mean", ascending=False)
    candidates = cv_rank.head(2)["model"].tolist()
    run_ablation(selected, X, y, encoder, labels)
    run_robustness(candidates, X, y, labels)
    run_error_analysis(candidates, aware, encoder)
    run_subgroups(selected, aware, encoder)
    run_calibration(candidates, y, encoder, aware["oof_probability"])

    # Selection is now fixed using CV only. SHAP is confined to its own directory.
    explain_selected_model(selected, aware, X, y, encoder, labels)
    selection_path = METRICS_DIR / "final_model_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["final_shap_not_run"] = False
    selection["shap_status"] = "completed after CV-based model selection"
    (selection_path).write_text(json.dumps(selection, indent=2), encoding="utf-8")
    write_comparison_notes(comparison, aware, selected, candidates)
    write_research_summary(selected, candidates, aware, comparison, encoder)
    print(f"Selected by CV Macro F1: {selected}")
    print(f"New experiment results: {OUTPUT}")


def write_comparison_notes(comparison, aware, selected, candidates):
    baseline_metadata = json.loads((BASELINE_METRICS / "run_metadata.json").read_text(encoding="utf-8"))
    notes = {
        "experiment": "imbalance-aware secondary experiment",
        "baseline_preserved": True,
        "baseline_outputs_written": False,
        "target": TARGET,
        "excluded_predictors": [f"PSS_Q{i}" for i in range(1, 11)] + ["PSS10_Total"],
        "features_used": list(aware["X_train"].columns),
        "same_train_test_split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "matches_baseline_metadata": baseline_metadata["test_size"] == TEST_SIZE and baseline_metadata["random_state"] == RANDOM_STATE,
        },
        "imbalance_handling": {
            "logistic_regression": "class_weight='balanced'; same setting as existing baseline",
            "svm": "class_weight='balanced'; same setting as existing baseline",
            "random_forest": "class_weight='balanced'; same setting as existing baseline",
            "xgboost": "compute_sample_weight('balanced', y_train) independently from each training partition; pass as classifier__sample_weight",
        },
        "cv": {"n_splits": N_SPLITS, "shuffle": True, "random_state": RANDOM_STATE},
        "selected_model": selected,
        "candidate_models_for_robustness_error_and_calibration": candidates,
        "shap": "run only after CV-based model selection; stability estimates use each held-out CV fold and fold-training background",
        "limitations": [
            "The three sklearn models already used class_weight='balanced' in the baseline implementation; those model settings are not a new treatment.",
            "Unweighted baseline XGBoost was refit on the same ten folds to enable the paired CV comparison; these recomputed values are separately recorded.",
            "Friedman and Wilcoxon tests on ten matched folds are reported with dependence caveat because CV training sets overlap.",
            "Subgroup metrics are withheld when the subgroup has fewer than 30 test cases or fewer than five cases in any class.",
        ],
    }
    (METRICS_DIR / "experiment_metadata.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")


def write_research_summary(selected, candidates, aware, comparison, encoder):
    test = aware["test_df"].set_index("model")
    cv = aware["cv_summary"]
    cv_wide = cv.pivot(index="model", columns="metric", values="mean")
    cv_sd = cv.pivot(index="model", columns="metric", values="std")
    baseline = pd.read_csv(BASELINE_METRICS / "model_comparison.csv").set_index("model")
    majority = pd.read_csv(METRICS_DIR / "majority_class_baseline.csv").iloc[0]
    ablation = pd.read_csv(ABLATION_DIR / "ablation_results.csv").sort_values("f1_macro_mean", ascending=False)
    stats = pd.read_csv(STATISTICS_DIR / "statistical_tests.csv")
    robustness = pd.read_csv(ROBUSTNESS_DIR / "robustness_results.csv")
    subgroup = pd.read_csv(SUBGROUP_DIR / "subgroup_results.csv")
    errors = pd.read_csv(ERRORS_DIR / "error_analysis.csv")
    calibration = pd.read_csv(CALIBRATION_DIR / "calibration_results.csv")
    shap_importance = pd.read_csv(SHAP_DIR / "global_feature_importance.csv").head(5)
    ci = pd.read_csv(METRICS_DIR / "confidence_intervals.csv")
    selected_ci = ci[ci.model == selected]
    macro_test_ci = selected_ci[selected_ci.metric == "f1_macro"].iloc[0]
    balanced_test_ci = selected_ci[selected_ci.metric == "balanced_accuracy"].iloc[0]
    macro_test_stat = stats[stats.metric == "f1_macro"].iloc[0]
    balanced_test_stat = stats[stats.metric == "balanced_accuracy"].iloc[0]
    selected_brier = calibration[
        (calibration.model == selected) & (calibration.measure == "multiclass_brier_score")
    ]["value"].iloc[0]
    adequate_subgroups = int(subgroup["adequate_for_macro_metrics"].sum())
    error_rows = errors[(errors.model == selected) & errors.error].sort_values("count", ascending=False).head(3)
    selected_values = test.loc[selected]
    lines = [
        "# Imbalance-aware experiment: generated results summary",
        "",
        "## A. Baseline and imbalance-aware performance",
        "",
        "The baseline outputs under `results/metrics/`, `results/figures/`, and `results/shap_plots/` were read as inputs and not overwritten. The three sklearn model specifications already used `class_weight='balanced'` in the baseline. XGBoost is the only estimator receiving a new training-partition-derived sample-weight treatment.",
        "",
        f"The Macro-F1/CV selection rule selected **{selected}** (10-fold Macro F1 {cv_wide.loc[selected, 'f1_macro']:.4f} ± {cv_sd.loc[selected, 'f1_macro']:.4f}; Balanced Accuracy {cv_wide.loc[selected, 'balanced_accuracy']:.4f} ± {cv_sd.loc[selected, 'balanced_accuracy']:.4f}). Its held-out Macro F1 is {selected_values['f1_macro']:.4f}, Balanced Accuracy {selected_values['balanced_accuracy']:.4f}.",
        "",
        "| Model | Baseline test Macro F1 | Aware test Macro F1 | Baseline CV Macro F1 | Aware CV Macro F1 | Aware test balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in test.index:
        base_row = baseline.loc[model]
        aware_cv_row = cv_wide.loc[model]
        base_cv_row = pd.read_csv(BASELINE_METRICS / "cross_validation_10fold.csv").query(
            "model == @model and metric == 'f1_macro'"
        ).iloc[0]
        lines.append(
            f"| {model} | {base_row['f1_macro']:.4f} | {test.loc[model, 'f1_macro']:.4f} | {base_cv_row['mean']:.4f} ± {base_cv_row['std']:.4f} | {aware_cv_row['f1_macro']:.4f} ± {cv_sd.loc[model, 'f1_macro']:.4f} | {test.loc[model, 'balanced_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## B. Majority-class baseline",
            "",
            f"Always predicting Moderate Stress yielded test accuracy {majority['accuracy']:.4f}, balanced accuracy {majority['balanced_accuracy']:.4f}, Macro F1 {majority['f1_macro']:.4f}, and weighted F1 {majority['f1_weighted']:.4f}. Per-class recall values are stored in `metrics/majority_class_baseline.csv`.",
            "",
            "## C. Ablation",
            "",
            f"Highest CV Macro F1 ablation: {ablation.iloc[0]['experiment']} ({ablation.iloc[0]['f1_macro_mean']:.4f} ± {ablation.iloc[0]['f1_macro_std']:.4f}); Low Stress recall {ablation.iloc[0]['low_recall_mean']:.4f}, High Stress recall {ablation.iloc[0]['high_recall_mean']:.4f}. The full results include every specified feature-group configuration.",
            "",
            "## D. Statistical comparisons and confidence intervals",
            "",
            f"Friedman Macro F1: statistic {macro_test_stat['statistic']:.4f}, p={macro_test_stat['p_value']:.6g}, Kendall W={macro_test_stat['effect_kendall_w']:.4f}; significant at 0.05: {bool(macro_test_stat['significant'])}. Balanced Accuracy: statistic {balanced_test_stat['statistic']:.4f}, p={balanced_test_stat['p_value']:.6g}, Kendall W={balanced_test_stat['effect_kendall_w']:.4f}; significant at 0.05: {bool(balanced_test_stat['significant'])}. Pairwise post-hoc results, when applicable, use Holm-adjusted Wilcoxon tests. The selected model's held-out Macro F1 95% stratified-bootstrap CI is [{macro_test_ci['ci_95_lower']:.4f}, {macro_test_ci['ci_95_upper']:.4f}], and Balanced Accuracy CI is [{balanced_test_ci['ci_95_lower']:.4f}, {balanced_test_ci['ci_95_upper']:.4f}]. Fold tests have overlapping training partitions, which limits the independence assumption.",
            "",
            "## E. Robustness",
            "",
            "The two highest CV Macro-F1 candidates were reevaluated using seeds 42, 123, 2024, 2025, and 2026. "
            + "; ".join(
                f"{r.model}: Macro F1 {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f}, balanced accuracy {r.balanced_accuracy_mean:.4f} ± {r.balanced_accuracy_std:.4f}"
                for r in robustness.itertuples()
            )
            + ".",
            "",
            "## F. Error and subgroup analysis",
            "",
            f"Structured actual-to-predicted error counts and descriptive profiles are in `errors/error_analysis.csv` for {', '.join(candidates)}. The largest error flows for selected {selected}: "
            + "; ".join(f"{r.actual_class} → {r.predicted_class}: {int(r['count'])}" for _, r in error_rows.iterrows())
            + f". Subgroup sample counts are reported in `subgroup/subgroup_results.csv`; {adequate_subgroups} subgroup(s) met the minimum total/class support thresholds. Other metrics were withheld. No causal or fairness claims are made.",
            "",
            "## G. Calibration",
            "",
            f"Out-of-fold multiclass Brier score for selected {selected}: {float(selected_brier):.4f}. Candidate Brier scores and one-vs-rest calibration curves are saved under `calibration/`; models were not recalibrated.",
            "",
            "## H. SHAP and SHAP stability",
            "",
            f"After CV-based selection, {selected} was explained. The five highest global SHAP features in this run are: "
            + ", ".join(f"{r.feature} ({r.mean_absolute_shap_value:.6g})" for r in shap_importance.itertuples())
            + ". Fold-wise feature importance, rank variability, and top-10 frequency are in `shap/shap_stability.csv`. SHAP values are model explanations, not causal effects.",
            "",
            "## Limitations",
            "",
            "The original baseline contains class weighting for Logistic Regression, SVM, and Random Forest; these are repeated settings rather than new interventions. XGBoost alone receives newly computed per-training-partition class sample weights. Subgroup results may be withheld for small groups. Fold-based inferential tests use ten overlapping CV training sets. Results describe this dataset and split protocol; they do not establish clinical validity or universal generalization.",
        ]
    )
    (OUTPUT / "research_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def slug(value):
    return str(value).lower().replace(" ", "_")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
