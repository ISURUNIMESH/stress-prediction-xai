"""Nested stratified CV experiment, isolated under results/nested_cv/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("Run nested CV with the project CPython 3.12 environment.")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from scipy.optimize import OptimizeWarning
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
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from models.train_evaluate import (
    DATA_FILE,
    EXCLUDED,
    RANDOM_STATE,
    build_models,
    make_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "nested_cv"
FIGURES = OUTPUT / "figures"
METRICS = OUTPUT / "metrics"
TARGET = "Stress_Level"
OUTER_SPLITS = 5
INNER_SPLITS = 5
SEED = 42
MODEL_NAMES = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]
CLASS_ORDER = ["High Stress", "Low Stress", "Moderate Stress"]

# SciPy 1.18 emits this benign option warning through scikit-learn's lbfgs
# adapter; suppress only this exact compatibility warning.
warnings.filterwarnings(
    "ignore", message="Unknown solver options: iprint", category=OptimizeWarning
)

# Deliberately small, fixed candidate grids.  The two candidates per model
# keep the complete 5 x 5 nested search reasonable and reproducible.
PARAMETER_GRIDS = {
    "Logistic Regression": [
        {"classifier__C": [0.1]},
        {"classifier__C": [1.0]},
    ],
    "SVM": [
        {"classifier__C": [0.5], "classifier__kernel": ["rbf"], "classifier__gamma": ["scale"]},
        {"classifier__C": [2.0], "classifier__kernel": ["rbf"], "classifier__gamma": ["scale"]},
    ],
    "Random Forest": [
        {
            "classifier__n_estimators": [200],
            "classifier__max_depth": [None],
            "classifier__min_samples_split": [2],
            "classifier__min_samples_leaf": [2],
        },
        {
            "classifier__n_estimators": [300],
            "classifier__max_depth": [12],
            "classifier__min_samples_split": [4],
            "classifier__min_samples_leaf": [2],
        },
    ],
    "XGBoost": [
        {
            "classifier__n_estimators": [150],
            "classifier__max_depth": [3],
            "classifier__learning_rate": [0.05],
            "classifier__subsample": [0.8],
            "classifier__colsample_bytree": [0.8],
        },
        {
            "classifier__n_estimators": [250],
            "classifier__max_depth": [4],
            "classifier__learning_rate": [0.05],
            "classifier__subsample": [0.8],
            "classifier__colsample_bytree": [0.8],
        },
    ],
}


def make_estimator(name: str, X: pd.DataFrame):
    """Build weighted estimator with preprocessing fitted by its pipeline."""
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    estimator = build_models()[name]
    if "n_jobs" in estimator.get_params(deep=False):
        estimator.set_params(n_jobs=1)
    return make_pipeline(
        estimator,
        numeric,
        categorical,
        scale_numeric=name in {"Logistic Regression", "SVM"},
    )


def fit_pipeline(name, X_train, y_train, params):
    pipeline = make_estimator(name, X_train)
    pipeline.set_params(**params)
    if name == "XGBoost":
        # Multiclass sample weights use this fit's training labels only.
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=weights)
    else:
        # LR, SVM and RF factories have class_weight="balanced" configured.
        pipeline.fit(X_train, y_train)
    return pipeline


def expand_grid(grid):
    return list(ParameterGrid(grid))


def aligned_probabilities(estimator, X, class_ids):
    raw = estimator.predict_proba(X)
    aligned = np.zeros((len(X), len(class_ids)), dtype=float)
    for source, label in enumerate(estimator.classes_):
        aligned[:, int(label)] = raw[:, source]
    return aligned


def score_predictions(y_true, prediction, probabilities, class_ids):
    metrics = {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision_macro": precision_score(y_true, prediction, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, prediction, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, prediction, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, prediction, average="weighted", zero_division=0),
        "roc_auc_macro_ovr": roc_auc_score(
            y_true, probabilities, labels=class_ids, multi_class="ovr", average="macro"
        ),
        "roc_auc_weighted_ovr": roc_auc_score(
            y_true, probabilities, labels=class_ids, multi_class="ovr", average="weighted"
        ),
    }
    per_class = {}
    for class_id, class_name in enumerate(CLASS_ORDER):
        per_class[class_name] = {
            "precision": precision_score(
                y_true, prediction, labels=[class_id], average="macro", zero_division=0
            ),
            "recall": recall_score(
                y_true, prediction, labels=[class_id], average="macro", zero_division=0
            ),
            "f1": f1_score(y_true, prediction, labels=[class_id], average="macro", zero_division=0),
        }
    return metrics, per_class


def select_parameters_inner(name, X_outer_train, y_outer_train):
    """Select by mean inner Macro F1, then mean Balanced Accuracy."""
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    candidates = expand_grid(PARAMETER_GRIDS[name])
    candidate_results = []
    for candidate_index, params in enumerate(candidates):
        fold_scores = []
        for inner_train_idx, inner_valid_idx in inner.split(X_outer_train, y_outer_train):
            X_inner_train = X_outer_train.iloc[inner_train_idx]
            X_inner_valid = X_outer_train.iloc[inner_valid_idx]
            y_inner_train = y_outer_train.iloc[inner_train_idx]
            y_inner_valid = y_outer_train.iloc[inner_valid_idx]
            fitted = fit_pipeline(name, X_inner_train, y_inner_train, params)
            prediction = fitted.predict(X_inner_valid)
            fold_scores.append(
                {
                    "f1_macro": f1_score(y_inner_valid, prediction, average="macro", zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(y_inner_valid, prediction),
                }
            )
        candidate_results.append(
            {
                "candidate_index": candidate_index,
                "params": params,
                "inner_macro_f1_mean": float(np.mean([row["f1_macro"] for row in fold_scores])),
                "inner_macro_f1_std": float(np.std([row["f1_macro"] for row in fold_scores], ddof=1)),
                "inner_balanced_accuracy_mean": float(
                    np.mean([row["balanced_accuracy"] for row in fold_scores])
                ),
            }
        )
    # Stable sort means exact ties preserve the declared candidate order.
    candidate_results.sort(
        key=lambda row: (row["inner_macro_f1_mean"], row["inner_balanced_accuracy_mean"]),
        reverse=True,
    )
    return candidate_results[0], candidate_results


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_FILE)
    if TARGET not in data.columns:
        raise ValueError(f"Processed dataset does not contain {TARGET}")
    excluded = [column for column in EXCLUDED if column in data.columns]
    X = data.drop(columns=excluded).copy()
    if TARGET in X or any(column.startswith("PSS_Q") for column in X) or "PSS10_Total" in X:
        raise ValueError("Leakage audit failed: target-derived/PSS columns are predictors")
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(data[TARGET].astype(str)), index=data.index, name=TARGET)
    if encoder.classes_.tolist() != CLASS_ORDER:
        raise ValueError(f"Unexpected class labels/order: {encoder.classes_.tolist()}")
    class_ids = np.arange(len(encoder.classes_))

    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    fold_rows, prediction_rows, parameter_rows = [], [], []
    confusion_rows, report_rows = [], []

    for model_name in MODEL_NAMES:
        print(f"Nested Stratified Cross-Validation: {model_name}", flush=True)
        all_true, all_pred = [], []
        for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(outer.split(X, y), start=1):
            X_outer_train, X_outer_valid = X.iloc[outer_train_idx], X.iloc[outer_valid_idx]
            y_outer_train, y_outer_valid = y.iloc[outer_train_idx], y.iloc[outer_valid_idx]

            selected, candidate_results = select_parameters_inner(
                model_name, X_outer_train, y_outer_train
            )
            # Outer validation is first accessed only after the inner choice is fixed.
            final_fold_model = fit_pipeline(
                model_name, X_outer_train, y_outer_train, selected["params"]
            )
            prediction = final_fold_model.predict(X_outer_valid)
            probabilities = aligned_probabilities(final_fold_model, X_outer_valid, class_ids)
            metrics, per_class = score_predictions(
                y_outer_valid, prediction, probabilities, class_ids
            )
            fold_record = {
                "model": model_name,
                "outer_fold": outer_fold,
                "outer_train_n": len(outer_train_idx),
                "outer_validation_n": len(outer_valid_idx),
                **metrics,
            }
            for class_name, class_metrics in per_class.items():
                for metric_name, value in class_metrics.items():
                    fold_record[f"{class_name.lower().replace(' ', '_')}_{metric_name}"] = value
            fold_rows.append(fold_record)

            for row_position, sample_index in enumerate(outer_valid_idx):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "sample_index": int(sample_index),
                        "outer_fold": outer_fold,
                        "actual": encoder.classes_[int(y.iloc[sample_index])],
                        "predicted": encoder.classes_[int(prediction[row_position])],
                        **{
                            f"probability_{encoder.classes_[class_id]}": float(probabilities[row_position, class_id])
                            for class_id in class_ids
                        },
                    }
                )
            all_true.extend(y_outer_valid.tolist())
            all_pred.extend(prediction.tolist())

            parameter_rows.append(
                {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "selected_parameters": json.dumps(selected["params"], sort_keys=True),
                    "inner_macro_f1_mean": selected["inner_macro_f1_mean"],
                    "inner_macro_f1_std": selected["inner_macro_f1_std"],
                    "inner_balanced_accuracy_mean": selected["inner_balanced_accuracy_mean"],
                    "outer_macro_f1": metrics["f1_macro"],
                    "outer_balanced_accuracy": metrics["balanced_accuracy"],
                    "random_state": SEED,
                    "candidate_scores": json.dumps(candidate_results, sort_keys=True),
                }
            )

        aggregate_matrix = confusion_matrix(all_true, all_pred, labels=class_ids)
        for actual_id, actual_name in enumerate(encoder.classes_):
            for predicted_id, predicted_name in enumerate(encoder.classes_):
                confusion_rows.append(
                    {
                        "model": model_name,
                        "actual_class": actual_name,
                        "predicted_class": predicted_name,
                        "count": int(aggregate_matrix[actual_id, predicted_id]),
                    }
                )
        report = classification_report(
            all_true,
            all_pred,
            labels=class_ids,
            target_names=encoder.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        )
        for label, values in report.items():
            report_rows.append(
                {
                    "model": model_name,
                    "class_or_average": label,
                    **(values if isinstance(values, dict) else {"value": values}),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUTPUT / "nested_cv_fold_results.csv", index=False)
    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(OUTPUT / "nested_cv_predictions.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(OUTPUT / "nested_cv_confusion_matrix.csv", index=False)
    pd.DataFrame(report_rows).to_csv(OUTPUT / "nested_cv_classification_report.csv", index=False)
    parameters_df = pd.DataFrame(parameter_rows)
    parameters_df.to_csv(OUTPUT / "nested_cv_selected_parameters.csv", index=False)

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_macro_ovr",
        "roc_auc_weighted_ovr",
    ]
    result_rows = []
    for model_name, group in fold_df.groupby("model", sort=False):
        for metric in metrics:
            result_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "std": float(group[metric].std(ddof=1)),
                    "n_outer_folds": OUTER_SPLITS,
                    "ci_95_lower": np.nan,
                    "ci_95_upper": np.nan,
                    "ci_note": "Not reported: five overlapping outer-CV training sets do not support a reliable independent-fold CI.",
                }
            )
        for class_name in CLASS_ORDER:
            suffix = class_name.lower().replace(" ", "_")
            for metric in ["precision", "recall", "f1"]:
                column = f"{suffix}_{metric}"
                result_rows.append(
                    {
                        "model": model_name,
                        "metric": f"{class_name}_{metric}",
                        "mean": float(group[column].mean()),
                        "std": float(group[column].std(ddof=1)),
                        "n_outer_folds": OUTER_SPLITS,
                        "ci_95_lower": np.nan,
                        "ci_95_upper": np.nan,
                        "ci_note": "Not reported: five outer folds; exploratory fold variability only.",
                    }
                )
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(OUTPUT / "nested_cv_results.csv", index=False)

    # Highest outer mean Macro F1, Balanced Accuracy as explicit tie-breaker.
    model_summary = fold_df.groupby("model").agg(
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    )
    selected_model = model_summary.sort_values(
        ["f1_macro_mean", "balanced_accuracy_mean"], ascending=False, kind="mergesort"
    ).index[0]

    previous = compare_previous_results(result_df)
    previous.to_csv(OUTPUT / "nested_cv_comparison_with_previous_results.csv", index=False)
    plot_confusions(confusion_rows, encoder.classes_)

    model_results = {}
    for model_name in MODEL_NAMES:
        model_results[model_name] = {
            row.metric: {"mean": float(row.mean), "std": float(row.std)}
            for row in result_df[result_df.model == model_name].itertuples(index=False)
        }
    nested_summary = {
        "experiment": "Nested Stratified Cross-Validation",
        "selected_model_by_nested_outer_mean": selected_model,
        "selection_criterion": "Highest mean outer Macro F1; mean outer Balanced Accuracy is the tie-breaker.",
        "model_results": model_results,
        "previous_cv_comparison": previous.to_dict(orient="records"),
        "outer_predictions_per_model": int(len(predictions_df) // len(MODEL_NAMES)),
        "confidence_intervals": None,
        "confidence_interval_reason": "Not produced: five outer folds have overlapping training sets; fold SD is descriptive.",
        "internal_validation_only": True,
        "external_validation": False,
    }
    (OUTPUT / "nested_cv_summary.json").write_text(
        json.dumps(nested_summary, indent=2), encoding="utf-8"
    )

    metadata = {
        "experiment": "Nested Stratified Cross-Validation",
        "purpose": "Internal validation with inner-fold model selection and outer-fold evaluation; not external validation.",
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "target": TARGET,
        "excluded_predictors": excluded,
        "predictors": list(X.columns),
        "outer_cv": {"type": "StratifiedKFold", "n_splits": OUTER_SPLITS, "shuffle": True, "random_state": SEED},
        "inner_cv": {"type": "StratifiedKFold", "n_splits": INNER_SPLITS, "shuffle": True, "random_state": SEED},
        "selection_rule": "Highest mean inner-CV Macro F1; mean inner Balanced Accuracy breaks ties.",
        "selected_model_by_outer_mean": selected_model,
        "outer_mean_selection_rule": "Highest mean outer Macro F1; outer Balanced Accuracy is the tie-breaker; interpret model ranking as exploratory due to comparing four candidates.",
        "parameter_grids": {name: expand_grid(grid) for name, grid in PARAMETER_GRIDS.items()},
        "imbalance_handling": {
            "logistic_regression": "class_weight='balanced'",
            "svm": "class_weight='balanced'",
            "random_forest": "class_weight='balanced'",
            "xgboost": "compute_sample_weight('balanced') from each fit's training labels only; no scale_pos_weight",
        },
        "preprocessing": "Imputation, scaling, and one-hot encoding are in each estimator pipeline and fitted on the current fit partition only.",
        "confidence_intervals": "Not produced; five outer folds have overlapping training sets and are insufficient for an independent-fold interval. Fold SD is reported descriptively.",
        "random_seeds": {"outer_cv": SEED, "inner_cv": SEED, "estimator_random_state": RANDOM_STATE},
        "command": ".venv\\Scripts\\python.exe -m experiments.nested_stratified_cv",
        "python_version": sys.version.split()[0],
    }
    (METRICS / "nested_cv_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_research_summary(result_df, previous, selected_model, metadata)
    print(f"Selected by nested-CV Macro F1: {selected_model}", flush=True)
    print(f"Nested CV artifacts: {OUTPUT}", flush=True)


def compare_previous_results(nested_results):
    previous_sources = {
        "Baseline 10-fold CV": ROOT / "results" / "metrics" / "cross_validation_10fold.csv",
        "Imbalance-aware 10-fold CV": ROOT / "results" / "imbalance_aware" / "metrics" / "imbalance_aware_cv_results.csv",
    }
    rows = []
    requested_metrics = {
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }
    nested = nested_results[nested_results.metric.isin(requested_metrics)]
    for source, path in previous_sources.items():
        previous = pd.read_csv(path)
        for _, row in nested.iterrows():
            metric = requested_metrics[row["metric"]]
            matched = previous[(previous.model == row["model"]) & (previous.metric == metric)]
            if matched.empty:
                raise ValueError(f"Missing previous result for {source}: {row['model']} / {metric}")
            previous_row = matched.iloc[0]
            rows.append(
                {
                    "model": row["model"],
                    "metric": metric,
                    "nested_cv_mean": row["mean"],
                    "nested_cv_std": row["std"],
                    "previous_experiment": source,
                    "previous_10fold_mean": float(previous_row["mean"]),
                    "previous_10fold_std": float(previous_row["std"]),
                    "mean_difference_nested_minus_previous": float(row["mean"] - previous_row["mean"]),
                    "note": "Different CV design; descriptive comparison only, not a paired significance test.",
                }
            )
    return pd.DataFrame(rows)


def plot_confusions(rows, class_names):
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, model_name in zip(axes.flat, MODEL_NAMES):
        table = frame[frame.model == model_name].pivot(
            index="actual_class", columns="predicted_class", values="count"
        ).reindex(index=class_names, columns=class_names)
        sns.heatmap(table, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(model_name)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
    fig.suptitle("Nested Stratified Cross-Validation: aggregate outer-fold confusion matrices")
    fig.tight_layout()
    fig.savefig(FIGURES / "nested_cv_aggregate_confusion_matrices.png", dpi=160)
    plt.close(fig)


def write_research_summary(results, comparison, selected_model, metadata):
    main = results[results.metric.isin(["f1_macro", "balanced_accuracy"])]
    pivot = main.pivot(index="model", columns="metric", values=["mean", "std"])
    predictions = pd.read_csv(OUTPUT / "nested_cv_predictions.csv")
    summary_lines = [
        "# Nested Stratified Cross-Validation",
        "",
        "## Purpose",
        "",
        "This experiment estimates candidate-model performance using nested stratified cross-validation. Nested CV is internal validation, not independent external validation; it does not establish performance at other universities or populations.",
        "",
        "## Method",
        "",
        f"The processed dataset contains {len(predictions) // len(MODEL_NAMES)} observations. Outer and inner loops both use 5-fold StratifiedKFold with shuffle=True and random_state={SEED}. Each outer fold is held out while hyperparameters are selected using only its outer-training partition.",
        "",
        "## Leakage prevention",
        "",
        "Stress_Level is the target. PSS_Q1–PSS_Q10, PSS10_Total, Timestamp, and Consent are excluded from predictors. Imputation, scaling, and one-hot encoding are fitted in the pipeline on each fit's training partition. XGBoost sample weights are recalculated from that fit's training labels only. No binary scale_pos_weight is used.",
        "",
        "## Model selection procedure",
        "",
        "Each outer training partition uses a 5-fold stratified inner CV. A small fixed two-candidate grid is evaluated for each model. Mean inner Macro F1 is primary and mean inner Balanced Accuracy is the tie-breaker. The selected parameters are then refit on the full outer training partition and evaluated once on its outer validation fold.",
        "",
        "## Nested CV results",
        "",
        "Mean ± SD across five outer folds:",
        "",
        "| Model | Macro F1 | Balanced Accuracy | Low recall | High recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        group = results[results.model == model_name].set_index("metric")
        fmt = lambda metric: f"{group.loc[metric, 'mean']:.4f} ± {group.loc[metric, 'std']:.4f}"
        summary_lines.append(
            f"| {model_name} | {fmt('f1_macro')} | {fmt('balanced_accuracy')} | {fmt('Low Stress_recall')} | {fmt('High Stress_recall')} |"
        )
    summary_lines.extend(
        [
            "",
            f"By the primary outer mean Macro F1 criterion (Balanced Accuracy tie-breaker), **{selected_model}** ranks highest among these candidates. Since the same outer estimates are compared across four model families, interpret the winner ranking as exploratory; nested CV protects hyperparameter selection within each family but does not remove all model-family selection optimism.",
            "",
            "No 95% confidence intervals are reported. Five outer folds have overlapping training sets; treating these five scores as independent observations would not justify a conventional fold-based interval. The reported SD describes fold-to-fold variation only.",
            "",
            "## Comparison with previous CV",
            "",
            "The CSV compares nested outer means with baseline and imbalance-aware 10-fold summaries. These are different CV designs and are reported descriptively, not as paired tests.",
            "",
            "| Model | Nested Macro F1 | Baseline 10-fold Macro F1 | Imbalance-aware 10-fold Macro F1 | Nested Balanced Accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    compare = comparison[comparison.metric == "f1_macro"]
    for model_name in MODEL_NAMES:
        row = compare[compare.model == model_name]
        baseline = row[row.previous_experiment == "Baseline 10-fold CV"].iloc[0]
        aware = row[row.previous_experiment == "Imbalance-aware 10-fold CV"].iloc[0]
        nested_f1 = results[(results.model == model_name) & (results.metric == "f1_macro")].iloc[0]
        nested_ba = results[(results.model == model_name) & (results.metric == "balanced_accuracy")].iloc[0]
        summary_lines.append(
            f"| {model_name} | {nested_f1['mean']:.4f} ± {nested_f1['std']:.4f} | {baseline['previous_10fold_mean']:.4f} ± {baseline['previous_10fold_std']:.4f} | {aware['previous_10fold_mean']:.4f} ± {aware['previous_10fold_std']:.4f} | {nested_ba['mean']:.4f} ± {nested_ba['std']:.4f} |"
        )
    summary_lines.extend(
        [
            "",
            "## Minority-class performance",
            "",
            "Low and High Stress precision, recall, and F1 are reported as outer-fold mean ± SD in `nested_cv_results.csv`; aggregate out-of-fold reports and confusion matrices are also saved. Interpret minority-class values in light of their smaller supports.",
            "",
            "## Limitations",
            "",
            "Nested CV is internal validation, not external validation. The dataset is from one study sample; these results do not demonstrate generalization to other universities or populations. Only two predeclared candidates per model were searched, so the selected settings are conditional on these small grids. Comparing four model families on the same outer folds may introduce winner-selection optimism. Five outer folds provide a limited view of performance variability, and no fold-independence confidence intervals are claimed.",
            "",
            f"Reproduction command: `{metadata['command']}`",
            "",
        ]
    )
    (OUTPUT / "research_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
