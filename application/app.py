"""Flask REST API for the stress prediction research prototype."""

from __future__ import annotations

import json
import logging
import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from application.services.explanation_service import ExplanationService
from application.services.prediction_service import InputValidationError, PredictionService

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FRONTEND = PROJECT_ROOT / "frontend"
RESULTS = PROJECT_ROOT / "results"
MODEL_ORDER = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_analytics_data(results_root=RESULTS):
    """Load already-generated research outputs without fitting or recalculating models."""
    root = Path(results_root)
    output = {"status": "ok", "errors": []}

    def read_json(relative):
        return json.loads((root / relative).read_text(encoding="utf-8"))

    def read_rows(relative):
        return _read_csv(root / relative)

    try:
        rows = read_rows("metrics/class_distribution.csv")
        counts = {row["Stress_Level"]: int(row["count"]) for row in rows}
        total = sum(counts.values())
        output["dataset"] = {
            "total_students": total,
            "classes": [
                {"name": name, "count": counts[name], "proportion": float(next(r["proportion"] for r in rows if r["Stress_Level"] == name))}
                for name in ["Low Stress", "Moderate Stress", "High Stress"]
            ],
            "source": "results/metrics/class_distribution.csv",
        }
    except Exception as exc:
        output["dataset"] = None
        output["errors"].append(f"Dataset overview unavailable: {exc}")

    try:
        nested = read_json("nested_cv/nested_cv_summary.json")
        metadata = read_json("nested_cv/metrics/nested_cv_metadata.json")
        rows = read_rows("nested_cv/nested_cv_results.csv")
        selected_name = nested.get("selected_model_by_nested_outer_mean", nested.get("selected_model_by_outer_mean"))
        if not selected_name:
            raise KeyError("selected model is not recorded in the nested-CV summary")
        model_rows = {}
        for row in rows:
            model_rows.setdefault(row["model"], {})[row["metric"]] = {
                "mean": float(row["mean"]), "std": float(row["std"]),
            }
        selected = model_rows[selected_name]
        output["selected_model"] = {
            "name": selected_name,
            "criterion_label": "Nested CV Macro F1",
            "selection_criterion": metadata["outer_mean_selection_rule"],
            "macro_f1": selected["f1_macro"],
            "balanced_accuracy": selected["balanced_accuracy"],
            "source": "results/nested_cv/nested_cv_results.csv and nested_cv_summary.json",
        }
        output["nested_cv_folds"] = {
            "outer": int(metadata["outer_cv"]["n_splits"]),
            "inner": int(metadata["inner_cv"]["n_splits"]),
        }
        output["model_comparison"] = [
            {
                "model": name,
                "macro_f1": model_rows[name]["f1_macro"],
                "balanced_accuracy": model_rows[name]["balanced_accuracy"],
            }
            for name in MODEL_ORDER if name in model_rows
        ]
        if len(output["model_comparison"]) != len(MODEL_ORDER):
            raise ValueError("Nested-CV results do not contain all four comparison models.")
    except Exception as exc:
        output["selected_model"] = None
        output["model_comparison"] = None
        output["errors"].append(f"Nested-CV metrics unavailable: {exc}")

    try:
        rows = read_rows("imbalance_aware/ablation/ablation_results.csv")
        output["ablation"] = [
            {
                "configuration": row["experiment"],
                "feature_groups": row["feature_groups"],
                "features": json.loads(row["features"]),
                "n_features": int(row["n_features"]),
                "macro_f1": {"mean": float(row["f1_macro_mean"]), "std": float(row["f1_macro_std"])},
                "balanced_accuracy": {"mean": float(row["balanced_accuracy_mean"]), "std": float(row["balanced_accuracy_std"])},
            }
            for row in rows
        ]
        if len(output["ablation"]) != 7:
            raise ValueError(f"Expected seven generated feature configurations; found {len(output['ablation'])}.")
    except Exception as exc:
        output["ablation"] = None
        output["errors"].append(f"Ablation results unavailable: {exc}")

    try:
        shap_meta = read_json("imbalance_aware/shap/shap_metadata.json")
        rows = read_rows("imbalance_aware/shap/global_feature_importance.csv")
        importance = sorted(
            ({"feature": row["feature"], "mean_absolute_shap_value": float(row["mean_absolute_shap_value"])} for row in rows),
            key=lambda row: row["mean_absolute_shap_value"], reverse=True,
        )
        output["shap"] = {
            "selected_model": shap_meta["selected_model"],
            "explainer": shap_meta["explainer"],
            "features": importance[:10],
            "source": "results/imbalance_aware/shap/global_feature_importance.csv",
            "note": "SHAP values describe model contributions and do not establish causality.",
        }
    except Exception as exc:
        output["shap"] = None
        output["errors"].append(f"Global SHAP importance unavailable: {exc}")

    validation_files = {
        "stratified_10_fold_cv": "metrics/cross_validation_10fold.csv",
        "nested_5x5_stratified_cv": "nested_cv/metrics/nested_cv_metadata.json",
        "multiple_seed_robustness": "imbalance_aware/robustness/robustness_results.csv",
        "held_out_test_evaluation": "imbalance_aware/metrics/imbalance_aware_model_comparison.csv",
    }
    output["validation_methods"] = [
        {"key": key, "available": (root / file).is_file()}
        for key, file in validation_files.items()
    ]
    try:
        rows = read_rows("imbalance_aware/metrics/per_class_test_metrics.csv")
        selected_name = (output.get("selected_model") or {}).get("name", "SVM")
        recalls = {
            row["class"]: float(row["recall"])
            for row in rows if row["model"] == selected_name and row["split"] == "test"
        }
        output["minority_class_recall"] = {
            name: recalls.get(name) for name in ["Low Stress", "High Stress"]
        }
    except Exception as exc:
        output["minority_class_recall"] = None
        output["errors"].append(f"Held-out per-class recall unavailable: {exc}")

    output["limitations"] = [
        "Dataset represents the collected Sri Lankan university-student sample.",
        "Nested cross-validation is internal validation and does not constitute external validation.",
        "The system is a research prototype and does not provide a clinical diagnosis.",
        "SHAP explanations describe model contributions, not causal explanations.",
    ]
    if output["errors"]:
        output["status"] = "partial"
    return output


def create_app():
    api = Flask(__name__)
    api.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    api.config["JSON_SORT_KEYS"] = False
    api.logger.setLevel(logging.INFO)

    api.extensions["prediction_service"] = None
    api.extensions["prediction_load_error"] = None
    api.extensions["explanation_service"] = None
    api.extensions["explanation_load_error"] = None
    api.extensions["metadata"] = None
    try:
        api.extensions["metadata"] = json.loads(
            (ARTIFACTS / "model_metadata.json").read_text(encoding="utf-8")
        )
        api.extensions["prediction_service"] = PredictionService(ARTIFACTS)
        api.logger.info("Final SVM pipeline loaded from application artifacts.")
    except Exception as exc:  # Keep server able to report unhealthy status.
        api.extensions["prediction_load_error"] = exc
        api.logger.exception("Could not load final SVM model artifacts.")

    @api.get("/")
    def frontend_index():
        return send_from_directory(FRONTEND, "index.html")

    @api.get("/analytics")
    def analytics_index():
        return send_from_directory(FRONTEND, "analytics.html")

    @api.get("/<path:asset_name>")
    def frontend_asset(asset_name):
        # Only serve the deliberate UI assets; never expose arbitrary files.
        if asset_name in {"style.css", "app.js", "analytics.css", "analytics.js", "visualizations.css", "recommendations.js", "profile-comparison.js", "theme.js"}:
            return send_from_directory(FRONTEND, asset_name)
        return not_found(None)

    @api.get("/api/health")
    def health():
        loaded = api.extensions["prediction_service"] is not None
        response = {
            "status": "ok" if loaded else "error",
            "service": "stress-prediction-api",
            "model_loaded": loaded,
        }
        return jsonify(response), (200 if loaded else 503)

    @api.post("/api/predict")
    def predict():
        service = api.extensions["prediction_service"]
        if service is None:
            return jsonify({"error": "Model unavailable"}), 503
        try:
            payload = get_json_payload()
            validated = service.validate(payload)
            prediction = service.predict(validated)
            prediction["input_quality"] = service.assess_input_quality(validated)
            return jsonify(prediction), 200
        except InputValidationError as exc:
            return jsonify({"error": "Invalid input", "details": exc.details}), 400
        except BadRequest:
            return jsonify({"error": "Invalid input", "details": ["Malformed JSON request body."]}), 400
        except Exception:
            api.logger.exception("Prediction request failed.")
            return jsonify({"error": "Internal prediction error"}), 500

    @api.post("/api/explain")
    def explain():
        service = api.extensions["prediction_service"]
        if service is None:
            return jsonify({"error": "Model unavailable"}), 503
        try:
            payload = get_json_payload()
            validated = service.validate(payload)
            prediction = service.predict(validated)
            explainer = api.extensions["explanation_service"]
            if explainer is None:
                try:
                    explainer = ExplanationService(service)
                    api.extensions["explanation_service"] = explainer
                except Exception as exc:
                    api.extensions["explanation_load_error"] = exc
                    raise
            explanation = explainer.explain(validated, prediction)
            return jsonify({**prediction, **explanation}), 200
        except InputValidationError as exc:
            return jsonify({"error": "Invalid input", "details": exc.details}), 400
        except BadRequest:
            return jsonify({"error": "Invalid input", "details": ["Malformed JSON request body."]}), 400
        except Exception:
            api.logger.exception("Explanation request failed.")
            return jsonify({"error": "Internal prediction error"}), 500

    @api.get("/api/model-info")
    def model_info():
        metadata = api.extensions["metadata"]
        if metadata is None or api.extensions["prediction_service"] is None:
            return jsonify({"error": "Model unavailable"}), 503
        return jsonify(
            {
                "model": metadata["model"],
                "model_type": metadata["model_type"],
                "target": metadata["target_name"],
                "classes": metadata["classes"],
                "features": metadata["feature_names"],
                "research_evaluation": metadata["research_evaluation_metrics"],
                "model_selection_criterion": metadata["model_selection_criterion"],
                "notice": metadata["research_notice"],
            }
        ), 200

    @api.get("/api/input-schema")
    def input_schema():
        service = api.extensions["prediction_service"]
        if service is None:
            return jsonify({"error": "Model unavailable"}), 503
        return jsonify({
            "features": service.features,
            "numeric_ranges": service.numeric_ranges,
            "numeric_distributions": service.numeric_distributions,
            "categorical_values": service.categorical_values,
            "categorical_frequencies": service.categorical_frequencies,
            "training_dataset_size": service.training_sample_size,
            "input_quality_method": service.input_quality_method,
        }), 200

    @api.get("/api/example-input")
    def example_input():
        service = api.extensions["prediction_service"]
        if service is None:
            return jsonify({"error": "Model unavailable"}), 503
        # Build a synthetic example from aggregate metadata; never return a
        # participant-level row from the research dataset.
        example = {}
        for feature in service.numeric_ranges:
            median = service.numeric_distributions[feature]["median"]
            example[feature] = int(median) if service.numeric_ranges[feature]["integer_only"] else float(median)
        for feature in service.categorical_values:
            counts = service.categorical_frequencies[feature]
            example[feature] = max(counts, key=counts.get)
        for feature in service.numeric_ranges:
            example[feature] = int(example[feature]) if service.numeric_ranges[feature]["integer_only"] else float(example[feature])
        validated = service.validate(example)
        return jsonify({"label": "Synthetic example using training-set medians and modes", "example": validated.iloc[0].to_dict()}), 200

    @api.get("/api/reference-profile")
    def reference_profile():
        service = api.extensions["prediction_service"]
        if service is None:
            return jsonify({"error": "Model unavailable"}), 503
        values = {}
        for feature in service.features:
            if feature in service.categorical_values:
                values[feature] = str(service.training_reference[feature].mode().iloc[0])
            else:
                median = service.numeric_distributions[feature]["median"]
                values[feature] = int(median) if service.numeric_ranges[feature]["integer_only"] else float(median)
        validated = service.validate(values)
        return jsonify({
            "label": "Reference Dataset Profile",
            "method": "Numeric predictors use processed-dataset medians; categorical predictors use the most frequent observed value.",
            "profile": validated.iloc[0].to_dict(),
            "prediction": service.predict(validated),
        }), 200

    @api.get("/api/research-analytics")
    def research_analytics():
        return jsonify(load_analytics_data(RESULTS)), 200

    @api.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Endpoint not found"}), 404

    @api.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error):
        return jsonify({"error": "Request too large", "details": ["Maximum request size is 16 KB."]}), 413

    @api.errorhandler(500)
    def internal_error(_error):
        api.logger.exception("Unhandled API error.")
        return jsonify({"error": "Internal prediction error"}), 500

    return api


def get_json_payload():
    if not request.is_json:
        raise BadRequest("Expected application/json")
    return request.get_json(silent=False)


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
