"""Cached Kernel SHAP explanation service for the selected SVM."""

from __future__ import annotations

import joblib
import numpy as np
import shap


class ExplanationService:
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service
        artifacts = prediction_service.artifacts_dir
        background_artifact = joblib.load(artifacts / "shap_background.joblib")
        self.background = np.asarray(background_artifact["data"], dtype=np.float64)
        self.transformed_feature_names = background_artifact["feature_names"]
        classifier = prediction_service.pipeline.named_steps["classifier"]
        # Construct once and reuse the stored training-only background for requests.
        self.explainer = shap.KernelExplainer(classifier.predict_proba, self.background)
        self.raw_feature_names = prediction_service.features

    def explain(self, validated_frame, prediction):
        transformed = self.prediction_service.transform(validated_frame)
        values = self.explainer.shap_values(transformed, nsamples=100)
        class_id = self.prediction_service.classes.index(prediction["stress_level"])
        class_values = self._class_values(values, class_id, len(self.raw_feature_names))
        grouped = {name: 0.0 for name in self.raw_feature_names}
        for transformed_name, value in zip(self.transformed_feature_names, class_values):
            original_name = self._original_feature(transformed_name)
            grouped[original_name] += float(value)
        contributions = []
        for feature, value in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True):
            if abs(value) < 1e-12:
                continue
            contributions.append(
                {
                    "feature": feature,
                    "shap_value": float(value),
                    "direction": "increases_prediction" if value > 0 else "decreases_prediction",
                    "interpretation": "contributes to the model prediction for the explained class",
                }
            )
            if len(contributions) == 10:
                break
        return {
            "explained_class": prediction["stress_level"],
            "method": "SHAP KernelExplainer on SVM class probabilities",
            "explanation": contributions,
            "note": "Contributions describe model behavior and are not causal effects.",
        }

    def _class_values(self, values, class_id, original_feature_count):
        class_count = len(self.prediction_service.classes)
        feature_count = len(self.transformed_feature_names)
        if isinstance(values, list):
            return np.asarray(values[class_id])[0].reshape(-1)
        array = np.asarray(values)
        if array.ndim == 3:
            if array.shape == (1, feature_count, class_count):
                result = array[0, :, class_id]
            elif array.shape == (class_count, 1, feature_count):
                result = array[class_id, 0, :]
            else:
                raise ValueError(f"Unexpected multiclass SHAP shape: {array.shape}")
        elif array.ndim == 2:
            if array.shape == (1, feature_count):
                result = array[0]
            elif array.shape == (feature_count, class_count):
                result = array[:, class_id]
            else:
                raise ValueError(f"Unexpected SHAP shape: {array.shape}")
        else:
            raise ValueError(f"Unexpected SHAP shape: {array.shape}")
        if len(result) != feature_count:
            raise ValueError(
                f"SHAP returned {len(result)} transformed values; expected {feature_count}"
            )
        return np.asarray(result, dtype=float)

    def _original_feature(self, transformed_name):
        if transformed_name.startswith("numeric__"):
            name = transformed_name.removeprefix("numeric__")
            if name in self.raw_feature_names:
                return name
        if transformed_name.startswith("categorical__"):
            encoded_name = transformed_name.removeprefix("categorical__")
            candidates = [
                feature
                for feature in self.raw_feature_names
                if encoded_name.startswith(f"{feature}_")
            ]
            if candidates:
                return max(candidates, key=len)
        raise ValueError(f"Cannot map transformed feature to input field: {transformed_name}")
