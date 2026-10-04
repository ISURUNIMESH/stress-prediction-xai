"""Load and validate requests for the exported stress-prediction pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class InputValidationError(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__("Invalid input")


class PredictionService:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.pipeline = joblib.load(self.artifacts_dir / "final_svm_pipeline.joblib")
        self.metadata = json.loads(
            (self.artifacts_dir / "model_metadata.json").read_text(encoding="utf-8")
        )
        self.features = self.metadata["feature_names"]
        self.classes = self.metadata["classes"]
        self.categorical_values = self.metadata["categorical_values"]
        self.numeric_ranges = self.metadata["numeric_ranges"]
        self.numeric_distributions = self.metadata.get("numeric_distributions", {})
        self.categorical_frequencies = self.metadata.get("categorical_frequencies", {})
        self.training_sample_size = int(self.metadata["training_dataset_size"])
        self.input_quality_method = (
            "Category support and marginal frequencies; low-frequency values flagged by the empirical Tukey lower fence (Q1 minus 1.5 times IQR); numeric Tukey 1.5-times-IQR fences; mixed-type nearest-neighbor distance flagged above the empirical 95th percentile of leave-one-out training distances."
        )
        processed_path = self.artifacts_dir.parent.parent / "dataset" / "processed" / "cleaned_stress_dataset.csv"
        if not processed_path.exists():
            raise FileNotFoundError(f"Training reference data not found: {processed_path}")
        self.training_reference = pd.read_csv(processed_path, usecols=self.features)
        if len(self.training_reference) != self.training_sample_size:
            raise ValueError("Training reference size does not match exported model metadata.")
        for feature, allowed in self.categorical_values.items():
            observed = sorted(self.training_reference[feature].dropna().astype(str).unique().tolist())
            if observed != sorted(allowed):
                raise ValueError(f"Training reference category mismatch for {feature}.")
        for feature, bounds in self.numeric_ranges.items():
            values = pd.to_numeric(self.training_reference[feature], errors="raise")
            if float(values.min()) != float(bounds["minimum"]) or float(values.max()) != float(bounds["maximum"]):
                raise ValueError(f"Training reference range mismatch for {feature}.")
        if not self.numeric_distributions:
            self.numeric_distributions = {}
            for feature in self.numeric_ranges:
                values = pd.to_numeric(self.training_reference[feature], errors="raise")
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                self.numeric_distributions[feature] = {
                    "median": float(values.median()),
                    "mean": float(values.mean()),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr_fence_minimum": float(q1 - 1.5 * iqr),
                    "iqr_fence_maximum": float(q3 + 1.5 * iqr),
                    "value_counts": {str(k): int(v) for k, v in values.value_counts().sort_index().items()},
                }
        if not self.categorical_frequencies:
            self.categorical_frequencies = {
                feature: {str(k): int(v) for k, v in self.training_reference[feature].astype(str).value_counts().items()}
                for feature in self.categorical_values
            }
        self.training_reference_distance_p95 = self._training_nearest_neighbor_p95()
        self.marginal_frequency_lower_fences = {}
        for feature in self.features:
            if feature in self.categorical_values:
                counts = np.asarray(list(self.categorical_frequencies.get(feature, {}).values()), dtype=float)
            else:
                counts = np.asarray(list(self.numeric_distributions[feature]["value_counts"].values()), dtype=float)
            if len(counts):
                q1, q3 = np.quantile(counts, [0.25, 0.75])
                self.marginal_frequency_lower_fences[feature] = float(q1 - 1.5 * (q3 - q1))

    def validate(self, payload):
        if not isinstance(payload, dict):
            raise InputValidationError(["JSON request body must be an object."])
        details = []
        missing = [feature for feature in self.features if feature not in payload]
        unknown = [feature for feature in payload if feature not in self.features]
        if missing:
            details.append(f"Missing required fields: {', '.join(missing)}")
        if unknown:
            details.append(f"Unknown or prohibited fields: {', '.join(map(str, unknown))}")
        if details:
            raise InputValidationError(details)

        row = {}
        for feature in self.features:
            value = payload[feature]
            if feature in self.categorical_values:
                allowed = self.categorical_values[feature]
                if not isinstance(value, str) or value not in allowed:
                    details.append(
                        f"{feature} must be one of the observed values: {', '.join(allowed)}"
                    )
                else:
                    row[feature] = value
                continue

            bounds = self.numeric_ranges[feature]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                details.append(f"{feature} must be a JSON number.")
                continue
            if not math.isfinite(float(value)):
                details.append(f"{feature} must be finite.")
                continue
            if value < bounds["minimum"] or value > bounds["maximum"]:
                details.append(
                    f"{feature} must be between {bounds['minimum']:g} and {bounds['maximum']:g}."
                )
                continue
            if bounds["integer_only"] and not float(value).is_integer():
                details.append(f"{feature} must be an integer based on the observed dataset values.")
                continue
            row[feature] = value

        if details:
            raise InputValidationError(details)
        return pd.DataFrame([row], columns=self.features)

    def predict(self, validated_frame):
        predicted_id = int(self.pipeline.predict(validated_frame)[0])
        probabilities = self.pipeline.predict_proba(validated_frame)[0]
        response_probabilities = {
            name: float(probabilities[index]) for index, name in enumerate(self.classes)
        }
        return {
            "stress_level": self.classes[predicted_id],
            "probabilities": response_probabilities,
        }

    def assess_input_quality(self, validated_frame):
        """Flag sparse marginal values and unusual numeric combinations.

        The OOD screen uses Tukey's 1.5*IQR fences per numeric feature,
        exact observed-category membership, a Tukey lower fence on observed
        value counts, and an empirical nearest-neighbor cutoff. These
        are transparent descriptive support checks, not learned decision
        boundaries, and never alter validation or the model prediction.
        """
        row = validated_frame.iloc[0]

        uncommon = []
        out_of_pattern = []
        for feature, value in row.items():
            if feature in self.categorical_values:
                count = self.categorical_frequencies.get(feature, {}).get(str(value), 0)
                if count == 0:
                    out_of_pattern.append(feature)
                elif count < self.marginal_frequency_lower_fences.get(feature, -1):
                    uncommon.append(feature)
                continue

            distribution = self.numeric_distributions.get(feature)
            if not distribution:
                continue
            number = float(value)
            if number < distribution["iqr_fence_minimum"] or number > distribution["iqr_fence_maximum"]:
                out_of_pattern.append(feature)
            count = distribution["value_counts"].get(str(int(number) if number.is_integer() else number), 0)
            if count < self.marginal_frequency_lower_fences.get(feature, -1):
                uncommon.append(feature)

        submitted_distance = self._distance_to_training(row)
        nearest_reference_distance = float(np.min(submitted_distance))
        unusual_combination = nearest_reference_distance > self.training_reference_distance_p95
        if unusual_combination:
            out_of_pattern.append("input_combination")

        caution = bool(uncommon or out_of_pattern)
        reasons = []
        if any(feature != "input_combination" for feature in out_of_pattern):
            reasons.append("One or more numeric features fall outside the training-data 1.5-times-IQR fences.")
        if unusual_combination:
            reasons.append("The mixed-type nearest-neighbor distance exceeds the empirical training-data 95th-percentile cutoff.")
        if uncommon:
            reasons.append("One or more values have frequency below the training-data Tukey lower fence.")
        return {
            "status": "Caution" if caution else "Good",
            "classification": "UNUSUAL" if caution else "VALID",
            "within_common_training_patterns": not caution,
            "uncommon_features": uncommon,
            "outside_distribution_features": out_of_pattern,
            "message": (
                "Input differs from patterns commonly observed in the training dataset. Interpret the prediction cautiously."
                if caution else "Input is within the observed training-data patterns."
            ),
            "method": self.input_quality_method,
            "marginal_frequency_lower_fences": self.marginal_frequency_lower_fences,
            "nearest_training_row_distance": nearest_reference_distance,
            "training_leave_one_out_distance_p95": self.training_reference_distance_p95,
            "unusual_feature_combination": unusual_combination,
            "reasons": reasons,
        }

    def _distance_to_training(self, row):
        n = len(self.training_reference)
        distance_sum = np.zeros(n, dtype=np.float64)
        feature_count = 0
        for feature, bounds in self.numeric_ranges.items():
            scale = max(float(bounds["maximum"]) - float(bounds["minimum"]), 1.0)
            reference = pd.to_numeric(self.training_reference[feature], errors="raise").to_numpy(dtype=float)
            distance_sum += np.abs(reference - float(row[feature])) / scale
            feature_count += 1
        for feature in self.categorical_values:
            reference = self.training_reference[feature].astype(str).to_numpy()
            distance_sum += (reference != str(row[feature])).astype(float)
            feature_count += 1
        return distance_sum / feature_count

    def _training_nearest_neighbor_p95(self):
        n = len(self.training_reference)
        if n < 2:
            return 0.0
        distance_matrix = np.zeros((n, n), dtype=np.float64)
        feature_count = 0
        for feature, bounds in self.numeric_ranges.items():
            scale = max(float(bounds["maximum"]) - float(bounds["minimum"]), 1.0)
            values = pd.to_numeric(self.training_reference[feature], errors="raise").to_numpy(dtype=float)
            distance_matrix += np.abs(values[:, None] - values[None, :]) / scale
            feature_count += 1
        for feature in self.categorical_values:
            values = self.training_reference[feature].astype(str).to_numpy()
            distance_matrix += (values[:, None] != values[None, :]).astype(float)
            feature_count += 1
        distance_matrix /= feature_count
        np.fill_diagonal(distance_matrix, np.inf)
        nearest = np.min(distance_matrix, axis=1)
        return float(np.quantile(nearest, 0.95))

    def transform(self, validated_frame):
        transformed = self.pipeline.named_steps["preprocess"].transform(validated_frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=np.float64)
