"""API tests use the exported artifact; they never retrain a model."""

import json
import unittest

from application.app import app


def valid_payload():
    metadata = app.extensions["metadata"]
    payload = {}
    for feature, bounds in metadata["numeric_ranges"].items():
        payload[feature] = int(bounds["minimum"])
    for feature, values in metadata["categorical_values"].items():
        payload[feature] = values[0]
    return payload


class StressPredictionAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if app.extensions["prediction_service"] is None:
            raise RuntimeError("Train/export application artifacts before running API tests")
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")
        self.assertTrue(response.get_json()["model_loaded"])

    def test_valid_prediction(self):
        response = self.client.post("/api/predict", json=valid_payload())
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn(data["stress_level"], app.extensions["metadata"]["classes"])
        self.assertAlmostEqual(sum(data["probabilities"].values()), 1.0, places=5)

    def test_typical_training_medians_predict_and_report_input_quality(self):
        service = app.extensions["prediction_service"]
        payload = {}
        for feature in service.numeric_ranges:
            if feature in service.numeric_distributions:
                payload[feature] = int(service.numeric_distributions[feature]["median"])
        for feature in service.categorical_values:
            counts = service.categorical_frequencies[feature]
            payload[feature] = max(counts, key=counts.get)
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertIn(response.get_json()["input_quality"]["status"], {"Good", "Caution"})

    def test_minimum_and_maximum_valid_age(self):
        for age in [18, 30]:
            payload = valid_payload()
            payload["Age"] = age
            response = self.client.post("/api/predict", json=payload)
            self.assertEqual(response.status_code, 200, response.get_json())

    def test_unusual_combination_warns_without_changing_prediction_contract(self):
        payload = valid_payload()
        for feature, bounds in app.extensions["metadata"]["numeric_ranges"].items():
            payload[feature] = int(bounds["minimum"])
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertIn(result["stress_level"], app.extensions["metadata"]["classes"])
        self.assertEqual(result["input_quality"]["status"], "Caution")
        self.assertTrue(result["input_quality"]["unusual_feature_combination"])

    def test_example_input_uses_aggregate_values_not_a_participant_row(self):
        response = self.client.get("/api/example-input")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["label"], "Synthetic example using training-set medians and modes")
        self.assertEqual(set(data["example"]), set(app.extensions["metadata"]["feature_names"]))
        self.assertFalse(any(name.startswith("PSS_") or name == "PSS10_Total" for name in data["example"]))
        service = app.extensions["prediction_service"]
        for feature in service.numeric_distributions:
            self.assertEqual(data["example"][feature], service.numeric_distributions[feature]["median"])
        for feature in service.categorical_values:
            counts = service.categorical_frequencies[feature]
            self.assertEqual(data["example"][feature], max(counts, key=counts.get))

    def test_input_schema_reports_dataset_distribution(self):
        response = self.client.get("/api/input-schema")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["training_dataset_size"], 1500)
        self.assertEqual(data["numeric_ranges"]["Age"], {"minimum": 18.0, "maximum": 30.0, "integer_only": True})
        self.assertEqual(data["categorical_values"], app.extensions["metadata"]["categorical_values"])
        self.assertIn("empirical 95th percentile", data["input_quality_method"])

    def test_reference_profile_uses_training_reference_and_saved_prediction(self):
        response = self.client.get("/api/reference-profile")
        self.assertEqual(response.status_code, 200)
        reference = response.get_json()
        service = app.extensions["prediction_service"]
        self.assertEqual(reference["label"], "Reference Dataset Profile")
        self.assertEqual(set(reference["profile"]), set(service.features))
        for feature in service.numeric_distributions:
            expected = service.numeric_distributions[feature]["median"]
            self.assertAlmostEqual(reference["profile"][feature], expected)
        for feature in service.categorical_values:
            expected = str(service.training_reference[feature].mode().iloc[0])
            self.assertEqual(reference["profile"][feature], expected)
        self.assertFalse(any(name.startswith("PSS_") or name in {"PSS10_Total", "Stress_Level"} for name in reference["profile"]))
        direct = self.client.post("/api/predict", json=reference["profile"]).get_json()
        self.assertEqual(reference["prediction"]["stress_level"], direct["stress_level"])
        self.assertEqual(reference["prediction"]["probabilities"], direct["probabilities"])

    def test_reference_profile_unavailable_without_model(self):
        original = app.extensions["prediction_service"]
        app.extensions["prediction_service"] = None
        try:
            response = self.client.get("/api/reference-profile")
            self.assertEqual(response.status_code, 503)
        finally:
            app.extensions["prediction_service"] = original

    def test_reference_profile_contract_remains_model_prediction_only(self):
        response = self.client.get("/api/reference-profile")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(set(result), {"label", "method", "profile", "prediction"})
        self.assertIn("probabilities", result["prediction"])
        self.assertEqual(set(result["profile"]), set(app.extensions["prediction_service"].features))

    def test_missing_field(self):
        payload = valid_payload()
        payload.pop("Age")
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Invalid input")

    def test_invalid_category(self):
        payload = valid_payload()
        payload["Gender"] = "Unknown category"
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_invalid_numeric_value(self):
        payload = valid_payload()
        payload["Age"] = 1000
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_pss_fields_rejected(self):
        payload = valid_payload()
        payload["PSS_Q1"] = 2
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("PSS_Q1", response.get_json()["details"][0])

    def test_explain_endpoint(self):
        response = self.client.post("/api/explain", json=valid_payload())
        self.assertEqual(response.status_code, 200, response.get_json())
        data = response.get_json()
        self.assertIn("explanation", data)
        self.assertTrue(data["explanation"])
        self.assertNotIn("PSS10_Total", [item["feature"] for item in data["explanation"]])
        self.assertIn("not causal", data["note"])
        json.dumps(data, allow_nan=False)

    def test_model_info(self):
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["model"], "SVM")
        self.assertEqual(data["target"], "Stress_Level")
        self.assertEqual(data["research_evaluation"]["protocol"], "Nested Stratified Cross-Validation; internal evaluation estimates, not training metrics")

    def test_malformed_json(self):
        response = self.client.post(
            "/api/predict", data="{bad json", content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Malformed JSON", response.get_json()["details"][0])


if __name__ == "__main__":
    unittest.main()
