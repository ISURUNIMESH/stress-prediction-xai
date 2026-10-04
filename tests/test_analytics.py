"""Research dashboard reads generated artifacts and does not run experiments."""

import csv
import unittest
from pathlib import Path
from unittest.mock import patch

from application.app import app

ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative):
    with (ROOT / "results" / relative).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ResearchAnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def get_analytics(self):
        response = self.client.get("/api/research-analytics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        response.close()
        return data

    def test_analytics_page_loads(self):
        response = self.client.get("/analytics")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        response.close()
        self.assertIn("Research Analytics", page)
        self.assertIn('id="analytics-main"', page)
        self.assertIn("/analytics.js", page)
        self.assertIn('id="kpi-macro-f1"', page)
        script = self.client.get("/analytics.js")
        stylesheet = self.client.get("/analytics.css")
        self.assertEqual(script.status_code, 200)
        self.assertEqual(stylesheet.status_code, 200)
        script.close()
        stylesheet.close()

    def test_research_metrics_load_from_nested_cv_results(self):
        data = self.get_analytics()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["selected_model"]["name"], "SVM")
        rows = read_csv("nested_cv/nested_cv_results.csv")
        expected = {row["metric"]: row for row in rows if row["model"] == "SVM"}
        self.assertAlmostEqual(data["selected_model"]["macro_f1"]["mean"], float(expected["f1_macro"]["mean"]))
        self.assertAlmostEqual(data["selected_model"]["macro_f1"]["std"], float(expected["f1_macro"]["std"]))
        self.assertAlmostEqual(data["selected_model"]["balanced_accuracy"]["mean"], float(expected["balanced_accuracy"]["mean"]))
        self.assertAlmostEqual(data["selected_model"]["balanced_accuracy"]["std"], float(expected["balanced_accuracy"]["std"]))

    def test_actual_dataset_counts_are_display_data(self):
        data = self.get_analytics()
        rows = read_csv("metrics/class_distribution.csv")
        expected = {row["Stress_Level"]: int(row["count"]) for row in rows}
        self.assertEqual(data["dataset"]["total_students"], sum(expected.values()))
        counts = {item["name"]: item["count"] for item in data["dataset"]["classes"]}
        self.assertEqual(counts, expected)
        self.assertAlmostEqual(sum(item["proportion"] for item in data["dataset"]["classes"]), 1.0)

    def test_four_model_comparison_loads(self):
        data = self.get_analytics()
        self.assertEqual([item["model"] for item in data["model_comparison"]], [
            "Logistic Regression", "SVM", "Random Forest", "XGBoost",
        ])
        svm = next(item for item in data["model_comparison"] if item["model"] == "SVM")
        rows = read_csv("nested_cv/nested_cv_results.csv")
        expected = {row["metric"]: row for row in rows if row["model"] == "SVM"}
        self.assertAlmostEqual(svm["macro_f1"]["mean"], float(expected["f1_macro"]["mean"]))
        self.assertAlmostEqual(svm["balanced_accuracy"]["mean"], float(expected["balanced_accuracy"]["mean"]))

    def test_seven_ablation_configurations_load(self):
        data = self.get_analytics()
        self.assertEqual(len(data["ablation"]), 7)
        self.assertTrue(all(row["features"] and row["macro_f1"]["mean"] >= 0 for row in data["ablation"]))
        self.assertTrue(all("balanced_accuracy" in row for row in data["ablation"]))

    def test_selected_model_global_shap_importance_loads(self):
        data = self.get_analytics()
        self.assertEqual(data["shap"]["selected_model"], "SVM")
        self.assertEqual(data["shap"]["explainer"], "KernelExplainer")
        self.assertEqual(len(data["shap"]["features"]), 10)
        expected = read_csv("imbalance_aware/shap/global_feature_importance.csv")
        self.assertEqual(data["shap"]["features"][0]["feature"], expected[0]["feature"])
        self.assertIn("do not establish causality", data["shap"]["note"])

    def test_missing_results_file_is_reported_without_crashing(self):
        missing_results = ROOT / "results" / "__analytics_missing_fixture__"
        self.assertFalse(missing_results.exists())
        with patch("application.app.RESULTS", missing_results):
            data = self.get_analytics()
        self.assertEqual(data["status"], "partial")
        self.assertIsNone(data["dataset"])
        self.assertTrue(data["errors"])
        self.assertTrue(any("Dataset overview unavailable" in error for error in data["errors"]))


if __name__ == "__main__":
    unittest.main()
