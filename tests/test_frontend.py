"""Static frontend integration checks using Flask's test client."""

import unittest
from html.parser import HTMLParser

from application.app import app


class FormSchemaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fields = set()
        self.categories = {}
        self.active_select = None
        self.active_option = False
        self.option_parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name"):
            self.fields.add(attrs["name"])
        elif tag == "select":
            self.active_select = attrs.get("name")
            self.fields.add(self.active_select)
            self.categories[self.active_select] = []
        elif tag == "option" and self.active_select:
            self.active_option = True
            self.option_parts = []
            self.option_disabled = "disabled" in attrs

    def handle_data(self, data):
        if self.active_option:
            self.option_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "option" and self.active_option:
            value = "".join(self.option_parts).strip()
            if value and not self.option_disabled:
                self.categories[self.active_select].append(value)
            self.active_option = False
        elif tag == "select":
            self.active_select = None


class FrontendIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_root_serves_dashboard(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        response.close()
        self.assertIn("Explainable AI Stress Prediction System", page)
        self.assertIn('id="prediction-form"', page)
        self.assertIn("does not provide a medical diagnosis", page)

    def test_frontend_assets_are_served(self):
        script = self.client.get("/app.js")
        stylesheet = self.client.get("/style.css")
        self.assertEqual(script.status_code, 200)
        self.assertEqual(stylesheet.status_code, 200)
        script_text = script.get_data(as_text=True)
        stylesheet_text = stylesheet.get_data(as_text=True)
        script.close()
        stylesheet.close()
        self.assertIn("/api/predict", script_text)
        self.assertIn("@media (max-width: 480px)", stylesheet_text)
        for asset in ["/visualizations.css", "/recommendations.js", "/profile-comparison.js", "/theme.js"]:
            response = self.client.get(asset)
            self.assertEqual(response.status_code, 200)
            response.close()

    def test_personalized_panels_start_idle_and_follow_prediction(self):
        response = self.client.get("/")
        page = response.get_data(as_text=True)
        response.close()
        self.assertIn('id="visualization-idle"', page)
        self.assertIn('id="profile-comparison-chart"', page)
        self.assertIn('id="stress-visualization-card"', page)
        self.assertIn('hidden>', page)
        self.assertIn('id="recommendation-card"', page)
        script_response = self.client.get("/app.js")
        script = script_response.get_data(as_text=True)
        script_response.close()
        self.assertIn("renderRecommendations(lastSubmittedInput, data)", script)
        self.assertIn("renderProfileComparison(lastSubmittedInput, data)", script)
        self.assertNotIn("stress-3d-canvas", page + script)

    def test_utf8_theme_and_comparison_ui(self):
        page = self.client.get("/")
        page_text = page.get_data(as_text=True)
        page.close()
        analytics = self.client.get("/analytics")
        analytics_text = analytics.get_data(as_text=True)
        analytics.close()
        self.assertIn('<meta charset="UTF-8">', page_text)
        self.assertIn('<meta charset="UTF-8">', analytics_text)
        self.assertIn('id="theme-toggle"', page_text)
        self.assertIn('id="theme-toggle"', analytics_text)
        self.assertIn("stress-theme", page_text)
        self.assertIn("stress-theme", analytics_text)
        for corrupted in ["Ãƒâ€”", "Ã¢â‚¬â€œ", "Ã‚Â±", "Ã¢â‚¬Å“"]:
            self.assertNotIn(corrupted, page_text + analytics_text)
        comparison = self.client.get("/profile-comparison.js")
        self.assertEqual(comparison.status_code, 200)
        self.assertIn("Sleep_Hours", comparison.get_data(as_text=True))
        comparison.close()

    def test_legacy_3d_asset_is_not_served(self):
        response = self.client.get("/stress-visualization.js")
        self.assertEqual(response.status_code, 404)

    def test_form_fields_and_categories_match_model_schema(self):
        response = self.client.get("/")
        page = response.get_data(as_text=True)
        response.close()
        parser = FormSchemaParser()
        parser.feed(page)
        metadata = app.extensions["metadata"]
        self.assertEqual(set(metadata["feature_names"]), parser.fields)
        schema = self.client.get("/api/input-schema").get_json()
        self.assertEqual(metadata["categorical_values"], schema["categorical_values"])
        self.assertEqual(set(schema["features"]), parser.fields)
        for field in ["Exercise_Frequency", "Academic_Pressure", "Financial_Stress", "Family_Support"]:
            self.assertIn(f'<select id="{field}"', page)

    def test_frontend_explains_probabilities_shap_and_limitations(self):
        response = self.client.get("/")
        page = response.get_data(as_text=True)
        response.close()
        self.assertIn("Model-predicted probabilities", page)
        self.assertIn("should not be interpreted as clinical certainty", page)
        self.assertIn("Contribution to the model prediction", page)
        self.assertIn("does not provide a medical diagnosis", page)
        self.assertIn("may differ for populations or situations", page)

    def test_reset_example_and_api_unavailable_states_are_wired(self):
        script_response = self.client.get("/app.js")
        script = script_response.get_data(as_text=True)
        script_response.close()
        self.assertIn('addEventListener("reset"', script)
        self.assertIn('addEventListener("click"', script)
        self.assertIn("loadExampleInput", script)
        original = app.extensions["prediction_service"]
        app.extensions["prediction_service"] = None
        try:
            unavailable = self.client.get("/api/input-schema")
            self.assertEqual(unavailable.status_code, 503)
            prediction_unavailable = self.client.post("/api/predict", json={})
            self.assertEqual(prediction_unavailable.status_code, 503)
        finally:
            app.extensions["prediction_service"] = original

    def test_frontend_static_route_does_not_expose_files(self):
        response = self.client.get("/application/artifacts/model_metadata.json")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json(), {"error": "Endpoint not found"})

    def test_api_routes_remain_available(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["model_loaded"])


if __name__ == "__main__":
    unittest.main()
