"""Regression checks for frontend/API/model prediction-path consistency."""

import base64
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from html.parser import HTMLParser

from application.app import app

ROOT = Path(__file__).resolve().parents[1]
JS_PATH = ROOT / "frontend" / "app.js"
METADATA = json.loads((ROOT / "application" / "artifacts" / "model_metadata.json").read_text(encoding="utf-8"))
PREDICTION_FIELDS = METADATA["feature_names"]
NUMERIC_FIELDS = list(METADATA["numeric_ranges"])
FORBIDDEN = {*(f"PSS_Q{i}" for i in range(1, 11)), "PSS10_Total", "Stress_Level"}


class FormFieldParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.form_fields = []
        self.in_prediction_form = False
        self.all_ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.all_ids.append(attrs["id"])
        if tag == "form" and attrs.get("id") == "prediction-form":
            self.in_prediction_form = True
        if self.in_prediction_form and tag in {"input", "select"}:
            self.form_fields.append((attrs.get("id"), attrs.get("name")))

    def handle_endtag(self, tag):
        if tag == "form":
            self.in_prediction_form = False


def node_collect_input(profile):
    """Execute the production collectInput function with a minimal form DOM."""
    source = JS_PATH.read_text(encoding="utf-8")
    start = source.index("function collectInput() {")
    end = source.index("\nfunction validateForm", start)
    function = source[start:end]
    code = r"""
const vm=require('node:vm');
const fn=Buffer.from(process.argv[1],'base64').toString('utf8');
const input=JSON.parse(Buffer.from(process.argv[2],'base64').toString('utf8'));
const numeric=Object.fromEntries(Object.keys(input).filter(k=>k in input.__numeric).map(k=>[k,{}]));
delete input.__numeric;
const form={elements:Object.entries(input).map(([name,value])=>({name,value:String(value)}))};
const inputSchema={numeric_ranges:numeric};
const result=vm.runInNewContext(fn+'\ncollectInput()', {form,inputSchema});
process.stdout.write(JSON.stringify(result));
"""
    prepared = {**profile, "__numeric": {key: True for key in NUMERIC_FIELDS}}
    encoded_function = base64.b64encode(function.encode()).decode()
    encoded_profile = base64.b64encode(json.dumps(prepared).encode()).decode()
    result = subprocess.run(
        ["node", "-e", code, encoded_function, encoded_profile],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def node_render_prediction(response, submitted_input):
    """Execute production renderPrediction against small DOM element doubles."""
    source = JS_PATH.read_text(encoding="utf-8")
    start = source.index("function renderPrediction(data) {")
    end = source.index("\nfunction renderRecommendations", start)
    function = source[start:end]
    code = r"""
const vm=require('node:vm');
const fn=Buffer.from(process.argv[1],'base64').toString('utf8');
const data=JSON.parse(Buffer.from(process.argv[2],'base64').toString('utf8'));
const elements=new Map();
function make(){return {textContent:'',hidden:false,disabled:false,children:[],style:{},classList:{toggle(){},add(){},remove(){}},setAttribute(){},append(...x){this.children.push(...x)},replaceChildren(){this.children=[]}}}
const document={getElementById(id){if(!elements.has(id))elements.set(id,make());return elements.get(id)},createElement(){return make()}};
const context={document,CLASS_ORDER:['Low Stress','Moderate Stress','High Stress'],data,
 resultCard:make(),explanationCard:make(),explainButton:make(),predictionError:make(),explanationError:make(),
 lastSubmittedInput:JSON.parse(Buffer.from(process.argv[3],'base64').toString('utf8')),
 inputSchema:{numeric_distributions:{}},formatProbability:v=>(v*100).toFixed(1)+'%',
 renderInputQuality(){},renderRecommendations(){},renderProfileComparison(){},hideMessage(){}};
vm.runInNewContext(fn+'\nrenderPrediction(data); JSON.stringify({label:document.getElementById("predicted-level").textContent,probabilities:Object.fromEntries(document.getElementById("probability-list").children.map(row=>[row.children[0].textContent,row.children[2].textContent]))})',context);
process.stdout.write(vm.runInNewContext('JSON.stringify({label:document.getElementById("predicted-level").textContent,probabilities:Object.fromEntries(document.getElementById("probability-list").children.map(row=>[row.children[0].textContent,row.children[2].textContent]))})',context));
"""
    encoded_function = base64.b64encode(function.encode()).decode()
    encoded_response = base64.b64encode(json.dumps(response).encode()).decode()
    encoded_input = base64.b64encode(json.dumps(submitted_input).encode()).decode()
    result = subprocess.run(
        ["node", "-e", code, encoded_function, encoded_response, encoded_input],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def extreme_profiles():
    categories = {name: values[0] for name, values in METADATA["categorical_values"].items()}
    profile_a = {
        "Age": 18, "Sleep_Hours": 4, "Exercise_Frequency": 0,
        "Screen_Time_Hours": 12, "Social_Media_Hours": 8, "Study_Hours": 10,
        "Academic_Pressure": 5, "Financial_Stress": 5, "Family_Support": 1,
        **categories,
    }
    profile_b = {
        "Age": 30, "Sleep_Hours": 9, "Exercise_Frequency": 7,
        "Screen_Time_Hours": 2, "Social_Media_Hours": 1, "Study_Hours": 1,
        "Academic_Pressure": 1, "Financial_Stress": 1, "Family_Support": 5,
        **categories,
    }
    return profile_a, profile_b


class PredictionDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = app.extensions["prediction_service"]
        cls.client = app.test_client()
        cls.data = pd.read_csv(ROOT / "dataset" / "processed" / "cleaned_stress_dataset.csv")
        cls.model = joblib.load(ROOT / "application" / "artifacts" / "final_svm_pipeline.joblib")

    def test_frontend_form_fields_exactly_match_model_features(self):
        parser = FormFieldParser()
        parser.feed((ROOT / "frontend" / "index.html").read_text(encoding="utf-8"))
        ids = [field_id for field_id, _ in parser.form_fields]
        names = [name for _, name in parser.form_fields]
        self.assertEqual(ids, names)
        self.assertEqual(set(names), set(PREDICTION_FIELDS))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(parser.all_ids), len(set(parser.all_ids)))
        self.assertFalse(FORBIDDEN.intersection(names))

    def test_frontend_payload_api_and_direct_service_match_for_extremes(self):
        for profile in extreme_profiles():
            frontend_payload = node_collect_input(profile)
            self.assertEqual(frontend_payload, profile)
            self.assertFalse(FORBIDDEN.intersection(frontend_payload))
            direct = self.service.predict(self.service.validate(frontend_payload))
            received = {}
            validate = self.service.validate

            def capture(payload):
                received.update(payload)
                return validate(payload)

            with patch.object(self.service, "validate", side_effect=capture):
                response = self.client.post("/api/predict", json=frontend_payload)
            self.assertEqual(response.status_code, 200)
            api_output = response.get_json()
            self.assertEqual(received, frontend_payload)
            self.assertEqual(api_output["stress_level"], direct["stress_level"])
            for class_name in METADATA["classes"]:
                self.assertAlmostEqual(api_output["probabilities"][class_name], direct["probabilities"][class_name], places=12)
            rendered = node_render_prediction(api_output, frontend_payload)
            self.assertEqual(rendered["label"], api_output["stress_level"])
            for class_name in ("Low Stress", "Moderate Stress", "High Stress"):
                self.assertEqual(rendered["probabilities"][class_name], f"{api_output['probabilities'][class_name] * 100:.1f}%")

    def test_model_probability_indices_follow_classifier_classes(self):
        classifier = self.model.named_steps["classifier"]
        self.assertEqual(classifier.classes_.tolist(), [0, 1, 2])
        self.assertEqual(METADATA["classes"], ["High Stress", "Low Stress", "Moderate Stress"])
        frame = self.service.validate(extreme_profiles()[0])
        raw = self.model.predict_proba(frame)[0]
        service = self.service.predict(frame)["probabilities"]
        expected = {METADATA["classes"][int(class_id)]: float(raw[index]) for index, class_id in enumerate(classifier.classes_)}
        self.assertEqual(set(service), {"Low Stress", "Moderate Stress", "High Stress"})
        for class_name in expected:
            self.assertAlmostEqual(service[class_name], expected[class_name], places=12)

    def test_actual_class_rows_use_only_predictors_and_direct_saved_model(self):
        for actual_label in ("Low Stress", "Moderate Stress", "High Stress"):
            row = self.data.loc[self.data["Stress_Level"] == actual_label].iloc[0]
            payload = row[PREDICTION_FIELDS].to_dict()
            self.assertEqual(set(payload), set(PREDICTION_FIELDS))
            self.assertFalse(FORBIDDEN.intersection(payload))
            result = self.service.predict(self.service.validate(payload))
            self.assertIn(result["stress_level"], METADATA["classes"])
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=10)

    def test_saved_model_probabilities_respond_to_each_controlled_feature(self):
        row = self.data.iloc[0][PREDICTION_FIELDS].to_dict()
        levels = {
            "Sleep_Hours": range(4, 10),
            "Academic_Pressure": range(1, 6),
            "Social_Media_Hours": range(1, 9),
            "Exercise_Frequency": range(0, 8),
            "Financial_Stress": range(1, 6),
            "Family_Support": range(1, 6),
        }
        for feature, values in levels.items():
            outputs = []
            for value in values:
                changed = row.copy()
                changed[feature] = value
                result = self.service.predict(self.service.validate(changed))
                outputs.append([result["probabilities"][label] for label in METADATA["classes"]])
            self.assertGreater(float(np.ptp(np.asarray(outputs), axis=0).max()), 1e-5, feature)

    def test_pss_target_recomputation_matches_without_mutating_source_responses(self):
        original = self.data[[f"PSS_Q{i}" for i in range(1, 11)]].copy(deep=True)
        scored = original.copy()
        for feature in ("PSS_Q4", "PSS_Q5", "PSS_Q7", "PSS_Q8"):
            scored[feature] = 4 - scored[feature]
        total = scored.sum(axis=1)
        expected_level = pd.cut(total, [-1, 13, 26, 40], labels=["Low Stress", "Moderate Stress", "High Stress"]).astype(str)
        self.assertTrue((total.to_numpy() == self.data["PSS10_Total"].to_numpy()).all())
        self.assertTrue((expected_level.to_numpy() == self.data["Stress_Level"].to_numpy()).all())
        pd.testing.assert_frame_equal(original, self.data[original.columns])


if __name__ == "__main__":
    unittest.main()
