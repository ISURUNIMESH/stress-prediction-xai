# Stress Prediction Research Prototype API

This API serves the selected SVM research pipeline. It accepts the 13 non-PSS predictor fields, returns an `Stress_Level` prediction and model probability estimates, and can return approximate Kernel SHAP contributions for a prediction.

> This system is a research prototype and does not provide a medical diagnosis.

Nested Stratified Cross-Validation is internal evaluation, not independent external validation. The recorded Macro F1 and Balanced Accuracy are research evaluation estimates; they are not training scores or confidence for an individual prediction.

## Setup

Use the project's Python 3.12 environment from the repository root. Install dependencies with:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `.venv` already exists, activate it and install the added Flask dependency using the same requirements command.

## Create the model artifact

Train and export the inference pipeline once:

```powershell
python application/train_final_model.py
```

This creates:

- `application/artifacts/final_svm_pipeline.joblib`: complete preprocessing and SVM pipeline.
- `application/artifacts/model_metadata.json`: predictor schema, observed input domains, nested-CV research estimates, and package versions.
- `application/artifacts/shap_background.joblib`: a fixed 30-row transformed background sampled from the training dataset.

The API loads the model once when the server process starts. API requests do not retrain it.

## Start locally

From the repository root:

```powershell
python application/app.py
```

The development server listens on `http://127.0.0.1:5000`. It is intended for local research prototyping, not production deployment.

## Input schema

Every prediction request must include these fields:

```json
{
  "Age": 21,
  "Gender": "Female",
  "University": "University of Colombo",
  "Degree_Program": "Computer Science",
  "Academic_Year": "2nd",
  "Sleep_Hours": 7,
  "Exercise_Frequency": 3,
  "Screen_Time_Hours": 6,
  "Social_Media_Hours": 4,
  "Study_Hours": 5,
  "Academic_Pressure": 3,
  "Financial_Stress": 2,
  "Family_Support": 4
}
```

This request example uses values observed in the processed dataset. Numeric inputs must be finite integers within each field's observed dataset range. Categorical inputs must exactly match a category observed during training. The service rejects missing fields, unknown fields, PSS fields, the PSS total, invalid types, out-of-range values, and unseen category values. The API does not derive the target from PSS responses.

Maximum request size is 16 KB. Send `Content-Type: application/json`.

## Endpoints

### `GET /api/health`

Returns HTTP 200 if the model pipeline loaded successfully, or HTTP 503 if model loading failed.

```json
{
  "status": "ok",
  "service": "stress-prediction-api",
  "model_loaded": true
}
```

### `GET /api/input-schema`

Returns the predictor names, dataset-observed numeric bounds and distributions, and exact category values/frequencies used by the frontend. The UI builds allowed entries from this response rather than maintaining a second hard-coded category list.

### `GET /api/example-input`

Returns a synthetic feature-only example assembled from training-set numeric medians and categorical modes. It does not return a participant-level row. The target and PSS fields are omitted.

### `POST /api/predict`

Accepts the input schema above. The response values are computed by the saved model at request time; exact output depends on the submitted inputs.

```json
{
  "stress_level": "<predicted class>",
  "input_quality": {
    "status": "<Good or Caution>",
    "message": "<training-distribution note>",
    "method": "<documented empirical input-support check>"
  },
  "probabilities": {
    "High Stress": "<model probability>",
    "Low Stress": "<model probability>",
    "Moderate Stress": "<model probability>"
  }
}
```

Probability values are the SVM pipeline's `predict_proba` outputs, not fabricated values. The input-quality annotation does not modify or reject an otherwise valid prediction. Numeric marginal outliers use Tukey 1.5?IQR fences; unusual mixed-type combinations are flagged when nearest-row distance is above the empirical 95th percentile of leave-one-out training distances. Low-frequency marginal values use the empirical Tukey lower fence on observed frequencies. These are descriptive warnings, not validated confidence estimates or accuracy claims.

### `POST /api/explain`

Accepts the same input and returns the prediction, probabilities, and the largest original-feature contributions for the predicted class. SHAP KernelExplainer uses the saved training-only background artifact and is initialized once per server process. One-hot encoded contributions are summed back to their source categorical field.

```json
{
  "stress_level": "<predicted class>",
  "probabilities": {
    "High Stress": "<runtime probability>",
    "Low Stress": "<runtime probability>",
    "Moderate Stress": "<runtime probability>"
  },
  "explained_class": "<predicted class>",
  "method": "SHAP KernelExplainer on SVM class probabilities",
  "explanation": [
    {
      "feature": "<input feature>",
      "shap_value": "<computed SHAP value>",
      "direction": "<increases_prediction or decreases_prediction>",
      "interpretation": "contributes to the model prediction for the explained class"
    }
  ],
  "note": "Contributions describe model behavior and are not causal effects."
}
```

The explanation is an approximate model explanation, not a causal effect. SHAP values and probabilities are generated at runtime and are not specified in this example.

### `GET /api/model-info`

Returns the model schema, class names, selection criterion, and research evaluation estimates clearly labeled as nested-CV estimates.

### `GET /api/research-analytics`

Reads the existing generated class-distribution, nested-CV, ablation, SHAP, robustness, and held-out metric files and returns dashboard-ready values. It does not train models or recalculate experiment metrics. If one or more expected files are unavailable, it returns the available sections with `status: "partial"` and error messages for the missing sections.

## Errors

Invalid or malformed JSON and invalid input return HTTP 400:

```json
{
  "error": "Invalid input",
  "details": ["<validation detail>"]
}
```

Unknown routes return HTTP 404 with `{"error":"Endpoint not found"}`. Requests over 16 KB return HTTP 413. If model files are missing or cannot load, prediction routes return HTTP 503 and health reports an error. Unexpected prediction or explanation failures return HTTP 500 with a safe message; details are logged server-side, not returned to the caller.

## Limitations

- This is a research prototype, not a clinical or production service.
- Nested CV is internal validation; external validity has not been established.
- The SVM probability estimates and approximate SHAP contributions can be uncertain, particularly for minority classes.
- Input ranges and categories are restricted to values observed in the current processed dataset.
- The Flask development server is for local testing only.
