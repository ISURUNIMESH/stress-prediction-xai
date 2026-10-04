# Frontend

Vanilla HTML, CSS, and JavaScript interface for the Explainable AI Stress Prediction System research prototype. The frontend collects predictor fields, calls the existing Flask API, and renders the API's prediction, probabilities, and SHAP explanation. It contains no model or preprocessing logic and does not save student inputs in browser storage.

## Run

From the repository root, start the saved model API:

```powershell
python application/app.py
```

Open <http://127.0.0.1:5000/> for prediction or <http://127.0.0.1:5000/analytics> for Research Analytics. Flask serves `index.html`, `style.css`, and `app.js` from `frontend/`; the browser and API share an origin, so no separate frontend server or CORS setup is needed.

## API URL configuration

The API base URL is set once near the top of `app.js` as `API_BASE_URL`. By default, the UI uses the current origin when served by Flask and `http://127.0.0.1:5000` when opened from a local file. Change the constant if the API is hosted elsewhere. Cross-origin hosting requires the API host to permit that frontend origin.

## API communication

The JavaScript API helpers call:

- `GET /api/health` for the connection status.
- `GET /api/model-info` for the model and nested-CV research metrics.
- `GET /api/input-schema` for data-derived ranges, distributions, and categories.
- `GET /api/example-input` for a synthetic feature-only example assembled from aggregate medians and modes.
- `POST /api/predict` for predictions and returned probabilities.
- `POST /api/explain` for the prediction and SHAP contributions.
- `GET /api/research-analytics` for a read-only view of the generated research metrics, ablation CSV, class counts, and selected-model SHAP output.

Inputs are kept in memory only while the page is open and are sent only when prediction or explanation is requested. The API is authoritative for validation. Numeric bounds and dropdown values are loaded from the exported model metadata and observed training data. Valid but unusual inputs remain eligible for prediction and receive an input-quality caution only. The check uses numeric Tukey 1.5?IQR fences, a Tukey lower fence for rare marginal values, and a mixed-type nearest-neighbor distance compared with its empirical 95th-percentile leave-one-out training reference. It does not modify prediction outputs. The example button loads a synthetic feature-only profile made from aggregate medians and modes.

The analytics page reads existing files under `results/`; it does not run experiments or write result files. A missing generated file leaves that section unavailable and is reported in the page.

## Limitations

This is a research prototype and does not provide a medical diagnosis. Nested CV is internal validation, not independent external validation. Probabilities are model outputs, not clinical certainty. Input quality is a descriptive support warning, not model accuracy. SHAP values describe contributions to this model's prediction and are not causes. The interface does not establish clinical validity or generalization to other populations.
