"use strict";

// Flask serves this page from the same origin. Change this single value when
// connecting to an API hosted at a different address.
const API_BASE_URL = window.location.protocol === "file:"
  ? "http://127.0.0.1:5000"
  : window.location.origin;
const REQUEST_TIMEOUT_MS = 12000;
const CLASS_ORDER = ["Low Stress", "Moderate Stress", "High Stress"];

const form = document.getElementById("prediction-form");
const predictButton = document.getElementById("predict-button");
const explainButton = document.getElementById("explain-button");
const statusText = document.getElementById("api-status");
const statusDot = document.getElementById("status-dot");
const connectionMessage = document.getElementById("connection-message");
const resultCard = document.getElementById("result-card");
const explanationCard = document.getElementById("explanation-card");
const recommendationCard = document.getElementById("recommendation-card");
const comparisonCard = document.getElementById("stress-visualization-card");
const predictionError = document.getElementById("predict-error");
const explanationError = document.getElementById("explain-error");
let lastSubmittedInput = null;
let requestGeneration = 0;
let inputSchema = null;
let referenceProfile = null;
let completedInput = null;
predictButton.disabled = true;
explainButton.disabled = true;

class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function requestJson(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      signal: controller.signal,
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
    let data;
    try {
      data = await response.json();
    } catch (_error) {
      throw new ApiError("The service returned an unreadable response.", response.status);
    }
    if (!response.ok) {
      const message = data && Array.isArray(data.details) && data.details.length
        ? data.details.join(" ")
        : friendlyStatusMessage(response.status);
      throw new ApiError(message, response.status);
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      throw new ApiError("The service returned an unexpected response format.", response.status);
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new ApiError("The request took too long. Please try again.");
    }
    if (error instanceof ApiError) throw error;
    throw new ApiError("Unable to connect to the prediction service.");
  } finally {
    window.clearTimeout(timeout);
  }
}

function friendlyStatusMessage(status) {
  if (status === 400) return "Some submitted values are invalid. Review the form and try again.";
  if (status === 404) return "The requested API endpoint was not found.";
  if (status >= 500) return "The prediction service encountered an error. Please try again later.";
  return "The request could not be completed. Please try again.";
}

async function checkHealth() {
  try {
    const health = await requestJson("/api/health");
    const online = health.status === "ok" && health.model_loaded === true;
    setApiStatus(online ? "Connected" : "Offline", online);
    connectionMessage.hidden = online;
    predictButton.disabled = !online || !inputSchema;
    return online;
  } catch (_error) {
    setApiStatus("Offline", false);
    connectionMessage.hidden = false;
    predictButton.disabled = true;
    return false;
  }
}

async function loadInputSchema() {
  const loading = document.getElementById("schema-loading");
  const error = document.getElementById("schema-error");
  try {
    const schema = await requestJson("/api/input-schema");
    if (!Array.isArray(schema.features) || !schema.numeric_ranges || !schema.categorical_values
      || !schema.numeric_distributions || !schema.categorical_frequencies) {
      throw new ApiError("Input schema is incomplete.");
    }
    inputSchema = schema;
    for (const [feature, values] of Object.entries(schema.categorical_values)) {
      const select = document.getElementById(feature);
      for (const value of values) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.append(option);
      }
    }
    for (const feature of ["Exercise_Frequency", "Academic_Pressure", "Financial_Stress", "Family_Support"]) {
      const select = document.getElementById(feature);
      for (const value of Object.keys(schema.numeric_distributions[feature].value_counts)) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.append(option);
      }
    }
    const units = {
      Age: "years", Sleep_Hours: "hours/night", Exercise_Frequency: "days/week",
      Screen_Time_Hours: "hours/day", Social_Media_Hours: "hours/day", Study_Hours: "hours/day",
    };
    for (const [feature, bounds] of Object.entries(schema.numeric_ranges)) {
      const field = document.getElementById(feature);
      if (field.type === "number") {
        field.min = String(bounds.minimum);
        field.max = String(bounds.maximum);
        field.step = bounds.integer_only ? "1" : "any";
        field.inputMode = bounds.integer_only ? "numeric" : "decimal";
      }
      const distribution = schema.numeric_distributions[feature];
      const rangeText = `${bounds.minimum}\u2013${bounds.maximum}`;
      const description = document.querySelector(`[data-help="${feature}"]`).textContent;
      document.querySelector(`[data-help="${feature}"]`).textContent = `Observed range: ${rangeText} ${units[feature] || "(scale)"}. ${description} Median: ${distribution.median}.`;
    }
    loading.hidden = true;
    error.hidden = true;
    predictButton.disabled = statusText.textContent !== "Connected";
  } catch (_error) {
    loading.hidden = true;
    error.hidden = false;
    inputSchema = null;
    predictButton.disabled = true;
    setApiStatus("Offline", false);
  }
}

function setApiStatus(label, online) {
  statusText.textContent = label;
  statusDot.classList.toggle("is-online", online);
  statusDot.classList.toggle("is-offline", !online);
  statusDot.classList.remove("is-pending");
}

async function getModelInfo() {
  const loading = document.getElementById("model-info-loading");
  const errorBox = document.getElementById("model-info-error");
  const content = document.getElementById("model-info-content");
  try {
    const info = await requestJson("/api/model-info");
    const evaluation = info.research_evaluation;
    if (!info.model || !info.target || !Array.isArray(info.classes) || !evaluation) {
      throw new ApiError("Model information is incomplete.");
    }
    document.getElementById("model-name").textContent = info.model;
    document.getElementById("selection-criterion").textContent = "Nested CV Macro F1";
    document.getElementById("macro-f1").textContent = formatEstimate(evaluation.macro_f1_mean, evaluation.macro_f1_std);
    document.getElementById("balanced-accuracy").textContent = formatEstimate(evaluation.balanced_accuracy_mean, evaluation.balanced_accuracy_std);
    document.getElementById("target-name").textContent = info.target;
    const orderedClasses = CLASS_ORDER.filter((name) => info.classes.includes(name));
    document.getElementById("class-names").textContent = orderedClasses.join("  ");
    loading.hidden = true;
    errorBox.hidden = true;
    content.hidden = false;
  } catch (_error) {
    loading.hidden = true;
    errorBox.hidden = false;
    content.hidden = true;
  }
}

function formatEstimate(mean, standardDeviation) {
  if (!Number.isFinite(mean) || !Number.isFinite(standardDeviation)) return "Unavailable";
  return `${mean.toFixed(4)} ? ${standardDeviation.toFixed(4)}`;
}

function collectInput() {
  const payload = {};
  for (const element of form.elements) {
    if (!element.name) continue;
    if (inputSchema && element.name in inputSchema.numeric_ranges) {
      payload[element.name] = element.value === "" ? null : Number(element.value);
    } else {
      payload[element.name] = element.value;
    }
  }
  return payload;
}

function validateForm() {
  form.classList.add("was-validated");
  if (!inputSchema) {
    showMessage(predictionError, "Input rules are not available. Check the API connection and try again.");
    return false;
  }
  if (!form.checkValidity()) {
    form.reportValidity();
    showMessage(predictionError, "Complete each field using the displayed valid range or options.");
    return false;
  }
  for (const [feature, value] of Object.entries(collectInput())) {
    if (feature in inputSchema.numeric_ranges) {
      const bounds = inputSchema.numeric_ranges[feature];
      if (typeof value !== "number" || !Number.isFinite(value)
        || value < bounds.minimum || value > bounds.maximum
        || (bounds.integer_only && !Number.isInteger(value))) {
        showMessage(predictionError, `${feature.replaceAll("_", " ")}: Please enter a value within the observed research-data range.`);
        document.getElementById(feature).focus();
        return false;
      }
    } else if (!(inputSchema.categorical_values[feature] || []).includes(value)) {
      showMessage(predictionError, `Choose a valid observed value for ${feature.replaceAll("_", " ")}.`);
      document.getElementById(feature).focus();
      return false;
    }
  }
  hideMessage(predictionError);
  return true;
}

async function predictStress(payload) {
  const data = await requestJson("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  validatePredictionResponse(data);
  return data;
}

function validatePredictionResponse(data) {
  if (typeof data.stress_level !== "string" || !CLASS_ORDER.includes(data.stress_level)) {
    throw new ApiError("The service returned an unexpected prediction format.");
  }
  if (data.probabilities !== undefined) {
    if (!data.probabilities || typeof data.probabilities !== "object") {
      throw new ApiError("The service returned invalid probability values.");
    }
    const values = CLASS_ORDER.map((name) => data.probabilities[name]);
    if (values.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
      throw new ApiError("The service returned invalid probability values.");
    }
    const total = values.reduce((sum, value) => sum + value, 0);
    if (Math.abs(total - 1) > 0.02) {
      throw new ApiError("The service returned an incomplete probability distribution.");
    }
    if (Math.abs(total - 1) > 1e-5) {
      throw new ApiError("The service returned probabilities that do not sum to approximately 1.");
    }
  }
  if (data.input_quality && !["Good", "Caution"].includes(data.input_quality.status)) {
    throw new ApiError("The service returned an invalid input quality status.");
  }
}

async function explainPrediction(payload) {
  const data = await requestJson("/api/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  validatePredictionResponse(data);
  if (!Array.isArray(data.explanation)) {
    throw new ApiError("The service returned an unexpected explanation format.");
  }
  for (const item of data.explanation) {
    if (!item || typeof item.feature !== "string" || !Number.isFinite(item.shap_value)
      || !["increases_prediction", "decreases_prediction"].includes(item.direction)) {
      throw new ApiError("The service returned an invalid explanation item.");
    }
  }
  return data;
}

function renderPrediction(data) {
  document.getElementById("predicted-level").textContent = data.stress_level;
  const probabilitySection = document.getElementById("probability-section");
  const list = document.getElementById("probability-list");
  list.replaceChildren();
  if (data.probabilities) {
    for (const className of CLASS_ORDER) {
      const probability = data.probabilities[className];
      const row = document.createElement("div");
      row.className = "probability-row";
      const label = document.createElement("span");
      label.className = "probability-label";
      label.textContent = className;
      const track = document.createElement("div");
      track.className = "probability-track";
      track.setAttribute("role", "img");
      track.setAttribute("aria-label", `${className}: ${formatProbability(probability)}`);
      const fill = document.createElement("div");
      fill.className = "probability-fill";
      fill.style.width = `${probability * 100}%`;
      track.append(fill);
      const value = document.createElement("span");
      value.className = "probability-value";
      value.textContent = formatProbability(probability);
      row.append(label, track, value);
      list.append(row);
    }
    probabilitySection.hidden = false;
  } else {
    probabilitySection.hidden = true;
  }
  resultCard.hidden = false;
  explanationCard.hidden = true;
  explainButton.disabled = false;
  renderInputQuality(data.input_quality);
  renderRecommendations(lastSubmittedInput, data);
  renderProfileComparison(lastSubmittedInput, data);
  hideMessage(predictionError);
  hideMessage(explanationError);
}

function renderRecommendations(input, prediction) {
  const result = window.StressRecommendations.buildRecommendations(
    input, prediction.stress_level, inputSchema.numeric_distributions,
  );
  const card = document.getElementById("recommendation-card");
  const badge = document.getElementById("recommendation-level");
  badge.textContent = result.stress_level;
  badge.dataset.level = result.stress_level;
  document.getElementById("recommendation-explanation").textContent = result.explanation;
  document.getElementById("recommendation-disclaimer").textContent = result.disclaimer;
  const tags = document.getElementById("priority-tags");
  tags.replaceChildren();
  for (const area of result.priority_areas) {
    const tag = document.createElement("span");
    tag.className = "priority-tag";
    tag.textContent = area;
    tags.append(tag);
  }
  const list = document.getElementById("recommendation-list");
  list.replaceChildren();
  for (const recommendation of result.recommendations) {
    const item = document.createElement("li");
    item.className = "recommendation-item";
    const icon = document.createElement("span");
    icon.className = "recommendation-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = recommendation.icon;
    const content = document.createElement("div");
    const heading = document.createElement("h3");
    heading.textContent = recommendation.area;
    const text = document.createElement("p");
    text.textContent = recommendation.text;
    content.append(heading, text);
    item.append(icon, content);
    list.append(item);
  }
  const support = document.getElementById("high-stress-support");
  support.hidden = !result.support_note;
  support.textContent = result.support_note || "";
  card.hidden = false;
}

async function loadReferenceProfile() {
  try {
    const data = await requestJson("/api/reference-profile");
    if (data.label !== "Reference Dataset Profile" || !data.profile || !data.prediction) {
      throw new ApiError("The reference profile response is incomplete.");
    }
    validatePredictionResponse(data.prediction);
    referenceProfile = data;
    if (completedInput) renderProfileComparison(completedInput);
  } catch (_error) {
    referenceProfile = null;
  }
}

function renderProfileComparison(input) {
  completedInput = { ...input };
  const idle = document.getElementById("visualization-idle");
  const content = document.getElementById("visualization-content");
  if (!referenceProfile || !inputSchema) {
    idle.hidden = false;
    idle.textContent = "Loading the reference dataset profile...";
    content.hidden = true;
    return;
  }
  const comparison = window.ProfileComparison.buildComparison(input, referenceProfile.profile, inputSchema.numeric_ranges);
  const chart = document.getElementById("profile-comparison-chart");
  chart.replaceChildren();
  for (const item of comparison) {
    const row = document.createElement("div");
    row.className = "comparison-row";
    row.setAttribute("role", "listitem");
    row.tabIndex = 0;
    row.dataset.tooltip = `${item.label}\nCurrent: ${item.currentLabel}\nReference: ${item.referenceLabel}\nRelative position within observed dataset range (%): current ${item.currentPercent}%, reference ${item.referencePercent}%`;
    row.setAttribute("aria-label", row.dataset.tooltip.replaceAll("\n", ". "));
    const heading = document.createElement("div");
    heading.className = "comparison-row-heading";
    const label = document.createElement("span");
    label.textContent = item.label;
    const values = document.createElement("span");
    values.textContent = `Current ${item.currentLabel} | Reference ${item.referenceLabel}`;
    heading.append(label, values);
    const tracks = document.createElement("div");
    tracks.className = "comparison-tracks";
    for (const [series, percent] of [["current", item.currentPercent], ["reference", item.referencePercent]]) {
      const track = document.createElement("div");
      track.className = `comparison-track ${series}`;
      track.setAttribute("role", "img");
      track.setAttribute("aria-label", `${series === "current" ? "Current Profile" : "Reference Dataset Profile"}: ${percent}% relative position`);
      const fill = document.createElement("span");
      fill.style.width = `${percent}%`;
      track.append(fill);
      tracks.append(track);
    }
    row.append(heading, tracks);
    row.addEventListener("mouseenter", () => showComparisonTooltip(row.dataset.tooltip));
    row.addEventListener("focus", () => showComparisonTooltip(row.dataset.tooltip));
    chart.append(row);
  }
  document.getElementById("current-profile-summary").textContent = comparison.map((item) => `${item.label}: current ${item.currentLabel}, reference ${item.referenceLabel}`).join("; ");
  document.getElementById("visualization-text-summary").textContent = "Current profile compared with the reference dataset profile across sleep, academic pressure, social media use, exercise frequency, study hours, and screen time. Relative positions show location within each observed feature range; they are not stress scores.";
  idle.hidden = true;
  content.hidden = false;
  comparisonCard.hidden = false;
}

function showComparisonTooltip(message) {
  document.getElementById("profile-comparison-tooltip").textContent = message;
}

function clearPersonalizedResults() {
  completedInput = null;
  recommendationCard.hidden = true;
  document.getElementById("recommendation-list").replaceChildren();
  document.getElementById("priority-tags").replaceChildren();
  document.getElementById("high-stress-support").hidden = true;
  comparisonCard.hidden = true;
  document.getElementById("visualization-content").hidden = true;
  const idle = document.getElementById("visualization-idle");
  idle.textContent = "Enter student information and run a prediction to compare profiles.";
  idle.hidden = false;
  document.getElementById("profile-comparison-chart").replaceChildren();
  document.getElementById("profile-comparison-tooltip").textContent = "Focus or point to a factor to inspect its values.";
}

async function loadProjectFacts() {
  try {
    const facts = await requestJson("/api/research-analytics");
    if (facts.dataset) document.getElementById("fact-students").textContent = facts.dataset.total_students.toLocaleString();
    if (facts.model_comparison) document.getElementById("fact-models").textContent = String(facts.model_comparison.length);
    if (facts.nested_cv_folds) document.getElementById("fact-cv").textContent = `${facts.nested_cv_folds.outer} \u00D7 ${facts.nested_cv_folds.inner}`;
  } catch (_error) {
    // Project fact cards stay unobtrusive if the generated research files are unavailable.
  }
}

function renderInputQuality(quality) {
  const section = document.getElementById("input-quality");
  section.hidden = !quality;
  if (!quality) return;
  document.getElementById("input-quality-status").textContent = quality.status;
  document.getElementById("input-quality-message").textContent = quality.message;
  document.getElementById("input-quality-method").textContent = quality.method;
  section.classList.toggle("is-caution", quality.status === "Caution");
}

async function loadExampleInput() {
  const data = await requestJson("/api/example-input");
  if (!data.example || !inputSchema) throw new ApiError("An example input is unavailable.");
  for (const feature of inputSchema.features) {
    const field = document.getElementById(feature);
    field.value = data.example[feature];
  }
  lastSubmittedInput = null;
  clearPersonalizedResults();
  resultCard.hidden = true;
  explanationCard.hidden = true;
  hideMessage(predictionError);
  hideMessage(explanationError);
}

function formatProbability(value) {
  return new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 }).format(value);
}

function renderExplanation(data) {
  const chart = document.getElementById("shap-chart");
  chart.replaceChildren();
  document.getElementById("explained-class").textContent = `Contributions for ${data.explained_class || data.stress_level}`;
  document.getElementById("explanation-note").textContent = data.note || "SHAP contributions describe model behavior; they are not causal effects.";
  const items = data.explanation;
  if (items.length === 0) {
    const empty = document.createElement("p");
    empty.className = "no-contributions";
    empty.textContent = "No non-zero feature contributions were returned for this prediction.";
    chart.append(empty);
  } else {
    const maxAbs = Math.max(...items.map((item) => Math.abs(item.shap_value)), Number.EPSILON);
    for (const item of items) chart.append(createContributionRow(item, maxAbs));
  }
  explanationCard.hidden = false;
  hideMessage(explanationError);
}

function createContributionRow(item, maxAbs) {
  const row = document.createElement("div");
  row.className = "shap-row";
  row.setAttribute("role", "listitem");
  const label = document.createElement("span");
  label.className = "shap-feature";
  label.textContent = item.feature.replaceAll("_", " ");
  const plot = document.createElement("div");
  plot.className = "shap-plot";
  plot.setAttribute("aria-hidden", "true");
  const negative = document.createElement("div");
  negative.className = "shap-half negative";
  const zero = document.createElement("span");
  zero.className = "shap-zero";
  const positive = document.createElement("div");
  positive.className = "shap-half positive";
  const bar = document.createElement("span");
  const magnitude = `${Math.max(2, (Math.abs(item.shap_value) / maxAbs) * 100)}%`;
  const isPositive = item.shap_value >= 0;
  bar.className = `shap-bar ${isPositive ? "positive" : "negative"}`;
  bar.style.width = magnitude;
  (isPositive ? positive : negative).append(bar);
  plot.append(negative, zero, positive);
  const values = document.createElement("div");
  values.className = "shap-value-row";
  const formatted = `${item.shap_value > 0 ? "+" : ""}${item.shap_value.toFixed(4)}`;
  const direction = document.createElement("span");
  direction.className = `direction-label ${isPositive ? "positive-text" : "negative-text"}`;
  direction.textContent = isPositive ? "Increases prediction" : "Decreases prediction";
  const amount = document.createElement("span");
  amount.textContent = formatted;
  values.append(direction, amount);
  row.append(label, plot, values);
  return row;
}

function showMessage(element, message) {
  element.textContent = message;
  element.hidden = false;
}

function hideMessage(element) {
  element.textContent = "";
  element.hidden = true;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!validateForm()) return;
  const generation = ++requestGeneration;
  const payload = collectInput();
  lastSubmittedInput = payload;
  clearPersonalizedResults();
  predictButton.disabled = true;
  predictButton.classList.add("is-loading");
  predictButton.querySelector("span:first-child").textContent = "Analyzing...";
  explainButton.disabled = true;
  resultCard.hidden = true;
  explanationCard.hidden = true;
  hideMessage(predictionError);
  try {
    const result = await predictStress(payload);
    if (generation !== requestGeneration) return;
    renderPrediction(result);
  } catch (error) {
    if (generation !== requestGeneration) return;
    showMessage(predictionError, error.message || "The prediction could not be completed.");
      if (error.status === 503 || error.message.toLowerCase().includes("connect")) {
        connectionMessage.hidden = false;
        setApiStatus("Offline", false);
    }
  } finally {
    if (generation === requestGeneration) {
      predictButton.disabled = !inputSchema || statusText.textContent !== "Connected";
      predictButton.classList.remove("is-loading");
      predictButton.querySelector("span:first-child").textContent = "Predict Stress Level";
    }
  }
});

explainButton.addEventListener("click", async () => {
  if (!lastSubmittedInput) return;
  const generation = requestGeneration;
  explainButton.disabled = true;
  explainButton.textContent = "Preparing explanation...";
  hideMessage(explanationError);
  try {
    const result = await explainPrediction(lastSubmittedInput);
    if (generation !== requestGeneration) return;
    renderExplanation(result);
  } catch (error) {
    if (generation !== requestGeneration) return;
    showMessage(explanationError, error.message || "The explanation could not be generated.");
  } finally {
    if (generation === requestGeneration) {
      explainButton.disabled = false;
      explainButton.textContent = "Explain This Prediction";
    }
  }
});

form.addEventListener("reset", () => {
  requestGeneration += 1;
  window.setTimeout(() => {
    lastSubmittedInput = null;
    clearPersonalizedResults();
    form.classList.remove("was-validated");
    resultCard.hidden = true;
    explanationCard.hidden = true;
    predictButton.disabled = !inputSchema || statusText.textContent !== "Connected";
    predictButton.querySelector("span:first-child").textContent = "Predict Stress Level";
    explainButton.disabled = true;
    explainButton.textContent = "Explain This Prediction";
    hideMessage(predictionError);
    hideMessage(explanationError);
    document.getElementById("probability-list").replaceChildren();
    document.getElementById("shap-chart").replaceChildren();
    document.getElementById("probability-section").hidden = true;
    document.getElementById("input-quality").hidden = true;
}, 0);
});

document.getElementById("load-example-button").addEventListener("click", async () => {
  try {
    await loadExampleInput();
  } catch (error) {
    showMessage(predictionError, error.message || "Example input could not be loaded.");
  }
});

form.addEventListener("input", () => {
  if (!lastSubmittedInput) return;
  lastSubmittedInput = null;
  clearPersonalizedResults();
  resultCard.hidden = true;
  explanationCard.hidden = true;
  explainButton.disabled = true;
  hideMessage(predictionError);
  hideMessage(explanationError);
});

window.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([checkHealth(), getModelInfo(), loadInputSchema(), loadReferenceProfile(), loadProjectFacts()]);
});
