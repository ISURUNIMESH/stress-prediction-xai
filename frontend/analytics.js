"use strict";

const METHOD_LABELS = {
  stratified_10_fold_cv: "Stratified 10-Fold Cross-Validation",
  nested_5x5_stratified_cv: "Nested 5×5 Stratified Cross-Validation",
  multiple_seed_robustness: "Multiple-seed robustness analysis",
  held_out_test_evaluation: "Held-out test evaluation",
};

function number(value, digits = 4) {
  return Number.isFinite(value) ? value.toFixed(digits) : "Unavailable";
}

function addText(parent, tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = content;
  parent.append(element);
  return element;
}

function displayFeature(name) {
  if (name.startsWith("numeric__")) return name.slice("numeric__".length).replaceAll("_", " ");
  if (name.startsWith("categorical__")) {
    const value = name.slice("categorical__".length);
    for (const feature of ["Academic_Year", "Degree_Program", "University", "Gender"]) {
      if (value.startsWith(`${feature}_`)) {
        return `${feature.replaceAll("_", " ")}: ${value.slice(feature.length + 1)}`;
      }
    }
    return value.replaceAll("_", " ");
  }
  return name.replaceAll("_", " ");
}

function renderDataset(dataset) {
  if (!dataset) return;
  document.querySelector(".analytics-kpis").classList.add("has-data");
  document.getElementById("total-students").textContent = dataset.total_students.toLocaleString();
  document.getElementById("kpi-students").textContent = dataset.total_students.toLocaleString();
  const chart = document.getElementById("class-distribution");
  chart.replaceChildren();
  for (const item of dataset.classes) {
    const row = document.createElement("div");
    row.className = "class-row";
    const heading = document.createElement("div");
    heading.className = "class-row-heading";
    addText(heading, "span", "", item.name);
    addText(heading, "strong", "", `${item.count.toLocaleString()} · ${(item.proportion * 100).toFixed(1)}%`);
    const track = document.createElement("div");
    track.className = "distribution-track";
    track.setAttribute("role", "img");
    track.setAttribute("aria-label", `${item.name}: ${item.count} students, ${(item.proportion * 100).toFixed(1)} percent`);
    const bar = document.createElement("span");
    bar.className = `distribution-fill ${item.name === "Moderate Stress" ? "moderate" : "minority"}`;
    bar.style.width = `${Math.max(0, Math.min(100, item.proportion * 100))}%`;
    track.append(bar);
    row.append(heading, track);
    chart.append(row);
  }
}

function renderSelectedModel(selected) {
  if (!selected) return;
  document.querySelector(".analytics-kpis").classList.add("has-data");
  document.getElementById("selected-model-name").textContent = selected.name;
  document.getElementById("selection-criterion").textContent = selected.selection_criterion;
  document.getElementById("selected-macro-f1").textContent = `${number(selected.macro_f1.mean)} ± ${number(selected.macro_f1.std)}`;
  document.getElementById("selected-balanced-accuracy").textContent = `${number(selected.balanced_accuracy.mean)} ± ${number(selected.balanced_accuracy.std)}`;
  document.getElementById("kpi-model").textContent = selected.name;
  document.getElementById("kpi-macro-f1").textContent = `${number(selected.macro_f1.mean)} ± ${number(selected.macro_f1.std)}`;
  document.getElementById("kpi-balanced-accuracy").textContent = `${number(selected.balanced_accuracy.mean)} ± ${number(selected.balanced_accuracy.std)}`;
}

function renderModelComparison(models) {
  if (!models) return;
  const chart = document.getElementById("model-comparison-chart");
  chart.replaceChildren();
  for (const model of models) {
    const group = document.createElement("div");
    group.className = "model-chart-row";
    addText(group, "h3", "model-chart-name", model.model);
    for (const [key, label, className] of [
      ["macro_f1", "Macro F1", "macro"],
      ["balanced_accuracy", "Balanced Accuracy", "balanced"],
    ]) {
      const metric = model[key];
      const line = document.createElement("div");
      line.className = "metric-bar-line";
      addText(line, "span", "metric-bar-label", label);
      const track = document.createElement("div");
      track.className = "metric-bar-track";
      const fill = document.createElement("span");
      fill.className = `metric-bar-fill ${className}`;
      fill.style.width = `${Math.max(0, Math.min(100, metric.mean * 100))}%`;
      track.append(fill);
      addText(line, "strong", "metric-bar-value", `${number(metric.mean)} ± ${number(metric.std)}`);
      line.insertBefore(track, line.lastElementChild);
      group.append(line);
    }
    chart.append(group);
  }
}

function renderAblation(rows) {
  if (!rows) return;
  const body = document.getElementById("ablation-rows");
  body.replaceChildren();
  for (const item of rows) {
    const row = document.createElement("tr");
    const configuration = document.createElement("td");
    addText(configuration, "strong", "", item.configuration.replaceAll("_", " "));
    addText(configuration, "small", "table-subtext", item.feature_groups);
    const features = document.createElement("td");
    addText(features, "span", "feature-count", `${item.n_features} predictors`);
    const details = document.createElement("details");
    addText(details, "summary", "", "Show columns");
    addText(details, "p", "feature-list", item.features.join(", "));
    features.append(details);
    const macro = document.createElement("td");
    macro.textContent = `${number(item.macro_f1.mean)} ± ${number(item.macro_f1.std)}`;
    const balanced = document.createElement("td");
    balanced.textContent = `${number(item.balanced_accuracy.mean)} ± ${number(item.balanced_accuracy.std)}`;
    row.append(configuration, features, macro, balanced);
    body.append(row);
  }
}

function renderShap(shap) {
  if (!shap) return;
  document.getElementById("shap-model-note").textContent = `${shap.selected_model} · ${shap.explainer} · Mean absolute SHAP values from the generated global importance file.`;
  const chart = document.getElementById("shap-importance-chart");
  chart.replaceChildren();
  const maximum = Math.max(...shap.features.map((item) => item.mean_absolute_shap_value), Number.EPSILON);
  for (const item of shap.features) {
    const row = document.createElement("div");
    row.className = "shap-importance-row";
    addText(row, "span", "shap-importance-name", displayFeature(item.feature));
    const track = document.createElement("div");
    track.className = "shap-importance-track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(1, item.mean_absolute_shap_value / maximum * 100)}%`;
    track.append(fill);
    addText(row, "strong", "shap-importance-value", number(item.mean_absolute_shap_value, 6));
    row.insertBefore(track, row.lastElementChild);
    chart.append(row);
  }
}

function renderMethods(methods) {
  const list = document.getElementById("validation-methods");
  list.replaceChildren();
  for (const method of methods || []) {
    const item = document.createElement("li");
    item.classList.toggle("method-unavailable", !method.available);
    item.textContent = `${METHOD_LABELS[method.key] || method.key}${method.available ? " · results available" : " · result file unavailable"}`;
    list.append(item);
  }
}

function renderLimitations(data) {
  const list = document.getElementById("research-limitations");
  list.replaceChildren();
  if (data.dataset) {
    const moderate = data.dataset.classes.find((item) => item.name === "Moderate Stress");
    if (moderate) addText(list, "li", "", `Severe class imbalance: Moderate Stress is ${(moderate.proportion * 100).toFixed(1)}% of the dataset.`);
  }
  if (data.minority_class_recall) {
    const low = data.minority_class_recall["Low Stress"];
    const high = data.minority_class_recall["High Stress"];
    if (Number.isFinite(low) && Number.isFinite(high)) {
      addText(list, "li", "", `Minority-class recall remains limited in the selected model's held-out test results: Low Stress ${number(low)}, High Stress ${number(high)}.`);
    }
  }
  for (const limitation of data.limitations || []) addText(list, "li", "", limitation);
}

async function loadAnalytics() {
  const loading = document.getElementById("analytics-loading");
  const errors = document.getElementById("analytics-errors");
  try {
    const response = await fetch("/api/research-analytics", { headers: { Accept: "application/json" } });
    const data = await response.json();
    if (!response.ok || !data || typeof data !== "object") throw new Error("Research results could not be loaded.");
    renderDataset(data.dataset);
    renderSelectedModel(data.selected_model);
    renderModelComparison(data.model_comparison);
    renderAblation(data.ablation);
    renderShap(data.shap);
    renderMethods(data.validation_methods);
    renderLimitations(data);
    if (data.errors && data.errors.length) {
      errors.textContent = `Some generated result files are unavailable: ${data.errors.join(" ")}`;
      errors.hidden = false;
    }
    document.getElementById("analytics-content").hidden = false;
  } catch (error) {
    errors.textContent = error.message || "Research analytics are temporarily unavailable.";
    errors.hidden = false;
  } finally {
    loading.hidden = true;
  }
}

window.addEventListener("DOMContentLoaded", loadAnalytics);
