"use strict";

(function attachProfileComparison(root, factory) {
  const api = factory();
  root.ProfileComparison = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(globalThis, function createProfileComparison() {
  const FEATURES = [
    ["Sleep_Hours", "Sleep Hours", "hours"],
    ["Academic_Pressure", "Academic Pressure", "/5"],
    ["Social_Media_Hours", "Social Media Hours", "hours/day"],
    ["Exercise_Frequency", "Exercise Frequency", "days/week"],
    ["Study_Hours", "Study Hours", "hours/day"],
    ["Screen_Time_Hours", "Screen Time Hours", "hours/day"],
  ];

  function buildComparison(current, reference, observedRanges) {
    return FEATURES.map(([feature, label, unit]) => {
      const range = observedRanges[feature];
      const span = range.maximum - range.minimum;
      const normalize = (value) => span === 0
        ? 0
        : Math.max(0, Math.min(100, ((Number(value) - range.minimum) / span) * 100));
      const currentValue = Number(current[feature]);
      const referenceValue = Number(reference[feature]);
      return {
        feature,
        label,
        unit,
        currentValue,
        referenceValue,
        currentPercent: Number(normalize(currentValue).toFixed(1)),
        referencePercent: Number(normalize(referenceValue).toFixed(1)),
        currentLabel: `${currentValue} ${unit}`,
        referenceLabel: `${referenceValue} ${unit}`,
      };
    });
  }

  return { buildComparison };
});
