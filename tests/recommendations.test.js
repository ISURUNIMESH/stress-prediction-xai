"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { buildRecommendations } = require("../frontend/recommendations.js");

const distributions = {
  Sleep_Hours: { median: 7, q1: 6, q3: 8 },
  Academic_Pressure: { median: 3, q1: 2, q3: 4 },
  Social_Media_Hours: { median: 4, q1: 2, q3: 6 },
  Screen_Time_Hours: { median: 6, q1: 4, q3: 8 },
  Exercise_Frequency: { median: 3, q1: 2, q3: 4 },
  Study_Hours: { median: 6, q1: 4, q3: 8 },
  Financial_Stress: { median: 3, q1: 2, q3: 4 },
  Family_Support: { median: 3, q1: 2, q3: 4 },
};

function typicalInput(overrides = {}) {
  return {
    Sleep_Hours: 7,
    Academic_Pressure: 3,
    Social_Media_Hours: 4,
    Screen_Time_Hours: 6,
    Exercise_Frequency: 3,
    Study_Hours: 6,
    Financial_Stress: 3,
    Family_Support: 3,
    ...overrides,
  };
}

test("recommendations target submitted factors and cap the list at three", () => {
  const input = typicalInput({
    Sleep_Hours: 4,
    Academic_Pressure: 5,
    Social_Media_Hours: 8,
    Screen_Time_Hours: 10,
    Exercise_Frequency: 1,
    Study_Hours: 10,
    Financial_Stress: 5,
    Family_Support: 1,
  });
  const original = { ...input };
  const result = buildRecommendations(input, "High Stress", distributions);
  assert.deepEqual(input, original);
  assert.equal(result.stress_level, "High Stress");
  assert.equal(result.recommendations.length, 3);
  assert.ok(result.recommendations.every((item) => item.feature in input));
  assert.ok(result.recommendations.some((item) => item.feature === "Sleep_Hours"));
  assert.ok(result.support_note);
  assert.match(result.disclaimer, /not a diagnosis or medical care/i);
});

test("recommendations adapt to a typical profile without inventing threshold alerts", () => {
  const result = buildRecommendations(typicalInput(), "Low Stress", distributions);
  assert.equal(result.stress_level, "Low Stress");
  assert.ok(result.recommendations.length >= 2 && result.recommendations.length <= 3);
  assert.ok(result.recommendations.every((item) => item.feature === null));
  assert.equal(result.support_note, null);
  assert.ok(!result.priority_areas.includes("Sleep"));
});

test("moderate stress suggestions respond to reported family support", () => {
  const result = buildRecommendations(typicalInput({ Family_Support: 1 }), "Moderate Stress", distributions);
  assert.equal(result.stress_level, "Moderate Stress");
  assert.ok(result.recommendations.some((item) => item.feature === "Family_Support"));
  assert.equal(result.support_note, null);
});
