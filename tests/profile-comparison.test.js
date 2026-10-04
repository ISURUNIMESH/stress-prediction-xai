"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { buildComparison } = require("../frontend/profile-comparison.js");

const ranges = {
  Sleep_Hours: { minimum: 3, maximum: 12 },
  Academic_Pressure: { minimum: 1, maximum: 5 },
  Social_Media_Hours: { minimum: 0, maximum: 12 },
  Exercise_Frequency: { minimum: 0, maximum: 7 },
  Study_Hours: { minimum: 0, maximum: 14 },
  Screen_Time_Hours: { minimum: 0, maximum: 16 },
};
const current = {
  Sleep_Hours: 5, Academic_Pressure: 5, Social_Media_Hours: 7,
  Exercise_Frequency: 1, Study_Hours: 8, Screen_Time_Hours: 9,
};
const reference = {
  Sleep_Hours: 7, Academic_Pressure: 3, Social_Media_Hours: 4,
  Exercise_Frequency: 3, Study_Hours: 6, Screen_Time_Hours: 6,
};

test("comparison normalizes each allowed factor against its observed range", () => {
  const rows = buildComparison(current, reference, ranges);
  assert.deepEqual(rows.map((row) => row.feature), Object.keys(ranges));
  assert.equal(rows.length, 6);
  assert.equal(rows.find((row) => row.feature === "Sleep_Hours").currentPercent, 22.2);
  assert.equal(rows.find((row) => row.feature === "Sleep_Hours").referencePercent, 44.4);
  assert.equal(rows.find((row) => row.feature === "Academic_Pressure").currentPercent, 100);
  assert.ok(rows.every((row) => row.currentPercent >= 0 && row.currentPercent <= 100));
  assert.ok(rows.every((row) => row.referencePercent >= 0 && row.referencePercent <= 100));
  assert.ok(!rows.some((row) => /PSS|Stress_Level|PSS10_Total/.test(row.feature)));
});

test("comparison handles a zero-width observed range without non-finite widths", () => {
  const fixedRanges = { ...ranges, Sleep_Hours: { minimum: 7, maximum: 7 } };
  const rows = buildComparison(current, reference, fixedRanges);
  const sleep = rows.find((row) => row.feature === "Sleep_Hours");
  assert.equal(sleep.currentPercent, 0);
  assert.equal(sleep.referencePercent, 0);
});
