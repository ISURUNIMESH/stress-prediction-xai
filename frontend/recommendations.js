"use strict";

(function attachRecommendationBuilder(root, factory) {
  const api = factory();
  root.StressRecommendations = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(globalThis, function createRecommendationBuilder() {
  const GENERIC_ROUTINES = [
    { area: "Weekly rhythm", icon: "◷", text: "Plan a balanced weekly rhythm that leaves room for study, personal time, and rest." },
    { area: "Supportive connections", icon: "◎", text: "Keep in touch with people and campus resources you find supportive." },
  ];

  function deviation(value, feature, distributions) {
    const summary = distributions && distributions[feature];
    if (!summary) return 0;
    const spread = Math.max(summary.q3 - summary.q1, 1);
    return Math.abs(value - summary.median) / spread;
  }

  function buildRecommendations(input, stressLevel, distributions) {
    const fallbackAreas = new Set();
    const candidates = [];
    const add = (feature, area, icon, text) => candidates.push({
      feature, area, icon, text, priority: deviation(Number(input[feature]), feature, distributions),
    });

    if (input.Sleep_Hours <= 5) {
      add("Sleep_Hours", "Sleep", "◷", `You reported ${input.Sleep_Hours} hours of sleep. Consider a consistent sleep and wake schedule and gradually increasing your sleep time.`);
    } else if (input.Sleep_Hours === 6) {
      add("Sleep_Hours", "Sleep", "◷", "You reported 6 hours of sleep. A steadier routine and gradual improvement in sleep duration may be worth considering.");
    }
    if (input.Academic_Pressure >= 4) {
      add("Academic_Pressure", "Academic pressure", "▤", `You reported academic pressure of ${input.Academic_Pressure}/5. Break larger tasks into smaller steps, schedule study sessions, and take short breaks.`);
    }
    if (input.Social_Media_Hours >= 6) {
      add("Social_Media_Hours", "Social media", "▣", `You reported ${input.Social_Media_Hours} hours of social-media use per day. Consider setting social-media-free periods during study or focus time.`);
    }
    if (input.Screen_Time_Hours >= 8) {
      add("Screen_Time_Hours", "Screen time", "▤", `You reported ${input.Screen_Time_Hours} hours of screen time per day. Consider regular screen breaks and reducing unnecessary screen exposure where practical.`);
    }
    if (input.Exercise_Frequency <= 2) {
      add("Exercise_Frequency", "Physical activity", "↗", `You reported exercise on ${input.Exercise_Frequency} day(s) per week. Consider adding manageable physical activity during the week.`);
    }
    if (input.Study_Hours >= 9) {
      add("Study_Hours", "Study balance", "▥", `You reported ${input.Study_Hours} study hours per day. Balance longer study periods with breaks, rest, and time away from study.`);
    }
    if (input.Financial_Stress >= 4) {
      add("Financial_Stress", "Financial support", "◇", `You reported financial stress of ${input.Financial_Stress}/5. Simple budgeting or planning may help organize concerns; ask about university or family support options where available.`);
    }
    if (input.Family_Support <= 2) {
      add("Family_Support", "Support network", "◎", `You reported family support of ${input.Family_Support}/5. Consider connecting with a trusted friend, relative, lecturer, or student-support service.`);
    }

    candidates.sort((left, right) => right.priority - left.priority);
    const chosen = candidates.slice(0, 3);
    for (const fallback of GENERIC_ROUTINES) {
      if (chosen.length >= 2) break;
      chosen.push({ ...fallback, feature: null, priority: 0 });
      fallbackAreas.add(fallback.area);
    }

    const explanations = {
      "Low Stress": "The model returned Low Stress for this submitted profile. Suggestions emphasize maintaining supportive routines.",
      "Moderate Stress": "The model returned Moderate Stress. Suggestions are drawn from the submitted factors and ordered by their deviation from observed dataset medians where available.",
      "High Stress": "The model returned High Stress. Suggestions focus on the submitted factors with the largest reference-profile deviations where available.",
    };
    return {
      stress_level: stressLevel,
      explanation: explanations[stressLevel] || "Suggestions are based on the submitted profile and model output.",
      priority_areas: chosen.filter((item) => !fallbackAreas.has(item.area)).map((item) => item.area),
      recommendations: chosen,
      support_note: stressLevel === "High Stress"
        ? "If stress feels difficult to manage or is affecting daily life, consider talking with a trusted person or an appropriate university or student-support service."
        : null,
      disclaimer: "These recommendations are general wellness guidance based on the submitted research-profile data. They are not a diagnosis or medical care.",
    };
  }

  return { buildRecommendations };
});
