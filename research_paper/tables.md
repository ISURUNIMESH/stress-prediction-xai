# Publication Tables

## Table 1. Observed stress-class distribution

| Operational class | Count | Proportion |
|---|---:|---:|
| Low Stress | 96 | 6.40% |
| Moderate Stress | 1,306 | 87.07% |
| High Stress | 98 | 6.53% |
| **Total** | **1,500** | **100.00%** |

Source: `results/metrics/class_distribution.csv`.

## Table 2. Predictor definitions

| Group | Predictors |
|---|---|
| Demographic | Age; Gender; University |
| Academic | Degree_Program; Academic_Year; Study_Hours; Academic_Pressure |
| Lifestyle | Sleep_Hours; Exercise_Frequency; Financial_Stress; Family_Support |
| Social/behavioural | Screen_Time_Hours; Social_Media_Hours |

PSS items, `PSS10_Total`, `Stress_Level`, `Timestamp`, and `Consent` are excluded from ML predictors.

## Table 3. Baseline held-out performance

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| SVM | 0.6533 | 0.3299 | 0.3244 | 0.6985 |
| Random Forest | 0.8700 | 0.3333 | 0.3102 | 0.8095 |
| XGBoost | 0.8700 | 0.3333 | 0.3102 | 0.8095 |
| Logistic Regression | 0.3667 | 0.3458 | 0.2496 | 0.4662 |

Source: `results/metrics/model_comparison.csv`.

## Table 4. Nested 5x5 cross-validation

| Model | Macro F1 (mean +/- SD) | Balanced Accuracy (mean +/- SD) | Low recall | High recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2473 +/- 0.0333 | 0.3578 +/- 0.0625 | 0.3021 +/- 0.1077 | 0.4174 +/- 0.1253 |
| SVM | **0.3425 +/- 0.0215** | 0.3451 +/- 0.0242 | 0.0937 +/- 0.1078 | 0.0711 +/- 0.0264 |
| Random Forest | 0.3102 +/- 0.0004 | 0.3331 +/- 0.0006 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| XGBoost | 0.3136 +/- 0.0255 | 0.3192 +/- 0.0260 | 0.0937 +/- 0.0940 | 0.0921 +/- 0.0237 |

Source: `results/nested_cv/research_summary.md`.

## Table 5. Imbalance-aware nested experiment

| Configuration | Macro F1 | Balanced Accuracy | Low recall | High recall |
|---|---:|---:|---:|---:|
| Current SVM | 0.3425 +/- 0.0215 | 0.3451 +/- 0.0242 | 0.0937 +/- 0.1078 | 0.0711 +/- 0.0264 |
| Tuned class-weighted SVM | 0.3340 +/- 0.0150 | 0.3433 +/- 0.0209 | 0.1353 +/- 0.1150 | 0.1016 +/- 0.0345 |
| RandomOverSampler + SVM | 0.3292 +/- 0.0190 | 0.3354 +/- 0.0126 | 0.0211 +/- 0.0288 | 0.0411 +/- 0.0230 |
| XGBoost training weights | 0.3048 +/- 0.0251 | 0.3300 +/- 0.0485 | 0.1663 +/- 0.0672 | 0.1942 +/- 0.1057 |

Source: `results/imbalance_improved/research_summary.md`.

## Table 6. Ablation results

| Feature group | Macro F1 | Balanced Accuracy | Low recall | High recall |
|---|---:|---:|---:|---:|
| Demographic | 0.2263 +/- 0.0328 | 0.3271 +/- 0.0734 | 0.2889 | 0.3700 |
| Demographic + Academic | 0.3119 +/- 0.0281 | 0.3501 +/- 0.0419 | 0.2256 | 0.2144 |
| Demographic + Lifestyle | 0.2853 +/- 0.0327 | 0.3439 +/- 0.0645 | 0.2367 | 0.2889 |
| Demographic + Social/Behavioural | 0.2434 +/- 0.0316 | 0.2764 +/- 0.0613 | 0.1756 | 0.2033 |
| Academic + Lifestyle | 0.3028 +/- 0.0319 | 0.3544 +/- 0.0648 | 0.2167 | 0.2844 |
| Academic + Social/Behavioural | 0.2996 +/- 0.0304 | **0.3929 +/- 0.0684** | 0.3544 | 0.3356 |
| All predictors | **0.3374 +/- 0.0397** | 0.3587 +/- 0.0590 | 0.1744 | 0.1833 |

Source: `results/imbalance_aware/ablation/ablation_results.csv`.

## Table 7. Global SHAP importance

| Rank | Transformed feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | numeric__Sleep_Hours | 0.00405994 |
| 2 | numeric__Academic_Pressure | 0.00375362 |
| 3 | numeric__Social_Media_Hours | 0.00244119 |
| 4 | numeric__Exercise_Frequency | 0.00175318 |
| 5 | numeric__Financial_Stress | 0.00161629 |

Source: `results/imbalance_aware/shap/global_feature_importance.csv`.
