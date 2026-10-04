# Improved imbalance experiment summary

This separate experiment leaves the baseline, imbalance-aware, nested-CV, SHAP, dataset, and application artifacts unchanged.

## Method

Outer and inner validation use stratified 5-fold CV (seed 42, shuffled). Selection uses inner mean Macro F1, then Balanced Accuracy as tie-breaker. Preprocessing and all model fits occur within each training fold. Validation folds are never resampled. Skipped because imbalanced-learn is not installed in the project Python 3.12 environment; no ordinary SMOTE on one-hot features.

## Nested CV results

Outer-fold mean +/- SD (five folds; descriptive only):

| Model | Macro F1 | Balanced Accuracy | Low recall | Moderate recall | High recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Current SVM | 0.3425 +/- 0.0215 | 0.3451 +/- 0.0242 | 0.0937 +/- 0.1078 | 0.8706 +/- 0.0213 | 0.0711 +/- 0.0264 | 0.7676 +/- 0.0107 |
| Tuned Class-Weighted SVM | 0.3340 +/- 0.0150 | 0.3433 +/- 0.0209 | 0.1353 +/- 0.1150 | 0.7932 +/- 0.0838 | 0.1016 +/- 0.0345 | 0.7311 +/- 0.0399 |
| RandomOverSampler + SVM | 0.3292 +/- 0.0190 | 0.3354 +/- 0.0126 | 0.0211 +/- 0.0288 | 0.9441 +/- 0.0120 | 0.0411 +/- 0.0230 | 0.7943 +/- 0.0075 |
| Random Forest Balanced | 0.3103 +/- 0.0003 | 0.3333 +/- 0.0000 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.8105 +/- 0.0021 |
| XGBoost Training Weights | 0.3048 +/- 0.0251 | 0.3300 +/- 0.0485 | 0.1663 +/- 0.0672 | 0.6294 +/- 0.0182 | 0.1942 +/- 0.1057 | 0.6446 +/- 0.0134 |
| Logistic Regression Balanced | 0.2532 +/- 0.0293 | 0.3603 +/- 0.0646 | 0.3021 +/- 0.1255 | 0.3714 +/- 0.0460 | 0.4074 +/- 0.1290 | 0.4671 +/- 0.0407 |

## Comparison against current SVM

By mean outer Macro F1, the highest measured configuration overall is **Current SVM**. The highest-Macro-F1 alternative is **Tuned Class-Weighted SVM**. Relative to Current SVM, that alternative changes Macro F1 by -0.0086, Balanced Accuracy by -0.0018, Low Stress recall by +0.0416, and High Stress recall by +0.0305.
Macro F1 was higher in 0 of five corresponding outer folds; this is descriptive, not an independent-fold significance test.

| Outer fold | Current SVM Macro F1 | Candidate Macro F1 | Difference |
|---:|---:|---:|---:|
| 1 | 0.3477 | 0.3477 | +0.0000 |
| 2 | 0.3215 | 0.3215 | +0.0000 |
| 3 | 0.3208 | 0.3147 | -0.0061 |
| 4 | 0.3714 | 0.3460 | -0.0254 |
| 5 | 0.3513 | 0.3399 | -0.0114 |

Five folds are not treated as independent participants. No p-value or confidence interval is calculated.

**Application model changed: No.** The separate nested comparison does not meet the conservative deployment gate (Macro F1 and Balanced Accuracy both higher, with Low and High Stress recall no worse). No model artifact has been replaced or exported in this experiment run.

## Limitations

This is internal validation on the same research sample, not external validation. Five outer folds give limited evidence about variability and no independent-fold uncertainty interval. The model-family winner is selected on these outer means and should be treated as exploratory. SMOTENC candidates were skipped because imbalanced-learn was unavailable; no invalid one-hot SMOTE was used.
