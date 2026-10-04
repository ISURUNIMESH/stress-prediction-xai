# Imbalance-aware experiment: generated results summary

## A. Baseline and imbalance-aware performance

The baseline outputs under `results/metrics/`, `results/figures/`, and `results/shap_plots/` were read as inputs and not overwritten. The three sklearn model specifications already used `class_weight='balanced'` in the baseline. XGBoost is the only estimator receiving a new training-partition-derived sample-weight treatment.

The Macro-F1/CV selection rule selected **SVM** (10-fold Macro F1 0.3374 ± 0.0397; Balanced Accuracy 0.3587 ± 0.0590). Its held-out Macro F1 is 0.3244, Balanced Accuracy 0.3299.

| Model | Baseline test Macro F1 | Aware test Macro F1 | Baseline CV Macro F1 | Aware CV Macro F1 | Aware test balanced accuracy |
|---|---:|---:|---:|---:|---:|
| SVM | 0.3244 | 0.3244 | 0.3374 ± 0.0397 | 0.3374 ± 0.0397 | 0.3299 |
| XGBoost | 0.3102 | 0.3211 | 0.3099 ± 0.0009 | 0.3290 ± 0.0533 | 0.3187 |
| Random Forest | 0.3102 | 0.3102 | 0.3103 ± 0.0007 | 0.3103 ± 0.0007 | 0.3333 |
| Logistic Regression | 0.2496 | 0.2496 | 0.2458 ± 0.0284 | 0.2458 ± 0.0284 | 0.3458 |

## B. Majority-class baseline

Always predicting Moderate Stress yielded test accuracy 0.8700, balanced accuracy 0.3333, Macro F1 0.3102, and weighted F1 0.8095. Per-class recall values are stored in `metrics/majority_class_baseline.csv`.

## C. Ablation

Highest CV Macro F1 ablation: G_All_Predictors (0.3374 ± 0.0397); Low Stress recall 0.1744, High Stress recall 0.1833. The full results include every specified feature-group configuration.

## D. Statistical comparisons and confidence intervals

Friedman Macro F1: statistic 17.1600, p=0.00065516, Kendall W=0.5720; significant at 0.05: True. Balanced Accuracy: statistic 0.8400, p=0.839878, Kendall W=0.0280; significant at 0.05: False. Pairwise post-hoc results, when applicable, use Holm-adjusted Wilcoxon tests. The selected model's held-out Macro F1 95% stratified-bootstrap CI is [0.2751, 0.3808], and Balanced Accuracy CI is [0.2641, 0.4077]. Fold tests have overlapping training partitions, which limits the independence assumption.

## E. Robustness

The two highest CV Macro-F1 candidates were reevaluated using seeds 42, 123, 2024, 2025, and 2026. SVM: Macro F1 0.3353 ± 0.0183, balanced accuracy 0.3479 ± 0.0233; XGBoost: Macro F1 0.3364 ± 0.0295, balanced accuracy 0.3365 ± 0.0320.

## F. Error and subgroup analysis

Structured actual-to-predicted error counts and descriptive profiles are in `errors/error_analysis.csv` for SVM, XGBoost. The largest error flows for selected SVM: Moderate Stress → High Stress: 45; Moderate Stress → Low Stress: 25; High Stress → Moderate Stress: 17. Subgroup sample counts are reported in `subgroup/subgroup_results.csv`; 3 subgroup(s) met the minimum total/class support thresholds. Other metrics were withheld. No causal or fairness claims are made.

## G. Calibration

Out-of-fold multiclass Brier score for selected SVM: 0.2344. Candidate Brier scores and one-vs-rest calibration curves are saved under `calibration/`; models were not recalibrated.

## H. SHAP and SHAP stability

After CV-based selection, SVM was explained. The five highest global SHAP features in this run are: numeric__Sleep_Hours (0.00405994), numeric__Academic_Pressure (0.00375362), numeric__Social_Media_Hours (0.00244119), numeric__Exercise_Frequency (0.00175318), numeric__Financial_Stress (0.00161629). Fold-wise feature importance, rank variability, and top-10 frequency are in `shap/shap_stability.csv`. SHAP values are model explanations, not causal effects.

## Limitations

The original baseline contains class weighting for Logistic Regression, SVM, and Random Forest; these are repeated settings rather than new interventions. XGBoost alone receives newly computed per-training-partition class sample weights. Subgroup results may be withheld for small groups. Fold-based inferential tests use ten overlapping CV training sets. Results describe this dataset and split protocol; they do not establish clinical validity or universal generalization.
