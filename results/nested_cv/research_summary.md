# Nested Stratified Cross-Validation

## Purpose

This experiment estimates candidate-model performance using nested stratified cross-validation. Nested CV is internal validation, not independent external validation; it does not establish performance at other universities or populations.

## Method

The processed dataset contains 1500 observations. Outer and inner loops both use 5-fold StratifiedKFold with shuffle=True and random_state=42. Each outer fold is held out while hyperparameters are selected using only its outer-training partition.

## Leakage prevention

Stress_Level is the target. PSS_Q1–PSS_Q10, PSS10_Total, Timestamp, and Consent are excluded from predictors. Imputation, scaling, and one-hot encoding are fitted in the pipeline on each fit's training partition. XGBoost sample weights are recalculated from that fit's training labels only. No binary scale_pos_weight is used.

## Model selection procedure

Each outer training partition uses a 5-fold stratified inner CV. A small fixed two-candidate grid is evaluated for each model. Mean inner Macro F1 is primary and mean inner Balanced Accuracy is the tie-breaker. The selected parameters are then refit on the full outer training partition and evaluated once on its outer validation fold.

## Nested CV results

Mean ± SD across five outer folds:

| Model | Macro F1 | Balanced Accuracy | Low recall | High recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2473 ± 0.0333 | 0.3578 ± 0.0625 | 0.3021 ± 0.1077 | 0.4174 ± 0.1253 |
| SVM | 0.3425 ± 0.0215 | 0.3451 ± 0.0242 | 0.0937 ± 0.1078 | 0.0711 ± 0.0264 |
| Random Forest | 0.3102 ± 0.0004 | 0.3331 ± 0.0006 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| XGBoost | 0.3136 ± 0.0255 | 0.3192 ± 0.0260 | 0.0937 ± 0.0940 | 0.0921 ± 0.0237 |

By the primary outer mean Macro F1 criterion (Balanced Accuracy tie-breaker), **SVM** ranks highest among these candidates. Since the same outer estimates are compared across four model families, interpret the winner ranking as exploratory; nested CV protects hyperparameter selection within each family but does not remove all model-family selection optimism.

No 95% confidence intervals are reported. Five outer folds have overlapping training sets; treating these five scores as independent observations would not justify a conventional fold-based interval. The reported SD describes fold-to-fold variation only.

## Comparison with previous CV

The CSV compares nested outer means with baseline and imbalance-aware 10-fold summaries. These are different CV designs and are reported descriptively, not as paired tests.

| Model | Nested Macro F1 | Baseline 10-fold Macro F1 | Imbalance-aware 10-fold Macro F1 | Nested Balanced Accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2473 ± 0.0333 | 0.2458 ± 0.0284 | 0.2458 ± 0.0284 | 0.3578 ± 0.0625 |
| SVM | 0.3425 ± 0.0215 | 0.3374 ± 0.0397 | 0.3374 ± 0.0397 | 0.3451 ± 0.0242 |
| Random Forest | 0.3102 ± 0.0004 | 0.3103 ± 0.0007 | 0.3103 ± 0.0007 | 0.3331 ± 0.0006 |
| XGBoost | 0.3136 ± 0.0255 | 0.3099 ± 0.0009 | 0.3290 ± 0.0533 | 0.3192 ± 0.0260 |

## Minority-class performance

Low and High Stress precision, recall, and F1 are reported as outer-fold mean ± SD in `nested_cv_results.csv`; aggregate out-of-fold reports and confusion matrices are also saved. Interpret minority-class values in light of their smaller supports.

## Limitations

Nested CV is internal validation, not external validation. The dataset is from one study sample; these results do not demonstrate generalization to other universities or populations. Only two predeclared candidates per model were searched, so the selected settings are conditional on these small grids. Comparing four model families on the same outer folds may introduce winner-selection optimism. Five outer folds provide a limited view of performance variability, and no fold-independence confidence intervals are claimed.

Reproduction command: `.venv\Scripts\python.exe -m experiments.nested_stratified_cv`
