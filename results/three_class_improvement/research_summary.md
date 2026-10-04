# Three-class improvement experiment

This experiment is separate from and does not overwrite baseline, prior imbalance, nested-CV, SHAP, dataset, or application results.

## Protocol

Nested stratified 5-fold outer and 5-fold inner CV, shuffled with random_state=42. All imputation, scaling, ordinal coding, one-hot encoding, SMOTENC, class-weight selection, SVM fitting, hyperparameter selection, and decision-threshold selection are confined to training/inner folds. Outer folds are not used for tuning.

SMOTENC operates on numeric-scaled plus ordinal-coded mixed training features, with categorical positions explicitly supplied; one-hot encoding occurs after resampling. SVM probabilities remain unchanged. Threshold vectors are selected from inner-validation probabilities and affect only the predicted label via argmax(p - threshold).

## Outer nested-CV results

Mean +/- SD across five outer folds (descriptive; folds are overlapping partitions, not independent populations):

| Model | Macro F1 | Balanced Accuracy | Low Recall | Moderate Recall | High Recall | Low F1 | Moderate F1 | High F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current SVM | 0.3425 +/- 0.0215 | 0.3451 +/- 0.0242 | 0.0937 +/- 0.1078 | 0.8706 +/- 0.0213 | 0.0711 +/- 0.0264 | 0.0886 +/- 0.0904 | 0.8698 +/- 0.0115 | 0.0691 +/- 0.0261 |
| SMOTENC + SVM | 0.3224 +/- 0.0114 | 0.3315 +/- 0.0078 | 0.0105 +/- 0.0235 | 0.9533 +/- 0.0127 | 0.0305 +/- 0.0279 | 0.0118 +/- 0.0263 | 0.9097 +/- 0.0041 | 0.0456 +/- 0.0416 |
| Threshold-tuned SMOTENC + SVM | 0.3303 +/- 0.0236 | 0.3358 +/- 0.0181 | 0.0211 +/- 0.0288 | 0.9242 +/- 0.0274 | 0.0621 +/- 0.0579 | 0.0225 +/- 0.0311 | 0.8963 +/- 0.0087 | 0.0723 +/- 0.0581 |
| SMOTENC + weighted SVM | 0.3304 +/- 0.0169 | 0.3312 +/- 0.0161 | 0.0621 +/- 0.0428 | 0.8698 +/- 0.0175 | 0.0616 +/- 0.0244 | 0.0633 +/- 0.0433 | 0.8688 +/- 0.0115 | 0.0591 +/- 0.0172 |
| Threshold-tuned SMOTENC + weighted SVM | 0.3264 +/- 0.0128 | 0.3270 +/- 0.0116 | 0.0516 +/- 0.0354 | 0.8782 +/- 0.0210 | 0.0511 +/- 0.0014 | 0.0542 +/- 0.0364 | 0.8718 +/- 0.0142 | 0.0532 +/- 0.0067 |
| Tuned weighted SVM | 0.3363 +/- 0.0140 | 0.3466 +/- 0.0191 | 0.1353 +/- 0.1150 | 0.7825 +/- 0.0702 | 0.1221 +/- 0.0256 | 0.0981 +/- 0.0711 | 0.8212 +/- 0.0402 | 0.0897 +/- 0.0276 |
| Threshold-tuned Tuned weighted SVM | 0.3363 +/- 0.0140 | 0.3466 +/- 0.0191 | 0.1353 +/- 0.1150 | 0.7825 +/- 0.0702 | 0.1221 +/- 0.0256 | 0.0981 +/- 0.0711 | 0.8212 +/- 0.0402 | 0.0897 +/- 0.0276 |
| SMOTENC + tuned SVM | 0.3191 +/- 0.0085 | 0.3213 +/- 0.0139 | 0.0521 +/- 0.0012 | 0.8186 +/- 0.0434 | 0.0932 +/- 0.0698 | 0.0472 +/- 0.0066 | 0.8396 +/- 0.0199 | 0.0703 +/- 0.0356 |
| Threshold-tuned SMOTENC + tuned SVM | 0.3220 +/- 0.0119 | 0.3224 +/- 0.0139 | 0.0626 +/- 0.0239 | 0.8530 +/- 0.0505 | 0.0516 +/- 0.0372 | 0.0626 +/- 0.0313 | 0.8576 +/- 0.0294 | 0.0457 +/- 0.0302 |

## Selection and interpretation

The highest-Macro-F1 non-reference candidate was **Tuned weighted SVM** (macro F1 0.3363; balanced accuracy 0.3466). Current SVM was 0.3425 Macro F1 and 0.3451 balanced accuracy.
Candidate Macro F1 exceeded the reference in 2/5 paired outer folds. Predeclared candidate gate: `{"all_classes_predicted_in_3_of_5_or_more_folds": true, "balanced_accuracy_not_lower": true, "macro_f1_higher_in_at_least_3_outer_folds": false, "macro_f1_improved": false, "mean_minority_recall_improved": true, "moderate_recall_drop_at_most_0_15": true}`.
**Candidate selected: No. Application model changed: No.** The candidate did not satisfy the full improvement gate; no candidate artifact was created.

No p-values or confidence intervals are calculated from five outer folds. This is internal validation on one research dataset and does not establish external generalization or causality.

SMOTENC sampling ratios and all class weights were predefined in experiment metadata. Threshold selection optimizes inner-fold Macro F1, with Balanced Accuracy as tie-breaker; outer labels were used only for evaluation.
