# A Comparative Explainable AI Framework for Stress Prediction Using Behavioral, Lifestyle, and Social Media Analytics Among Sri Lankan University Students

**Author information:** [AUTHOR INPUT REQUIRED: author names, affiliations, corresponding author, ORCID, funding, conflicts of interest]

## Abstract

### Background
University-student stress is a multifactorial phenomenon involving academic, lifestyle, social, and demographic context. This project evaluates whether routinely collected, non-PSS survey variables can support reproducible prediction of operational stress categories derived from the Perceived Stress Scale (PSS-10), while making the model behavior inspectable.

### Objective
The study compares Logistic Regression, radial-basis-function Support Vector Machine (SVM), Random Forest, and XGBoost classifiers; evaluates the effect of severe class imbalance; and uses SHAP explanations to describe model-attributed feature contributions.

### Methods
The implementation uses 1,500 Google Forms responses from Sri Lankan university students and 13 predictors spanning demographics, academic context, lifestyle, behaviour, and social-media use. PSS-10 items are reverse-scored exactly as implemented and mapped to Low (0–13), Moderate (14–26), and High (27–40) operational categories. Predictor preprocessing is contained in scikit-learn pipelines. Baseline evaluation uses a stratified 80/20 holdout and shuffled stratified 10-fold cross-validation. Model selection for the exported artifact uses nested 5x5 stratified cross-validation, with outer Macro F1 as the primary criterion and Balanced Accuracy as the tie-breaker. Additional experiments examine weighting, oversampling, SMOTENC, threshold tuning, ablation, robustness, calibration, and error patterns.

### Results
The class distribution was 96 Low, 1,306 Moderate, and 98 High observations. In nested 5x5 cross-validation, SVM had the highest mean Macro F1 (0.3425 +/- 0.0215); its Balanced Accuracy was 0.3451 +/- 0.0242, with Low and High recall of 0.0937 and 0.0711, respectively. The exported artifact is an RBF SVM selected by this criterion. In the imbalance-aware experiment, a Moderate-only majority baseline achieved 0.8700 accuracy but only 0.3102 Macro F1 and 0.3333 Balanced Accuracy. The highest mean Macro F1 non-reference candidate in the three-class improvement experiment was a tuned weighted SVM (0.3363 +/- 0.0140), which did not pass the predefined deployment gate and therefore did not replace the current artifact. The five largest mean absolute SHAP values in the imbalance-aware run were associated with Sleep_Hours, Academic_Pressure, Social_Media_Hours, Exercise_Frequency, and Financial_Stress.

### Conclusion
The repository supports a reproducible comparative research prototype, not a clinically validated diagnostic system. SVM ranking is conditional on the dataset and protocol, and minority-class recognition remains weak. The results show why accuracy alone is inadequate under the observed class distribution and why external, multi-university, longitudinal, and expert-reviewed validation is required.

**Keywords:** student stress; explainable artificial intelligence; SHAP; support vector machine; class imbalance; PSS-10; Sri Lanka; nested cross-validation

## 1. Introduction

### 1.1 Background
Perceived stress is commonly measured as a subjective appraisal of demands and coping resources. The PSS-10 is used here as a survey instrument and source of an operational target, not as a clinical diagnosis [1]. Student stress can be related to academic demands, financial circumstances, sleep, exercise, family support, and digital-media behaviour. A Sri Lankan medical-undergraduate study illustrates the local relevance of perceived-stress research, while differing from this project's population and predictors [2]. These variables are potentially useful for screening research questions, but observed model associations do not establish causes.

### 1.2 Problem statement
The project addresses multiclass prediction of PSS-10-derived stress categories from variables that are deliberately separated from the PSS questions. This separation is essential because using the PSS items or their total would directly reveal the target. The implementation also addresses the practical difficulty that Moderate Stress dominates the observed sample.

### 1.3 Research gap
The repository combines comparative classification, leakage-safe nested validation, explicit imbalance experiments, ablation, robustness and calibration analyses, and SHAP explanations in one reproducible implementation. Recent work shows that student-stress machine-learning evidence often uses physiological or wearable data and should be interpreted with attention to measurement and validation context [3]. The contribution is methodological and empirical for this sample; it does not establish population-wide or clinical performance.

### 1.4 Research questions

* **RQ1:** How do the four implemented classifiers compare for stress-level prediction using demographic, academic, behavioural, lifestyle, and social-media-related predictors?
* **RQ2:** Which transformed predictors receive the greatest model-attributed importance under SHAP?
* **RQ3:** How does severe class imbalance affect recognition of Low and High Stress classes?
* **RQ4:** How robust are the findings under leakage-safe nested cross-validation and imbalance-aware evaluation?

### 1.5 Contributions

1. A reproducible implementation using a Sri Lankan university-student survey sample.
2. A comparison of four classifier families under common preprocessing and evaluation rules.
3. Nested 5x5 validation and separate imbalance-aware experiments, including negative findings.
4. Feature-group ablation, robustness, calibration, error, and limited-support subgroup analyses.
5. SHAP-based global and local model explanations and a research-prototype API.

### 1.6 Paper organization
Section 2 reviews the methodological foundations used by the implementation. Section 3 describes the data, target construction, models, validation, imbalance procedures, explainability, ethics boundary, and reproducibility. Section 4 reports frozen results. Section 5 interprets them conservatively. Section 6 describes the prototype. Section 7 concludes.

## 2. Related Work

### 2.1 Student stress and measurement
Cohen, Kamarck, and Mermelstein introduced the Perceived Stress Scale as a global measure of perceived stress [1]. The present project uses the implemented PSS-10 scoring and explicitly treats its three ranges as study-operational categories. The repository does not contain evidence that these cutoffs are universally validated clinical thresholds; that justification remains an author input.

### 2.2 Machine-learning prediction
The model set deliberately spans a linear classifier, a kernel method, a bagged tree ensemble, and gradient boosting. Support-vector networks provide the theoretical basis for the SVM family [4], while XGBoost is a scalable gradient-boosting implementation [5]. The repository's comparison is an internal empirical comparison, not a claim that one family is universally superior.

### 2.3 Behavioural and lifestyle predictors
The implementation records sleep, exercise, screen time, social-media hours, study hours, academic pressure, financial stress, and family support alongside demographic and academic variables. Because the data are cross-sectional survey responses, the models can describe predictive associations but cannot identify temporal direction or causality.

### 2.4 Social-media analytics
Social-media hours and screen time are included as self-reported predictors. Reviews indicate that associations between social-media use and mental health are heterogeneous and depend on use patterns and study design [6]. Longitudinal student evidence also distinguishes problematic use from simple duration measures [7]. The repository does not contain platform-level posts, passive sensing, temporal logs, or validated social-media scales. Accordingly, this paper uses “social-media-related predictors” rather than claiming direct social-media content analysis.

### 2.5 Explainable AI
SHAP assigns additive feature-attribution values to model outputs and was introduced as a unified approach to interpreting predictions [8]. Reviews of explainable AI in healthcare describe SHAP as a widely used attribution method while also emphasizing interpretive and validation limitations [9]. In this project, SHAP values explain fitted model behaviour after preprocessing. They are not causal effects, clinical explanations, or evidence that changing a feature would change stress.

### 2.6 Class imbalance and validation
Imbalanced-data research shows that accuracy can be misleading when classes have unequal prevalence [10]. SMOTE is a synthetic oversampling method [11]; the project's three-class experiment uses training-fold-only SMOTENC with explicit categorical positions. Nested cross-validation is used to reduce optimistic bias from tuning and evaluation reuse [12]. The repository nevertheless cautions that five outer folds are overlapping and that the results are internal validation rather than external validation.

### 2.7 Research gap
The repository contains no prior literature review, dataset provenance citation, or validated local PSS-category justification. The preceding discussion therefore combines foundational methods with selected external context references; [AUTHOR INPUT REQUIRED: additional Sri Lankan/South Asian student-stress literature that directly matches the study population and dataset provenance references].

## 3. Methodology

### 3.1 Research design
This is a retrospective, cross-sectional, supervised multiclass modelling study using a fixed processed survey dataset. All reported results are generated artifacts in the repository. The raw and processed datasets are local inputs and are not part of the tracked source distribution.

### 3.2 Dataset and participants
The source is `dataset/raw/Stress Prediction Form.csv`, described by the repository as a Google Forms export with 1,500 responses and 25 original columns. The observed processed class counts are 96 Low, 1,306 Moderate, and 98 High. The repository does not document recruitment, eligibility, response rate, geographic sampling frame, or university-wise recruitment; these details are required before publication.

### 3.3 Data collection and variables
The documented fields include Timestamp, Consent, ten PSS items, and 13 non-PSS predictors. The predictors are Age, Gender, University, Degree_Program, Academic_Year, Sleep_Hours, Exercise_Frequency, Screen_Time_Hours, Social_Media_Hours, Study_Hours, Academic_Pressure, Financial_Stress, and Family_Support. Timestamp and Consent are administrative fields and are not predictors.

### 3.4 PSS-10 scoring
PSS_Q4, PSS_Q5, PSS_Q7, and PSS_Q8 are transformed as `4 - response`. The other six items remain unchanged. `PSS10_Total` is the sum of the ten scored responses and is validated against the 0–40 instrument range. The observed totals in this dataset range from 7 to 33.

### 3.5 Target construction
The implementation assigns Low Stress to 0–13, Moderate Stress to 14–26, and High Stress to 27–40. These are the operational categories used in this study and must not be described as universal clinical cutoffs without additional evidence.

### 3.6 Feature selection and leakage prevention
The target, PSS_Q1–PSS_Q10, PSS10_Total, Timestamp, and Consent are excluded. Consequently, no predictor is a direct component of the target calculation. The exclusions are persisted in `results/metrics/run_metadata.json` and application model metadata.

### 3.7 Preprocessing
Numeric predictors use median imputation; categorical predictors use most-frequent imputation and one-hot encoding with unknown-category handling. Logistic Regression and SVM additionally use StandardScaler for numeric predictors. These operations are inside pipelines and are fitted only on the relevant training partition or fold. No blanket outlier removal is applied.

### 3.8 Model development
The baseline models are:

* Logistic Regression (`C=1.0`, `max_iter=2000`, balanced class weights);
* RBF SVM with probability estimates (`C=1.0`, balanced class weights);
* Random Forest (`n_estimators=300`, `min_samples_leaf=2`, balanced class weights);
* XGBoost (`n_estimators=250`, `max_depth=4`, learning rate 0.05, subsample 0.8, column subsample 0.8).

The final exported model is an RBF SVM with `C=2.0`, `gamma=scale`, balanced class weights, and probability estimates, trained on all 1,500 processed rows after selection by nested CV.

### 3.9 Baseline and nested validation
Baseline evaluation uses a stratified 80/20 holdout with random state 42 shared across models and shuffled stratified 10-fold cross-validation with random state 42. Nested evaluation uses shuffled stratified 5-fold outer and inner loops, also with random state 42. The inner mean Macro F1 is primary and Balanced Accuracy is the tie-breaker; the selected model is refit on each outer training partition and evaluated once on its outer fold.

Nested CV is internal validation. It does not demonstrate generalization to other universities or populations. The five outer folds overlap in training observations, so fold standard deviations are descriptive and are not treated as independent-population uncertainty intervals.

### 3.10 Class-imbalance handling
The observed distribution is 87.07% Moderate Stress, with approximately 6.40% Low and 6.53% High. Baseline Logistic Regression, SVM, and Random Forest already use balanced class weights. Separate experiments evaluate XGBoost training-fold weights, tuned class-weighted SVM, RandomOverSampler, SMOTENC, and inner-fold threshold tuning. Resampling and threshold selection are confined to training/inner folds; validation folds are not resampled. SMOTENC candidates were unavailable in one improved-imbalance run because `imbalanced-learn` was not installed at that run's execution time.

### 3.11 Evaluation metrics
Reported metrics include accuracy, Balanced Accuracy, macro and weighted precision/recall/F1, and one-vs-rest ROC-AUC. Macro F1 gives each class equal weight and is the primary selection metric. Balanced Accuracy averages per-class recall. A majority-class baseline is reported to contextualize accuracy. Calibration is summarized using multiclass Brier score; models were not recalibrated.

### 3.12 SHAP explainability
The implementation uses TreeExplainer for Random Forest and XGBoost, LinearExplainer for Logistic Regression, and KernelExplainer for SVM. The final SVM uses a background artifact derived from 30 rows sampled from the stratified 80% training partition; the held-out 20% partition is excluded from that background. Global mean absolute SHAP values and local contributions are exported. All interpretations are model-attributed contributions, not causal claims.

### 3.13 Ethics, privacy, and data governance
The repository contains a Consent field and keeps participant datasets local rather than distributing them in Git. It does not document an ethics committee approval, exemption, recruitment procedure, anonymisation protocol, data owner, or consent wording. Therefore: **[AUTHOR INPUT REQUIRED: ethics approval or exemption, consent procedure, privacy/de-identification procedure, data ownership/access statement, and author-approved publication permissions]**. The application is explicitly documented as a research prototype and not a medical diagnosis.

### 3.14 Reproducibility
The environment is pinned for CPython 3.12, with versions recorded in `requirements.txt` and model metadata. The documented commands run preprocessing, baseline evaluation, SHAP analysis, nested CV, imbalance-aware experiments, and three-class improvement experiments. The final artifact is `application/artifacts/final_svm_pipeline.joblib`; its metadata records seed 42, 1,500 training rows, and the nested-CV selection rule. Existing Python and Node tests are documented in the README.

## 4. Results

### 4.1 Dataset characteristics
The dataset contains 1,500 responses and 25 original columns. The target distribution is shown in Table 1. No missing values or duplicate rows were present in the inspected source according to the repository documentation. The paper does not infer recruitment representativeness from these facts.

### 4.2 Baseline results
On the fixed held-out test set, SVM had Macro F1 0.3244 and Balanced Accuracy 0.3299. Random Forest and XGBoost each reached accuracy 0.8700 and Macro F1 0.3102, reflecting the dominance of Moderate Stress. Logistic Regression had the highest held-out Balanced Accuracy among the four (0.3458) but lower Macro F1 (0.2496). These are held-out results and must not be conflated with nested-CV estimates.

In 10-fold CV, SVM had Macro F1 0.3374 +/- 0.0397, compared with 0.2458 +/- 0.0284 for Logistic Regression, 0.3103 +/- 0.0007 for Random Forest, and 0.3099 +/- 0.0009 for XGBoost. The baseline selection rule used held-out Macro F1 with weighted F1 as tie-breaker; the deployed artifact instead follows the later nested-CV rule.

### 4.3 Nested 5x5 results
Table 2 reports the frozen outer-fold means and standard deviations. SVM ranked first by Macro F1 (0.3425 +/- 0.0215). Logistic Regression had the highest Balanced Accuracy (0.3578 +/- 0.0625), while its Macro F1 was lower. SVM Low and High recall were 0.0937 and 0.0711; Moderate recall was 0.8706. This is not evidence of reliable minority-class detection.

### 4.4 Imbalance and per-class results
A Moderate-only classifier achieved 0.8700 accuracy, but Balanced Accuracy was 0.3333 and Macro F1 was 0.3102. In the imbalance-aware 10-fold experiment, XGBoost training-fold weights increased its Macro F1 from 0.3099 +/- 0.0009 to 0.3290 +/- 0.0533, but SVM remained the selected model. The selected SVM's stratified-bootstrap held-out intervals were [0.2751, 0.3808] for Macro F1 and [0.2641, 0.4077] for Balanced Accuracy; these intervals belong to that specific held-out bootstrap analysis, not nested CV.

The Friedman test for Macro F1 was 17.1600 with p=0.00065516 and Kendall's W=0.5720. The corresponding Balanced Accuracy test was not significant (statistic 0.8400, p=0.839878, W=0.0280). Because CV training partitions overlap, these tests are reported as repository outputs with the documented independence limitation.

### 4.5 Improvement experiments
The tuned weighted SVM was the strongest non-reference candidate in the three-class improvement experiment (Macro F1 0.3363 +/- 0.0140; Balanced Accuracy 0.3466 +/- 0.0191; Low recall 0.1353 +/- 0.1150; High recall 0.1221 +/- 0.0256). It exceeded the current SVM in only 2 of 5 paired outer folds and failed the predefined candidate gate. SMOTENC and threshold variants also did not replace the artifact. The application model was unchanged.

### 4.6 Ablation
The full predictor group (demographic + academic + lifestyle + social-behavioural) had the highest 10-fold Macro F1: 0.3374 +/- 0.0397 and Balanced Accuracy 0.3587 +/- 0.0590. The Academic+Social_Behavioural configuration had the highest ablation Balanced Accuracy (0.3929 +/- 0.0684), but lower Macro F1 (0.2996 +/- 0.0304). The full group produced Low and High recall of 0.1744 and 0.1833 in that ablation output. These comparisons are descriptive and do not establish causal value for any feature group.

### 4.7 SHAP results
In the imbalance-aware SHAP run, the five largest mean absolute transformed-feature values were Sleep_Hours (0.00405994), Academic_Pressure (0.00375362), Social_Media_Hours (0.00244119), Exercise_Frequency (0.00175318), and Financial_Stress (0.00161629). These values quantify contribution to the fitted SVM output in the analysed data and representation. They do not indicate that these variables cause stress.

### 4.8 Robustness, errors, calibration, and subgroup analysis
Across seeds 42, 123, 2024, 2025, and 2026, SVM achieved Macro F1 0.3353 +/- 0.0183 and Balanced Accuracy 0.3479 +/- 0.0233; weighted XGBoost achieved 0.3364 +/- 0.0295 and 0.3365 +/- 0.0320. The selected SVM out-of-fold multiclass Brier score was 0.2344 and no recalibration was performed. The largest selected-SVM error flows were Moderate-to-High (45), Moderate-to-Low (25), and High-to-Moderate (17). Only three subgroups met the repository's minimum support thresholds; other subgroup metrics were withheld.

## 5. Discussion

### 5.1 Main findings
The nested protocol selected SVM by Macro F1, but its minority-class recall remained low. The results therefore support a constrained conclusion: the SVM was the strongest measured candidate under the stated protocol, not a highly accurate or clinically validated stress predictor.

### 5.2 Model comparison
The disagreement between accuracy and Macro F1 is central. Tree models often predicted the majority class and achieved high accuracy, whereas SVM and Logistic Regression traded accuracy for more varied predictions. Logistic Regression's higher nested Balanced Accuracy did not overcome its lower Macro F1 under the predeclared ranking rule.

### 5.3 Class imbalance
The Moderate-only baseline demonstrates that accuracy can largely reflect prevalence. Weighting and resampling changed the recall trade-off but did not improve the complete deployment gate. Increasing Low or High predictions alone would not establish improvement; the repository appropriately retained the existing SVM.

### 5.4 Explainability
Sleep, academic pressure, social-media hours, exercise, and financial stress received the highest global SHAP values in one frozen run. These are hypotheses about fitted-model behaviour that can motivate data-quality and validation work. They should not be interpreted as intervention targets or causes.

### 5.5 Context and implications
The predictors reflect domains plausibly relevant to student life, but the repository does not provide a sampling frame or external comparison. Practical use is therefore limited to research prototyping, reproducibility, and model-audit demonstrations. The API's input validation and explanation endpoints do not convert the system into a clinical tool.

### 5.6 Strengths
Strengths include explicit leakage exclusions, fold-contained preprocessing, a shared seed, nested selection, frozen negative findings, a majority baseline, ablation and robustness checks, and a clear non-diagnostic application boundary.

### 5.7 Limitations
The evidence supports the following limitations: one cross-sectional sample; severe class imbalance; only 96 Low and 98 High observations; self-reported survey variables; unavailable recruitment and response-rate documentation; no external or clinical validation; no causal inference; limited hyperparameter grids; winner-selection optimism across four families; descriptive rather than independent-fold uncertainty; weak and unstable minority-class recognition; withheld small-subgroup metrics; and no demonstrated fairness. The operational category thresholds also require author justification.

## 6. Prototype Implementation

### 6.1 System architecture
The repository separates preprocessing, model training, experiments, explainability, application services, and a browser frontend. The final artifact is a saved SVM pipeline rather than a dynamically retrained model.

### 6.2 API
The Flask REST API accepts the 13 allowed predictors, rejects PSS fields and invalid or unsupported values, returns the saved SVM prediction and probabilities, and exposes approximate runtime Kernel SHAP explanations. Input-quality checks include numeric Tukey fences, category support checks, and a mixed-type nearest-neighbour distance rule.

### 6.3 User interfaces
The HTML/CSS/JavaScript frontend provides a prediction interface, an explanation interface, and an analytics dashboard. These interfaces display research-prototype outputs and must not be presented as diagnosis or clinical deployment.

### 6.4 Reproducibility
The repository includes Python and Node test commands, pinned dependencies, experiment scripts, metadata, frozen CSV/JSON outputs, and explicit instructions not to overwrite archived results during reproduction. Access to the source dataset remains restricted to an authorized local copy.

## 7. Conclusion and Future Work

This study investigated four classifiers for PSS-10-derived operational stress categories using 13 non-PSS predictors from 1,500 Sri Lankan university-student responses. Under nested 5x5 validation, SVM had the highest Macro F1 (0.3425 +/- 0.0215), but Low and High recall were only 0.0937 and 0.0711. The class distribution and majority baseline show why accuracy is insufficient. SHAP made the fitted model's feature attributions inspectable, with Sleep_Hours and Academic_Pressure among the highest-ranked transformed features in the imbalance-aware run. No imbalance-aware candidate passed the deployment gate, so the application artifact was not changed.

Future work should obtain ethics and provenance documentation, justify the operational thresholds, collect larger and better-balanced samples, validate across universities, use longitudinal designs, improve minority-class recognition, evaluate subgroup fairness with sufficient support, and seek appropriate expert or clinical validation before any health-related use. These are future requirements, not properties established by this repository.

## References

See [references.md](./references.md). Tables and figure inventory are provided in [tables.md](./tables.md) and [figures.md](./figures.md). The evidence and consistency audit is provided in [paper_consistency_check.md](./paper_consistency_check.md).
