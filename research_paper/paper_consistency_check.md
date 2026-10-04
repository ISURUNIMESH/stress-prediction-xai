# Paper Consistency Check

## Completed checks

- Dataset size is consistently reported as 1,500 responses and 25 original columns.
- Class counts are consistently reported as Low 96, Moderate 1,306, and High 98.
- PSS reverse-scoring is consistently reported for Q4, Q5, Q7, and Q8 as `4 - response`.
- Operational thresholds are consistently reported as 0–13, 14–26, and 27–40.
- The 13 ML predictors are listed consistently; PSS items, total, target, Timestamp, and Consent are excluded.
- Baseline holdout, baseline 10-fold CV, and nested 5x5 CV are explicitly separated.
- Nested SVM Macro F1 and Balanced Accuracy are reported as 0.3425 +/- 0.0215 and 0.3451 +/- 0.0242.
- SVM Low and High nested recall are reported as 0.0937 and 0.0711.
- The exported model is described as the nested-CV-selected RBF SVM and not as clinically validated.
- Imbalance-aware negative findings and the unchanged application artifact are reported.
- SHAP is described as model explanation, not causal inference.
- No ethics approval, recruitment, provenance, threshold-validity, or clinical claims were invented.
- Reference numbering in the manuscript matches `references.md`.
- References [1]–[12] are cited in the manuscript; no numbered reference is orphaned.
- Added literature is limited to verified sources covering student-stress ML context [3], Sri Lankan perceived-stress context [2], healthcare XAI [9], and social-media/mental-health evidence [6], [7].

## Source files checked

- `README.md`
- `preprocessing/preprocess.py`
- `models/train_evaluate.py`
- `results/metrics/model_comparison.csv`
- `results/metrics/cross_validation_10fold.csv`
- `results/metrics/run_metadata.json`
- `results/nested_cv/research_summary.md`
- `results/imbalance_aware/research_summary.md`
- `results/imbalance_aware/ablation/ablation_results.csv`
- `results/imbalance_aware/shap/global_feature_importance.csv`
- `results/imbalance_improved/research_summary.md`
- `results/three_class_improvement/research_summary.md`
- `application/artifacts/model_metadata.json`

## Author input required before submission

1. Author and affiliation information.
2. Ethics approval/exemption and consent procedure.
3. Recruitment, eligibility, response-rate, and sampling details.
4. Dataset provenance, ownership, anonymisation, and sharing statement.
5. PSS-10 instrument version/translation validation and citation.
6. Justification and citation for the operational category thresholds.
7. Sri Lankan/South Asian related-work references.
8. Funding, conflict-of-interest, and publication-permission statements.
9. Final journal-specific tables and figures.
10. Author-approved interpretation of minority-class performance and deployment appropriateness.

## Final journal-readiness review

### Marker-by-marker author-input requirements

1. **Author information** (`research_paper.md`, title block): supply author names, affiliations, corresponding-author contact, ORCID identifiers, funding statement, and conflicts-of-interest statement.
2. **Sri Lankan/South Asian literature and dataset provenance** (`research_paper.md`, Section 2.7): supply additional population-matched peer-reviewed sources and the citable provenance/ownership record for the Google Forms dataset. This is distinct from the external Sri Lankan context reference [2].
3. **Ethics, consent, privacy, and data governance** (`research_paper.md`, Section 3.13): supply ethics approval or exemption, approval identifier/date, consent wording and procedure, recruitment permissions, anonymisation/de-identification method, data owner, access restrictions, and publication permission.
4. **PSS instrument and thresholds** (`research_paper.md`, Sections 2.1, 3.4, 3.5; `references.md`): supply the exact instrument version and language/translation validation, plus a real citation and justification for the 0–13/14–26/27–40 operational categories. Reference [1] supports the PSS instrument's origin but does not by itself validate these cutoffs for this study.
5. **Final figures and tables** (`figures.md`, figure authoring marker): supply the target journal, final figure selection, resolution/dimensions, permissions, and author approval of dataset-derived visualisations. This marker remains in `figures.md` because the requested final-review edits were limited to the manuscript, references, and consistency audit.
6. **Submission declarations** (`research_paper.md`, title block and Section 7; `paper_consistency_check.md`): supply funding, conflicts, data-availability, code/artifact-availability, and publication-permission statements, and approve the interpretation of minority-class performance and deployment appropriateness.

### Literature coverage decision

- **Introduction/student stress:** adequate foundational measurement plus one Sri Lankan context study, but additional recent and population-matched literature is still recommended.
- **ML stress prediction:** partially adequate; [3] supplies a recent student-stress ML review, but it concerns wearable AI and is not directly equivalent to this survey-based design.
- **Social-media analytics:** adequate for a cautious background discussion through [6] and [7]; neither source validates the project's self-reported hour variables.
- **Explainable AI/SHAP:** adequate for a concise methodological discussion through [8] and [9].
- **Class imbalance:** adequate foundational coverage through [10] and [11]; the paper's repository-specific experiments remain the primary evidence.
- **Sri Lankan/South Asian context:** insufficient for a strong regional literature review; author-supplied, population-matched sources are still required.
- **Dataset provenance and PSS thresholds:** absent from the repository and still require author input.

### Final status

The paper has **12 references**, all numbered citations are consistent, and no orphan reference remains. Literature coverage is sufficient for an evidence-bounded author review, but not sufficient for final journal submission in the Sri Lankan/South Asian context until the missing regional literature, dataset provenance, ethics, instrument/threshold justification, authorship, and declarations are supplied. The package is ready for **author review**, not submission-ready.

## Status

The paper package is internally consistent with the inspected repository artifacts, but it is **not submission-ready** until the author-input items above are completed and the final figures/references are approved.
