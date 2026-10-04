# Figure Inventory

The following repository figures are suitable for manuscript review after checking resolution and journal formatting:

1. **Figure 1. System architecture.** `images/architecture_diagram.png`. The end-to-end separation of data preparation, model evaluation, SHAP analysis, API services, and frontend.
2. **Figure 2. Stress-class distribution.** `results/figures/class_distribution.png`. The observed Low, Moderate, and High class counts; the caption must state the severe imbalance.
3. **Figure 3. Baseline model comparison.** `results/figures/model_performance_comparison.png`. Compare metrics without implying that accuracy is sufficient.
4. **Figure 4. Baseline confusion matrices.** `results/figures/*_confusion_matrix.png`. Label each panel with model name, actual class, and predicted class.
5. **Figure 5. Global SHAP importance.** `results/shap_plots/shap_feature_importance.png`. Explain that values are model-attributed contributions.
6. **Figure 6. SHAP summary plot.** `results/shap_plots/shap_summary.png`. Do not interpret direction as causality.

The repository contains CSV/JSON results for calibration, robustness, error, subgroup, and SHAP stability analyses. No additional figure should be created from those results without preserving the exact experimental design and stating whether it is descriptive.

**Figure authoring inputs required:** [AUTHOR INPUT REQUIRED: final figure selection, source-resolution approval, target-journal dimensions, and permission to reproduce any dataset-derived visualisation.]
