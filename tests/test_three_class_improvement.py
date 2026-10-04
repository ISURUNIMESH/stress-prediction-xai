"""Integrity tests for the isolated SMOTENC and threshold-tuning experiment."""

import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiments import three_class_improvement as experiment


class ThreeClassImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.encoder = experiment.load_predictors()

    def test_pss_and_target_are_excluded(self):
        self.assertFalse(set(experiment.PSS_FIELDS + [experiment.TARGET]).intersection(self.X.columns))
        self.assertEqual(len(self.X), 1500)
        self.assertEqual(self.encoder.classes_.tolist(), experiment.CLASS_NAMES)

    def test_outer_inner_splits_are_stratified_and_reproducible(self):
        splitter = StratifiedKFold(5, shuffle=True, random_state=42)
        first = list(splitter.split(self.X, self.y))
        second = list(StratifiedKFold(5, shuffle=True, random_state=42).split(self.X, self.y))
        self.assertTrue(all(np.array_equal(a[1], b[1]) for a, b in zip(first, second)))
        for train, test in first:
            self.assertEqual(set(self.y.iloc[train].unique()), {0, 1, 2})
            self.assertEqual(set(self.y.iloc[test].unique()), {0, 1, 2})
            inner = StratifiedKFold(5, shuffle=True, random_state=42)
            self.assertEqual(len(list(inner.split(self.X.iloc[train], self.y.iloc[train]))), 5)

    def test_smotenc_strategy_uses_training_labels_and_conservative_ratio(self):
        training_labels = np.array([0] * 8 + [1] * 9 + [2] * 80)
        strategy = experiment.smotenc_strategy(training_labels, 0.25)
        self.assertEqual(strategy, {0: 20, 1: 20})
        self.assertNotIn(2, strategy)

    def test_smotenc_fit_receives_only_training_rows(self):
        train_indices, validation_indices = next(
            StratifiedKFold(5, shuffle=True, random_state=42).split(self.X, self.y)
        )
        X_train = self.X.iloc[train_indices]
        y_train = self.y.iloc[train_indices]
        observed = {}
        real_sampler = experiment.SMOTENC

        def recording_sampler(*args, **kwargs):
            observed["categorical_features"] = kwargs["categorical_features"]
            observed["strategy"] = kwargs["sampling_strategy"]
            return real_sampler(*args, **kwargs)

        with patch.object(experiment, "SMOTENC", side_effect=recording_sampler):
            fitted = experiment.fit_model(
                X_train, y_train,
                {"C": 1.0, "gamma": "scale", "class_weight": "balanced", "sampling_ratio": 0.25},
                sampler_ratio=0.25,
                seed=42,
            )
        self.assertTrue(observed["categorical_features"])
        self.assertTrue(observed["strategy"])
        probabilities = fitted.predict_proba(self.X.iloc[validation_indices[:6]])
        self.assertEqual(probabilities.shape, (6, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_thresholds_change_labels_not_probabilities(self):
        probabilities = np.array([[0.28, 0.31, 0.41], [0.35, 0.30, 0.35]])
        before = probabilities.copy()
        thresholds = [0.20, 0.20, 1 / 3]
        predicted = experiment.apply_thresholds(probabilities, thresholds)
        np.testing.assert_array_equal(probabilities, before)
        self.assertEqual(predicted.shape, (2,))

    def test_threshold_search_uses_inner_training_partitions_only(self):
        sampled = pd.concat([
            self.X.loc[self.y[self.y == class_id].index[:120]]
            for class_id in range(3)
        ]).sort_index()
        sampled_y = self.y.loc[sampled.index]
        outer_train, outer_valid = next(
            StratifiedKFold(5, shuffle=True, random_state=42).split(sampled, sampled_y)
        )
        X_train, y_train = sampled.iloc[outer_train], sampled_y.iloc[outer_train]
        outer_validation_ids = set(sampled.iloc[outer_valid].index)
        original_fit = experiment.fit_model
        fitted_indices = []

        def recording_fit(X_part, y_part, *args, **kwargs):
            fitted_indices.extend(X_part.index.tolist())
            return original_fit(X_part, y_part, *args, **kwargs)

        threshold_family = {
            "sampler": None,
            "grid": [experiment.CURRENT_SVM_PARAMS],
            "threshold_variant": True,
        }
        trace = []
        with patch.object(experiment, "fit_model", side_effect=recording_fit):
            _, chosen_threshold, all_candidates = experiment.select_inner_configurations(
                "Threshold isolation test", threshold_family, X_train, y_train, 1, trace
            )
        self.assertTrue(fitted_indices)
        self.assertTrue(set(fitted_indices).isdisjoint(outer_validation_ids))
        self.assertGreater(len(all_candidates), 1)
        self.assertIn(chosen_threshold["threshold_name"], {item["threshold_name"] for item in trace})

    def test_metrics_include_each_class_and_multiclass_auc(self):
        actual = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.array([
            [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8],
        ])
        metrics, _ = experiment.fold_scores(actual, probabilities, [1 / 3] * 3)
        for class_name in ("high", "low", "moderate"):
            self.assertIn(f"{class_name}_precision", metrics)
            self.assertIn(f"{class_name}_recall", metrics)
            self.assertIn(f"{class_name}_f1", metrics)
        self.assertIn("roc_auc_macro_ovr", metrics)
        self.assertIn("roc_auc_weighted_ovr", metrics)

    def test_prediction_distributions_and_candidate_decision_are_guarded(self):
        prior_directory = experiment.ROOT / "application" / "artifacts" / "final_svm_pipeline.joblib"
        digest_before = hashlib.sha256(prior_directory.read_bytes()).hexdigest()
        metadata_file = experiment.OUTPUT / "experiment_metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            self.assertEqual(bool(metadata.get("candidate_selected")), experiment.CANDIDATE_ARTIFACT_DIR.exists())
            self.assertFalse(metadata.get("application_model_changed"))
            distributions = pd.read_csv(experiment.OUTPUT / "prediction_distributions" / "predicted_counts_by_outer_fold.csv")
            self.assertEqual(distributions.groupby("model").size().min(), 5)
            self.assertTrue({"low_recall", "moderate_recall", "high_recall"}.issubset(
                set(pd.read_csv(experiment.OUTPUT / "nested_cv_results.csv").columns)
            ))
        self.assertEqual(hashlib.sha256(prior_directory.read_bytes()).hexdigest(), digest_before)


if __name__ == "__main__":
    unittest.main()
