"""Fast integrity checks for the isolated imbalance experiment."""

import numpy as np
import pandas as pd
import unittest
from sklearn.model_selection import StratifiedKFold
from unittest.mock import patch

from experiments import improved_imbalance_cv as experiment


class ImprovedImbalanceIntegrityTests(unittest.TestCase):
 def test_predictors_exclude_pss_and_target(self):
    data = pd.read_csv(experiment.DATA_FILE)
    X, y, encoder = experiment.load_predictors(data)
    self.assertFalse(set(experiment.PSS_FIELDS).intersection(X.columns))
    self.assertNotIn(experiment.TARGET, X.columns)
    self.assertEqual(len(X), len(y))
    self.assertEqual(len(X), len(data))
    self.assertEqual(encoder.classes_.tolist(), experiment.CLASS_NAMES)


 def test_nested_splits_are_stratified_reproducible(self):
    data = pd.read_csv(experiment.DATA_FILE)
    X, y, _ = experiment.load_predictors(data)
    outer_a = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    outer_b = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    self.assertTrue(all(np.array_equal(a[1], b[1]) for a, b in zip(outer_a, outer_b)))
    for train, valid in outer_a:
        self.assertEqual(set(y.iloc[train].unique()), {0, 1, 2})
        self.assertEqual(set(y.iloc[valid].unique()), {0, 1, 2})
        inner = StratifiedKFold(5, shuffle=True, random_state=42)
        inner_splits = list(inner.split(X.iloc[train], y.iloc[train]))
        self.assertEqual(len(inner_splits), 5)
        self.assertTrue(all(set(y.iloc[train].iloc[v].unique()) == {0, 1, 2} for _, v in inner_splits))


 def test_random_oversampling_is_deterministic_and_balances_only_input(self):
    features = np.arange(24).reshape(12, 2)
    labels = np.array([0] * 2 + [1] * 3 + [2] * 7)
    x1, y1 = experiment.random_oversample_training(features, labels, seed=42)
    x2, y2 = experiment.random_oversample_training(features, labels, seed=42)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    self.assertEqual(len(x1), 21)
    self.assertEqual(np.bincount(y1).tolist(), [7, 7, 7])
    self.assertTrue(set(map(tuple, x1)).issubset(set(map(tuple, features))))


 def test_oversampler_receives_only_training_partition(self):
    data = pd.read_csv(experiment.DATA_FILE).iloc[:240].copy()
    X, y, _ = experiment.load_predictors(data)
    train_idx, valid_idx = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    observed = {}
    original = experiment.random_oversample_training

    def recording_oversampler(train_x, train_y, seed):
        observed["rows"] = len(train_y)
        observed["label_ids"] = set(np.asarray(train_y).tolist())
        return original(train_x, train_y, seed)

    with patch.object(experiment, "random_oversample_training", recording_oversampler):
        fit = experiment.fit_candidate(
            "RandomOverSampler + SVM", X.iloc[train_idx], y.iloc[train_idx],
            {"C": 2.0, "gamma": "scale", "class_weight": "balanced"}, seed=42,
        )
    prediction, probability = experiment.predict_candidate(fit, X.iloc[valid_idx])
    self.assertEqual(observed["rows"], len(train_idx))
    self.assertEqual(observed["label_ids"], {0, 1, 2})
    self.assertEqual(len(prediction), len(valid_idx))
    self.assertEqual(probability.shape, (len(valid_idx), 3))
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)


 def test_multiclass_metrics_and_saved_comparison_are_present(self):
    y = np.array([0, 1, 2, 0, 1, 2])
    prediction = np.array([0, 1, 1, 0, 2, 2])
    probabilities = np.eye(3)[prediction] * 0.8 + 0.2 / 3
    metrics = experiment.calculate_metrics(y, prediction, probabilities)
    self.assertTrue(0 <= metrics["macro_f1"] <= 1)
    self.assertTrue(0 <= metrics["balanced_accuracy"] <= 1)
    self.assertTrue(all(0 <= metrics[key] <= 1 for key in ("low_recall", "moderate_recall", "high_recall", "roc_auc")))
    comparison = pd.read_csv(experiment.OUTPUT / "model_comparison.csv")
    self.assertIn("Current SVM", comparison["Model"].tolist())
    self.assertEqual(comparison["macro_f1_mean"].notna().sum(), 6)
    self.assertGreater(len(pd.read_csv(experiment.OUTPUT / "classification_reports.csv")), 0)
    self.assertEqual(len(pd.read_csv(experiment.OUTPUT / "confusion_matrices.csv")), 6 * 9)
