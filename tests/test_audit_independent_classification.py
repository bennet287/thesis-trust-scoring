"""Data integrity checks for the internal classification audit; no real dataset needed."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_baseline"))
from audit_independent_classification import (  # noqa: E402
    build_profiles, choose_threshold, retrieval_scores, split_indices, three_scores,
)


class AuditExperimentTests(unittest.TestCase):
    def test_disjoint_stratified_splits(self):
        labels = np.array([0] * 70 + [1] * 70)
        fit, calibration, audit = split_indices(labels, seed=17)
        self.assertEqual(len(fit) + len(calibration) + len(audit), len(labels))
        self.assertEqual(len(set(fit) | set(calibration) | set(audit)), len(labels))
        self.assertFalse(set(fit) & set(calibration))
        self.assertFalse(set(calibration) & set(audit))
        self.assertFalse(set(fit) & set(audit))
        self.assertEqual(len(fit), 84)
        self.assertEqual(len(calibration), 28)
        self.assertEqual(len(audit), 28)

    def test_retrieval_matches_training_centroids(self):
        X_fit = np.array([[1., 0.], [0.9, 0.1], [0., 1.], [0.1, 0.9]])
        probs = np.array([0.02, 0.04, 0.98, 0.99])
        profiles = build_profiles(X_fit, probs)
        self.assertEqual(len(profiles), 2)
        values = retrieval_scores(np.array([[1., 0.], [0., 1.]]), profiles)
        np.testing.assert_allclose(values, [0.95, 0.05])

    def test_no_stateful_history_is_used_for_independent_rows(self):
        risks = three_scores(np.array([0.0, 1.0]), np.array([0.95, 0.05]))
        for scores in risks.values():
            self.assertEqual(scores.shape, (2,))
            self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))
        self.assertEqual(float(risks["model_only"][0]), 0.0)
        self.assertEqual(float(risks["model_only"][1]), 1.0)

    def test_calibration_threshold_selection_is_reproducible(self):
        labels = np.array([0, 0, 1, 1])
        risk = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(choose_threshold(labels, risk), 0.5)


if __name__ == "__main__":
    unittest.main()
