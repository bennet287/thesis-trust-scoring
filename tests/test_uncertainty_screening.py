"""No-dataset regression checks for the Phase 1B review-signal audit."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_baseline"))
from audit_uncertainty_screening import (  # noqa: E402
    calibrated_threshold, compute_review_flags, partitions,
)


class TestUncertaintyScreening(unittest.TestCase):
    def setUp(self):
        self.categories = np.array(["Normal"] * 40 + ["DoS"] * 40 + ["Fuzzers"] * 20)
        self.labels = np.array([0] * 40 + [1] * 60)

    def test_withheld_category_excluded(self):
        train, cal, audit, unknown = partitions(self.labels, self.categories, "Fuzzers", 42)
        self.assertEqual(len(unknown), 20)
        self.assertEqual(len(set(train) | set(cal) | set(audit) | set(unknown)), 100)
        self.assertTrue(np.all(self.categories[unknown] == "Fuzzers"))

    def test_invalid_withheld_normal_category_rejected(self):
        with self.assertRaises(ValueError):
            partitions(self.labels, self.categories, "Normal", 42)

    def test_review_budget_ties_never_exceed(self):
        normal_scores = np.array([0.] * 950 + [1.] * 50)
        threshold = calibrated_threshold(normal_scores)
        self.assertEqual(np.mean(normal_scores > threshold), 0.)

    def test_review_direction_threshold(self):
        threshold, flags = compute_review_flags(np.linspace(0, 1, 100), np.array([-3, 0.3, 1.0, 3.0]))
        self.assertTrue(0.9 < threshold < 1.0)
        self.assertEqual(flags.tolist(), [False, False, True, True])


if __name__ == "__main__":
    unittest.main()
