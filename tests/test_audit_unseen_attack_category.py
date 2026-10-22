"""Synthetic tests for the withheld-category experiment."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_baseline"))
from audit_unseen_attack_category import challenge_indices  # noqa: E402


class WithheldCategoryTests(unittest.TestCase):
    def test_withheld_attack_never_enters_train_or_validation(self):
        categories = np.array(["Normal"] * 20 + ["DoS"] * 20 + ["Fuzzers"] * 10)
        labels = np.array([0] * 20 + [1] * 30)
        fit, valid, withheld = challenge_indices(categories, labels, "Fuzzers", 42)
        self.assertEqual(len(withheld), 10)
        self.assertEqual(len(fit) + len(valid), 40)
        self.assertEqual(len(set(fit) | set(valid) | set(withheld)), 50)
        self.assertTrue(np.all(categories[withheld] == "Fuzzers"))

    def test_reject_normal_category(self):
        with self.assertRaisesRegex(ValueError, "malicious"):
            challenge_indices(np.array(["Normal", "Fuzzers"]), np.array([0, 1]), "Normal", 42)

    def test_reject_non_attack_rows_in_withheld(self):
        cats = np.array(["Normal"] * 20 + ["Fuzzers"] * 10 + ["DoS"] * 20)
        labels = np.array([0] * 20 + [1] * 10 + [1] * 19 + [0])
        with self.assertRaisesRegex(ValueError, "attack-class"):
            challenge_indices(cats, labels, "DoS", 42)


if __name__ == "__main__":
    unittest.main()
