"""Guard the evaluation split from retrieval-profile construction."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_baseline"))

from build_retrieval_kb import (  # noqa: E402
    TRAIN_FILE,
    verify_profile_source,
)


class RetrievalProfileSourceTest(unittest.TestCase):
    def test_training_source_is_permitted(self):
        verify_profile_source(TRAIN_FILE)

    def test_test_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "designated training split"):
            verify_profile_source(ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv")


if __name__ == "__main__":
    unittest.main()
