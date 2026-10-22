"""Retrieval and centroid construction must share fitted preprocessing."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_baseline"))
from ltca_experiment import (  # noqa: E402
    HistoryBuffer, PolicyConfig, RetrievalProfile, ltca_pipeline,
    retrieval, transform_retrieval_features,
)


class RetrievalScaleTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"f1": [0.0, 1.0, 100.0, 90.0], "f2": [0.0, 1.0, 1.0, 0.0]})
        self.y = [0, 0, 1, 1]
        self.pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]).fit(self.df, self.y)

    def test_transformed_sample_matches_fitted_preprocessor(self):
        raw = self.df.iloc[2].to_numpy()
        expected = self.pipe.named_steps["scaler"].transform(
            self.pipe.named_steps["imputer"].transform(self.df.iloc[[2]])
        )[0]
        np.testing.assert_allclose(transform_retrieval_features(raw, self.pipe), expected)

    def test_pipeline_retrieves_in_scaled_not_raw_space(self):
        raw = self.df.iloc[2].to_numpy()
        scaled = transform_retrieval_features(raw, self.pipe)
        profiles = [
            RetrievalProfile("scaled-match", 0.7, scaled.copy()),
            RetrievalProfile("raw-match", 0.2, raw.copy()),
        ]
        expected = retrieval(scaled, profiles)["profile_name"]
        self.assertEqual(expected, "scaled-match")
        result = ltca_pipeline(raw, self.pipe, profiles, HistoryBuffer(), PolicyConfig())
        self.assertEqual(result["retrieval_profile"], expected)


if __name__ == "__main__":
    unittest.main()
