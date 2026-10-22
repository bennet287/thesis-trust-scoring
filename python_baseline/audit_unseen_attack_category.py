"""Explore how a classifier behaves on a withheld attack *category*.

Exploratory dataset-internal challenge only. Not an external OOD benchmark,
not an uncertainty detector, and not a Zero Trust authorization test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from audit_independent_classification import DEFAULT_INPUT, sha256_file

DEFAULT_CATEGORIES = ["DoS", "Reconnaissance", "Fuzzers", "Exploits", "Generic"]
DEFAULT_SEEDS = [7, 17, 42, 73, 123]


def challenge_indices(categories, labels, category, seed):
    """Withhold all rows of one positive attack category from model fitting."""
    categories = np.asarray(categories)
    labels = np.asarray(labels, dtype=int)
    if category == "Normal":
        raise ValueError("Withheld category must be a malicious attack category")
    withheld = np.flatnonzero(categories == category)
    known = np.flatnonzero(categories != category)
    if not len(withheld) or not np.all(labels[withheld] == 1):
        raise ValueError("Withheld category must exist and contain attack-class rows only")
    fit, known_validation = train_test_split(
        known, test_size=0.2, stratify=labels[known], random_state=seed
    )
    if len(set(fit) & set(withheld)) or len(set(known_validation) & set(withheld)):
        raise AssertionError("Withheld attack category leaked into training or validation")
    return fit, known_validation, withheld


def evaluate_category(X, y, categories, category, seed):
    fit, known_val, withheld = challenge_indices(categories, y, category, seed)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("model", DecisionTreeClassifier(
            max_depth=8, min_samples_split=10, min_samples_leaf=5,
            criterion="entropy", random_state=seed,
        )),
    ])
    model.fit(X.iloc[fit], y[fit])
    known_risk = model.predict_proba(X.iloc[known_val])[:, 1]
    unseen_risk = model.predict_proba(X.iloc[withheld])[:, 1]
    known_predictions = known_risk >= 0.5
    return {
        "category": category,
        "seed": int(seed),
        "train_count": int(len(fit)),
        "known_validation_count": int(len(known_val)),
        "withheld_attack_count": int(len(withheld)),
        "known_validation_f1": float(f1_score(y[known_val], known_predictions)),
        "known_validation_attack_recall": float(recall_score(y[known_val], known_predictions)),
        "withheld_category_attack_recall": float(np.mean(unseen_risk >= 0.5)),
        "withheld_missed_count": int(np.count_nonzero(unseen_risk < 0.5)),
        "withheld_predicted_attack_probability_le_0_1": float(np.mean(unseen_risk <= 0.1)),
        "withheld_probability_median": float(np.median(unseen_risk)),
    }


def run(source, categories=DEFAULT_CATEGORIES, seeds=DEFAULT_SEEDS):
    if source.name != "UNSW_NB15_training-set.csv":
        raise ValueError("Restricted to the development/training source; not the official testing set")
    df = pd.read_csv(source)
    for col in ("label", "attack_cat", "id"):
        if col not in df:
            raise ValueError(f"Missing source column: {col}")
    X = df.drop(columns=["id", "label", "attack_cat"]).select_dtypes(include=["number"])
    labels = df["label"].to_numpy(int)
    names = df["attack_cat"].to_numpy()
    reports = [evaluate_category(X, labels, names, category, int(seed)) for category in categories for seed in seeds]
    return {
        "protocol": "within-dataset withheld-attack-category exploratory challenge; NOT external distribution-shift validation",
        "dataset_sha256": sha256_file(source),
        "categories": list(categories),
        "seeds": [int(s) for s in seeds],
        "limitations": [
            "The dataset has already been used in this project's earlier development.",
            "Categories are attack-class labels; no authorization or device identities are measured.",
            "A held-out attack category is only a proxy for novelty, not an independent deployment domain.",
            "Decision Tree probability estimates are not necessarily calibrated: <=0.1 is descriptive, not a guaranteed confidence level.",
            "No test of uncertainty estimation or uncertainty-aware policy mechanism is performed here.",
        ],
        "results": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(run(args.input, args.categories, args.seeds), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
