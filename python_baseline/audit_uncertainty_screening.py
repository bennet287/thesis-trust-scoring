"""Exploratory known-vs-withheld attack novelty challenge; no authorization claims.

For each withheld attack category, fit only on remaining known categories. Threshold
three review signals on an independent calibration set's known-benign flows.
Assess false review rate on held-out known-benign flows and sensitivity to
previously-unseen-category attack flows, especially those misclassified benign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
CATEGORIES = ("Fuzzers", "Reconnaissance", "DoS")
METHODS = ("low_classification_confidence", "isolation_forest_novelty", "pca_reconstruction_novelty")


def partitions(labels, categories, withheld, seed):
    labels, categories = np.asarray(labels), np.asarray(categories)
    omitted = np.flatnonzero(categories == withheld)
    known = np.flatnonzero(categories != withheld)
    if not len(omitted) or not np.all(labels[omitted] == 1):
        raise ValueError("A withheld category must exist and consist of malicious flows")
    train, remainder = train_test_split(known, stratify=labels[known], test_size=0.4, random_state=seed)
    calibration, evaluation = train_test_split(remainder, stratify=labels[remainder], test_size=0.5, random_state=seed + 1)
    assert len(set(train) | set(calibration) | set(evaluation) | set(omitted)) == len(labels)
    assert len(set(train) & set(omitted)) == 0
    return train, calibration, evaluation, omitted


def calibrated_threshold(scores, fraction=0.05):
    """Flag strictly above known-benign calibration 95th percentile; ties stay unflagged.

    This is a calibration-set review budget, NOT a distribution-free guarantee
    for shifted populations or even exactly 5% on a held-out benign set.
    """
    score = np.asarray(scores)
    if score.size == 0 or not np.isfinite(score).all():
        raise ValueError("Calibration scores missing/nonfinite")
    return float(np.quantile(score, 1.0 - fraction, method="higher"))


def compute_review_flags(calibration_scores, other_scores):
    threshold = calibrated_threshold(calibration_scores)
    return threshold, np.asarray(other_scores) > threshold


def score_signals(model, prep, X_fit, X_cal, X_eval, X_unseen, seed):
    fit_z = np.asarray(prep.transform(X_fit))
    cal_z = np.asarray(prep.transform(X_cal))
    eval_z = np.asarray(prep.transform(X_eval))
    unseen_z = np.asarray(prep.transform(X_unseen))
    rng = np.random.default_rng(seed)
    subsample = rng.choice(len(fit_z), size=min(10000, len(fit_z)), replace=False)
    known_fit_z = fit_z[subsample]
    detector = IsolationForest(n_estimators=75, max_samples=1024, random_state=seed, n_jobs=-1)
    detector.fit(known_fit_z)
    pca = PCA(n_components=min(10, fit_z.shape[1]), random_state=seed, svd_solver="randomized").fit(known_fit_z)

    def feature_scores(data, data_z):
        attack_p = model.predict_proba(data)[:, 1]
        ambiguity = 1.0 - np.abs(attack_p - 0.5) * 2.0
        iforest = -detector.score_samples(data_z)
        reconstructed = pca.inverse_transform(pca.transform(data_z))
        reconstruction = np.mean(np.square(data_z - reconstructed), axis=1)
        return attack_p, {
            "low_classification_confidence": ambiguity,
            "isolation_forest_novelty": iforest,
            "pca_reconstruction_novelty": reconstruction,
        }
    return feature_scores(X_cal, cal_z), feature_scores(X_eval, eval_z), feature_scores(X_unseen, unseen_z)


def evaluate(df, withheld, seed=42, false_review_budget=0.05):
    y = df["label"].to_numpy(dtype=int)
    names = df["attack_cat"].to_numpy()
    X = df.drop(columns=["label", "id", "attack_cat"]).select_dtypes(include=["number"])
    train, cal, evaluation, omitted = partitions(y, names, withheld, seed)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("model", DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=5, criterion="entropy", random_state=seed)),
    ])
    pipeline.fit(X.iloc[train], y[train])
    prep = pipeline[:-1]
    (pc, sc), (pe, se), (pu, su) = score_signals(pipeline, prep, X.iloc[train], X.iloc[cal], X.iloc[evaluation], X.iloc[omitted], seed)
    known_benign_cal = y[cal] == 0
    known_benign_eval = y[evaluation] == 0
    unseen_missed = pu < 0.5
    results = {}
    for method in METHODS:
        threshold = calibrated_threshold(sc[method][known_benign_cal], false_review_budget)
        cal_flags = sc[method][known_benign_cal] > threshold
        benign_flags = se[method][known_benign_eval] > threshold
        unknown_flags = su[method] > threshold
        results[method] = {
            "review_threshold": threshold,
            "calibration_benign_review_fraction": float(cal_flags.mean()),
            "held_out_known_benign_review_fraction": float(benign_flags.mean()),
            "withheld_attacks_reviewed_fraction": float(unknown_flags.mean()),
            "missed_withheld_attacks_reviewed_fraction": float(unknown_flags[unseen_missed].mean()) if unseen_missed.any() else None,
            "missed_withheld_attacks_reviewed_count": int(unknown_flags[unseen_missed].sum()),
        }
    return {
        "withheld_category": withheld,
        "seed": seed,
        "n_fit": len(train), "n_calibration": len(cal), "n_known_evaluation": len(evaluation),
        "n_withheld_category": len(omitted),
        "n_model_missed_withheld": int(unseen_missed.sum()),
        "model_attack_recall_on_withheld": float(np.mean(~unseen_missed)),
        "n_benign_calibration": int(known_benign_cal.sum()),
        "n_benign_evaluation": int(known_benign_eval.sum()),
        "nominal_calibration_false_review_budget": false_review_budget,
        "results": results,
    }


def run(path, categories=("Fuzzers",), seeds=(42,)):
    if path.name != "UNSW_NB15_training-set.csv":
        raise ValueError("Only the historic development/training CSV may be used here")
    df = pd.read_csv(path)
    if not {"label", "id", "attack_cat"}.issubset(df.columns):
        raise ValueError("Expected dataset columns absent")
    dig = hashlib.sha256(path.read_bytes()).hexdigest()
    records = [evaluate(df, cat, seed) for cat in categories for seed in seeds]
    return {
        "protocol": "exploratory category-withheld, calibrated benign-review-budget comparison; NOT independent OOD validation or access control",
        "source_sha256": dig,
        "methods": METHODS,
        "limitations": [
            "Categories and source data already known to prior project work; exploratory only.",
            "Category exclusion within the same capture is only a proxy for unfamiliar deployment conditions.",
            "Calibration budget is empirical on benign validation, not a distribution-free guarantee on held-out or shifted traffic.",
            "Intrusion label is not authorization ground truth; a review signal is not an enforced block.",
            "Random flow splits may leak correlated captures/sessions; new independent device captures are required.",
            "Confidence score is an uncalibrated tree-prediction margin; IsolationForest and PCA scores have no intrinsic security meaning.",
            "These methods do not identify whether an authorized principal is entitled to access a particular resource.",
        ],
        "records": records,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DATA_PATH)
    p.add_argument("--category", action="append", dest="categories", choices=CATEGORIES)
    p.add_argument("--seed", type=int, action="append", dest="seeds")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = run(args.input, args.categories or ["Fuzzers"], args.seeds or [42])
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
