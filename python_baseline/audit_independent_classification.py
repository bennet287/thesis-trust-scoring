"""Internal, leakage-controlled comparison of flow *classification* scores.

This is NOT an access-control benchmark. It intentionally ignores manually
specified scenario retrieval values and does not pretend independent flows
are temporal device histories. The 20% internal audit split is only an
exploratory holdout; past project work has already used this public dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
TRUST_ANCHORS = {"trusted": 0.95, "moderate": 0.65, "suspicious": 0.35, "malicious": 0.05}


def split_indices(labels, seed=42):
    """60% fit, 20% calibration, 20% internal audit, no overlapping indices."""
    all_idx = np.arange(len(labels))
    fit_idx, remainder = train_test_split(all_idx, test_size=0.4, stratify=labels, random_state=seed)
    cal_idx, audit_idx = train_test_split(remainder, test_size=0.5, stratify=np.asarray(labels)[remainder], random_state=seed + 1)
    if len(set(fit_idx) & set(cal_idx)) or len(set(fit_idx) & set(audit_idx)) or len(set(cal_idx) & set(audit_idx)):
        raise AssertionError("Data splits overlap")
    return fit_idx, cal_idx, audit_idx


def build_profiles(training_vectors, training_attack_probability):
    """Construct profile centroids using fitted TRAIN data only."""
    trusts = 1.0 - np.asarray(training_attack_probability)
    profiles = []
    for name, anchor in TRUST_ANCHORS.items():
        if name == "trusted":
            mask = trusts >= 0.80
        elif name == "moderate":
            mask = (trusts >= 0.50) & (trusts < 0.80)
        elif name == "suspicious":
            mask = (trusts >= 0.20) & (trusts < 0.50)
        else:
            mask = trusts < 0.20
        if np.any(mask):
            profiles.append((name, float(anchor), np.asarray(training_vectors)[mask].mean(axis=0)))
    if not profiles:
        raise ValueError("No train-only retrieval profiles generated")
    return profiles


def retrieval_scores(transformed_vectors, profiles):
    """Return fixed anchor trusts of closest profile in the same scaled feature space."""
    matrix = np.vstack([centroid for _, _, centroid in profiles])
    chosen = cosine_similarity(transformed_vectors, matrix).argmax(axis=1)
    values = np.asarray([trust for _, trust, _ in profiles], dtype=float)
    return values[chosen]


def three_scores(model_attack_probability, retrieval_trust):
    """Model, naive fusion, adaptive fusion with neutral non-temporal H=0.5."""
    m = 1.0 - np.asarray(model_attack_probability)
    r = np.asarray(retrieval_trust)
    h = np.full_like(m, 0.5)
    naive = 0.50 * m + 0.30 * r + 0.20 * h
    confidence = np.clip(np.abs(m - 0.5) * 2.0, 0.0, 1.0)
    agreement = np.clip(1.0 - (np.abs(m - r) + np.abs(m - h)) / 2.0, 0.0, 1.0)
    wm = 0.5 + 0.5 * confidence
    wr = 0.3 * agreement
    wh = 0.2 + 0.2 * (1.0 - confidence)
    total = wm + wr + wh
    adaptive = (wm * m + wr * r + wh * h) / total
    return {"model_only": 1.0 - m, "naive_fusion": 1.0 - naive, "adaptive_fusion": 1.0 - adaptive}


def choose_threshold(y_true, risk):
    """Select on calibration only; never use audit labels for tuning."""
    thresholds = np.linspace(0.05, 0.95, 19)
    return float(max(thresholds, key=lambda t: (f1_score(y_true, risk >= t, zero_division=0), -abs(t - 0.5))))


def classification_metrics(y_true, risk, threshold):
    y_pred = np.asarray(risk) >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_descriptive_not_calibrated": float(brier_score_loss(y_true, np.clip(risk, 0, 1))),
        "confusion_matrix_tn_fp_fn_tp": [int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()],
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(source: Path, seed=42):
    if source.name != "UNSW_NB15_training-set.csv":
        raise ValueError("This internal audit is restricted to the development/training dataset")
    df = pd.read_csv(source)
    needed = {"label", "id", "attack_cat"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Expected UNSW-NB15 columns missing: {needed - set(df.columns)}")
    y = df["label"].to_numpy(dtype=int)
    X = df.drop(columns=["id", "attack_cat", "label"]).select_dtypes(include=["number"])
    fit_idx, cal_idx, audit_idx = split_indices(y, seed)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("model", DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=5, criterion="entropy", random_state=seed)),
    ])
    model.fit(X.iloc[fit_idx], y[fit_idx])
    preprocessing = model[:-1]
    train_vectors = preprocessing.transform(X.iloc[fit_idx])
    profiles = build_profiles(train_vectors, model.predict_proba(X.iloc[fit_idx])[:, 1])
    scores = {}
    for name, index in (("calibration", cal_idx), ("internal_audit", audit_idx)):
        prob = model.predict_proba(X.iloc[index])[:, 1]
        retrieval = retrieval_scores(preprocessing.transform(X.iloc[index]), profiles)
        scores[name] = three_scores(prob, retrieval)
    comparisons = {}
    for name in scores["calibration"]:
        threshold = choose_threshold(y[cal_idx], scores["calibration"][name])
        comparisons[name] = {
            "selected_calibration_threshold": threshold,
            "calibration": classification_metrics(y[cal_idx], scores["calibration"][name], threshold),
            "internal_audit": classification_metrics(y[audit_idx], scores["internal_audit"][name], threshold),
        }
    return {
        "protocol": "internal exploratory flow-classification audit; NOT authorization or generalization proof",
        "input_file": source.name,
        "input_sha256": sha256_file(source),
        "seed": int(seed),
        "rows": {"fit": len(fit_idx), "calibration": len(cal_idx), "internal_audit": len(audit_idx)},
        "n_features": int(X.shape[1]),
        "n_profiles": len(profiles),
        "history_handling": "neutral H=0.5 for independent rows; not a temporal device-history evaluation",
        "calibration_method": "maximize F1 on calibration partition; threshold fixed before internal audit",
        "limitations": [
            "Original project used this public dataset in prior development; internal audit is not an untouched external benchmark.",
            "No genuine device identities, access requests, resource permissions or enforcement are evaluated.",
            "IoT/edge deployment, uncertainty under distribution shift, and temporal history remain untested.",
            "Fusion risk values are heuristic scores; Brier scores are descriptive, not evidence of calibrated probabilities.",
        ],
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run(args.input, args.seed)
    output = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
