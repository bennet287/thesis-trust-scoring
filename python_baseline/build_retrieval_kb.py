from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "numeric_dt_pipeline.joblib"
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
OUT_FILE = PROJECT_ROOT / "results" / "models" / "retrieval_kb.json"


def detect_target_column(df: pd.DataFrame) -> str:
    for col in ["label", "Label", "target", "class", "Class"]:
        if col in df.columns:
            return col
    raise ValueError("Could not detect target column.")


def verify_profile_source(source_file: Path) -> None:
    """Prevent retrieval profiles from being built using non-training splits."""
    if source_file.resolve() != TRAIN_FILE.resolve():
        raise ValueError("Retrieval profiles must use the designated training split only")


def main() -> None:
    verify_profile_source(TRAIN_FILE)
    pipeline = joblib.load(MODEL_PATH)

    # Retrieval profiles are built only from training/development data.
    df = pd.read_csv(TRAIN_FILE)
    target_col = detect_target_column(df)

    drop_candidates = ["id", "attack_cat", target_col]
    X_full = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")
    X = X_full.select_dtypes(include=["int64", "float64"]).copy()

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]

    # Transform numeric inputs the same way as training
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    probs = pipeline.predict_proba(X)[:, 1]
    trusts = 1.0 - probs

    # Trust bands
    bands = {
        "trusted": trusts >= 0.80,
        "moderate": (trusts >= 0.50) & (trusts < 0.80),
        "suspicious": (trusts >= 0.20) & (trusts < 0.50),
        "malicious": trusts < 0.20,
    }

    # Fixed target trust anchors for retrieval
    profile_trusts = {
        "trusted": 0.95,
        "moderate": 0.65,
        "suspicious": 0.35,
        "malicious": 0.05,
    }

    descriptions = {
        "trusted": "Centroid of high-trust normal-like behavior",
        "moderate": "Centroid of moderately trusted or borderline behavior",
        "suspicious": "Centroid of suspicious behavior likely needing quarantine",
        "malicious": "Centroid of highly malicious behavior likely needing block",
    }

    profiles = []
    for name, mask in bands.items():
        mask = np.asarray(mask)
        if mask.sum() == 0:
            continue

        centroid = X_scaled[mask].mean(axis=0).tolist()

        profiles.append(
            {
                "name": name,
                "description": descriptions[name],
                "profile_trust": profile_trusts[name],
                "count": int(mask.sum()),
                "centroid": centroid,
            }
        )

    kb = {
        "data_provenance": {
            "split": "training",
            "source_file": TRAIN_FILE.name,
            "n_source_rows": int(len(df)),
        },
        "feature_names": X.columns.tolist(),
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
        "profiles": profiles,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)

    print("=== Retrieval KB Built ===")
    print(f"Output: {OUT_FILE}")
    for p in profiles:
        print(
            f"{p['name']}: count={p['count']}, profile_trust={p['profile_trust']:.2f}"
        )


if __name__ == "__main__":
    main()