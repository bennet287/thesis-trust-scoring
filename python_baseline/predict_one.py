from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "dt_pipeline.joblib"

TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"


def trust_to_action(trust_score: float) -> str:
    if trust_score >= 0.80:
        return "ALLOW"
    if trust_score >= 0.50:
        return "LIMIT"
    if trust_score >= 0.20:
        return "QUARANTINE"
    return "BLOCK"


def detect_target_column(df: pd.DataFrame) -> str:
    candidate_targets = ["label", "Label", "target", "class", "Class"]
    for col in candidate_targets:
        if col in df.columns:
            return col
    raise ValueError(f"Could not detect target column. Columns: {list(df.columns)}")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    print("Loading saved model...")
    pipeline = joblib.load(MODEL_PATH)

    print("Loading one sample from test set...")
    df = pd.read_csv(TEST_FILE)
    target_col = detect_target_column(df)

    drop_candidates = ["id", "attack_cat"]
    X = df.drop(columns=[target_col] + [c for c in drop_candidates if c in df.columns], errors="ignore")

    sample = X.iloc[[0]]  # keep as DataFrame
    attack_probability = float(pipeline.predict_proba(sample)[0, 1])
    trust_score = 1.0 - attack_probability
    action = trust_to_action(trust_score)

    print("\n=== Single-sample inference ===")
    print(f"Attack probability: {attack_probability:.4f}")
    print(f"Trust score      : {trust_score:.4f}")
    print(f"Policy action    : {action}")


if __name__ == "__main__":
    main()