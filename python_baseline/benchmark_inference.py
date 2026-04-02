from __future__ import annotations

from pathlib import Path
from time import perf_counter
import statistics

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "dt_pipeline.joblib"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"


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

    print("Loading model...")
    pipeline = joblib.load(MODEL_PATH)

    print("Loading test data...")
    df = pd.read_csv(TEST_FILE)
    target_col = detect_target_column(df)
    drop_candidates = ["id", "attack_cat"]
    X = df.drop(columns=[target_col] + [c for c in drop_candidates if c in df.columns], errors="ignore")

    # warm-up
    _ = pipeline.predict_proba(X.iloc[[0]])

    # single-sample latency over first 100 samples
    latencies_ms = []
    for i in range(100):
        sample = X.iloc[[i]]
        t0 = perf_counter()
        _ = pipeline.predict_proba(sample)
        t1 = perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    # batch inference for first 1000 samples
    batch = X.iloc[:1000]
    t0 = perf_counter()
    probs = pipeline.predict_proba(batch)
    t1 = perf_counter()
    batch_time_ms = (t1 - t0) * 1000

    print("\n=== Inference Benchmark ===")
    print(f"Single-sample mean latency   : {statistics.mean(latencies_ms):.4f} ms")
    print(f"Single-sample median latency : {statistics.median(latencies_ms):.4f} ms")
    print(f"Single-sample max latency    : {max(latencies_ms):.4f} ms")
    print(f"Batch size                   : {len(batch)}")
    print(f"Batch total latency          : {batch_time_ms:.4f} ms")
    print(f"Batch avg per sample         : {batch_time_ms / len(batch):.6f} ms")
    print(f"Predictions shape            : {probs.shape}")


if __name__ == "__main__":
    main()