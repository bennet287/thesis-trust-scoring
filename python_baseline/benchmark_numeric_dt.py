from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json
import statistics

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "numeric_dt_pipeline.joblib"
TEST_FILE = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"
OUT_FILE = PROJECT_ROOT / "results" / "metrics" / "python_numeric_dt_benchmark.json"


def detect_target_column(df: pd.DataFrame) -> str:
    for col in ["label", "Label", "target", "class", "Class"]:
        if col in df.columns:
            return col
    raise ValueError("Could not detect target column.")


def main() -> None:
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_FILE)

    target_col = detect_target_column(df)
    drop_candidates = ["id", "attack_cat", target_col]
    X_full = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")
    X = X_full.select_dtypes(include=["int64", "float64"]).copy()

    # warm-up
    _ = pipeline.predict_proba(X.iloc[[0]])

    single_latencies_ms = []
    for i in range(100):
        sample = X.iloc[[i]]
        t0 = perf_counter()
        _ = pipeline.predict_proba(sample)
        t1 = perf_counter()
        single_latencies_ms.append((t1 - t0) * 1000)

    # repeated one-sample inference
    sample = X.iloc[[0]]
    repeat_count = 1000
    t0 = perf_counter()
    for _ in range(repeat_count):
        _ = pipeline.predict_proba(sample)
    t1 = perf_counter()
    repeat_total_ms = (t1 - t0) * 1000
    repeat_avg_ms = repeat_total_ms / repeat_count

    # multi-sample sequence benchmark
    sequence = X.iloc[:5]
    seq_repeat_count = 1000
    t0 = perf_counter()
    for _ in range(seq_repeat_count):
        _ = pipeline.predict_proba(sequence)
    t1 = perf_counter()
    sequence_total_ms = (t1 - t0) * 1000
    sequence_avg_ms = sequence_total_ms / seq_repeat_count

    results = {
        "single_mean_ms": statistics.mean(single_latencies_ms),
        "single_median_ms": statistics.median(single_latencies_ms),
        "single_max_ms": max(single_latencies_ms),
        "repeat_count": repeat_count,
        "repeat_total_ms": repeat_total_ms,
        "repeat_avg_ms": repeat_avg_ms,
        "sequence_batch_size": len(sequence),
        "sequence_repeat_count": seq_repeat_count,
        "sequence_total_ms": sequence_total_ms,
        "sequence_avg_ms": sequence_avg_ms,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== Python Numeric DT Benchmark ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print(f"Saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()