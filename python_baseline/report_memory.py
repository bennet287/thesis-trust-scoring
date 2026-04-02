from __future__ import annotations

import os
from pathlib import Path

import joblib
import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "numeric_dt_pipeline.joblib"


def main() -> None:
    process = psutil.Process(os.getpid())

    mem_before_mb = process.memory_info().rss / (1024 * 1024)
    model = joblib.load(MODEL_PATH)
    mem_after_mb = process.memory_info().rss / (1024 * 1024)

    print("=== Python Memory Report ===")
    print(f"Memory before load : {mem_before_mb:.3f} MB")
    print(f"Memory after load  : {mem_after_mb:.3f} MB")
    print(f"Approx model delta : {mem_after_mb - mem_before_mb:.3f} MB")

    # prevent lint-like complaints
    print(f"Loaded object type : {type(model)}")


if __name__ == "__main__":
    main()