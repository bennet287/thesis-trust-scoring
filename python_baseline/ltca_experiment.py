import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PATH = MODELS_DIR / "numeric_dt_pipeline.joblib"
KB_PATH = MODELS_DIR / "retrieval_kb.json"
SEQUENCE_PATH = MODELS_DIR / "sequence_input.json"
LOG_PATH = LOGS_DIR / "ltca_experiment.jsonl"


# -----------------------------
# Policy
# -----------------------------
def policy(trust: float) -> str:
    if trust >= 0.80:
        return "ALLOW"
    elif trust >= 0.50:
        return "LIMIT"
    elif trust >= 0.20:
        return "QUARANTINE"
    return "BLOCK"


# -----------------------------
# History buffer (non-circular)
# -----------------------------
class HistoryBuffer:
    def __init__(self, window_size: int = 4, neutral_prior: float = 0.5):
        self.window_size = window_size
        self.neutral_prior = neutral_prior
        self.buffer = []

    def compute(self) -> float:
        if not self.buffer:
            return self.neutral_prior
        return float(sum(self.buffer) / len(self.buffer))

    def update(self, trust_value: float) -> None:
        self.buffer.append(float(trust_value))
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def snapshot(self):
        return list(self.buffer)


# -----------------------------
# Retrieval profiles
# -----------------------------
@dataclass
class RetrievalProfile:
    name: str
    profile_trust: float
    centroid: np.ndarray
    count: int = 0
    description: str = ""


def load_profiles(kb_path: Path):
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    profiles = []
    for item in kb["profiles"]:
        profiles.append(
            RetrievalProfile(
                name=item["name"],
                profile_trust=float(item["profile_trust"]),
                centroid=np.array(item["centroid"], dtype=np.float64),
                count=int(item.get("count", 0)),
                description=item.get("description", "")
            )
        )
    return profiles


def retrieval(x: np.ndarray, profiles):
    if len(profiles) == 0:
        raise ValueError("No retrieval profiles loaded.")

    x2 = x.reshape(1, -1)
    best_profile = None
    best_similarity = -1.0

    for p in profiles:
        cent = p.centroid.reshape(1, -1)
        sim = float(cosine_similarity(x2, cent)[0][0])
        if sim > best_similarity:
            best_similarity = sim
            best_profile = p

    return {
        "profile_name": best_profile.name,
        "profile_trust": float(best_profile.profile_trust),
        "similarity": float(best_similarity),
        "R": float(best_profile.profile_trust),
    }


# -----------------------------
# Confidence regime + fusion
# -----------------------------
def confidence_regime(M: float) -> str:
    if M >= 0.9 or M <= 0.1:
        return "HIGH"
    return "UNCERTAIN"


def gated_weights(regime: str):
    if regime == "HIGH":
        return (0.80, 0.05, 0.15)
    return (0.50, 0.30, 0.20)


def fuse(M: float, R: float, H: float, regime: str) -> float:
    wM, wR, wH = gated_weights(regime)
    return float(wM * M + wR * R + wH * H)


def naive_fusion(M: float, R: float, H: float) -> float:
    return float(0.50 * M + 0.30 * R + 0.20 * H)


# -----------------------------
# Model inference
# -----------------------------
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


def model_trust(pipeline, x: np.ndarray):
    x2 = x.reshape(1, -1)
    attack_prob = float(pipeline.predict_proba(x2)[0][1])
    M = float(1.0 - attack_prob)
    return attack_prob, M


# -----------------------------
# Sequence loading
# -----------------------------
def load_sequence(sequence_path: Path):
    with open(sequence_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "samples" in data:
        samples = data["samples"]
    elif "sequence" in data:
        samples = data["sequence"]
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError(
            f"Unsupported sequence format in {sequence_path}. "
            "Expected {'samples': [...]}, {'sequence': [...]}, or a raw list."
        )

    parsed = []
    for row in samples:
        if isinstance(row, dict):
            if "values" in row:
                parsed.append(np.array(row["values"], dtype=np.float64))
            elif "features" in row:
                parsed.append(np.array(row["features"], dtype=np.float64))
            elif "x" in row:
                parsed.append(np.array(row["x"], dtype=np.float64))
            elif "input" in row:
                parsed.append(np.array(row["input"], dtype=np.float64))
            else:
                raise ValueError(
                    "Sequence entry is a dict but has no supported key. "
                    "Expected one of: 'values', 'features', 'x', 'input'."
                )
        else:
            parsed.append(np.array(row, dtype=np.float64))

    return parsed


# -----------------------------
# Logging
# -----------------------------
def log_result(step: int, result: dict):
    entry = {
        "step": step,
        "timestamp": time.time(),
        **result
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# -----------------------------
# LTCA pipeline
# -----------------------------
def ltca_pipeline(x: np.ndarray, pipeline, profiles, history_buffer: HistoryBuffer):
    attack_prob, M = model_trust(pipeline, x)

    rinfo = retrieval(x, profiles)
    R = rinfo["R"]

    H = history_buffer.compute()
    regime = confidence_regime(M)

    T_naive = naive_fusion(M, R, H)
    A_naive = policy(T_naive)

    T = fuse(M, R, H, regime)
    A = policy(T)

    result = {
        "attack_probability": attack_prob,
        "M": M,
        "R": R,
        "H": H,
        "T_naive": T_naive,
        "A_naive": A_naive,
        "T": T,
        "A": A,
        "regime": regime,
        "weights": gated_weights(regime),
        "retrieval_profile": rinfo["profile_name"],
        "retrieval_similarity": rinfo["similarity"],
        "history_before_update": history_buffer.snapshot(),
    }

    history_buffer.update(T)
    result["history_after_update"] = history_buffer.snapshot()

    return result


# -----------------------------
# Main experiment
# -----------------------------
def main():
    print("=== LTCA Experiment Runner ===")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Pipeline     : {PIPELINE_PATH}")
    print(f"KB           : {KB_PATH}")
    print(f"Sequence     : {SEQUENCE_PATH}")
    print(f"Log output   : {LOG_PATH}")
    print()

    pipeline = load_pipeline()
    profiles = load_profiles(KB_PATH)
    sequence = load_sequence(SEQUENCE_PATH)

    history = HistoryBuffer(window_size=4, neutral_prior=0.5)

    if LOG_PATH.exists():
        LOG_PATH.unlink()

    rows = []

    for i, x in enumerate(sequence, start=1):
        res = ltca_pipeline(x, pipeline, profiles, history)
        log_result(i, res)

        print(f"--- t{i} ---")
        print(f"Attack probability : {res['attack_probability']:.6f}")
        print(f"M(x)               : {res['M']:.6f}")
        print(
            f"R(x)               : {res['R']:.6f}  "
            f"({res['retrieval_profile']}, sim={res['retrieval_similarity']:.6f})"
        )
        print(f"H(x)               : {res['H']:.6f}")
        print(f"Regime             : {res['regime']}")
        print(f"Naive T(x)         : {res['T_naive']:.6f}  -> {res['A_naive']}")
        print(f"LTCA  T(x)         : {res['T']:.6f}  -> {res['A']}")
        print(f"Weights            : {res['weights']}")
        print(f"History before     : {res['history_before_update']}")
        print(f"History after      : {res['history_after_update']}")
        print()

        rows.append({
            "t": i,
            "attack_probability": res["attack_probability"],
            "M": res["M"],
            "R": res["R"],
            "H": res["H"],
            "T_naive": res["T_naive"],
            "A_naive": res["A_naive"],
            "T_ltca": res["T"],
            "A_ltca": res["A"],
            "regime": res["regime"],
            "retrieval_profile": res["retrieval_profile"],
            "retrieval_similarity": res["retrieval_similarity"],
        })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "metrics" / "ltca_experiment_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("Saved summary to:", out_csv)
    print("Saved JSONL log to:", LOG_PATH)


if __name__ == "__main__":
    main()