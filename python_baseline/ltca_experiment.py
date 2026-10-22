import json
import time
import argparse
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
@dataclass
class PolicyConfig:
    allow_threshold: float = 0.80
    limit_threshold: float = 0.50
    block_threshold: float = 0.20
    mild_trend_threshold: float = -0.05
    strong_trend_threshold: float = -0.10
    mild_penalty: float = 0.10
    strong_penalty: float = 0.20


def policy(trust: float, cfg: PolicyConfig) -> str:
    if trust >= cfg.allow_threshold:
        return "ALLOW"
    elif trust >= cfg.limit_threshold:
        return "LIMIT"
    elif trust >= cfg.block_threshold:
        return "QUARANTINE"
    return "BLOCK"


def adjust_for_trend(trust: float, history_values, cfg: PolicyConfig):
    if len(history_values) < 3:
        return trust

    trend = history_values[-1] - history_values[-3]
    if trend <= cfg.strong_trend_threshold:
        return max(0.0, trust - cfg.strong_penalty)
    if trend <= cfg.mild_trend_threshold:
        return max(0.0, trust - cfg.mild_penalty)
    return trust


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


def retrieval_with_override(x: np.ndarray, profiles, override_r=None):
    rinfo = retrieval(x, profiles)
    if override_r is None:
        return rinfo

    forced_r = float(override_r)
    rinfo["R"] = forced_r
    rinfo["profile_name"] = f"OVERRIDE({forced_r:.3f})"
    return rinfo


# -----------------------------
# Confidence regime + fusion
# -----------------------------
def confidence_regime(M: float) -> str:
    if M >= 0.9 or M <= 0.1:
        return "HIGH"
    return "UNCERTAIN"


def is_degrading(history_values) -> bool:
    if len(history_values) < 3:
        return False
    return history_values[-1] < history_values[-2] < history_values[-3]


def compute_model_confidence(M: float) -> float:
    return max(0.0, min(1.0, abs(M - 0.5) * 2.0))


def compute_signal_agreement(M: float, R: float, H: float) -> float:
    agreement = 1.0 - (abs(M - R) + abs(M - H)) / 2.0
    return max(0.0, min(1.0, agreement))


def compute_trend_value(history_values):
    if len(history_values) < 3:
        return 0.0
    return float(history_values[-1] - history_values[-3])


def compute_weights(M: float, R: float, H: float, history_values):
    confidence = compute_model_confidence(M)
    agreement = compute_signal_agreement(M, R, H)

    if len(history_values) >= 2:
        previous_h = float(sum(history_values[:-1]) / len(history_values[:-1]))
        trend = max(0.0, previous_h - H)
    else:
        trend = 0.0

    wM = 0.5 + 0.5 * confidence + 0.3 * trend
    wR = 0.3 * agreement
    wH = 0.2 + 0.2 * (1.0 - confidence)

    total = wM + wR + wH
    if total <= 0.0:
        return (0.5, 0.3, 0.2)

    return (wM / total, wR / total, wH / total)


def fuse(M: float, R: float, H: float, weights) -> float:
    wM, wR, wH = weights
    return float(wM * M + wR * R + wH * H)


def naive_fusion(M: float, R: float, H: float) -> float:
    return float(0.50 * M + 0.30 * R + 0.20 * H)


# -----------------------------
# Model inference
# -----------------------------
def load_pipeline(pipeline_path: Path):
    return joblib.load(pipeline_path)


def _as_model_input(pipeline, x: np.ndarray):
    x2 = x.reshape(1, -1)
    feature_names = getattr(pipeline, "feature_names_in_", None)
    if feature_names is not None:
        if len(feature_names) != x2.shape[1]:
            raise ValueError(
                f"Feature length mismatch: pipeline expects {len(feature_names)} features, "
                f"but sample has {x2.shape[1]}."
            )
        return pd.DataFrame(x2, columns=list(feature_names))
    return x2


def model_trust(pipeline, x: np.ndarray):
    model_input = _as_model_input(pipeline, x)
    attack_prob = float(pipeline.predict_proba(model_input)[0][1])
    M = float(1.0 - attack_prob)
    return attack_prob, M


def transform_retrieval_features(x: np.ndarray, pipeline) -> np.ndarray:
    """Apply the fitted numeric preprocessor used to create retrieval centroids.

    Input features are raw, while retrieval_kb.json stores centroids in the
    imputed + MinMax-scaled space. Both sides must use the same representation.
    """
    steps = pipeline.named_steps
    if "imputer" not in steps or "scaler" not in steps:
        raise ValueError("Retrieval requires a numeric pipeline with imputer and scaler steps")
    model_input = _as_model_input(pipeline, np.asarray(x, dtype=np.float64))
    imputed = steps["imputer"].transform(model_input)
    transformed = steps["scaler"].transform(imputed)
    return np.asarray(transformed, dtype=np.float64).reshape(-1)


# -----------------------------
# Sequence loading
# -----------------------------
def load_sequence(sequence_path: Path):
    with open(sequence_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenario_name = sequence_path.stem
    retrieval_overrides = None
    expected_labels = None

    # Scenario bundle format: [{"name": ..., "samples": ..., "retrieval": ...}, ...]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "name" in data[0] and "samples" in data[0]:
        raise ValueError(
            "Scenario bundle detected. Please pass a single scenario file, or use a dedicated loader."
        )

    # Single scenario format: {"name": ..., "samples": [...], "retrieval": [...], "expected": [...]}
    if isinstance(data, dict) and "samples" in data:
        scenario_name = data.get("name", scenario_name)
        retrieval_overrides = data.get("retrieval")
        expected_labels = data.get("expected")
        samples = data["samples"]
    elif "samples" in data:
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

    if retrieval_overrides is not None and len(retrieval_overrides) != len(parsed):
        raise ValueError(
            "Length mismatch: retrieval overrides must match number of samples."
        )

    if expected_labels is not None and len(expected_labels) != len(parsed):
        raise ValueError(
            "Length mismatch: expected labels must match number of samples."
        )

    return {
        "name": scenario_name,
        "samples": parsed,
        "retrieval": retrieval_overrides,
        "expected": expected_labels,
    }


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
def ltca_pipeline(
    x: np.ndarray,
    pipeline,
    profiles,
    history_buffer: HistoryBuffer,
    policy_cfg: PolicyConfig,
    retrieval_override=None,
):
    attack_prob, M = model_trust(pipeline, x)

    retrieval_x = transform_retrieval_features(x, pipeline)
    rinfo = retrieval_with_override(retrieval_x, profiles, retrieval_override)
    R = rinfo["R"]

    H = history_buffer.compute()
    regime = confidence_regime(M)
    history_before = history_buffer.snapshot()
    weights = compute_weights(M, R, H, history_before)
    trend_sensitive = is_degrading(history_before)
    model_confidence = compute_model_confidence(M)
    signal_agreement = compute_signal_agreement(M, R, H)
    trend_value = compute_trend_value(history_before)

    T_naive = naive_fusion(M, R, H)
    A_naive = policy(T_naive, policy_cfg)

    T_model = M
    A_model = policy(T_model, policy_cfg)

    T = fuse(M, R, H, weights)
    T_adjusted = adjust_for_trend(T, history_before, policy_cfg)
    A = policy(T_adjusted, policy_cfg)
    trend_penalty = max(0.0, float(T - T_adjusted))

    result = {
        "attack_probability": attack_prob,
        "M": M,
        "T_model": T_model,
        "A_model": A_model,
        "R": R,
        "H": H,
        "T_naive": T_naive,
        "A_naive": A_naive,
        "T": T,
        "T_adjusted": T_adjusted,
        "A": A,
        "model_confidence": model_confidence,
        "signal_agreement": signal_agreement,
        "trend_value": trend_value,
        "trend_penalty": trend_penalty,
        "t_fused": T,
        "t_final": T_adjusted,
        "policy_action": A,
        "regime": regime,
        "weights": weights,
        "trend_sensitive": trend_sensitive,
        "retrieval_profile": rinfo["profile_name"],
        "retrieval_similarity": rinfo["similarity"],
        "history_before_update": history_before,
    }

    history_buffer.update(T)
    result["history_after_update"] = history_buffer.snapshot()

    return result


def evaluate_decisions(rows, expected_labels):
    if expected_labels is None:
        return None

    model_errors = 0
    naive_errors = 0
    ltca_errors = 0
    for row, expected in zip(rows, expected_labels):
        if row["A_model"] != expected:
            model_errors += 1
        if row["A_naive"] != expected:
            naive_errors += 1
        if row["A_ltca"] != expected:
            ltca_errors += 1

    return {
        "model_errors": model_errors,
        "naive_errors": naive_errors,
        "ltca_errors": ltca_errors,
        "total_steps": len(expected_labels),
    }


# -----------------------------
# Main experiment
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run LTCA experiment for one scenario file.")
    parser.add_argument(
        "--sequence",
        type=Path,
        default=SEQUENCE_PATH,
        help="Path to sequence file (raw list or scenario dict with samples/retrieval/expected).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional scenario label to use in output file names.",
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=PIPELINE_PATH,
        help="Path to sklearn pipeline joblib used to compute M(x).",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Optional model tag for output file names (e.g., dt, rf).",
    )
    parser.add_argument("--allow-threshold", type=float, default=0.80)
    parser.add_argument("--limit-threshold", type=float, default=0.50)
    parser.add_argument("--block-threshold", type=float, default=0.20)
    parser.add_argument("--mild-trend-threshold", type=float, default=-0.05)
    parser.add_argument("--strong-trend-threshold", type=float, default=-0.10)
    parser.add_argument("--mild-penalty", type=float, default=0.10)
    parser.add_argument("--strong-penalty", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    sequence_path = args.sequence
    pipeline_path = args.pipeline

    policy_cfg = PolicyConfig(
        allow_threshold=args.allow_threshold,
        limit_threshold=args.limit_threshold,
        block_threshold=args.block_threshold,
        mild_trend_threshold=args.mild_trend_threshold,
        strong_trend_threshold=args.strong_trend_threshold,
        mild_penalty=args.mild_penalty,
        strong_penalty=args.strong_penalty,
    )

    sequence_bundle = load_sequence(sequence_path)
    scenario_name = args.label if args.label else sequence_bundle["name"]
    model_tag = args.model_tag if args.model_tag else pipeline_path.stem.replace("_pipeline", "")
    safe_name = scenario_name.replace(" ", "_")
    run_log_path = LOGS_DIR / f"ltca_experiment_{model_tag}_{safe_name}.jsonl"

    print("=== LTCA Experiment Runner ===")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Pipeline     : {pipeline_path}")
    print(f"KB           : {KB_PATH}")
    print(f"Sequence     : {sequence_path}")
    print(f"Model tag    : {model_tag}")
    print(f"Scenario     : {scenario_name}")
    print(f"Log output   : {run_log_path}")
    print(f"Policy       : {policy_cfg}")
    print()

    pipeline = load_pipeline(pipeline_path)
    profiles = load_profiles(KB_PATH)
    sequence = sequence_bundle["samples"]
    retrieval_overrides = sequence_bundle["retrieval"]
    expected_labels = sequence_bundle["expected"]

    history = HistoryBuffer(window_size=4, neutral_prior=0.5)

    if run_log_path.exists():
        run_log_path.unlink()

    rows = []

    for i, x in enumerate(sequence, start=1):
        retrieval_override = None if retrieval_overrides is None else retrieval_overrides[i - 1]
        res = ltca_pipeline(
            x,
            pipeline,
            profiles,
            history,
            policy_cfg=policy_cfg,
            retrieval_override=retrieval_override,
        )

        entry = {
            "step": i,
            "timestamp": time.time(),
            "scenario": scenario_name,
            **res,
        }
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"--- t{i} ---")
        print(f"Attack probability : {res['attack_probability']:.6f}")
        print(f"M(x)               : {res['M']:.6f}")
        print(
            f"R(x)               : {res['R']:.6f}  "
            f"({res['retrieval_profile']}, sim={res['retrieval_similarity']:.6f})"
        )
        print(f"H(x)               : {res['H']:.6f}")
        print(f"Regime             : {res['regime']}")
        print(f"Model T(x)         : {res['T_model']:.6f}  -> {res['A_model']}")
        print(f"Naive T(x)         : {res['T_naive']:.6f}  -> {res['A_naive']}")
        print(f"LTCA fused T(x)    : {res['t_fused']:.6f}")
        print(f"LTCA final T(x)    : {res['t_final']:.6f}  -> {res['policy_action']}")
        print(f"Model confidence   : {res['model_confidence']:.6f}")
        print(f"Signal agreement   : {res['signal_agreement']:.6f}")
        print(f"Trend penalty      : {res['trend_penalty']:.6f}")
        print(f"Weights            : {res['weights']}")
        print(f"History before     : {res['history_before_update']}")
        print(f"History after      : {res['history_after_update']}")
        print()

        rows.append({
            "t": i,
            "scenario": scenario_name,
            "attack_probability": res["attack_probability"],
            "M": res["M"],
            "T_model": res["T_model"],
            "A_model": res["A_model"],
            "R": res["R"],
            "H": res["H"],
            "T_naive": res["T_naive"],
            "A_naive": res["A_naive"],
            "model_confidence": res["model_confidence"],
            "signal_agreement": res["signal_agreement"],
            "trend_penalty": res["trend_penalty"],
            "T_fused": res["t_fused"],
            "T_final": res["t_final"],
            "policy_action": res["policy_action"],
            "T_ltca": res["t_final"],
            "A_ltca": res["A"],
            "regime": res["regime"],
            "retrieval_profile": res["retrieval_profile"],
            "retrieval_similarity": res["retrieval_similarity"],
        })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "metrics" / f"ltca_experiment_summary_{model_tag}_{safe_name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    metrics_path = RESULTS_DIR / "metrics" / f"ltca_metrics_{model_tag}_{safe_name}.json"
    metrics_payload = {
        "model_tag": model_tag,
        "scenario": scenario_name,
        "steps": len(rows),
        "policy": {
            "allow_threshold": policy_cfg.allow_threshold,
            "limit_threshold": policy_cfg.limit_threshold,
            "block_threshold": policy_cfg.block_threshold,
            "mild_trend_threshold": policy_cfg.mild_trend_threshold,
            "strong_trend_threshold": policy_cfg.strong_trend_threshold,
            "mild_penalty": policy_cfg.mild_penalty,
            "strong_penalty": policy_cfg.strong_penalty,
        },
    }
    eval_metrics = evaluate_decisions(rows, expected_labels)
    if eval_metrics is not None:
        metrics_payload.update(eval_metrics)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Saved summary to:", out_csv)
    print("Saved JSONL log to:", run_log_path)
    print("Saved metrics to:", metrics_path)

    if eval_metrics is not None:
        print("\nScenario error metrics")
        print(f"Model errors : {eval_metrics['model_errors']}/{eval_metrics['total_steps']}")
        print(f"Naive errors : {eval_metrics['naive_errors']}/{eval_metrics['total_steps']}")
        print(f"LTCA errors  : {eval_metrics['ltca_errors']}/{eval_metrics['total_steps']}")


if __name__ == "__main__":
    main()