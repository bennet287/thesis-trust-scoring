import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "results" / "models"

ACTION_ORDER = {
    "ALLOW": 0,
    "LIMIT": 1,
    "QUARANTINE": 2,
    "BLOCK": 3,
}


@dataclass
class RetrievalProfile:
    name: str
    trust: float
    centroid: np.ndarray


@dataclass
class EvalResult:
    total_steps: int
    error_count: int
    under_enforcement: int
    over_enforcement: int


class HistoryBuffer:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.values: List[float] = []

    def compute(self) -> float:
        if not self.values:
            return 0.5
        return float(sum(self.values) / len(self.values))

    def update(self, value: float) -> None:
        self.values.append(float(value))
        if len(self.values) > self.window_size:
            self.values.pop(0)


class PolicyRuntime:
    def __init__(self, policy: Dict):
        self.policy = policy

    def _band(self, model_trust: float) -> Dict[str, float]:
        gates = self.policy["confidence_gates"]
        fusion = self.policy["fusion"]

        if model_trust >= gates["high"] or model_trust <= (1.0 - gates["high"]):
            return fusion["high_confidence"]

        if (
            model_trust >= gates["medium"]
            or model_trust <= gates["low_center_min"]
            or model_trust <= (1.0 - gates["high_center_max"])
        ):
            return fusion["medium_confidence"]

        return fusion["low_confidence"]

    def final_trust(self, model_trust: float, retrieval_trust: float, history_score: float) -> float:
        b = self._band(model_trust)
        return (
            b["w_m"] * model_trust
            + b["w_r"] * retrieval_trust
            + b["w_h"] * history_score
        )

    def action(self, trust: float) -> str:
        t = self.policy["thresholds"]
        if trust >= t["allow"]:
            return "ALLOW"
        if trust >= t["limit"]:
            return "LIMIT"
        if trust >= t["quarantine"]:
            return "QUARANTINE"
        return "BLOCK"


def load_policy(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_policy(path: Path, policy: Dict) -> None:
    path.write_text(json.dumps(policy, indent=2), encoding="utf-8")


def get_pipeline_path(model_tag: str) -> Path:
    if model_tag == "dt":
        return MODELS_DIR / "numeric_dt_pipeline.joblib"
    if model_tag == "rf":
        return MODELS_DIR / "numeric_rf_pipeline.joblib"
    raise ValueError(f"unsupported model: {model_tag}")


def load_pipeline(model_tag: str):
    return joblib.load(get_pipeline_path(model_tag))


def load_profiles(kb_path: Path) -> List[RetrievalProfile]:
    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    out: List[RetrievalProfile] = []
    for item in kb["profiles"]:
        out.append(
            RetrievalProfile(
                name=item["name"],
                trust=float(item["profile_trust"]),
                centroid=np.array(item["centroid"], dtype=np.float64),
            )
        )
    return out


def _model_trust(pipeline, x: np.ndarray) -> float:
    x2 = x.reshape(1, -1)
    feature_names = getattr(pipeline, "feature_names_in_", None)
    if feature_names is not None:
        import pandas as pd

        x2 = pd.DataFrame(x2, columns=list(feature_names))

    attack_probability = float(pipeline.predict_proba(x2)[0][1])
    return float(1.0 - attack_probability)


def _retrieval_trust(x: np.ndarray, profiles: List[RetrievalProfile]) -> float:
    x2 = x.reshape(1, -1)
    best_sim = -1.0
    best_trust = 0.5
    for p in profiles:
        sim = float(cosine_similarity(x2, p.centroid.reshape(1, -1))[0][0])
        if sim > best_sim:
            best_sim = sim
            best_trust = p.trust
    return best_trust


def load_scenario(path: Path) -> Tuple[List[np.ndarray], List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = [np.array(row["values"], dtype=np.float64) for row in payload["samples"]]
    expected = [str(x).upper() for x in payload["expected"]]
    return samples, expected


def evaluate_policy_on_scenario(
    scenario_path: Path,
    pipeline,
    profiles: List[RetrievalProfile],
    policy: Dict,
) -> EvalResult:
    samples, expected = load_scenario(scenario_path)
    runtime = PolicyRuntime(policy)
    history = HistoryBuffer(window_size=5)

    errors = 0
    under = 0
    over = 0

    for x, exp in zip(samples, expected):
        m = _model_trust(pipeline, x)
        r = _retrieval_trust(x, profiles)
        h = history.compute()
        t = runtime.final_trust(m, r, h)
        pred = runtime.action(t)
        history.update(m)

        if pred != exp:
            errors += 1

        pred_sev = ACTION_ORDER[pred]
        exp_sev = ACTION_ORDER[exp]
        if pred_sev < exp_sev:
            under += (exp_sev - pred_sev)
        elif pred_sev > exp_sev:
            over += (pred_sev - exp_sev)

    return EvalResult(
        total_steps=len(samples),
        error_count=errors,
        under_enforcement=under,
        over_enforcement=over,
    )


def objective(per_scenario: Dict[str, EvalResult]) -> float:
    normal = per_scenario["normal"]
    attack = per_scenario["attack"]
    noise = per_scenario["noise"]

    score = 0.0
    score += 1.0 * normal.error_count + 1.0 * normal.over_enforcement + 3.0 * normal.under_enforcement
    score += 3.0 * attack.error_count + 1.0 * attack.over_enforcement + 4.0 * attack.under_enforcement
    score += 2.0 * noise.error_count + 1.0 * noise.over_enforcement + 4.0 * noise.under_enforcement
    return score


def normalize_band(w_m: float, w_r: float) -> Dict[str, float]:
    w_h = 1.0 - w_m - w_r
    if w_h < 0.0:
        raise ValueError("invalid band weights: w_h would be negative")
    return {
        "w_m": round(float(w_m), 6),
        "w_r": round(float(w_r), 6),
        "w_h": round(float(w_h), 6),
    }
