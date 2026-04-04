import argparse
import json
import itertools
from pathlib import Path

import joblib
import pandas as pd

from ltca_experiment import (
    HistoryBuffer,
    PolicyConfig,
    evaluate_decisions,
    load_profiles,
    load_sequence,
    ltca_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

KB_PATH = MODELS_DIR / "retrieval_kb.json"
DT_PIPELINE_PATH = MODELS_DIR / "numeric_dt_pipeline.joblib"
RF_PIPELINE_PATH = MODELS_DIR / "numeric_rf_pipeline.joblib"


def parse_args():
    parser = argparse.ArgumentParser(description="Policy-only calibration for LTCA trend control.")
    parser.add_argument("--model", choices=["dt", "rf"], default="dt")
    parser.add_argument("--prefix", default="calibration", help="Calibration scenario file prefix.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["normal", "attack", "noise"],
        help="Scenario names to use for calibration.",
    )
    return parser.parse_args()


def get_pipeline_path(model_tag: str) -> Path:
    return DT_PIPELINE_PATH if model_tag == "dt" else RF_PIPELINE_PATH


def run_one_scenario(sequence_path: Path, pipeline, profiles, cfg: PolicyConfig):
    bundle = load_sequence(sequence_path)
    samples = bundle["samples"]
    retrieval_overrides = bundle["retrieval"]
    expected_labels = bundle["expected"]

    history = HistoryBuffer(window_size=4, neutral_prior=0.5)
    rows = []

    for i, x in enumerate(samples):
        retrieval_override = None if retrieval_overrides is None else retrieval_overrides[i]
        res = ltca_pipeline(
            x,
            pipeline,
            profiles,
            history,
            policy_cfg=cfg,
            retrieval_override=retrieval_override,
        )

        rows.append(
            {
                "A_model": res["A_model"],
                "A_naive": res["A_naive"],
                "A_ltca": res["A"],
            }
        )

    metrics = evaluate_decisions(rows, expected_labels)
    if metrics is None:
        raise ValueError(f"Scenario {sequence_path} must provide expected labels.")
    return metrics


def objective(per_scenario_metrics: dict):
    # Prioritize attack/noise behavior over normal for security relevance.
    attack = per_scenario_metrics["attack"]
    noise = per_scenario_metrics["noise"]
    normal = per_scenario_metrics["normal"]

    score = 0
    score += 3 * attack["ltca_errors"]
    score += 2 * noise["ltca_errors"]
    score += 1 * normal["ltca_errors"]

    # Penalize regressions vs naive to avoid overfitting to absolute counts only.
    score += max(0, attack["ltca_errors"] - attack["naive_errors"]) * 3
    score += max(0, noise["ltca_errors"] - noise["naive_errors"]) * 2
    score += max(0, normal["ltca_errors"] - normal["naive_errors"]) * 1

    return score


def build_grid():
    allow_values = [0.80]
    limit_values = [0.50]
    block_values = [0.20, 0.25]

    mild_trend_values = [-0.04, -0.05, -0.06]
    strong_trend_values = [-0.10, -0.12]

    mild_penalty_values = [0.08, 0.10, 0.12]
    strong_penalty_values = [0.16, 0.20, 0.24]

    grid = []
    for vals in itertools.product(
        allow_values,
        limit_values,
        block_values,
        mild_trend_values,
        strong_trend_values,
        mild_penalty_values,
        strong_penalty_values,
    ):
        (
            allow_t,
            limit_t,
            block_t,
            mild_tt,
            strong_tt,
            mild_p,
            strong_p,
        ) = vals

        if strong_tt >= mild_tt:
            continue
        if strong_p < mild_p:
            continue

        grid.append(
            PolicyConfig(
                allow_threshold=allow_t,
                limit_threshold=limit_t,
                block_threshold=block_t,
                mild_trend_threshold=mild_tt,
                strong_trend_threshold=strong_tt,
                mild_penalty=mild_p,
                strong_penalty=strong_p,
            )
        )

    return grid


def main():
    args = parse_args()

    pipeline = joblib.load(get_pipeline_path(args.model))
    profiles = load_profiles(KB_PATH)

    scenario_paths = {
        s: MODELS_DIR / f"{args.prefix}_{s}.json" for s in args.scenarios
    }

    missing = [str(p) for p in scenario_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing calibration scenarios: " + ", ".join(missing)
        )

    grid = build_grid()

    rows = []
    best = None

    for cfg in grid:
        per_scenario = {}
        for name, path in scenario_paths.items():
            per_scenario[name] = run_one_scenario(path, pipeline, profiles, cfg)

        score = objective(per_scenario)

        row = {
            "model": args.model,
            "score": score,
            "allow_threshold": cfg.allow_threshold,
            "limit_threshold": cfg.limit_threshold,
            "block_threshold": cfg.block_threshold,
            "mild_trend_threshold": cfg.mild_trend_threshold,
            "strong_trend_threshold": cfg.strong_trend_threshold,
            "mild_penalty": cfg.mild_penalty,
            "strong_penalty": cfg.strong_penalty,
        }

        for scenario_name, m in per_scenario.items():
            row[f"{scenario_name}_model_errors"] = m["model_errors"]
            row[f"{scenario_name}_naive_errors"] = m["naive_errors"]
            row[f"{scenario_name}_ltca_errors"] = m["ltca_errors"]

        rows.append(row)

        if best is None or score < best["score"]:
            best = {"score": score, "config": cfg, "per_scenario": per_scenario, "row": row}

    out_grid = METRICS_DIR / f"policy_calibration_grid_{args.model}.csv"
    pd.DataFrame(rows).sort_values(by=["score"]).to_csv(out_grid, index=False)

    best_out = {
        "model": args.model,
        "score": best["score"],
        "config": {
            "allow_threshold": best["config"].allow_threshold,
            "limit_threshold": best["config"].limit_threshold,
            "block_threshold": best["config"].block_threshold,
            "mild_trend_threshold": best["config"].mild_trend_threshold,
            "strong_trend_threshold": best["config"].strong_trend_threshold,
            "mild_penalty": best["config"].mild_penalty,
            "strong_penalty": best["config"].strong_penalty,
        },
        "per_scenario": best["per_scenario"],
    }

    out_best = METRICS_DIR / f"policy_calibration_best_{args.model}.json"
    with open(out_best, "w", encoding="utf-8") as f:
        json.dump(best_out, f, indent=2)

    print("Saved grid:", out_grid)
    print("Saved best:", out_best)
    print("Best score:", best["score"])
    print("Best config:", best_out["config"])


if __name__ == "__main__":
    main()
