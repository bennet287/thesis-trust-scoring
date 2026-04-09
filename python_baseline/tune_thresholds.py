import argparse
import json
from pathlib import Path

from policy_tuning_common import (
    MODELS_DIR,
    evaluate_policy_on_scenario,
    load_pipeline,
    load_policy,
    load_profiles,
    objective,
    save_policy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune decision thresholds for LTCA policy.json")
    parser.add_argument("--model", choices=["dt", "rf"], default="dt")
    parser.add_argument("--policy", type=Path, default=MODELS_DIR / "policy.json")
    parser.add_argument("--kb", type=Path, default=MODELS_DIR / "retrieval_kb.json")
    parser.add_argument("--scenario-prefix", default="calibration")
    parser.add_argument("--scenarios", nargs="+", default=["normal", "attack", "noise"])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/metrics/policy_calibration_thresholds_best_dt.json"),
    )
    parser.add_argument("--apply", action="store_true", help="Write best thresholds back to policy file")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.out.name.endswith("_dt.json") or args.out.name.endswith("_rf.json"):
        out_path = args.out
    else:
        out_path = args.out.with_name(f"policy_calibration_thresholds_best_{args.model}.json")

    base_policy = load_policy(args.policy)
    pipeline = load_pipeline(args.model)
    profiles = load_profiles(args.kb)

    scenario_paths = {
        s: MODELS_DIR / f"{args.scenario_prefix}_{s}.json" for s in args.scenarios
    }

    missing = [str(p) for p in scenario_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing scenario files: " + ", ".join(missing))

    allow_values = [0.80, 0.85, 0.90, 0.95, 0.97]
    limit_values = [0.55, 0.60, 0.65, 0.70, 0.75]
    quarantine_values = [0.30, 0.35, 0.40, 0.45, 0.50]

    best = None
    evaluated = 0

    for allow_t in allow_values:
        for limit_t in limit_values:
            for quarantine_t in quarantine_values:
                if not (allow_t > limit_t > quarantine_t):
                    continue

                candidate = json.loads(json.dumps(base_policy))
                candidate["thresholds"] = {
                    "allow": round(float(allow_t), 6),
                    "limit": round(float(limit_t), 6),
                    "quarantine": round(float(quarantine_t), 6),
                }

                per_scenario = {
                    s: evaluate_policy_on_scenario(path, pipeline, profiles, candidate)
                    for s, path in scenario_paths.items()
                }
                score = objective(per_scenario)
                evaluated += 1

                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "policy": candidate,
                        "per_scenario": per_scenario,
                    }

    if best is None:
        raise RuntimeError("No valid candidate thresholds were evaluated")

    out_payload = {
        "model": args.model,
        "evaluated_candidates": evaluated,
        "score": best["score"],
        "thresholds": best["policy"]["thresholds"],
        "fusion": best["policy"]["fusion"],
        "confidence_gates": best["policy"]["confidence_gates"],
        "per_scenario": {
            s: {
                "total_steps": r.total_steps,
                "errors": r.error_count,
                "under_enforcement": r.under_enforcement,
                "over_enforcement": r.over_enforcement,
            }
            for s, r in best["per_scenario"].items()
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print(f"Saved best thresholds: {out_path}")
    print(f"Evaluated candidates: {evaluated}")
    print(f"Best score: {best['score']}")

    if args.apply:
        save_policy(args.policy, best["policy"])
        print(f"Applied tuned thresholds to: {args.policy}")


if __name__ == "__main__":
    main()
