import argparse
import itertools
import json
from pathlib import Path

from policy_tuning_common import (
    MODELS_DIR,
    evaluate_policy_on_scenario,
    load_pipeline,
    load_policy,
    load_profiles,
    normalize_band,
    objective,
    save_policy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune fusion weights for LTCA policy.json")
    parser.add_argument("--model", choices=["dt", "rf"], default="dt")
    parser.add_argument("--policy", type=Path, default=MODELS_DIR / "policy.json")
    parser.add_argument("--kb", type=Path, default=MODELS_DIR / "retrieval_kb.json")
    parser.add_argument("--scenario-prefix", default="calibration")
    parser.add_argument("--scenarios", nargs="+", default=["normal", "attack", "noise"])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/metrics/policy_calibration_weights_best_dt.json"),
    )
    parser.add_argument("--apply", action="store_true", help="Write best weights back to policy file")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.out.name.endswith("_dt.json") or args.out.name.endswith("_rf.json"):
        out_path = args.out
    else:
        out_path = args.out.with_name(f"policy_calibration_weights_best_{args.model}.json")

    base_policy = load_policy(args.policy)
    pipeline = load_pipeline(args.model)
    profiles = load_profiles(args.kb)

    scenario_paths = {
        s: MODELS_DIR / f"{args.scenario_prefix}_{s}.json" for s in args.scenarios
    }

    missing = [str(p) for p in scenario_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing scenario files: " + ", ".join(missing))

    high_wm = [0.75, 0.80, 0.85, 0.90]
    high_wr = [0.03, 0.05, 0.08, 0.10]

    med_wm = [0.55, 0.60, 0.65, 0.70]
    med_wr = [0.10, 0.15, 0.20, 0.25]

    low_wm = [0.40, 0.45, 0.50, 0.55]
    low_wr = [0.20, 0.25, 0.30, 0.35]

    best = None
    evaluated = 0

    for hwm, hwr, mwm, mwr, lwm, lwr in itertools.product(
        high_wm,
        high_wr,
        med_wm,
        med_wr,
        low_wm,
        low_wr,
    ):
        try:
            candidate = json.loads(json.dumps(base_policy))
            candidate["fusion"]["high_confidence"] = normalize_band(hwm, hwr)
            candidate["fusion"]["medium_confidence"] = normalize_band(mwm, mwr)
            candidate["fusion"]["low_confidence"] = normalize_band(lwm, lwr)
        except ValueError:
            continue

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
        raise RuntimeError("No valid candidate weights were evaluated")

    out_payload = {
        "model": args.model,
        "evaluated_candidates": evaluated,
        "score": best["score"],
        "fusion": best["policy"]["fusion"],
        "thresholds": best["policy"]["thresholds"],
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

    print(f"Saved best fusion weights: {out_path}")
    print(f"Evaluated candidates: {evaluated}")
    print(f"Best score: {best['score']}")

    if args.apply:
        save_policy(args.policy, best["policy"])
        print(f"Applied tuned fusion weights to: {args.policy}")


if __name__ == "__main__":
    main()
