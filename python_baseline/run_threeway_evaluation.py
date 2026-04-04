import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Build three-way evaluation table from LTCA metrics JSON files.")
    parser.add_argument("--models", nargs="+", default=["dt", "rf"])
    parser.add_argument("--scenarios", nargs="+", default=["normal", "attack", "noise"])
    parser.add_argument(
        "--output",
        type=Path,
        default=METRICS_DIR / "ltca_threeway_model_scenario_comparison.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    for model in args.models:
        for scenario in args.scenarios:
            p = METRICS_DIR / f"ltca_metrics_{model}_{scenario}.json"
            if not p.exists():
                continue

            d = json.loads(p.read_text(encoding="utf-8"))
            rows.append(
                {
                    "Model": model.upper(),
                    "Scenario": scenario.capitalize(),
                    "Model-only Errors": d.get("model_errors"),
                    "Naive Errors": d.get("naive_errors"),
                    "LTCA Errors": d.get("ltca_errors"),
                    "Steps": d.get("total_steps", d.get("steps")),
                    "Allow Threshold": d.get("policy", {}).get("allow_threshold"),
                    "Limit Threshold": d.get("policy", {}).get("limit_threshold"),
                    "Block Threshold": d.get("policy", {}).get("block_threshold"),
                    "Mild Trend Thresh": d.get("policy", {}).get("mild_trend_threshold"),
                    "Strong Trend Thresh": d.get("policy", {}).get("strong_trend_threshold"),
                    "Mild Penalty": d.get("policy", {}).get("mild_penalty"),
                    "Strong Penalty": d.get("policy", {}).get("strong_penalty"),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
