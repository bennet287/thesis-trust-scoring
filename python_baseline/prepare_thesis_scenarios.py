import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
PIPELINE_PATH = MODELS_DIR / "numeric_dt_pipeline.joblib"


def load_numeric_features():
    df = pd.read_csv(DATA_PATH)

    target_candidates = ["label", "Label", "target", "class", "Class"]
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break

    drop_candidates = ["id", "attack_cat"]
    if target_col:
        drop_candidates.append(target_col)

    x_full = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")
    x = x_full.select_dtypes(include=["int64", "float64"]).copy()
    return x


def select_rows_for_targets(x: pd.DataFrame, model, target_m_values):
    probs = model.predict_proba(x)[:, 1]
    m_values = 1.0 - probs

    score_df = x.copy()
    score_df["M"] = m_values

    selected = []
    used_indices = set()

    for target_m in target_m_values:
        candidates = score_df.copy()
        if used_indices:
            candidates = candidates.drop(index=list(used_indices), errors="ignore")

        candidates["distance"] = (candidates["M"] - target_m).abs()
        idx = candidates["distance"].idxmin()
        used_indices.add(idx)

        row = score_df.loc[idx]
        selected.append({
            "values": [float(row[col]) for col in x.columns],
            "target_M": float(target_m),
            "selected_M": float(row["M"]),
        })

    return selected


def write_scenario_file(name, target_m_values, retrieval, expected, model, x):
    selected = select_rows_for_targets(x, model, target_m_values)

    scenario = {
        "name": name,
        "samples": [{"values": item["values"]} for item in selected],
        "retrieval": retrieval,
        "expected": expected,
        "meta": {
            "target_M": [item["target_M"] for item in selected],
            "selected_M": [item["selected_M"] for item in selected],
        },
    }

    out_path = MODELS_DIR / f"sequence_{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2)

    print(f"Saved: {out_path}")
    print("Selected M values:", [round(v, 4) for v in scenario["meta"]["selected_M"]])


def main():
    model = joblib.load(PIPELINE_PATH)
    x = load_numeric_features()

    write_scenario_file(
        name="normal",
        target_m_values=[0.9, 0.85, 0.88],
        retrieval=[0.9, 0.9, 0.95],
        expected=["ALLOW", "ALLOW", "ALLOW"],
        model=model,
        x=x,
    )

    write_scenario_file(
        name="attack",
        target_m_values=[0.9, 0.7, 0.5, 0.2, 0.0],
        retrieval=[0.2, 0.2, 0.1, 0.1, 0.1],
        expected=["ALLOW", "LIMIT", "QUARANTINE", "BLOCK", "BLOCK"],
        model=model,
        x=x,
    )

    write_scenario_file(
        name="noise",
        target_m_values=[0.1, 0.0],
        retrieval=[0.9, 0.95],
        expected=["BLOCK", "BLOCK"],
        model=model,
        x=x,
    )


if __name__ == "__main__":
    main()
