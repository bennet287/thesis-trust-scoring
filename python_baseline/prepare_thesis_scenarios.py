import json
import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"
MODELS_DIR = PROJECT_ROOT / "results" / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Build deterministic label-grounded scenario files.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Input CSV dataset path used for scenario row selection.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sequence",
        help="Output file prefix. Example: 'sequence' or 'calibration'.",
    )
    return parser.parse_args()


def load_dataset(data_path: Path):
    df = pd.read_csv(data_path)

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
    return df, x, target_col


def pick_row_by_quantile(candidates: pd.DataFrame, q: float, used_indices: set[int]):
    available = candidates.loc[~candidates.index.isin(used_indices)]
    if available.empty:
        raise ValueError("No available rows left after excluding used indices.")

    ordered = available.sort_index()
    pos = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return int(ordered.index[pos])


def pick_category_row(df: pd.DataFrame, category: str, used_indices: set[int], q: float = 0.5):
    candidates = df[df["attack_cat"] == category]
    if candidates.empty:
        return None
    idx = pick_row_by_quantile(candidates, q=q, used_indices=used_indices)
    used_indices.add(idx)
    return idx


def build_normal_samples(df: pd.DataFrame, used_indices: set[int]):
    normal_df = df[df["attack_cat"] == "Normal"]
    quantiles = [0.2, 0.5, 0.8]
    indices = []
    for q in quantiles:
        idx = pick_row_by_quantile(normal_df, q=q, used_indices=used_indices)
        used_indices.add(idx)
        indices.append(idx)
    return indices


def build_attack_samples(df: pd.DataFrame, used_indices: set[int]):
    steps = [
        ["Reconnaissance", "Fuzzers"],
        ["Analysis", "Backdoor"],
        ["Exploits"],
        ["DoS", "Generic"],
        ["Worms", "Shellcode", "Generic"],
    ]

    quantiles = [0.2, 0.4, 0.5, 0.6, 0.8]
    indices = []

    for category_options, q in zip(steps, quantiles):
        selected = None
        for category in category_options:
            selected = pick_category_row(df, category, used_indices=used_indices, q=q)
            if selected is not None:
                break

        if selected is None:
            attack_df = df[df["attack_cat"] != "Normal"]
            selected = pick_row_by_quantile(attack_df, q=q, used_indices=used_indices)
            used_indices.add(selected)

        indices.append(selected)

    return indices


def build_noise_samples(df: pd.DataFrame, used_indices: set[int]):
    indices = []
    candidates = [
        ("Reconnaissance", 0.65),
        ("Analysis", 0.7),
        ("Backdoor", 0.7),
    ]

    for category, q in candidates:
        idx = pick_category_row(df, category, used_indices=used_indices, q=q)
        if idx is not None:
            indices.append(idx)
        if len(indices) == 2:
            break

    if len(indices) < 2:
        attack_df = df[df["attack_cat"] != "Normal"]
        while len(indices) < 2:
            q = 0.75 if len(indices) == 0 else 0.85
            idx = pick_row_by_quantile(attack_df, q=q, used_indices=used_indices)
            used_indices.add(idx)
            indices.append(idx)

    return indices


def rows_to_samples(x: pd.DataFrame, source_df: pd.DataFrame, indices: list[int]):
    samples = []
    meta = []
    for idx in indices:
        row_x = x.loc[idx]
        row_src = source_df.loc[idx]
        samples.append({"values": [float(row_x[col]) for col in x.columns]})
        meta.append(
            {
                "index": int(idx),
                "attack_cat": str(row_src.get("attack_cat", "UNKNOWN")),
                "label": int(row_src.get("label", row_src.get("Label", 0))),
            }
        )
    return samples, meta

def write_scenario_file(name, sample_indices, retrieval, expected, x, source_df, prefix):
    samples, selection_meta = rows_to_samples(x, source_df, sample_indices)

    scenario = {
        "name": name,
        "samples": samples,
        "retrieval": retrieval,
        "expected": expected,
        "meta": {
            "selection_strategy": "label_grounded_attack_category",
            "selected_rows": selection_meta,
        },
    }

    out_path = MODELS_DIR / f"{prefix}_{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2)

    print(f"Saved: {out_path}")
    print("Selected rows:", selection_meta)


def main():
    args = parse_args()
    source_df, x, target_col = load_dataset(args.data)

    if "attack_cat" not in source_df.columns:
        raise ValueError("Expected column 'attack_cat' for label-grounded scenario construction.")

    if target_col and target_col != "label":
        source_df = source_df.rename(columns={target_col: "label"})

    used_indices: set[int] = set()
    normal_indices = build_normal_samples(source_df, used_indices)
    attack_indices = build_attack_samples(source_df, used_indices)
    noise_indices = build_noise_samples(source_df, used_indices)

    write_scenario_file(
        name="normal",
        sample_indices=normal_indices,
        retrieval=[0.9, 0.9, 0.95],
        expected=["ALLOW", "ALLOW", "ALLOW"],
        x=x,
        source_df=source_df,
        prefix=args.prefix,
    )

    write_scenario_file(
        name="attack",
        sample_indices=attack_indices,
        retrieval=[0.2, 0.2, 0.1, 0.1, 0.1],
        expected=["ALLOW", "LIMIT", "QUARANTINE", "BLOCK", "BLOCK"],
        x=x,
        source_df=source_df,
        prefix=args.prefix,
    )

    write_scenario_file(
        name="noise",
        sample_indices=noise_indices,
        retrieval=[0.9, 0.95],
        expected=["BLOCK", "BLOCK"],
        x=x,
        source_df=source_df,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
