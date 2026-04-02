import json
import pandas as pd
import joblib

# Load trained numeric pipeline
model = joblib.load("../results/models/numeric_dt_pipeline.joblib")

# Load dataset
df = pd.read_csv("../data/raw/UNSW_NB15_testing-set.csv")

# Detect target column
target_candidates = ["label", "Label", "target", "class", "Class"]
target_col = None
for col in target_candidates:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError("Could not detect target column in dataset.")

# Drop non-feature columns exactly like training
drop_candidates = ["id", "attack_cat", target_col]
X_full = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")

# Keep numeric-only features exactly like numeric_dt_pipeline training
X = X_full.select_dtypes(include=["int64", "float64"]).copy()

# Predict probabilities
probs = model.predict_proba(X)[:, 1]  # attack probability
trusts = 1.0 - probs

# Build helper dataframe
df_scores = X.copy()
df_scores["attack_probability"] = probs
df_scores["trust"] = trusts

# Target trust levels for a realistic progression
target_trusts = [1.00, 0.80, 0.60, 0.30, 0.00]

sequence_list = []
used_indices = set()

for target in target_trusts:
    candidate_df = df_scores.copy()
    candidate_df["distance"] = (candidate_df["trust"] - target).abs()

    # Exclude already used rows
    if used_indices:
        candidate_df = candidate_df.drop(index=list(used_indices), errors="ignore")

    best_idx = candidate_df["distance"].idxmin()
    used_indices.add(best_idx)

    row = df_scores.loc[best_idx]

    sequence_list.append({
        "values": [float(row[col]) for col in X.columns],
        "attack_probability": float(row["attack_probability"]),
        "trust": float(row["trust"])
    })

# Save sequence
out_path = "../results/models/sequence_input.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(sequence_list, f, indent=2)

print(f"Sequence file generated: {out_path}")
print("Selected trust levels:")
for i, item in enumerate(sequence_list, start=1):
    print(
        f"t{i}: attack_probability={item['attack_probability']:.6f}, "
        f"trust={item['trust']:.6f}"
    )