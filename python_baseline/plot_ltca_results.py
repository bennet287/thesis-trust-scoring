import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "python_baseline" else CURRENT_DIR
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_PATH = RESULTS_DIR / "metrics" / "ltca_experiment_summary.csv"
PLOT_DIR = RESULTS_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)

# 1. Comparison Table (thesis-ready format)
print("\n=== LTCA vs Naïve Fusion Comparison ===")
table_cols = ["t", "M", "R", "H", "T_naive", "A_naive", "T_ltca", "A_ltca"]
print(df[table_cols].to_string(index=False))

# Save table as CSV for LaTeX/Word
df_table = df[table_cols]
df_table.to_csv(PLOT_DIR / "comparison_table.csv", index=False)

# 2. Plot: Trust evolution over time
plt.figure(figsize=(10, 6))
plt.plot(df["t"], df["M"], marker="o", linewidth=2, markersize=8, label="Model Trust M(x)")
plt.plot(df["t"], df["T_naive"], marker="s", linewidth=2, markersize=8, label="Naïve Fusion T_naive")
plt.plot(df["t"], df["T_ltca"], marker="^", linewidth=2, markersize=8, label="LTCA T(x)")

# Highlight key point t5
plt.axvline(x=5, linestyle="--", alpha=0.5, label="Attack case (t5)")
t5_ltca = df.loc[df["t"] == 5, "T_ltca"].iloc[0]
plt.scatter([5], [t5_ltca], s=100, zorder=5)

plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Trust Score", fontsize=12)
plt.title("LTCA vs Naïve Fusion: Trust Evolution Over Time", fontsize=14)
plt.xticks(df["t"])
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_DIR / "trust_evolution.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOT_DIR / "trust_evolution.pdf", bbox_inches="tight")
plt.show()

print(f"\nPlot saved to {PLOT_DIR / 'trust_evolution.png'}")
print(f"Table saved to {PLOT_DIR / 'comparison_table.csv'}")