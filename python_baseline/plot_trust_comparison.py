import json
import matplotlib.pyplot as plt

# === Manually copy values from your latest Rust output ===

time_steps = ["t1", "t2", "t3", "t4", "t5"]

model_trust = [
    1.000000,
    0.784264,
    0.592857,
    0.285714,
    0.000000
]

final_trust = [
    0.958319,
    0.762897,
    0.466857,
    0.430314,
    0.081217
]

# === Plot ===

plt.figure(figsize=(8, 5))

plt.plot(time_steps, model_trust, marker='o', label="Model Trust M(x)")
plt.plot(time_steps, final_trust, marker='s', linestyle='--', label="Final Trust T(x)")

# Highlight key transitions
for i in range(len(time_steps)):
    plt.text(i, model_trust[i] + 0.03, f"{model_trust[i]:.2f}", ha='center')
    plt.text(i, final_trust[i] - 0.08, f"{final_trust[i]:.2f}", ha='center')

plt.title("Trust Evolution: Model vs Calibrated Retrieval-Augmented Trust")
plt.xlabel("Time Step")
plt.ylabel("Trust Score")

plt.ylim(0, 1.05)

plt.grid(True)
plt.legend()

plt.tight_layout()

# Save figure
output_path = "../results/metrics/trust_comparison_plot.png"
plt.savefig(output_path, dpi=300)

print(f"Plot saved to: {output_path}")

plt.show()