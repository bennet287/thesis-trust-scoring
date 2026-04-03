from pathlib import Path
import matplotlib.pyplot as plt

# ===== Data from your latest calibrated Rust output =====

time_steps = ["t1", "t2", "t3", "t4", "t5"]
x = list(range(1, len(time_steps) + 1))

attack_probability = [
    0.000000,
    0.215736,
    0.407143,
    0.714286,
    1.000000,
]

model_trust = [
    1.000000,
    0.784264,
    0.592857,
    0.285714,
    0.000000,
]

final_trust = [
    0.958319,
    0.762897,
    0.466857,
    0.430314,
    0.081217,
]

# ===== Output path =====

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

png_path = OUT_DIR / "trust_comparison_publication.png"
pdf_path = OUT_DIR / "trust_comparison_publication.pdf"

# ===== Plot =====

plt.figure(figsize=(10, 6))

# Decision regions
plt.axhspan(0.80, 1.00, alpha=0.10, label="ALLOW region")
plt.axhspan(0.50, 0.80, alpha=0.08, label="LIMIT region")
plt.axhspan(0.20, 0.50, alpha=0.06, label="QUARANTINE region")
plt.axhspan(0.00, 0.20, alpha=0.05, label="BLOCK region")

# Boundary lines
plt.axhline(0.80, linestyle="--", linewidth=1)
plt.axhline(0.50, linestyle="--", linewidth=1)
plt.axhline(0.20, linestyle="--", linewidth=1)

# Curves
plt.plot(x, attack_probability, marker="^", linewidth=2, label="Attack Probability")
plt.plot(x, model_trust, marker="o", linewidth=2, label="Model Trust M(x)")
plt.plot(x, final_trust, marker="s", linewidth=2, linestyle="--", label="Final Trust T(x)")

# Point labels
for xi, y in zip(x, attack_probability):
    plt.annotate(f"{y:.2f}", (xi, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

for xi, y in zip(x, model_trust):
    plt.annotate(f"{y:.2f}", (xi, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

for xi, y in zip(x, final_trust):
    plt.annotate(f"{y:.2f}", (xi, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9)

# Axis formatting
plt.xticks(x, time_steps)
plt.ylim(0, 1.05)
plt.xlabel("Time Step")
plt.ylabel("Score")
plt.title("Trust Evolution with Calibrated Retrieval-Augmented Fusion")

# Region labels
plt.text(5.35, 0.90, "ALLOW", va="center", fontsize=10)
plt.text(5.35, 0.65, "LIMIT", va="center", fontsize=10)
plt.text(5.35, 0.35, "QUARANTINE", va="center", fontsize=10)
plt.text(5.35, 0.10, "BLOCK", va="center", fontsize=10)

plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")

plt.show()