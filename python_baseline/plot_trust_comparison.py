import json
import matplotlib.pyplot as plt

# === Load sequence (ground truth + model reference) ===
with open("../results/models/sequence_input.json", "r") as f:
    sequence = json.load(f)

# === Manually paste final trust values from Rust output ===
# (you just ran it — copy from terminal)
final_trust = [
    0.958319,  # t1
    0.762897,  # t2
    0.466857,  # t3
    0.430314,  # t4
    0.081217   # t5
]

model_trust = [step["trust"] for step in sequence]

time_steps = list(range(1, len(sequence) + 1))

# === Plot ===
plt.figure()

plt.plot(time_steps, model_trust, marker='o', label="Model Trust M(x)")
plt.plot(time_steps, final_trust, marker='s', linestyle='--', label="Final Trust T(x)")

plt.xlabel("Time Step")
plt.ylabel("Trust Score")
plt.title("Model Trust vs Final Trust (After Retrieval Calibration)")
plt.legend()
plt.grid()

plt.show()