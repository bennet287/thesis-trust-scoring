# Project Status Update — Zero Trust Edge Trust Scoring
_Date: 2026-04-02_

## What we completed today

### 1. Project setup and environment
- Created the project structure for:
  - `python_baseline/`
  - `rust_inference/`
  - `wasm_module/`
  - `host_runner/`
  - `results/`
- Set up Python virtual environment and installed required packages.
- Set up Rust toolchain and WebAssembly target.
- Initialized Git repository and started milestone-based commits.

### 2. Python baseline model
- Loaded the **UNSW-NB15** dataset.
- Trained a **Decision Tree** baseline.
- Evaluated the model and saved metrics.
- Converted attack probability into a continuous **trust score**:
  - `trust = 1 - attack_probability`
- Mapped trust to policy actions:
  - `ALLOW`
  - `LIMIT`
  - `QUARANTINE`
  - `BLOCK`

### 3. Python inference and benchmarking
- Built single-sample inference pipeline.
- Built Python benchmark for latency.
- Built numeric-only benchmark for fairer comparison with deployment core.
- Added Python memory report.

### 4. Rust inference
- Exported a Rust-friendly numeric Decision Tree representation.
- Reimplemented inference in Rust.
- Verified exact parity with Python:
  - attack probability delta = `0.000000`
  - trust score delta = `0.000000`

### 5. WebAssembly deployment
- Compiled the Rust inference core into **WebAssembly**.
- Built a Wasmtime-based host runner.
- Verified exact parity across Python → Rust → Wasm.

### 6. Dynamic Wasm inference
- Generalized the Wasm module to accept dynamic 39-feature input through Wasm memory.
- Verified that arbitrary real inputs can be passed from host to Wasm and inferred correctly.

### 7. Temporal trust / behavior memory
- Added short-term behavioral memory using recent trust history.
- Computed history-aware trust:
  - `H(x)` = history score
- Built fused trust:
  - `T(x) = 0.7 * M(x) + 0.3 * H(x)` in the first version
- Ran temporal trust simulations.

### 8. Real dataset sequence simulation
- Replaced synthetic drift with a real trust progression sequence extracted from dataset samples.
- Simulated:
  - high-trust
  - medium-trust
  - suspicious
  - malicious progression
- Verified time-evolving trust and action transitions.

### 9. Evaluation package
- Benchmarked:
  - Python latency
  - Wasm latency
  - repeated one-sample inference
  - repeated multi-sample sequence inference
- Reported artifact sizes:
  - Wasm binary
  - exported JSON model
  - Python pipeline artifact
- Reported Python memory footprint.

### 10. Retrieval layer
- Built a local retrieval KB with trust-profile centroids:
  - trusted
  - moderate
  - suspicious
  - malicious
- Added retrieval score:
  - `R(x)`
- Integrated full hybrid trust fusion:
  - `M(x) + R(x) + H(x)`

### 11. Retrieval calibration
- Identified that naïve retrieval fusion could destabilize highly confident model outputs.
- Implemented **confidence-gated fusion**:
  - high-confidence model outputs dominate
  - retrieval contributes more in uncertain regions
- Fixed critical cases:
  - safe input remained `ALLOW`
  - malicious input remained `BLOCK`

### 12. Visualization
- Created trust comparison plots:
  - `Model Trust M(x)` vs `Final Trust T(x)`
- Generated publication-ready figure outputs.

## Current system status

The current prototype now supports:

```text
Input sample
→ Wasm model inference
→ Model trust M(x)
→ Retrieval trust R(x)
→ History score H(x)
→ Confidence-gated trust fusion
→ Final trust T(x)
→ Policy action
```

This means the project already includes the **core thesis prototype**.

## Milestones completed

- **Milestone 1** — Python → Rust → Wasm pipeline with identical trust scoring
- **Milestone 2** — Generalized dynamic-input Wasm inference
- **Milestone 3** — Wasm benchmarking
- **Milestone 4** — Behavior-aware trust scoring with history-based fusion
- **Milestone 5** — Temporal trust simulation with history-aware fusion
- **Milestone 6** — Evaluation baseline (Python vs Wasm latency, memory, artifact size, dynamic inference, behavior-aware trust)
- **Milestone 7** — Retrieval-augmented trust scoring (`M + R + H` fusion with local KB)
- **Milestone 8** — Retrieval calibration with confidence gating + visualization

## Key achievements from today

1. Built an end-to-end lightweight trust-scoring pipeline.
2. Demonstrated exact Python/Rust/Wasm parity.
3. Added a behavior-aware temporal layer.
4. Added a local retrieval layer.
5. Fixed retrieval instability with confidence gating.
6. Created evaluation outputs and visualizations.

## What is left to do

### 1. Write-up / documentation
- Write a proper **Results** section.
- Write a **Discussion** section.
- Document:
  - architecture
  - formulas
  - implementation choices
  - experimental setup

### 2. Final evaluation improvements
- Build a clearer comparison between:
  - model-only trust
  - behavior-aware trust
  - retrieval-augmented trust
  - calibrated trust fusion
- Possibly test more real sequences near policy thresholds.

### 3. Optional model extension
- Add **Random Forest** version for comparison.
- Compare DT vs RF under the same Wasm deployment path.

### 4. Retrieval improvements
- Improve KB quality:
  - more profiles
  - better prototype selection
  - possibly top-k retrieval instead of single-centroid match

### 5. Thesis packaging
- Finalize publication-quality figures.
- Convert the implementation and evaluation into thesis chapter format.
- Prepare seminar slides / 5-minute presentation.

## Recommended next session

When work resumes, the best next step is:

1. Write the **Results + Discussion** section
2. Build final comparison tables
3. Prepare thesis-ready figures and explanation text

## Git push commands

From project root:

```bash
git status
git push origin main
```

If remote is not configured yet:

```bash
git remote add origin <YOUR_REPO_URL>
git branch -M main
git push -u origin main
```

## Closing note

Today’s work moved the project from a baseline IDS implementation to a **real hybrid trust-scoring edge prototype** with:
- Wasm deployment
- behavioral memory
- retrieval augmentation
- confidence-aware fusion
- evaluation and visualization

This is already strong Master’s thesis progress.
