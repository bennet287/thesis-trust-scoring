# Phase 0B — Internal classification audit (2026-10-22)

**Status:** exploratory experiment only. The separately approved thesis topic is unchanged. This compares network-flow **classification**, not real Zero Trust authorization.

## Goal
Test whether retrieval-based score fusion clearly outperforms an ML-only baseline **before** proposing a new Edge AI authorization architecture. The old demo scenarios used hand-set retrieval overrides and cannot answer this question.

## Reproducible protocol
- Dataset: `data/raw/UNSW_NB15_training-set.csv`; SHA-256 `7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c`
- Seed 42; stratified, pairwise-disjoint indices: **49,399 training**, **16,466 calibration**, **16,467 internal audit**.
- Model: freshly fitted numeric-only Decision Tree (max_depth=8, min_samples_split=10, min_samples_leaf=5, entropy).
- Imputer and MinMaxScaler fitted on training only; retrieval centroids constructed from training predictions only and compared to equivalently scaled calibration/audit rows.
- Three flow risk scores: model-only, naive 50/30/20 fusion, adaptive fusion based on model confidence and signal agreement.
- **No manual retrieval overrides**. Independent rows have neutral `H=0.5`; this is *not* a test of genuine chronological history, trust trends, or the full stateful LTCA pipeline.
- For each score, the threshold maximizing attack-class F1 is selected on calibration rows only, and then held fixed for the internal audit partition.
- Each score's raw Brier statistic is descriptive; the fused scores are not calibrated probabilities.
- Uses the former repository's training file only. This dataset has been used in prior model development, so **the internal split is not an untouched external test set**.

## Local observed results
All three selected decision thresholds were approximately 0.50. The internal-audit partition contained 16,467 samples.

| Method | F1 | TN | FP | FN | TP | Errors = FP + FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model only | **0.95244** | 7,018 | 382 | 476 | 8,591 | **858** |
| Naive fusion | 0.95026 | 6,980 | 420 | 479 | 8,588 | 899 |
| Adaptive fusion | 0.95102 | 7,000 | 400 | 484 | 8,583 | 884 |

In this **one exploratory split**, fusion did not improve F1 or total classification errors. This is *not proof that fusion never helps*: these settings exclude temporal history, use heuristic profile trusts, and were not tested on independent environments or time/device-held-out data.

## Sensitivity check across five random seeds
The same procedure (re-fitting the classifier, centroids and per-method calibration thresholds each time) was repeated with seeds 7, 17, 42, 73 and 123, using the same public source file:

| Seed | Model F1 | Naive F1 | Adaptive F1 | Adaptive - Model |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.94760 | 0.94874 | 0.95007 | +0.00247 |
| 17 | 0.95290 | 0.95143 | 0.95219 | -0.00071 |
| 42 | 0.95244 | 0.95026 | 0.95102 | -0.00142 |
| 73 | 0.95208 | 0.95030 | 0.95174 | -0.00034 |
| 123 | 0.94933 | 0.94813 | 0.94863 | -0.00070 |

Adaptive fusion exceeded the model-only F1 on **one of five** random splits and trailed on four. This reinforces that the improvement is **not reliably demonstrated** here, but the five evaluations reuse one source dataset and are correlated; do not interpret these as independent external replications or a formal statistical significance test.

## How to reproduce
```bash
python -m pip install numpy pandas scikit-learn joblib
python -m unittest discover -s tests -v
python python_baseline/audit_independent_classification.py --seed 42 --output results/audit/phase0b_internal_classification.json
```

Local unit tests added in this PR: **4/4 passed**; experiment reproduced in a local workspace. GitHub CI has not yet been verified, and the generated results are not committed by default.

## Key validity restrictions
1. UNSW-NB15 labels classify *network-flow attack status*, not whether an authenticated subject is authorized for an operation on a resource. Do **not** call these access-control success rates.
2. No evidence here of prediction uncertainty under distribution shift, unknown attack categories, edge-device latency, enforceable policy decisions, or real device behavior over time.
3. A random stratified split can overestimate generalization when flows share capture contexts, devices, sessions, or other correlated artifacts. An independent capture/testbed is required for thesis conclusions.
4. Because the wider project used the available datasets while tuning, the internal audit split is useful for **engineering debugging only**, not a pristine final research test.
5. Further comparisons need repeated seeds/uncertainty intervals; the small difference between F1 scores here may not be statistically reliable.

## Research decision
Move next to a **separate** two-part testbed: (A) ML confidence/calibration on unfamiliar device/network behavior, and (B) explicit identities/resources/actions/policy enforcement. Only then investigate whether uncertainty-aware fallback improves protected access outcomes over static policy and ordinary ML-based risk decisions.

Related: audit issue #1, data leakage PR #2, feature-space repair PR #4.