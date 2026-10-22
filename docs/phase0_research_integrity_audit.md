# Phase 0 — Research integrity audit (2026-10-22)

**Status:** preliminary engineering audit; no scientifically validated improvement or approved thesis-topic change. Work from the archived repository snapshot (April 2026) plus isolated draft fixes. Keep `main` unchanged until review.

## Why this audit is needed

The current prototype combines model prediction M, a retrieved behavior-profile trust R, and history H. Its security-action outputs are *demo decisions*; they do not yet demonstrate an independently evaluated authorization system.

## Confirmed defects and evidence

### 1. Retrieval profile evaluation leakage (P0)
- Original `python_baseline/build_retrieval_kb.py` loaded `UNSW_NB15_testing-set.csv` to build retrieval centroids.
- Draft PR #2 switches to the designated training file and adds provenance and a source-split guard.
- **Local smoke test:** generated 4 centroids from 82,332 training rows, using 39 numeric features. Unit tests: 2/2 passed.
- This correction does not validate historical metrics. Saved JSON/CBOR KB artifacts still reflect previous generation until intentionally regenerated, versioned, and retested.

### 2. Retrieval feature-space inconsistency (P0)
- KB centroids are constructed in the fitted imputer + MinMax-scaled feature space.
- The original `ltca_pipeline` compared those centroids to raw sample vectors using cosine similarity.
- Draft PR #4 applies the **same fitted preprocessing** to incoming features before retrieval.
- **Local check:** nearest-centroid identity differs on 301 of the first 1,000 test rows when comparing raw-input retrieval against correctly transformed-input retrieval, with a new training-only KB.
- This is a *software consistency* result, not a test of anomaly detection or authorization success. Tests of the new scale code: 2/2 passed (4/4 including PR #2 tests in the local audit workspace).

### 3. Demo retrieval scores are manually overridden (P0)
- `prepare_thesis_scenarios.py` assembles rows by attack category and writes hand-set `retrieval` values and `expected` actions.
- `retrieval_with_override` substitutes the provided value for R. Hence the stored demo comparisons are **not evidence of performance of the learned/retrieved centroids**.
- With both local fixes active and a new training-only KB, the three tiny Decision Tree demo scenarios yielded these *illustrative* outcomes:

| Scenario | Hand-set R (first values) | Calculated R (first values) | LTCA actions changed? |
| --- | --- | --- | --- |
| normal (3 rows) | 0.90, 0.90, 0.95 | 0.05, 0.95, 0.95 | No |
| attack (5 rows) | 0.20, 0.20, 0.10 | 0.65, 0.95, 0.95 | No |
| noise (2 rows) | 0.90, 0.95 | 0.65, 0.05 | No |

There are only 10 constructed observations total. They are **not** chronological device traces. The `expected` actions are scenario author choices, not access-policy ground truth. Changing R without changing the actions in this tiny demo is not a meaningful robustness conclusion.

### 4. Calibration and held-out independence (P0)
- `prepare_thesis_scenarios.py` defaults to the **testing** dataset.
- `calibrate_policy.py`, `tune_weights.py`, and `tune_thresholds.py` can optimize using constructed scenario expectations.
- Do not report calibrated-on-scenario results as independent held-out evaluation. Create separate development/calibration and final evaluation data and pre-register the settings evaluated.

### 5. Detection is not authorization (P0)
- UNSW-NB15 attack/normal labels indicate intrusion-detection classes, not permissions.
- A policy decision must separately depend on a verified principal, requested action, resource, and applicable authorization rule.
- The current decision labels ALLOW/LIMIT/QUARANTINE/BLOCK do not establish actual enforced network-access outcomes.

## Local environment and source provenance
- Python 3.13.5, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.2.3, joblib 1.5.3.
- Training CSV SHA-256: `7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c`.
- Testing CSV SHA-256: `0ebd9eb06d53ae223fbe7948737985e851e00e9c20140508d653de4b1655fdd6`.
- Local Python tests and run outputs were **not** uploaded as prevalidated CI outputs. Test results must be reproduced in GitHub CI.
- Source data and saved models originate from the uploaded project archive; the tested local source incorporated both proposed changes. This document is not a claim that either PR has been merged.

## Next reproducible experiment (pending)
1. Separate training, validation/calibration, and fully held-out data with documented partition logic and hashes.
2. Rebuild models, retrieval profiles, and policy calibration from **development data only**.
3. Disable all hard-coded retrieval overrides during scientific evaluations.
4. Measure independent intrusion-classification metrics and error calibration before testing downstream authorization.
5. Define a protected resource, authenticated identities and explicit access-policy ground truth; send real requests through an enforcement point.
6. Benchmark static policies, ML risk-only rules, original fusion, and (if justified by evidence) uncertainty-aware policy-constrained authorization.
7. Report security errors, disruption to legitimate requests, runtime and memory under precisely identified hardware/conditions.
8. Compare against relevant publications before making novelty claims.

## Public repository cautions
Check committed history for secrets, third-party dataset redistribution rights, and provenance/AI-assistance statements. `.gitignore` does not remove already committed content or history. Review before publishing further artifacts.

Track: #1, #3; draft fixes: #2, #4.
