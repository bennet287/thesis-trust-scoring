# Experiment Design: Model-Agnostic LTCA Validation (DT vs RF)

## Objective
Evaluate whether LTCA improves trust decision reliability across two classifier families:
- Decision Tree (DT): lightweight baseline model for edge deployment.
- Random Forest (RF): higher-capacity ensemble model.

The focus is decision reliability under uncertainty, not raw classifier benchmarking alone.

## Controlled Scenarios
Three controlled scenario families are used:
1. Normal behavior: high trust inputs and stable behavior.
2. Attack escalation: gradually degrading malicious progression.
3. Noisy retrieval: low model trust with misleadingly high retrieval trust.

Fairness rule:
- Scenario rows are selected from labels and attack categories only (model-independent, deterministic quantile picks).
- No model confidence score is used for selecting rows.

Scenario files:
- results/models/sequence_normal.json
- results/models/sequence_attack.json
- results/models/sequence_noise.json

## Protocol
For each model (DT and RF), run the same LTCA pipeline with:
- Model-only decision branch (base-paper style reference).
- Naive fusion decision branch.
- LTCA fusion + trend-aware policy branch.

Policy-only calibration discipline:
- Calibration performed on separate training-derived scenarios (prefix: calibration_).
- Frozen benchmark scenarios (prefix: sequence_) are never modified during calibration.
- Retrieval profiles and fusion mechanism are unchanged.

Core LTCA execution script:
- python_baseline/ltca_experiment.py

Calibration scripts:
- python_baseline/prepare_thesis_scenarios.py --data data/raw/UNSW_NB15_training-set.csv --prefix calibration
- python_baseline/calibrate_policy.py
- python_baseline/run_threeway_evaluation.py

Model pipelines:
- results/models/numeric_dt_pipeline.joblib
- results/models/numeric_rf_pipeline.joblib

## Primary Decision Metric
Count action-level errors against scenario expectations:
- Model-only Errors
- Naive Errors
- LTCA Errors

Output table:
- results/metrics/ltca_threeway_model_scenario_comparison.csv

## Classifier Reference Metrics
For context (not as primary thesis metric):
- Accuracy
- Precision
- Recall
- F1

Output table:
- results/metrics/classifier_metrics_dt_rf.csv

## Calibration Outcome
Calibration outputs:
- results/metrics/policy_calibration_grid_dt.csv
- results/metrics/policy_calibration_grid_rf.csv
- results/metrics/policy_calibration_best_dt.json
- results/metrics/policy_calibration_best_rf.json

Best configuration (both DT and RF):
- allow_threshold = 0.80
- limit_threshold = 0.50
- block_threshold = 0.20
- mild_trend_threshold = -0.04
- strong_trend_threshold = -0.10
- mild_penalty = 0.08
- strong_penalty = 0.16

## Current Results Summary
From results/metrics/ltca_threeway_model_scenario_comparison.csv:

- DT Normal: Model-only 1, Naive 1, LTCA 1
- DT Attack: Model-only 3, Naive 2, LTCA 3
- DT Noise: Model-only 0, Naive 2, LTCA 0
- RF Normal: Model-only 1, Naive 1, LTCA 1
- RF Attack: Model-only 2, Naive 3, LTCA 3
- RF Noise: Model-only 0, Naive 2, LTCA 1

From results/metrics/classifier_metrics_dt_rf.csv:

- DT F1: 0.9147
- RF F1: 0.9181

## Interpretation Guidance
- Under fair (model-independent) scenario construction, LTCA gains are mixed and scenario-dependent.
- LTCA improves both models on noisy-retrieval robustness (DT: 2 -> 0, RF: 2 -> 1).
- LTCA does not improve attack-escalation versus naive in this fair run (DT: 2 -> 3, RF: 3 -> 3).
- Classifier quality (RF F1 > DT F1) remains distinct from trust-policy reliability outcomes.
- Current evidence supports a bounded claim: LTCA is beneficial for retrieval-noise mitigation, while attack-escalation behavior remains a policy calibration problem under fair evaluation.

## Thesis-Safe Claim
Under fair, model-independent scenario construction and policy-only calibration on a separate split, LTCA consistently reduces errors in retrieval-noise scenarios for both DT and RF, while attack-escalation improvements are not automatic. This indicates that LTCA is currently most effective for contextual-noise control, and that attack-path responsiveness requires additional policy calibration.
