# Phase 3A — Policy-first risk-gate baseline (2026-10-25)

**Status:** research engineering baseline only. No deployed hardware, production-grade credentials, validated uncertainty method, or novel thesis contribution has been demonstrated.

## Why this milestone exists
Our Phase 1 experiments showed a network intrusion classifier can miss unfamiliar attack traffic and assign very low apparent attack probability. Phase 2 proved a small HTTP server can actually enforce authenticated device permissions.

The immediate question is: **Can we introduce a second, server-controlled ML-risk decision layer without ever bypassing explicit access permissions?**

This code tests the *decision mechanism and enforcement invariant*, not the quality of a deployed model.

## Baselines for later fair experiments
1. **Policy only:** authenticate bearer token, check principal/action/resource permissions, then execute permitted request.
2. **Policy + risk gate (this PR):** authenticate and check permissions *first*. For otherwise permitted requests, consult server-side risk evidence. If score is high, deny; if invalid/unavailable, fail closed (HTTP 503) instead of allowing. Only a valid lower-risk result proceeds.
3. **Policy + uncertainty-aware gate (not implemented):** a separate experimentally validated uncertainty signal could lead to different *restrictive* handling on uncertain evidence. Define its comparison protocol before claiming an advantage.

**Invariant:** Every accepted request must satisfy explicit policy permissions. A benign ML score cannot grant `sensor-A` the right to POST configuration.

## Architecture

```text
request -> server-verified bearer token -> principal permissions
                           |
                  policy denied -> HTTP 403; no mutation
                           |
                  policy allowed -> [optional server-side risk provider]
                                       |
                    invalid/offline -> HTTP 503; no mutation
                    high risk       -> HTTP 403; no mutation
                    below threshold -> protected action proceeds
```

## Implementation
- `policy_testbed/risk_policy.py`: deterministic policy-first logic with numeric and finite risk score validation, explicit risk threshold.
- `policy_testbed/server.py`: optional `risk_mode="risk_gated"`, server-injected `risk_provider(principal,method,path)`. Existing policy-only mode remains the default.
- `tests/test_policy_risk_gate.py`: HTTP integration and pure decision tests. Covers risk-based veto, no state mutation, no privilege expansion, no trust of `X-Risk` headers, invalid/unavailable risk and exceptions.
- The production-looking handler **does not accept a client-supplied risk score**. It also does not currently query an actual trained model. Tests use deterministic server-side stand-ins.

## Verification
```bash
python -m unittest discover -s tests -v
```

Tests execute real localhost HTTP calls. The project's existing Python audit workflow also runs the full test suite in GitHub Actions on pull requests. Record passing CI explicitly before claiming hosted verification.

## Crucial research limitations
- **No genuine ML inference is connected** yet. A risk provider receives (verified principal, HTTP method, resource), not a network-flow feature vector or validated device-bound telemetry.
- Decision Tree output from UNSW-NB15 is a *network-flow attack classification score*, not a principal-specific authorization probability. NEVER fabricate correspondence by arbitrarily assigning dataset flows to testbed device IDs.
- Threshold 0.70 is an **illustrative baseline setting** (not optimized from held-out authorization outcomes). In serious experiments, tune thresholds on development data, keep fixed on final test, and report error/disruption trade-offs over a range.
- The risk gate denies operations on high risk and can interrupt legitimate access; it doesn't automatically improve security. In policy-only mode an unauthorized operation is already blocked, which may limit what can be improved.
- This basic server uses loopback HTTP and demonstration tokens; do not expose it to network clients or describe it as production Zero Trust.

## Engineering next steps
1. Build a trustworthy **device-bound telemetry ingestion path** that attributes measurements to an authenticated principal. Provide controlled timestamping, freshness and provenance.
2. Train/calibrate a model with data from these devices or a clearly documented testbed, avoiding traffic/identity association assumptions.
3. Add a reproducible request workload with ground-truth *legitimate authorized requests* and *unsafe behavior* to measure extra risk-gate protection versus disruption.
4. Evaluate optional uncertainty methods separately, including missing evidence, false positives and latency/edge memory.
5. Ask supervisors to validate the problem, novelty and feasible threat model **before** treating any tentative topic revision as approved.

## Glossary for learning
- **Authentication:** proving which device or person is making a request.
- **Authorization:** deciding what that identity is permitted to do.
- **Risk assessment:** optional evidence about suspicious behavior; not a permission grant.
- **Policy decision point:** logic that decides based on identity, policy and extra evidence.
- **Policy enforcement point:** server component that actually allows or blocks operations.
- **Fail closed:** when required evidence is missing or invalid, do not perform a protected operation.

Reference: NIST SP 800-207, https://doi.org/10.6028/NIST.SP.800-207 (PE, PA, PEP model).
