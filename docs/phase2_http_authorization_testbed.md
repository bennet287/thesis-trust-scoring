# Phase 2 — A minimal, **enforced** Zero Trust authorization baseline

**Status (2026-10-25):** local testbed, not a production Zero Trust architecture, not a neural-network experiment, and not an approved thesis-topic change. This is the **authorization reference implementation** needed before connecting potentially unreliable ML evidence to access decisions.

## 1. Research question
Can an edge gateway uphold explicit identity and resource permissions independently of ML predictions? If so, we can later test how optional behavioral risk/uncertainty signals affect *otherwise permitted* operations, without ever granting privileges beyond policy.

**Important:** Intrusion classification is different from authorization. A benign-looking flow does not establish that a device may change machine settings. This testbed defines separate, auditable identity/permission ground truth.

## 2. Minimal threat model
- **Protected resources:** simulated temperature telemetry (`GET /v1/telemetry`), simulated sampling interval configuration (`POST /v1/config`).
- **Identity evidence:** a secret bearer token mapped server-side to a principal. Missing/wrong tokens are rejected. A client-supplied `X-Principal` does not grant a different identity.
- **Policy:** `sensor-A` may only read telemetry; `operator-B` may read telemetry and modify configuration.
- **Enforcement:** rejected writes return HTTP 401/403 and **do not change** the in-memory sampling interval. Successful authorized writes change it.
- **Default-deny:** principals without an explicitly permitted method-resource pair cannot perform that operation.
- **Audit evidence:** stores principal (if verified), method, resource and decision. Raw tokens and request bodies are not logged.

The server listens only on `127.0.0.1`. Tokens are provided at runtime through `EDGE_TESTBED_TOKENS_JSON`; none are committed.

## 3. Architecture
```text
HTTP client --Bearer token--> token verification --> identity (principal)
                                |
                                v
                    explicit permission policy
                                |
                 +--------------+--------------+
                 |                             |
               denied                         allowed
                 |                             |
              HTTP 401/403              HTTP endpoint executes
              no mutation             GET telemetry or POST config
                 |                             |
                 +---------------+-------------+
                                 v
                      token-free audit entry
```

An **ML classifier is deliberately not connected**. Adding a probabilistic prediction to this system must not let a sensor bypass its permissions.

## 4. Run tests
From repository root:

```bash
python -m unittest discover -s tests -p 'test_policy_enforcement.py' -v
```

The nine tests issue live HTTP requests against a temporary loopback server and verify:
- authorized sensor read,
- unauthorized sensor write is rejected **and does not mutate state**,
- authorized operator write actually changes state,
- missing/incorrect bearer token rejected,
- identity-header spoofing rejected,
- unrecognized method/resource/query rejected,
- invalid configuration rejected without mutation,
- audit trail omits credentials,
- missing/duplicated credential registry rejected.

## 5. Start a manual loopback-only demonstration
On Windows PowerShell, from the repository root:

```powershell
$env:EDGE_TESTBED_TOKENS_JSON = python -c "import json,secrets; print(json.dumps({'sensor-A':secrets.token_urlsafe(32),'operator-B':secrets.token_urlsafe(32)}))"
python -m policy_testbed.server --port 8765
```

From a **second** PowerShell terminal (with the same environment variable set in that terminal, e.g. explicitly copied within the same machine), run:

```powershell
python -m unittest discover -s tests -p "test_policy_enforcement.py" -v
```

Note that the tests intentionally use **their own ephemeral server and tokens**; they validate behavior separately and do not contact the manual `8765` instance. To manually call the service, send local HTTP requests with `Authorization: Bearer <actual random token>` from the first terminal. Do not paste tokens into issues or screenshots.

## 6. Local evidence
- On 2026-10-25, **9/9 Python integration tests passed locally**, including actual HTTP policy enforcement and state-mutation checks.
- Hosted GitHub Actions result must be checked separately; passing tests do **not** prove production security.
- No external network connections, factories, sensors, real credentials, cryptographic device attestation, TLS, remote users, persistence, or production-grade identity lifecycle were tested.

## 7. Limitations and next experiment
This is an intentionally narrow proof-of-function and **not** secure to expose as a general network service. Bearer tokens over unencrypted HTTP are **unsafe over networks**, so the host is deliberately restricted to loopback. Production deployments require TLS/mTLS, proper identity provisioning/revocation, policy management, authenticated configuration updates, logging controls, threat analysis, and resilience testing.

For thesis-level experiments, introduce an evidence-based behavioral-risk *advisory layer* only after evaluating its reliability. Define fixed authorization cases (principal, action, resource) and compare:
1. static explicit policy (this testbed),
2. policy + conventional ML risk gate,
3. policy + uncertainty-aware restricted fallback (if justified by research).

No AI model should grant access a policy would otherwise deny. Measure unauthorized actions, legitimate operations disrupted, decision overhead, and predictable behavior when the model is wrong or unavailable. Independently validate on genuinely distinct traffic conditions before claiming improvement.
