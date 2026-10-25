# Literature comparison v0.1 — traditional, existing and candidate research
_Date: 2026-10-25 · Status: working literature matrix, NOT a completed systematic review_

## Approved topic and decision point

Approved topic: **Multi-Signal Fusion-Based Scoring for Zero Trust at the Edge**.

Candidate research question (unapproved):
> When an edge traffic detector's evidence is untrustworthy or stale, can an explicitly bounded response strategy improve actual authorization outcomes **without overriding identity/permissions or disrupting legitimate work excessively**, at tolerable edge overhead?

**Novelty is not established.** Several closely related methods already exist, especially TrustEdge-AI (2026).

## A. Traditional approaches — baseline category

### A1. Perimeter/network-location controls
- Principle: control entry through perimeter zones/firewalls/VPN; may treat internal network position as a partial trust signal.
- Strength: familiar, comparatively straightforward boundary enforcement.
- Limitation: network location alone does not demonstrate a device/user should access a particular resource; lateral movement and dynamic edge access challenge perimeter assumptions.
- Comparator: perimeter/segment-based allowlist under matched identities/resources, if useful to threat model.
- Primary reference: NIST SP 800-207 (2020), https://doi.org/10.6028/NIST.SP.800-207.

### A2. Explicit identity-based permissions (RBAC, ABAC, least privilege)
- Principle: authenticate identity and enforce roles or attributes/conditions on principal-action-resource requests.
- Strength: actual permission ground truth can be enforced; no classifier needed to define entitlement.
- Limitation relevant to *our question*: a static role-only rule does not independently capture unknown malicious device behavior; ABAC, however, can already incorporate dynamic attributes. Do not falsely imply all traditional access control is static.
- Comparator: deterministic role/attribute policy baseline; our Phase 2 localhost testbed demonstrates this narrow property.

## B. Existing approaches — concrete precedents

### B1. NIST SP 800-207 Zero Trust Architecture (2020)
- Standard reference: policy engine (PE) makes decision based on policy and contextual evidence; policy administrator (PA) executes instructions; policy enforcement point (PEP) guards resource.
- Existing capability: continuous/contextual authorization, not simply static permissions.
- Therefore **contextual scoring plus enforcement is not new**.
- Sources: https://www.nist.gov/publications/zero-trust-architecture ; https://pages.nist.gov/zero-trust-architecture/VolumeB/architecture.html

### B2. NIST SP 800-207A (2023) application identity/policy
- Existing capability: service identities, fine-grained access policies, enforcement using gateways/sidecars in cloud-native systems.
- Difference of target environment: edge/IoT device constraints, which require evidence rather than a novelty claim.
- https://csrc.nist.gov/pubs/sp/800/207/a/final

### B3. Yang, 'Design and Verification of a Zero-Trust Access Control Framework for Edge Computing Environments' (Aug 2026)
- Existing approach: dynamic identity validation, behavior/historical/contextual trust scoring, adaptive trust threshold, contextual policy enforcement and consensus verification in simulated edge environments.
- Compares RBAC and ABAC on simulated access requests; includes latency and trust evaluation.
- **Very close precedent for combining multiple trust signals with enforcement.**
- Evaluate rigor: assess simulated labels, external validity, enforced outcomes, resource overhead and assumption about independent signals.
- https://doi.org/10.3233/ATDE250871

### B4. Ayari et al., 'TrustEdge-AI: Uncertainty-Aware Dynamic Trust Intelligence for Adaptive Cyber Defense in IoT Edge Networks' (Sep 2026)
- **Critical closest precedent**: joins reliability-weighted classifier/anomaly evidence, behavioral consistency, bounded evolving trust and graduated allow/observe/restrict/isolate policies.
- Its abstract explicitly reports withheld-family stress tests, calibration, trust dynamics, ablation and low-latency evaluation over BoT-IoT, TON-IoT and UNSW-NB15.
- **Direct overlap with both our originally proposed fusion and revised uncertainty-aware concept.** We must inspect the full paper and implementation/reproduction assumptions before claiming any distinct research contribution.
- https://doi.org/10.14569/IJACSA.2026.0170996

### B5. Ovadia et al., 'Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty under Dataset Shift' (NeurIPS 2019)
- Existing approach/evidence: broad benchmarks demonstrate model uncertainty/calibration often degrades under distribution shift, with differences across estimation methods.
- Foundational *motivation*, not authorization innovation.
- https://papers.nips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

### B6. Li et al., 'Position: Supervised Classifiers Answer the Wrong Questions for OOD Detection' (ICML 2025)
- Challenges the premise that high uncertainty necessarily detects novel samples or that distance in supervised feature space identifies new threats.
- Warns not to assume naive confidence-margin or distance threshold will solve confidently wrong predictions.
- https://proceedings.mlr.press/v267/li25ec.html

### B7. Selective/reject-option approaches
- Existing idea: classifiers can abstain from doubtful predictions. Conformal reject-option approaches analyze error/rejection tradeoffs under assumptions; shift can violate exchangeability assumptions.
- The *abstain/restrict* decision by itself cannot be claimed new.
- https://arxiv.org/abs/2506.21802 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC12470254/

## C. What we propose — hypothesis, not an established advance
1. **Protected-action-centric question:** Connect *authenticated* device/principal and the actual resource/action to detector evidence; evaluate authorization consequences, not merely an intrusion-label F1 or abstract trust band. Need to verify whether B3/B4 already have this.
2. **Mandatory policy bound:** AI evidence may *restrict* an otherwise permitted operation, never grant beyond explicit rights. Note this is already standard security design; not novelty by itself.
3. **Defined evidence failure handling:** distinguish valid, stale, missing, conflicting, confidently wrong and out-of-distribution evidence. Make retry/step-up/restrict/revoke behaviors explicit; determine whether available methods already provide these.
4. **Measurable tradeoffs:** compare fixed role/attribute policy, normal risk gate, strong published dynamic trust approaches, and *if justified* uncertainty-aware response with the same request ground truth, real enforcement and cost metrics.
5. **Edge feasibility:** measure processing latency, CPU/memory, throughput and legitimate operation disruption on specified hardware. Loopback HTTP alone is not proof of actual edge feasibility.

Possible original result: a **well-tested constrained design and empirical failure/utility analysis** for uncertainty and stale-risk evidence under authenticated IoT access decisions, but its novelty and value remain unverified until full related-work analysis and new experiments.

## D. Initial comparison table

| Criterion | Traditional perimeter/RBAC | Existing dynamic ZTA & edge research | Candidate thesis experiment |
| --- | --- | --- | --- |
| Identity + permissions | static network rules or explicit roles/attributes | explicit identity plus contextual policy | enforce concrete principal/action/resource permissions |
| Traffic ML evidence | often absent from simple baseline | present in many frameworks | model evidence attached to correct authenticated principal |
| Multi-signal score | not always used | **already widely used** (Yang 2026, TrustEdge-AI 2026) | **not new alone**; treat as baseline |
| Uncertainty | not part of RBAC alone | **already explored**, including at IoT edge | empirically compare difficult *confident mistakes* and evidence failure |
| Response | allow/deny; conditional policies possible | dynamic allow/restrict/isolate and other actions | explicitly bounded actions, meaningful enforcement, no privilege expansion |
| Evaluation | policy correctness | typically detection/trust and some simulated access/latency | real protected operations; security-vs-disruption cost; independent shift evaluation |
| Novelty status | established baseline | published prior work | **unconfirmed** |

## E. Our repository evidence to date — keep separate from published claims
- Phase 0: original retrieval KB included test-set data and retrieval used inconsistent feature scale; draft correction PRs #2/#4.
- Phase 0B: clean exploratory internal split did not show stable advantage for multi-signal fusion over ML alone; the public dataset was used previously, so no pristine external-performance claim.
- Phase 1A: on one withheld-Fuzzers configuration, 3,125 of 6,062 attack flows were classified benign; some assigned very low attack probability.
- Phase 1B: confidence margin, IsolationForest and PCA missed most of the classifier's confidently wrong predictions; no validated uncertainty method yet.
- Phase 2: localhost protected HTTP testbed enforces bearer-token identity and resource permissions.
- Phase 3A: server-side risk gate vetoes allowed requests above a risk threshold; risk cannot grant rights; risk provider is a **test double**, not a real authenticated device-attached ML model.
- **UNSW-NB15 attack labels are NOT access-control ground truth.** Never fabricate device/identity mapping from unrelated dataset flow rows.

## F. Remaining concrete work before proposing change to supervisors
- [ ] Read and extract full methods/results from TrustEdge-AI and Yang (2026), with strengths and limits; verify journal/venue/paper quality and availability of implementations.
- [ ] Read at least 5 more closely related peer-reviewed security/IoT authorization works, not just general ML uncertainty.
- [ ] Define threat model, identities, behavior telemetry provenance, and the exact protected access outcomes to improve.
- [ ] Compare proposed method against closest existing papers, not weak or fictional baselines; choose metrics before experiments.
- [ ] Use genuinely independent capture/testbed conditions, estimate false restrictions and overhead, repeat experiments with uncertainty intervals.
- [ ] Confirm novelty and thesis title/scope with supervisors; no formal change yet.

**Do not tell supervisors we already found a research gap.** We have identified candidate failure conditions and substantial prior-work overlap.
