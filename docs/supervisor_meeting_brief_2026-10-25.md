# Supervisor meeting brief — Zero Trust at the Edge (working draft, 2026-10-25)

**Approved thesis title:** Multi-Signal Fusion-Based Scoring for Zero Trust at the Edge

**This is a research discussion, not a formally approved thesis title change.**

## 1. What was the original idea?
Combine a classifier's traffic-risk evidence (M), similarity to historical behavior profiles (R), and trust history (H) into a numerical score supporting edge-network access decisions.

## 2. Why should we reconsider how the research problem is framed?
A new weighted score is not a scientifically meaningful contribution by itself. The original prototype had evaluation leakage, an inconsistent retrieval feature scale, synthetic history demonstrations, and no independently grounded authorization test. Repaired and exploratory results have not demonstrated reliable classification improvements from fusion.

## 3. Specific practical question
A traffic classifier can label previously withheld attack categories as benign, sometimes with extreme apparent confidence. In exploratory category-exclusion tests, 3,125/6,062 Fuzzers attack flows were missed in one configuration. Three simple review signals (confidence margin, IsolationForest and PCA reconstruction) missed most of these errors. **These results are dataset-internal engineering evidence, not generalizable production attack rates.**

## 4. Candidate research problem
**Under unfamiliar or shifting traffic, can a bounded, uncertainty-aware risk-evidence handling strategy reduce *unsafe decisions* in a Zero Trust authorization workflow without disrupting legitimate access excessively?**

Important: An NIDS output is a traffic-class probability, NOT a device permission decision. Any model evidence must be attributed to the correct authenticated principal/request before it can be used as a security signal. The explicit authorization policy remains binding.

## 5. Current working prototypes
- Python training-only retrieval profile fix, retrieval scaling fix, and clean internal split comparative scripts — draft PRs #2, #4 and #5.
- Withheld-category/novelty experiments — draft PRs #6, #7.
- Local HTTP access-control testbed: authenticated principal, method, protected endpoint and policy enforcement, actually rejecting writes — draft PR #8.
- **Experimental** server-side risk veto after permission check; invalid risk fails closed; risk cannot grant unauthorized access — draft PR #9. Test risks are stand-ins, **not yet trained-model evidence tied to real device identities**.

No branch has been merged by this work; GitHub Actions tests check engineering code, not scientific effectiveness.

## 6. Baselines to study
1. Explicit resource policy only.
2. Policy with traditional fixed ML risk gate.
3. Policy with well-specified uncertainty/freshness/recovery mechanism, *only if validated*.

Compare mis-authorized actions; unnecessary blocks of legitimate requests; extra authentication/verification burden; response latency and edge memory/cpu; model failures; robustness on independent devices/captures. For any quality-of-life metric, define legitimate workloads and what failure means.

## 7. What is NOT novel by itself
NIST SP 800-207 already supports contextual signals and policy decisions. IoT Zero Trust, adaptive access control, lightweight edge AI and uncertainty estimation have prior work. Simply combining these concepts cannot be claimed as novelty. New contribution requires a precise research gap, comparable published methods, a validated design, and reproducible better trade-offs.

## 8. Related works to read critically (preliminary starter set)
- Rose et al., **NIST SP 800-207: Zero Trust Architecture (2020)**, https://doi.org/10.6028/NIST.SP.800-207 — policy engine, administrator, enforcement point, and contextual trust algorithms (standard reference).
- UK NCSC, **Zero trust architecture design principles: Use policies to authorise requests**, https://www.ncsc.gov.uk/collection/zero-trust/architecture-design-principles/use-policies-to-authorise-requests — explicit authorization of each request and multiple signals (government guidance).
- Kasa et al., **Adapting Prediction Sets to Distribution Shifts Without Labels (UAI 2025)**, https://proceedings.mlr.press/v286/kasa25a.html — uncertainty/prediction sets under distribution shift; methodological context, *not* directly Zero Trust.
- Stodt et al., **A Context-Aware and Real-Time Dynamic Zero Trust ... (2025)**, https://hal.science/hal-05108783v1 — relevant dynamic policy/ZTA prior work; carefully inspect before gap claims.
- Lotto et al., **Safety-Aware Zero Trust Enforcement for IoT and Cyber-Physical Systems (Sep 2026 arXiv preprint)**, https://arxiv.org/abs/2609.28170 — consequences of restriction/enforcement; **preprint, peer-review status not established**.

No paper above proves our proposed contribution is new. Full related-work quality comparison and independent experiment design still needed.

## 9. Questions to ask supervisors
- Should we stay under the approved multi-signal fusion title and narrow the problem, or seek a formal title amendment?
- What precise system/threat model and independent IoT environment is realistic for master's scope?
- Is unreliable ML evidence under unknown conditions a suitable research problem, and what novelty threshold is expected?
- Which 3–5 comparator papers and baseline implementations should we prioritize?
- Should we focus on safe authorization trade-offs, calibration/uncertainty, or another more defensible topic?

## 10. Plain-English elevator pitch
'Our original project asks how multiple security signals can be combined. But a combined trust score does not guarantee a better access decision. We found examples where an ML detector misclassifies unfamiliar malicious activity and ordinary uncertainty checks miss it. I want to investigate how an edge authorization system should respond to such unreliable evidence without ever bypassing explicit identity permissions. Before settling on a solution, I need help confirming the gap against existing research and selecting a feasible experiment.'
