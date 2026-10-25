# Phase 1B — Are confidently wrong ML predictions detectable?

**Status:** exploratory research diagnostic, 2026-10-25. This is **not** an approved thesis-topic change, an independent deployment benchmark, a validated uncertainty estimator, or an enforced Zero Trust authorization test.

## The question in plain English
The existing classifier misses some unfamiliar attack traffic while assigning low attack likelihood to it. Can simple uncertainty/novelty signals identify these mistakes *without constantly flagging legitimate network traffic*?

## Methods tested
All three use the same training-only preprocessing and a fresh Decision Tree classifier (depth 8, entropy, seed recorded).

1. **Low classifier confidence:** uncertainty = `1 - 2*abs(p_attack - 0.5)`. This is an *uncalibrated probability-margin heuristic*, not a validated uncertainty estimate.
2. **IsolationForest novelty:** fit on a seeded subset (up to 10,000) of known-category training vectors after fitting the imputer and MinMaxScaler on train. Flag larger `-score_samples` values.
3. **PCA reconstruction novelty:** fit ten principal components on that known training subset. Flag larger average squared reconstruction error.

For a fair operating point, each alert threshold is set on **known benign flows in the calibration partition**, targeting at most approximately 5% alert rate there. The threshold is fixed before evaluating a separate set of known-benign flows and all rows from the withheld attack category.

The test asks **how many attacks initially classified benign the extra signal would flag for REVIEW**, not how many would be blocked.

## Data and split
- Source: `UNSW_NB15_training-set.csv`
- SHA-256: `7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c`
- For each withheld attack category, *all* its records are excluded from model fitting and known-condition threshold calibration.
- Remaining known-category records split into 60% fitting, 20% calibration, 20% known evaluation (stratified random splitting, seeds 7,17,42).
- Source has been used previously by this project, and random flow-level splits may share correlated sessions/contexts. **This is a software/research exploratory result, not a pristine independent test.**

## Results: percent of misclassified withheld attacks flagged for review
| Withheld category | Seed | Number of missed attacks | Confidence margin | IsolationForest | PCA reconstruction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fuzzers | 7 | 3,283 | 13.95% | 5.48% | 9.63% |
| Fuzzers | 17 | 2,866 | 7.19% | 8.41% | 10.50% |
| Fuzzers | 42 | 3,182 | 16.91% | 6.54% | 8.83% |
| Reconnaissance | 7 | 1,250 | 18.48% | 0.00% | 0.16% |
| Reconnaissance | 17 | 1,155 | 4.16% | 0.00% | 0.35% |
| Reconnaissance | 42 | 1,288 | 18.79% | 0.00% | 0.16% |

Mean sensitivity over these **three correlated seeds**:
- Fuzzers: confidence 12.68%; IsolationForest 6.81%; PCA 9.65%.
- Reconnaissance: confidence 13.81%; IsolationForest **0.00%**; PCA 0.22%.

For comparison, observed flagged fraction of **held-out known-benign** flows on Fuzzers seed 42:
- Confidence margin: **5.18%**
- IsolationForest: **4.64%**
- PCA: **4.59%**

The benign review rate varies across seeds and is *not guaranteed* to equal 5% outside the calibration partition.

## Why this is important — and what it does NOT establish
- The three simple techniques catch **only a minority of false-benign predictions** in these trials, even while flagging roughly 4–5% of known benign evaluation flows.
- IsolationForest and PCA are not reliable detectors of withheld attack novelty by default: **unseen categories are not necessarily feature-space outliers**.
- The confidence-margin method is unstable by split and can be near zero when the classifier is confidently wrong.
- Review flags do NOT equal access restrictions. Authorization must separately evaluate principal, requested resource/action, verified credentials and mandatory policy.
- This experiment does not establish novel research, safety benefit, actual uncertainty calibration, distribution-free error guarantees, or edge resource feasibility.
- No result here warrants claiming the proposed uncertainty-aware authorization is superior. Next, test more appropriate calibrated/selective methods and/or focus on enforceable-policy behavior with genuine authorization outcomes.

## Reproduce
```bash
python -m pip install numpy pandas scikit-learn joblib
python -m unittest discover -s tests -v
python python_baseline/audit_uncertainty_screening.py --category Fuzzers --category Reconnaissance --seed 7 --seed 17 --seed 42 --output results/audit/phase1b.json
```

The source CSV is expected to be available locally at the repository's documented `data/raw` path. Do not publish this CSV or generated artifacts before checking redistribution rights and public-repository hygiene.

Local development: four new focused synthetic tests passed. Python 3.13.5, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.2.3. CI status should be checked separately.

## Next scientific decision
1. **Do not build policy enforcement on unreliable score thresholds without stronger evidence.**
2. Evaluate calibrated predictive uncertainty/ensembles and error-rejection tradeoffs using development-only tuning, ideally on independently captured devices/sites.
3. In parallel construct a minimal enforceable authorization testbed with principal/action/resource labels, since NIDS predictions cannot establish actual access permission.
4. Review recent literature with Prof. Gebser and Dr. Farahani **after** narrowing the falsifiable, real-world research question.

Related research stack: #1 (audit), PR #2 (train-only KB), PR #4 (scale), PR #5 (independent split), PR #6 (withheld category).
