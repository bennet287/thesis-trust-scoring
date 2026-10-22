# Phase 1A — Withheld attack category challenge (2026-10-22)

**Research status: preliminary diagnostic only.** This experiment evaluates a classifier on an attack category entirely omitted during training, to test whether previously unobserved attack types can trigger confidently wrong predictions. It does NOT implement or validate uncertainty estimation, authorization or a new Zero Trust framework.

## Protocol
- Use `data/raw/UNSW_NB15_training-set.csv`, SHA-256 `7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c`.
- For each category, exclude **all** rows labeled with that attack category from model fitting and known-condition validation.
- Split the remaining rows 80%/20% with stratification, selected seed.
- Train a numeric-only imputer → MinMaxScaler → Decision Tree pipeline using fitting data only (depth 8, entropy, seed recorded).
- Compare attack recall on the known-condition validation partition against recall on the **withheld positive-class category**; threshold is 0.5 for this diagnostic.
- Report fraction of withheld attacks assigned an attack probability <=0.1. This is a descriptive signal of **apparent confidence**, not a calibrated probability or validated uncertainty estimate.
- Do not use `attack_cat` as a feature, do not use the separately supplied official testing-set file, and do not create any fabricated attack sequences.

## Reproduction
```bash
python -m pip install numpy pandas scikit-learn
python -m unittest discover -s tests -v
python python_baseline/audit_unseen_attack_category.py --output results/audit/phase1a_categories.json
```
The CLI's defaults check DoS, Reconnaissance, Fuzzers, Exploits, and Generic under seeds 7, 17, 42, 73, 123.

## One-split observations, seed 42
| Withheld attack category | Withheld records | Attack recall | Missed | Attack probability <= 0.1 |
| --- | ---: | ---: | ---: | ---: |
| DoS | 4,089 | 94.52% | ~224 | 1.10% |
| Reconnaissance | 3,496 | 64.30% | ~1,248 | 1.32% |
| Fuzzers | 6,062 | **48.45%** | **3,125** | **39.90%** |
| Exploits | 11,132 | 84.72% | ~1,701 | 1.73% |
| Generic | 18,871 | 99.59% | ~77 | 0.02% |

Only the exact Fuzzers missed count is independently calculated and recorded here; other approximate missed counts are estimated from rounded rates. The source CLI emits exact integers.

### Fuzzers sensitivity across five seeds
| Seed | Attack recall | Attack probability <=0.1 |
| --- | ---: | ---: |
| 7 | 46.70% | 41.01% |
| 17 | 50.48% | 41.87% |
| 42 | 48.45% | 39.90% |
| 73 | 47.99% | 37.30% |
| 123 | 47.13% | 40.93% |

This held-category exercise reveals a failure case worth studying: the classifier misses many withheld Fuzzers cases and sometimes assigns very low attack likelihood to them.

## Interpretation constraints
- Holding out a known attack category creates a synthetic *novel-category* evaluation challenge, not genuine deployment distribution shift.
- The public dataset was used in earlier project work, so this is exploratory and cannot be presented as a pristine independent test.
- Tree leaf probabilities may be poorly calibrated. A numeric 0.1 score must not be interpreted as a reliable real-world 10% risk estimate.
- Other categories show very different performance. There is **no evidence yet** that one uncertainty estimation method detects these errors, that all unfamiliar threats behave like Fuzzers, or that any authorization scheme improves outcomes.
- Network flow labels do not encode identity, requested operation, device permission or enforcement. Nothing here evaluates real access authorization.
- The five seeds reuse the same source and withheld category; they are sensitivity checks, not independent benchmark replications.

## Next research experiment
Compare calibration/uncertainty measures on known vs category-withheld data without tuning on the withheld challenge. Then create a separate toy authorization testbed with verified principals/resources/actions, explicit policy and a measurable actual enforcement point. Reassess novelty against published work before pitching an approved thesis change.