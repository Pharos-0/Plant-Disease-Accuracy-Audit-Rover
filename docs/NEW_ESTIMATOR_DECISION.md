# New Estimator — Final Decision

**2026-09-10.** Evidence: `docs/NEW_ESTIMATOR_RESEARCH_REPORT.md`,
`docs/NEW_ESTIMATOR_NOVELTY_AUDIT.md`, `docs/NEW_ESTIMATOR_FEASIBILITY.md`.

---

## 1. Executive Decision

### **DO NOT PROCEED** with a new estimator. **CONDITIONAL PROCEED** with the audit as a diagnostic contribution.

No candidate improved on the existing baselines; the strongest one made 7 of 9 base
estimators significantly worse. The measurement framework is sound and produced a real
finding, but the metric is established mathematics and does not predict reliability.

## 2. Candidate Methods Investigated

| family | description | outcome |
|---|---|---|
| A | target-information-weighted calibration | rejected on derivation — empirical-Bayes shrinkage |
| B | anchor/target decomposition + reliability rule | rejected — identical to A once written down |
| C | target-sensitivity-calibrated estimator | rejected — shrinkage with a non-standard weight |
| D | robust target correction with source fallback | rejected — shrinkage with a fallback |
| E | reliability-weighted combination | rejected — ensemble weighting, excluded by the brief |
| **F** | **TI-adaptive shrinkage** | **implemented, benchmarked, ablated — empirically rejected** |

Family F was implemented rather than dismissed on paper, because rejecting the strongest
candidate by argument alone would have been weaker evidence.

## 3. Mathematical Novelty Assessment

| item | verdict |
|---|---|
| TI as a metric | **MATHEMATICALLY EQUIVALENT** to a first-order Sobol' index over two input groups |
| target-removal check | **EQUIVALENT** to group-level permutation importance (Breiman 2001) |
| Families A–E | **DERIVATIVE** |
| Family F | **MATHEMATICALLY EQUIVALENT** to James–Stein / empirical-Bayes shrinkage |
| TI applied to label-free accuracy estimators | **POSSIBLY NOVEL — NEEDS HUMAN REVIEW** |

## 4. Existing Methods Most Closely Related

- **Sobol' (1993); functional ANOVA** — the variance decomposition itself.
- ***Look at the Variance!*** (2021) — Sobol' indices applied to neural networks, for
  explaining *predictions* rather than auditing *estimators*.
- **Breiman (2001)** — permutation importance.
- **James–Stein / empirical Bayes** — the shrinkage candidate.
- **Distance-Matters** (ICCV-W 2023) — nearest work in the accuracy-estimation literature;
  read in full, diagnoses *which samples* to distrust, not how much of the estimate comes
  from the target.

## 5. Empirical Results

TI, and MAE, over 20 models on the full 1,616-crop pool
(`results/ti_audit/ti_summary.csv`):

| estimator | TI | MAE |
|---|---:|---:|
| confidence | 1.0000\* | 0.3214 |
| entropy | 1.0000\* | 0.1337 |
| doc | 0.4567 | 0.3407 |
| atc_ne | 0.0731 | 0.3109 |
| atc_mc | 0.0624 | 0.3268 |
| cot | 0.0587 | 0.1339 |
| nuclear_norm | 0.0084 | 0.3089 |
| gdscore | 0.0068 | 0.4543 |
| mano | 0.0066 | 0.4165 |

\* no anchor exists — definitional, not measured.

## 6. Best Baseline

**`entropy`, MAE 0.1337**, closely followed by **`cot`, 0.1339**. Both are pre-existing.

## 7. Proposed Method

**None.** Family F: `w·base(T) + (1−w)·source_accuracy`, `w` = measured TI. Bases improved
**0 of 9**; bases significantly harmed **7 of 9**. Best candidate MAE 0.1337, reached only
where it degenerates to entropy.

## 8. Statistical Evidence

- Candidate vs base, paired Wilcoxon, n = 20 per comparison, Holm-corrected across 9:
  **7 significantly worse, 0 better.** Largest harm `cot` +0.3099 (95% CI 0.276–0.342).
- Q4, real: Spearman(TI, MAE) = **−0.469, p = 0.203** (n = 9); **−0.321, p = 0.482**
  excluding the two anchor-free estimators.
- Q4, synthetic: **+0.886, p = 0.003** — **opposite sign.**
- Within-estimator TI vs error, Holm over 7: only `doc` survives (−0.624, p = 0.0033).

## 9. Generalization Evidence

The sign of TI's association with error **reverses** between a mild-shift synthetic world
(anchor–truth gap 0.028) and the real agricultural setting (gap 0.463). TI therefore does
not generalise as a reliability signal, and the gap that governs it is unobservable without
target labels — which is the whole difficulty.

## 10. Failure Modes

1. **Shrinkage formalises the pathology.** It leans harder on the anchor exactly when
   target information is weak.
2. **The anchor is very wrong here.** `anchor_only` MAE = 0.4629.
3. **TI is a variance share, not a magnitude share** — TI(α=0.5) = 0.078. It must not be
   quoted as "x% of the output comes from the target", and it is **not** the same quantity
   as the Phase-7b 93–98% intercept share.
4. **TI depends on anchor precision.** A degenerate synthetic world (source accuracy 1.000,
   bootstrap sd 0.0000) drove TI to 1.0 for every probe until the world was fixed.
5. **TI = 1 is definitional** for estimators with no anchor.

## 11. Leakage / Validity Checks

- Audit functions accept **no argument** that could carry a target label — enforced by test.
- `src/estimators/ti_audit.py` passes the repository AST import firewall (which **rejected**
  it initially and forced a relative import).
- Candidate's anchor is source-validation accuracy — source labels only, permitted.
- Target labels used **only** to score already-written estimates.
- `w` is measured, never tuned; ablations fix it at 0.5 and 0.
- Cluster bootstrap at photograph level on target and leaf level on source.

## 12. Reproducibility Status

```bash
python scripts/ti_synthetic.py     # Phase 5
python scripts/ti_audit.py --n-boot 100   # Phases 6-8
python scripts/ti_candidate.py     # Phases 9-13
```

Manifests in `results/ti_*/manifest*.json` record git commit, seeds, package versions and
cache fingerprints. **Test suite: 126 passing**, executed 2026-09-10. No historical result
was read for writing or modified.

## 13. Thesis Contribution Potential

**Not a new estimator.** What the evidence supports:

> **A validity audit for label-free accuracy-estimation benchmarks.** Two estimators can
> post indistinguishable MAE (entropy 0.1337, COT 0.1339) while one reacts to the target
> and the other barely does — COT moves 0.062 when the entire target is replaced by random
> logits. On a model population whose true accuracies have sd 0.075, **MAE cannot separate
> a target-informed estimator from an anchor that lands near the mean.**

This strengthens the existing thesis narrative rather than replacing it, and it sits
alongside the Phase-7b calibration-anchor finding as a *methodological* contribution.

## 14. Remaining Risks

| risk | severity |
|---|---|
| TI is a Sobol' index — any "new metric" claim will be refuted immediately | **high** if claimed; nil if stated correctly |
| Sobol-equivalence check was arXiv-only; Scopus/WoS/IEEE/ACM unsearched | medium |
| One source dataset, one target, three classes, 20 models, n = 5 seeds | **high** for generality |
| No real-data N-sweep for TI | medium |
| Families A–E rejected by derivation, not experiment | medium — stated openly |

## 15. Recommended Next Step

1. **Do not build an estimator.** Do not implement A–E.
2. **Keep the audit** as a diagnostic, reported with its equivalence to Sobol' indices
   stated plainly, and with the Q4 negative result kept prominent.
3. **Report the entropy/COT pair** as the headline — it is the sharpest evidence in the
   whole project that this benchmark's primary metric is insufficient on its own.
4. If breadth is ever needed, the cheapest additions remain **Distance-Matters** and
   **Agreement-on-the-Line** (see `docs/estimator_selection.md`), not a new method.
