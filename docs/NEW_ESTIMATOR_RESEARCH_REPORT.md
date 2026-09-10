# Research Report — Auditing Target Informativeness

**Executed 2026-09-10.** Every number is traceable to a named output file. Nothing is
predicted, estimated, or filled in.

---

## Table A — Target vs anchor sensitivity, and MAE

Nine estimators, 20 models (4 backbones × 5 seeds), full 1,616-crop Arm-D pool,
100 cluster bootstraps per side. Source: `results/ti_audit/ti_summary.csv`.

| estimator | target sd | anchor sd | anchor/target | **TI** | **MAE** |
|---|---:|---:|---:|---:|---:|
| confidence | 0.0061 | 0.0000 | 0.0 | **1.0000**\* | 0.3214 |
| entropy | 0.0103 | 0.0000 | 0.0 | **1.0000**\* | 0.1337 |
| doc | 0.0060 | 0.0081 | 1.3 | 0.4567 | 0.3407 |
| atc_ne | 0.0128 | 0.1271 | 10.0 | 0.0731 | 0.3109 |
| atc_mc | 0.0118 | 0.0638 | 5.4 | 0.0624 | 0.3268 |
| cot | 0.0077 | 0.0326 | 4.2 | 0.0587 | 0.1339 |
| nuclear_norm | 0.0048 | 0.0589 | 12.4 | 0.0084 | 0.3089 |
| gdscore | 0.0012 | 0.0163 | 13.7 | 0.0068 | 0.4543† |
| mano | 0.0020 | 0.0290 | 14.2 | 0.0066 | 0.4165 |

\* `confidence` and `entropy` have `needs_source = False` — no anchor exists, so TI = 1 is
**definitional**, not measured. † GdScore in **eval** mode, for consistency with the other
eight; its official train-mode MAE is 0.1599 (`results/rq7_gdscore/`).

**For six of nine estimators the estimate moves 4–14× more when the source calibration is
resampled than when the target is resampled.**

## Table B — Target-information removal

Mean |estimate change| when target information is destroyed, source held fixed.
Source: `results/ti_audit/ti_removal_summary.csv`.

| estimator | shuffled | moment-matched noise | duplicated | **random logits** |
|---|---:|---:|---:|---:|
| cot | 0.0000 | 0.0701 | 0.3041 | **0.0618** |
| mano | 0.0000 | 0.0231 | 0.0372 | **0.0651** |
| gdscore | 0.0015 | 0.0068 | 0.0263 | **0.0729** |
| nuclear_norm | 0.0000 | 0.0526 | 0.2384 | **0.1061** |
| confidence | 0.0000 | 0.0640 | 0.1258 | 0.2248 |
| doc | 0.0000 | 0.0637 | 0.1242 | 0.2250 |
| entropy | 0.0000 | 0.1162 | 0.1869 | 0.3810 |
| atc_mc | 0.0000 | 0.0911 | 0.2317 | 0.4448 |
| atc_ne | 0.0000 | 0.0956 | 0.2721 | 0.5560 |

`shuffled ≈ 0` for every permutation-invariant estimator confirms the machinery; GdScore's
0.0015 correctly exposes its fixed-batch dependence.

> **COT — joint-best on MAE — changes by only 0.062 when the entire target is replaced by
> random logits.** MaNo 0.065, GdScore 0.073.

## Table C — Synthetic validation against a known answer

`MixtureProbe(α) = α·mean_conf(T) + (1−α)·source_acc`, 5 reps.
Source: `results/ti_synthetic/v1_known_alpha.csv`.

| α | 0.00 | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 | 1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TI** | 0.0000 | 0.0010 | 0.0094 | 0.0781 | 0.4270 | 0.8671 | 1.0000 |

Spearman(α, TI) = **1.000** (p < 0.0001). TI(0) = 0, TI(1) = 1. **TI measures what it
claims.** Note it is a *variance* share — TI(0.5) = 0.078, not 0.5.

## Table D — Candidate Family F, benchmarked

`w·base(T) + (1−w)·source_accuracy`, `w` = measured TI. 20 models.
Source: `results/ti_candidate/candidate_summary.csv`, `candidate_stats.csv`.

| base | base MAE | **candidate** | fixed w=0.5 | anchor only | Δ | Wilcoxon p | Holm |
|---|---:|---:|---:|---:|---:|---:|---|
| entropy | 0.1337 | 0.1337 | 0.2758 | 0.4629 | 0.0000 | — | — |
| cot | 0.1339 | **0.4438** | 0.2984 | 0.4629 | **+0.3099** | <0.0001 | **worse** |
| nuclear_norm | 0.3089 | 0.4613 | 0.3859 | 0.4629 | +0.1524 | <0.0001 | **worse** |
| atc_ne | 0.3109 | 0.4486 | 0.3869 | 0.4629 | +0.1377 | <0.0001 | **worse** |
| confidence | 0.3214 | 0.3214 | 0.3921 | 0.4629 | 0.0000 | — | — |
| atc_mc | 0.3268 | 0.4552 | 0.3948 | 0.4629 | +0.1284 | <0.0001 | **worse** |
| doc | 0.3407 | 0.4025 | 0.4018 | 0.4629 | +0.0618 | <0.0001 | **worse** |
| mano | 0.4165 | 0.4625 | 0.4397 | 0.4629 | +0.0460 | <0.0001 | **worse** |
| gdscore | 0.4543 | 0.4628 | 0.4586 | 0.4629 | +0.0085 | 0.0083 | **worse** |

**Bases improved: 0 of 9. Bases significantly harmed: 7 of 9**, all surviving Holm
correction. The two unchanged are those with TI = 1, where `w = 1` and the candidate
reduces to the base by construction.

**Why:** `anchor_only` MAE is **0.4629** — the source anchor is a very poor predictor here
(source accuracy 0.963, target accuracy 0.500). Shrinking toward it can only hurt, and the
candidate shrinks hardest exactly when TI is low.

## The twelve questions

**Q1. Can target informativeness be measured rigorously?**
**Yes.** TI recovers a known α with Spearman 1.000 and correct endpoints (Table C). Two
caveats: it is a *variance* share, and it depends on how precisely the anchor is
determined, which is a property of the source sample. **VERIFIED.**

**Q2. Are existing estimators actually target-informed?**
**Mostly not, in this setting.** Six of nine move 4–14× more under source resampling than
target resampling (Table A). **VERIFIED.**

**Q3. Which estimators are dominated by source anchors?**
mano (14.2×), gdscore (13.7×), nuclear_norm (12.4×), atc_ne (10.0×), atc_mc (5.4×),
cot (4.2×). doc is balanced (1.3×); confidence and entropy have no anchor. **VERIFIED.**

**Q4. Does target informativeness predict estimator reliability?**
**No.** Real data: Spearman(TI, MAE) = **−0.469, p = 0.203** across nine; **−0.321,
p = 0.482** across the seven with a real anchor. Within-estimator, only `doc` survives Holm
(−0.624, p = 0.0033). Synthetic: **+0.886, p = 0.003** — the **opposite sign**.
**VERIFIED. This is the decisive negative result.**

**The contradiction is resolved, not ignored.** What determines whether anchor reliance
helps is the **anchor–truth gap**:

| world | source acc | target acc | gap | "predict the anchor" MAE | Spearman(TI, MAE) |
|---|---:|---:|---:|---:|---:|
| agricultural (real) | 0.963 | 0.500 | **0.463** | 0.463 | −0.469 (n.s.) |
| synthetic | ~0.72 | 0.692 | **0.028** | 0.051 | +0.886 (p=0.003) |

Under mild shift the anchor is nearly correct, so anchor-dominated estimators win and high
TI looks *bad*. Under severe shift the anchor is badly wrong, so they lose and high TI
looks *good*. **TI measures how much an estimator leans on the anchor; it cannot tell you
whether leaning will help, because that depends on a gap you cannot see without target
labels.**

**Q5. Can target-information auditing improve accuracy estimation?**
**No evidence that it can.** The one implemented candidate made 7 of 9 bases significantly
worse and improved none (Table D). **VERIFIED.**

**Q6. Can a genuinely new estimator be constructed from this observation?**
**Not on this analysis.** All six candidate families reduce to empirical-Bayes shrinkage or
ensemble weighting (novelty audit §5), and shrinkage responds to weak target information by
leaning *harder* on the anchor — formalising the pathology rather than repairing it. The
tested family behaved exactly that way. **DERIVATION + VERIFIED.**

**Q7. Does it outperform existing estimators?**
**No.** Best overall remains `entropy` at MAE 0.1337 and `cot` at 0.1339, both existing
baselines. Best candidate variant: 0.1337, achieved only by reducing to entropy.
**VERIFIED.**

**Q8. Generalise across architectures?** Not applicable — no candidate survived. The audit
itself was run across all four backbones; per-architecture TI is in
`results/ti_audit/ti_per_model.csv`. n = 5 seeds per backbone is **underpowered**.

**Q9. Generalise across target sample sizes?** **NOT RUN** on real data. Synthetic
N = 100 vs 400 showed MAE roughly halving for most estimators
(`results/ti_synthetic/v2_shift_and_size.csv`).

**Q10. Generalise across shift severity?** Synthetic sweep run (shift 0.0–1.0). The
decisive finding is the **sign reversal** in Q4 between the mild-shift synthetic world and
the severe-shift real world.

**Q11. Is the improvement statistically credible?** There is **no improvement** to test.
The *harm* is credible: 7 of 9 significant under paired Wilcoxon with Holm correction,
n = 20 per comparison.

**Q12. Is the method genuinely different from existing literature?**
**The metric is not.** TI is a first-order Sobol' index over two input groups —
established mathematics. **The application** — auditing label-free accuracy estimators
this way — returned **no matching prior work** in the sources searched
(`abs:"sensitivity analysis" AND abs:"accuracy estimation" AND abs:"distribution shift"` →
0 on arXiv). That supports *"no matching work identified"*, never *"first"*.

## The result that matters most

Two estimators, essentially identical MAE, entirely different mechanisms:

| | TI | MAE | change under random target logits |
|---|---:|---:|---:|
| **entropy** | 1.0000 | **0.1337** | 0.3810 |
| **cot** | 0.0587 | **0.1339** | **0.0618** |

MAE differs by 0.0002. TI differs by a factor of **17**. COT achieves its score while
barely reacting to the target being replaced by noise.

> **MAE on a 20-model population whose true accuracies have sd 0.075 cannot distinguish a
> target-informed estimator from an anchor that happens to land near the mean.** That is a
> benchmark-validity finding, and it is what this work actually established.
