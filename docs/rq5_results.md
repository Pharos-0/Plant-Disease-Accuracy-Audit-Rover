# RQ5 Results — how much labelling replaces a label-free estimate?

**Experiment `rq5`, 2026-09-09.** Pre-registered in `docs/rq5_plan.md`, committed before
the run (commit `eefb777`). Tables in `results/rq5/`, figures `rq5_fig01`–`rq5_fig05`.

No retraining, no new estimator, no modification to any existing estimator, no historical
table touched.

---

## 1. Executive summary

Under the evaluated PlantVillage → Tomato-Village setting, **labelling about 20 field
photographs produced a more accurate estimate of the classifier's field accuracy than
every one of the eight label-free estimators applied to all 1,616 unlabelled crops.**
Against the six weaker baselines, **5 photographs sufficed**.

That headline is real but it is **not uniform across architectures**, and the
qualification matters: for ResNet-50 and ViT-S/16, COT on the unlabelled pool reaches MAE
0.048 and 0.039 — competitive with 75–100 labelled photographs — and **no crossover was
established for those backbones within the tested budgets**. The pooled result is driven
by the two weaker backbones, where label-free estimation fails badly.

A secondary finding with direct practical consequence: **the confidence interval a
practitioner would naturally quote from a labelled sample is badly wrong here.** Nominal
95% Wilson intervals achieved **63–75% actual coverage**, because crops within a
photograph are correlated and the interval treats them as independent.

**Outcome classification: A (small labelled budget wins), qualified by architecture.**

---

## 2. Research question

> How many labelled field observations are required before a small labelled sample
> provides a more reliable estimate of field accuracy than the evaluated label-free
> estimators?

Arising from the Phase-5 decision gate (Outcome 2): the failure mechanism is real but not
measurable without labels, so the practical question becomes what labelling costs.

---

## 3. Experimental protocol

| | |
|---|---|
| Target arm | D (Tomato-Village, real Indian field), `SHARED_3` |
| Models | 20 — V2 + V9, four backbones × five seeds |
| Estimand | true crop-level accuracy of each frozen model on the **complete** 1,616-crop pool |
| Labelled estimator | ratio estimator: correct crops ÷ total crops among sampled photographs |
| Label-free baselines | the eight already in the thesis, unmodified, on **all 1,616 unlabelled crops** |
| Budgets | 5, 10, 20, 30, 50, 75, 100 photographs (+391 for convergence) |
| Repetitions | 100 per budget |
| Seeds | `20260909 + repetition_index`, shared across models so comparisons are paired |
| Primary unit | the trained model (n = 20), paired |
| Multiplicity | Holm across the 8 baselines within each budget |
| Runtime | **2.1 minutes** |

Both estimators target the *same* quantity. One spends labels; the other spends unlabelled
data. That is what makes the comparison fair.

## 4. Independent-unit definition

**The photograph.** Verified from `src/data/index.py` and the cache, not assumed:

- Arm D = **1,616 crops from 391 photographs**; 0 groups failed the `TV_STEM` parse.
- **0 of 391 photographs span more than one true class**, so photograph-level labelling is
  well defined.
- All 20 models share byte-identical Arm-D ordering, `group_ids` and labels.

**Cluster sizes are bimodal, not uniform** — every healthy photograph yields exactly 1
crop, every blight photograph exactly 8:

| class | photographs | crops | crops/photograph |
|---|---:|---:|---:|
| healthy | 216 | 216 | 1.0 |
| early_blight | 62 | 496 | 8.0 |
| late_blight | 113 | 904 | 8.0 |

Class balance therefore differs sharply by level — photograph 55/16/29, crop 13/31/56 —
which is why the estimator is a **ratio estimator over unequal clusters** and why a fixed
photograph budget yields a *variable* number of labelled crops (Table 2).

> **Recorded so it is never exploited:** crop count perfectly identifies the healthy class
> in this dataset (1 crop ⟺ healthy). That is an artefact of how Tomato-Village was
> cropped, is unavailable to a deployed rover, and **is not used as a feature anywhere.**

## 5. Sampling procedure

Uniformly sample *n* photographs without replacement from the 391; take **all** their
crops, since an annotator labels a photograph once. The same photograph subsets are reused
across all 20 models and all estimators, so comparisons are paired.

## 6–8. Budgets, baselines, metrics

Budgets as above. Baselines: `confidence`, `entropy`, `atc_mc`, `atc_ne`, `doc`, `cot`,
`nuclear_norm`, `mano` — estimator suite version 1.0.0, unmodified. Primary metric:
absolute error against the true pool accuracy (MAE, median). Pre-specified tail metric:
**95th percentile absolute error**. One tail metric only.

## 9. Statistical procedure

Paired at the model level (n = 20): per model, labelled error is the mean over 100
repetitions; label-free error is a single value. Differences summarised by a percentile
bootstrap (2,000 resamples) plus a Wilcoxon signed-rank test, Holm-corrected across the 8
baselines within each budget.

**Pre-registered crossover:** the smallest budget where the mean paired difference is
negative *and* the 95% bootstrap CI excludes zero.

## 10. Leakage checks

- The estimator is given only the sampled photographs' labels. A **poisoning test** flips
  every *unsampled* label and asserts the estimate is unchanged — it passes.
- Label-free estimators receive logits and features only; no label reaches them.
- The AST firewall test and its negative control still behave correctly.
- Full labels are used solely as post-hoc evaluation truth.

## 11. Reproducibility

`results/rq5/manifest.json` records experiment id, git commit and dirty state, Python and
PyTorch versions, backbones, seeds and the seed convention, arm, label set, sampling unit,
photograph count, budgets, repetitions, estimator suite version, metric and CI definitions,
input cache fingerprint, output paths and runtime.

---

## 12. Main results

### Table 2 — MAE by labelled budget (label-free baselines are constant; they use 0 labels)

| photographs | mean crops | **labelled** | entropy | cot | nuclear_norm | atc_ne | confidence | atc_mc | doc | mano |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 20.7 | 0.2000 | 0.1337 | 0.1339 | 0.3089 | 0.3109 | 0.3214 | 0.3268 | 0.3407 | 0.4165 |
| 10 | 41.6 | 0.1255 | " | " | " | " | " | " | " | " |
| **20** | **81.3** | **0.0795** | " | " | " | " | " | " | " | " |
| 30 | 125.9 | 0.0670 | " | " | " | " | " | " | " | " |
| 50 | 206.3 | 0.0513 | " | " | " | " | " | " | " | " |
| 75 | 311.0 | 0.0384 | " | " | " | " | " | " | " | " |
| 100 | 410.9 | 0.0328 | " | " | " | " | " | " | " | " |

At 5 photographs the labelled estimate (0.200) is **significantly worse** than entropy and
COT (paired difference +0.066, CI [+0.016, +0.112]). At 10 it is a tie. At 20 it wins.

### Table 4 — tail risk (95th percentile absolute error)

| photographs | 5 | 10 | 20 | 30 | 50 | 75 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 95th pct error | 0.499 | 0.305 | 0.190 | 0.174 | 0.136 | 0.101 | 0.079 |

Tail risk matters more than the mean for a deployment gate. At 5 photographs the worst 5%
of samples are wrong by ~0.50 — a coin flip on a three-class problem. Only from ~50
photographs does the tail fall below the *mean* error of the best label-free estimator.

## 13. Crossover analysis

### Table 3 — crossover budget by baseline

| baseline | MAE (0 labels) | crossover (pre-registered CI criterion) | additionally Holm-significant from |
|---|---:|---:|---:|
| mano | 0.4165 | **5** | 5 |
| doc | 0.3407 | **5** | 5 |
| atc_mc | 0.3268 | **5** | 5 |
| confidence | 0.3214 | **5** | 5 |
| atc_ne | 0.3109 | **5** | 5 |
| nuclear_norm | 0.3089 | **5** | 5 |
| entropy | 0.1337 | **20** | 30 |
| cot | 0.1339 | **20** | 50 |

**Headline: 20 labelled photographs (≈ 81 crops) beat all eight.** Five suffice against the
six weaker methods.

**A tension reported rather than smoothed:** against entropy and COT at N = 20, the
bootstrap CI excludes zero (the pre-registered criterion is met) but the Holm-corrected
Wilcoxon p is 0.21. The two disagree because one tests the mean and the other ranks. The
conservative reading is therefore **20 photographs by the pre-registered criterion,
30–50 if one additionally demands Holm-corrected significance**. Both are stated; neither
is hidden.

## 14. Per-backbone analysis

| backbone | best label-free (0 labels) | its MAE | labelled MAE @20 | @100 | crossover vs that best |
|---|---|---:|---:|---:|---|
| mobilenetv3_large | cot | 0.2016 | 0.0729 | 0.0298 | **10** |
| efficientnet_b0 | entropy | 0.1115 | 0.0681 | 0.0289 | **20** |
| resnet50 | cot | **0.0475** | 0.0845 | 0.0367 | **none within budgets** |
| vit_small_patch16 | cot | **0.0388** | 0.0927 | 0.0359 | **none within budgets** |

**This is the most important qualification in RQ5.** On the two stronger backbones COT
estimates field accuracy to within ~0.04 using **no labels at all** — roughly what 75–100
labelled photographs achieve — and no crossover was established within the tested range.
The per-backbone tests use n = 5 seeds and are **underpowered**; absence of a detected
crossover is not evidence of absence.

The pooled "20 photographs" result is therefore driven by MobileNetV3-L and
EfficientNet-B0, the backbones where label-free estimation fails worst. This is consistent
with the Phase-5 finding that COT is strongly architecture-dependent.

## 15. Capture-day analysis (Setting B)

Labels from one day only; estimand is that day's accuracy.

| day | photographs | true accuracy | best label-free | its MAE | labelled MAE @10 | @50 | crossover |
|---|---:|---:|---|---:|---:|---:|---:|
| 2022-03-23 | 88 | 0.592 | cot | 0.1264 | 0.0878 | 0.0267 | **10** |
| 2022-03-24 | 154 | 0.484 | cot | 0.1425 | 0.1217 | 0.0454 | **10** |
| 2022-03-25 | 149 | 0.378 | entropy | 0.1927 | 0.1401 | 0.0554 | **10** |

**About ten labelled photographs beat the best label-free estimator on every day.**

The pattern is directional and consistent: **label-free error grows as the day gets harder**
(best 0.126 → 0.143 → 0.193; MaNo 0.326 → 0.433 → 0.535), while the labelled estimator's
error does not track difficulty in the same way. This is the operational restatement of the
Phase-5 capture-day finding — the label-free estimate degrades exactly when the session is
worst, which is when a deployment gate most needs to be right.

Caveat: budgets of 50 on the 88-photograph day consume 57% of that day, so the largest
day-level budgets are not independent of the estimand. The N = 10 crossover, at 11% of the
smallest day, is not affected.

## 16. Sanity checks

| # | check | result |
|---|---|---|
| 1 | no duplicate photograph within a sample | **PASS** (asserted over 25 seeds) |
| 2 | whole photographs only, never split | **PASS** |
| 3 | crop-level duplication creates no leakage | **PASS** — photographs are atomic |
| 4 | sampling reproducible from seed | **PASS** |
| 5 | different seed changes the sample, not the pipeline | **PASS** |
| 6 | **large-N convergence** | **PASS** — max abs error at N=391 is **0.00e+00** |
| 7 | error decreases with budget | **PASS** — 0.200 → 0.033 monotonically |
| 8 | **baseline reproduction** | **PASS** — worst abs diff **1.11e-16** over 160 cells |
| 9 | no target label reaches a label-free estimator | **PASS** |
| 10 | firewall negative control still fails correctly | **PASS** |
| 11 | manifest identifies the run | **PASS** |
| 12 | no historical result file modified | **PASS** — outputs confined to `results/rq5/` |

### Wilson interval coverage — a practical warning

| photographs | 5 | 10 | 20 | 30 | 50 | 75 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| actual coverage of a nominal 95% interval | 62.5% | 66.3% | 71.9% | 70.7% | 71.3% | 75.2% | 75.0% |

Pre-registered as expected to be anti-conservative, and it is — badly. Crops within a
photograph are correlated; the interval treats them as independent, so it is far too
narrow. **A practitioner quoting a naive binomial interval on clustered field crops would
be substantially overconfident.** A cluster-aware interval is needed; specifying one is
future work, not something to retrofit here.

## 17. Limitations

1. **One source and one target field dataset**, 391 photographs, three classes, four
   backbones. No claim generalises beyond this setting.
2. **Per-backbone comparisons use n = 5 seeds** and are underpowered; "no crossover" for
   ResNet-50 and ViT-S/16 means *not established*, not *absent*.
3. **The pooled crossover is driven by the weaker backbones.** Reporting only the pooled
   number would overstate the case for labelling.
4. **Uniform sampling only.** Stratified or active labelling would very likely do better
   and is not tested.
5. **Cluster structure is peculiar to this dataset** (1 vs 8 crops, perfectly aligned with
   healthy vs blight). Budgets in *photographs* transfer more safely than budgets in crops.
6. The largest day-level budgets consume a large fraction of the smallest day.
7. **Crossover depends on the criterion**: 20 photographs by the pre-registered CI rule,
   30–50 if Holm-corrected significance is also demanded.

## 18. Practical interpretation

For this lab-to-field setting, the operational answer is:

> **Labelling roughly 20–30 field photographs gives a more trustworthy estimate of a
> classifier's field accuracy than any of the eight evaluated label-free estimators
> applied to 1,616 unlabelled crops** — unless the model happens to be one for which COT
> works well, which cannot be determined without labels.

That last clause is the sting, and it should be stated in the thesis. **Deciding whether
you are in the regime where label-free estimation works requires exactly the labels you
were trying to avoid.** A practitioner cannot know in advance whether their backbone is a
ResNet-like case (COT MAE 0.04) or a MobileNet-like case (COT MAE 0.20).

Twenty photographs is a modest cost — an afternoon's annotation — against a 46-point
accuracy collapse that no label-free estimator in this study reliably detected.

## 19. Thesis-ready conclusions

1. Under the evaluated setting, a labelled sample of **≈20 independent field photographs**
   outperformed all eight evaluated label-free estimators on absolute estimation error,
   by the pre-registered criterion.
2. Against the six weaker estimators, **5 photographs** sufficed.
3. **Architecture matters:** for two of four backbones the best label-free estimator was
   competitive with 75–100 labelled photographs, and no crossover was established.
4. In a **per-day holdout**, ~10 labelled photographs beat the best label-free estimator on
   each of three real acquisition days, and label-free error grew as the day got harder.
5. **Naive binomial intervals on clustered field samples are badly anti-conservative**
   (63–75% coverage at nominal 95%).

## 20. What RQ5 does **not** establish

- Any universal labelling budget for agricultural deployment.
- That labelling is preferable in general — only how the two error magnitudes compare here.
- Anything about stratified, active or adaptive labelling.
- Anything per-image; every quantity is session- or pool-level.
- That label-free estimation is useless — on two backbones it was competitive with a
  substantial labelling effort.
- Any causal claim. This is an observational comparison over 20 fixed models.

## 21. Recommended next step

**Do not develop a new estimator.** RQ5 did not produce a deployable failure signal and
Phase-5's Outcome 2 stands.

The most valuable next work, in order:

1. **A cluster-aware confidence interval** for the labelled estimator. The naive interval
   is demonstrably wrong here, and a practitioner needs an honest one. This is a small,
   well-posed statistical task, not method invention.
2. **Stratified or class-balanced labelling**, which the bimodal cluster structure suggests
   could beat uniform sampling at the same budget.
3. Then the outstanding items: Arm E, a second source dataset, and the database literature
   search.
