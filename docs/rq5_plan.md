# RQ5 Pre-Registration

**Committed before the experiment is run.** Fixed in advance so the analysis is
confirmatory. Nothing below is adjusted after seeing results; deviations, if any, are
recorded as deviations in `docs/rq5_results.md`.

---

## A. Research question

> How many labelled field observations are required before a small labelled sample
> provides a more reliable estimate of field accuracy than the evaluated label-free
> estimators?

This is a practical-guidance question, arising from the Phase-5 decision gate
(**Outcome 2**: the failure mechanism is real but not measurable without labels).
**No new estimator is developed.** The labelled estimator is deliberately the most
transparent one available — sample accuracy.

---

## B. Experimental units — verified, not assumed

Verified against `src/data/index.py` and the Arm-D cache before writing this plan:

- `group_id` for Tomato-Village is `ts:IMG<YYYYMMDD><HHMMSS>`, i.e. **the source
  photograph**, assigned by `TV_STEM` in `src/data/index.py`.
- Arm D `SHARED_3`: **1,616 crops from 391 photographs**. 0 groups failed to parse.
- **0 of 391 photographs contain crops of more than one true class**, so a photograph
  has a single label and photograph-level labelling is well defined.
- All 20 models (V2 + V9, `SHARED_3`) share **identical** Arm-D ordering, `group_ids`
  and labels — verified by array equality — so comparisons can be paired across models.

### B.1 A structural property that materially shapes this experiment

The crops-per-photograph distribution is **bimodal, not uniform**:

| class | photographs | crops | crops per photograph |
|---|---:|---:|---:|
| healthy | 216 | 216 | **1.0** |
| early_blight | 62 | 496 | **8.0** |
| late_blight | 113 | 904 | **8.0** |

Every healthy photograph yields exactly one crop; every blight photograph yields exactly
eight. Consequently the class balance differs sharply by level:

- **photograph level:** healthy 55.2%, early blight 15.9%, late blight 28.9%
- **crop level:** healthy 13.4%, early blight 30.7%, late blight 55.9%

Two consequences, both pre-registered:

1. A uniform sample of photographs is a **cluster sample with unequal cluster sizes**.
   The estimator is therefore a **ratio estimator** (§F), which is consistent but has
   higher variance than a simple binomial sample of the same crop count.
2. **Crop count perfectly identifies the healthy class in this dataset** (1 crop ⟺
   healthy). This is a dataset artefact of how Tomato-Village was cropped. It is
   recorded here so that it is never used as a signal, and it is not available to a
   deployed rover. **No analysis in RQ5 may use crop count as a feature.**

---

## C. Labelled sample budgets

**N_label ∈ {5, 10, 20, 30, 50, 75, 100} photographs**, plus **391** (the full pool) as
the convergence check required by §28 of the brief.

All are feasible: the largest tested budget (100) is 25.6% of the 391 independent
photographs. No budget is removed.

Budgets are stated in **photographs**, the independent unit. The corresponding number of
labelled *crops* is random (between 1 and 8 per photograph) and will be reported
alongside, since it is the quantity a human annotator actually inspects.

---

## D. Sampling protocol

For each repetition *r* and budget *n*:

1. draw *n* photographs **uniformly without replacement** from the 391;
2. take **all crops** belonging to those photographs — a photograph is labelled once and
   all its crops inherit that label, which is what a human annotator would actually do;
3. no photograph may appear twice in a sample (asserted in tests);
4. the same photograph subsets are **reused across all 20 models and all estimators**, so
   comparisons are paired rather than needlessly noisy.

Sampling is at photograph level throughout. Crops are never sampled independently.

---

## E. Evaluation protocol

### Setting A — random field sample (primary)

**Estimand:** the true crop-level accuracy of a given frozen model on the **complete
Arm-D pool** (1,616 crops). One value per model; 20 models.

- **Labelled estimator** sees only the sampled photographs' labels.
- **Label-free estimators** see the **entire unlabelled pool** (all 1,616 crops) and no
  labels, fitted on source validation exactly as in the existing pipeline.
- Full labels are used **only** as post-hoc evaluation truth.

Both therefore estimate the *same* quantity, which is what makes the comparison fair: one
spends labels, the other spends unlabelled data.

### Setting B — capture-day holdout (secondary)

Per-day photograph counts are 88 / 154 / 149, so a day-level analysis is feasible with
reduced budgets. **Estimand:** that day's crop-level accuracy. Labels from one day are
never used to estimate another. Budgets are capped at **N_label ∈ {5, 10, 20, 30, 50}**
so that no budget exceeds ~57% of the smallest day (88 photographs). If any day-level
result rests on a sample exceeding half that day's photographs, it is reported with that
caveat attached.

---

## F. Estimands and estimators

**Labelled estimator — ratio estimator over sampled clusters:**

```
            correct crops among the sampled photographs
acc_hat  =  -------------------------------------------
             total crops among the sampled photographs
```

Deliberately simple and transparent (§6 of the brief). No learned calibration, no
weighting, no correction.

**Label-free baselines:** the eight already in the thesis — `confidence`, `entropy`,
`atc_mc`, `atc_ne`, `doc`, `cot`, `nuclear_norm`, `mano` — used **exactly as implemented**
(`src/estimators/methods.py`, suite version 1.0.0). None is modified, retuned or added to.

---

## G. Baselines and reproduction requirement

Before any comparison, the existing Arm-D session-level results must be reproduced from
the same cached inputs (brief §29). **Tolerance: 1e-6 on the reproduced MAE.** If
reproduction fails, the experiment stops and the discrepancy is reported.

---

## H. Repetitions

**R = 100** per (budget, setting). Chosen over 200 deliberately: Phase 5 cost 5.3 hours,
and RQ5 is designed to stay cheap. 100 repetitions × 8 budgets × 20 models is adequate
for the paired comparison specified in §K.

---

## I. Random seeds

```
seed = 20260909 + repetition_index          # repetition_index = 0 … R-1
```

The same seed sequence is used for every budget, model and estimator, so the photograph
subsets are shared and the comparison is paired. Seeds are recorded in the manifest.

---

## J. Confidence interval method

Two distinct quantities, never conflated:

1. **Uncertainty available at deployment** — a **Wilson score interval** on the labelled
   sample, computed on the number of labelled *crops*. Chosen in advance.
   **It ignores within-photograph clustering and is therefore expected to be
   anti-conservative**; its **empirical coverage of the true pool accuracy is measured
   and reported** rather than assumed. This is the honest way to present it.
2. **Actual estimation error**, measured post-hoc against the full labels.

For aggregate metrics (MAE, tail error) intervals come from a **percentile bootstrap over
models (2,000 resamples)**, the model being the primary unit (§K).

---

## K. Primary comparison

**Unit: the trained model (n = 20).** For each model:

- labelled error at budget *n* = mean over R repetitions of \|acc_hat − pool accuracy\|;
- label-free error = \|estimate on the full pool − pool accuracy\| (one value).

This yields 20 paired differences per (budget, baseline). Reported as the mean paired
difference with a percentile bootstrap CI, plus a **Wilcoxon signed-rank** test
(no normality assumption at n = 20).

### Crossover definition — fixed in advance

> The **practical crossover budget** is the smallest N_label at which the labelled
> estimator's mean absolute error is lower than the baseline's **and** the 95% bootstrap
> CI of the paired difference excludes zero.

A point estimate alone is **not** sufficient. If no budget satisfies both conditions,
the result is reported as **"no crossover observed within the tested budgets"**, with no
extrapolation beyond N_label = 100.

---

## L. Secondary analyses

1. **Tail risk:** the **95th percentile absolute error** across repetitions — one
   pre-specified tail metric, no others.
2. **Per-backbone crossover** (ResNet-50, EfficientNet-B0, MobileNetV3-L, ViT-S/16),
   using only the clean V2 + V9 `SHARED_3` protocol. Arm B remains excluded.
3. **Capture-day analysis** (Setting B).
4. **Wilson interval coverage**, as described in §J.

---

## M. Multiple-comparison policy

At each budget the labelled estimator is compared against 8 baselines. **Holm correction
is applied across the 8 baselines within each budget.** Effect sizes (mean paired
difference in absolute error, with bootstrap CI) are reported for every comparison
regardless of significance. No comparison is reported selectively.

---

## N. Exclusion rules

- Arm B excluded (ResNet-50 only — no architecture claim possible).
- V1 excluded from the primary analysis (image-level split; V2 is the leakage-free
  protocol). V2S7 excluded (different label set).
- Photographs failing the `TV_STEM` parse would be excluded — there are **0**.
- No repetition is discarded after the fact for any reason.

---

## O. Stopping / decision rule

The experiment **stops and reports a blocker** if any of the following occur (brief §37):
baseline reproduction fails; the labelled estimator does not converge to the full-pool
accuracy at N_label = 391; a photograph appears twice in a sample; any label reaches a
label-free estimator; any historical result file is modified.

Outcome is then classified as **A** (small budget wins), **B** (moderate), **C** (large),
or **D** (no crossover) — per brief §30, chosen on the evidence and not on which is most
convenient.

---

## P. Expected computational cost

Cheap by design, and deliberately unlike Phase 5:

- Label-free estimates on the full pool: 20 models × 8 estimators = **160 estimates**.
  COT solves a linear programme over 1,616 × 3 and is the slowest term.
- Labelled sampling: 8 budgets × 100 repetitions × 20 models = 16,000 ratio computations,
  which are trivial array operations.
- **No MMD or Fréchet computation** — that was what made Phase 5 expensive.

Estimated total well under one hour. If it exceeds two hours, that is a signal something
is wrong and will be investigated rather than left to run.

---

## Q. What RQ5 will not establish

- Any universal number of labels for agricultural deployment. One source dataset, one
  field dataset, 391 photographs, three classes, four backbones.
- Anything about active or stratified labelling — sampling here is uniform by design.
- Anything per-image.
- Any claim that labelling is *preferable*; only how the two error magnitudes compare
  under the stated protocol.
