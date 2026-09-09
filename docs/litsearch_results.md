# Systematic Literature Search — Interim Results

**Status: INCOMPLETE. Do not freeze the title or any novelty claim on this.**

Stage 1 (arXiv + Crossref) executed 2026-09-09, all 222 pre-registered queries, zero
failed requests. Stage 2 (OpenAlex) is **blocked until midnight UTC** by a daily quota and
is the best-coverage database of the three. Protocol: `docs/litsearch_protocol.md`.

---

## 1. Two bugs found, both of which flattered the thesis

### 1a. A retrieval bug that fabricated the headline finding

The first attempt produced, in its log, exactly the result the thesis wanted:

```
[68/222] S2 unsupervised model evaluation AND plant disease    oa=0
... 112 of 163 queries reporting oa=0
```

Read literally: *no published work applies label-free accuracy estimation to plant
disease* — the load-bearing claim of the entire direction.

**It was false.** Those were HTTP 429s. OpenAlex allows 1,000 requests/day, and
`search_openalex()` returned a count of `0` whenever a request failed; the retry logic then
hit each 429 four more times, burning the remaining budget. Caught because
`plant disease AND domain shift` also returned zero, which is not credible — this search
has since found 147 examples.

Fixed in `c8c9313`: a failed request returns `None` and a status string, never `0`; a
quota refusal aborts instead of retrying; per-query status is recorded and summarised; and
if nothing succeeds the script writes nothing rather than an empty "result". No results
from the bad run were ever written.

### 1b. A screening bug found by the protocol's own audit

Protocol section 8 requires hand-sampling the excluded set. Doing so revealed that
`wrong_sense_of_accuracy` was excluding **prediction-powered-inference papers**:

- *Semi-Supervised Learning via Cross-Prediction-Powered Inference for Wireless Systems*
- *Prediction-Powered Semi-Supervised Learning with Online Power Tuning*
- *Debiased Inference for AI-Generated Data without Gold-Standard Labels*

The false-positive guards read `and not TASK.search(text)`, which is True exactly when
only the **named-method** regex matched — so a named-estimator hit was thrown out by the
filters meant to remove noise, because its abstract also contained a phrase like "state
estimation". A named method is the strongest positive signal there is and must never be
overridden. Fixed; 8 records recovered; included went 329 → 346.

Both bugs pushed in the same direction: **towards a cleaner-looking gap.** That is the
direction to be most suspicious of, and it is why the audit steps are worth running.

## 2. What has been searched

| database | status | queries | failed |
|---|---|---:|---:|
| **arXiv** | complete | 174 (48 not run by design) | **0** |
| **Crossref** | complete | 222 | **0** |
| **OpenAlex** | **blocked, resets midnight UTC** | 0 of 222 | — |
| Scopus, Web of Science, IEEE Xplore, ACM DL | **no institutional access** | — | — |
| DBLP, Semantic Scholar | refused automated access | — | — |

**22,340 records identified → 8,617 unique works → 346 included, 153 adjacent, 8,118
excluded.**

Bounding caveats:

- **Crossref's `query.bibliographic` is a relevance ranking, not a boolean filter.** It
  ignores the `AND` and reports totals in the millions; its totals are never quoted as
  counts.
- **73% of records arrive with no abstract** and are screened on **title alone**, which
  has lower recall. OpenAlex has far better abstract coverage.
- **arXiv counts are genuine boolean phrase searches over abstracts** and are the most
  trustworthy numbers here.

## 3. Q2 — the load-bearing question

> Has any prior work applied label-free accuracy estimation to plant-disease or
> agricultural imagery?

**No such work found. But the surrounding space is more occupied than assumed.**

| search | queries | arXiv hits |
|---|---:|---:|
| **S2** task × agriculture | 88 | **1** |
| **S3** task × agriculture × shift | 60 | **0** |

Every flagged candidate was read in full:

| record | verdict |
|---|---|
| *Automatic Detection of Rice Disease in Images of Various Leaf Sizes* (2022) | estimates **rice leaf width** to size image tiles. **Rejected.** |
| *A double copy from twisted (co)homology at genus one* (2023) | string theory; "leaf" is foliation theory, `MaNo` matched by accident. **Rejected.** |
| *Design-Based Prediction-Powered Inference for Spatial Data* (2026) | statistical methodology, not plant disease — **but highly relevant, section 5** |
| **Scalable Vision-Guided Crop Yield Estimation (2025)** | **genuine. See below.** |

### The find that narrows the gap

*Scalable Vision-Guided Crop Yield Estimation* (2025) applies **PPI in agriculture**: it
supplements slow gold-standard crop cuts with fast field **photographs**, trains a vision
model to predict yield from the photos, learns a spatially-indexed control function to
recalibrate them, and uses ~20,000 real crop cuts and photos of rice and maize in
sub-Saharan Africa.

That is agriculture + a vision model + a small gold-standard sample + field photographs —
structurally very close to this thesis's setting. **The claim "labelled-sample-assisted
estimation has not been done in agriculture" is false and must not be made.**

What it does *not* do: estimate a **classifier's accuracy** on unlabelled field images.
Its estimand is average crop yield, a regression quantity measured with a designed survey,
not the accuracy of a fixed deployed classifier. That distinction is the surviving gap,
and it is narrower than the one assumed at the start of this search.

## 4. Q3 — is the agricultural lab-to-field setting occupied?

**Yes, heavily — but always *with* labels.** 147 records are `agricultural_shift_no_estimation`:

- *From Benchmark Accuracy to Field Performance: Hybrid Deep Learning-Based Plant Disease
  Classification* (2026)
- *Explainable Transfer Learning for Multi-Crop Plant Disease Identification Under
  Real-Field Conditions* (2026)
- *Enhancing Generalization of Tomato Leaf Disease Classification via TDR Model and
  Field-Condition* (2026)
- *A multi-scale supervised contrastive framework for cross-domain soybean disease
  classification* (2026)

None estimates a fixed model's field accuracy without labels. The gap is real — but
Related Work must engage this set rather than claim the setting is unstudied.

## 5. The PPI family — read, not just counted

102 records in the corpus. Four read in full:

**Power Analysis for Prediction-Powered Inference (2026).** Derives closed-form formulas
for *how many labelled samples are needed* — literally the RQ5 question. Its rule of
thumb: the reduction in required labels **scales roughly with the R² between predictions
and ground truth.**

> This is the single most useful sentence the search has produced. Our estimators have
> R² = 0.026 (GdScore) and calibration r ≈ 0. Under this rule, PPI theory predicts
> **essentially no labelling saving** from them — which is exactly what RQ5 measured
> empirically. The RQ5 result stops being an isolated observation and becomes consistent
> with established theory, with the R² we measured as the reason.

**Statistical Optimality of PPI (2026).** PPI attains the semiparametric efficiency bound
when the predictor is **score-calibrated**. Our Phase-7b finding is precisely that the
predictor is *not* calibrated — so the condition under which PPI is optimal is the
condition our estimators fail.

**Design-Based PPI for Spatial Data (2026).** Recasts PPI when labels arrive by **survey
design rather than i.i.d.**, with correlated errors, stratified sampling and exact design
variances — the photograph-cluster problem. Notes coverage can deteriorate as labels
accumulate, which is directly relevant to the 63–75% coverage we measured for naive
crop-level intervals.

**Prediction-Powered Risk Monitoring (2026).** Anytime-valid lower bounds on running risk
from synthetic labels plus a few true ones, to detect harmful shift in deployment. This is
the closest published competitor to a deployment-gate framing and must be cited.

**Consequence:** RQ5's crossover cannot be presented as a novel discovery, and RQ5's ratio
estimator is a hand-rolled special case of an established framework. Position RQ5 as a
practical, agricultural, photograph-level measurement *within* PPI's frame — not as the
novelty. This also settles the earlier objection (*"isn't that universal… how is this
novel"*): correct, it is not, and there is now a citation for why.

## 6. Q4 — is the calibration-anchor pathology known?

**Nothing found.** All four S6 queries returned 0 on arXiv; no included record came from
S6.

This is **weak** evidence — four queries, one database, and a pathology is hard to name in
advance. It is consistent with Phase 7b being a genuine contribution and is **not**
sufficient to claim priority. This is the question most in need of the subscription
databases.

## 7. Q1 — are the nine estimators representative?

| method | arXiv hits | in our nine |
|---|---:|---|
| prediction-powered inference | 86 | **no — adjacent family** |
| projection norm | 36 | **no** |
| agreement-on-the-line | 31 | **no** |
| difference of confidences | 25 | yes (DoC) |
| AutoEval | 15 | **no** |
| average thresholded confidence | 4 | yes (ATC) |
| confidence optimal transport | 2 | yes (COT) |
| GdScore | 1 | yes |
| MaNo | 134 → **3 genuine** | yes |

Read, to judge whether the omissions matter:

- **Projection Norm** (Yu et al., 2022) — pseudo-labels the test set, **trains a new
  model**, and measures parameter divergence. Genuinely label-free, but it requires
  retraining per test set, so it does not fit the "expensive once, cheap millions of
  times" caching design. A defensible omission, and one to state.
- **Agreement-on-the-line** (Baek et al., 2022; Miller et al., 2021) — estimates OOD
  performance from ID/OOD **agreement across a collection of models**. We have 20 models
  across 4 architectures, so this is *applicable to our setup* and is the most notable
  omission. It is also known to break on some shifts, which is itself relevant.
- **AutoEval** — the term is overloaded: the arXiv hits are *AutoEval-Video*, an unrelated
  LLM benchmark. The original (Deng & Zheng, CVPR 2021) did not surface in an arXiv
  abstract search and needs to be retrieved by name.

Screening artefacts recorded: `MaNo` matches the **MANO parametric hand model** and
Spanish *"la mano"*; only 3 of 134 hits are the estimator.

## 8. Draft literature-gap table

Provisional — the `?` rows are exactly what stage 2 and the subscription databases must
settle.

| Area | Existing work? | What it does | What is missing |
|---|---|---|---|
| Plant disease lab→field | **Yes** (147 found) | Measures the gap with labels, or adapts the model to close it | Assessing it **without** field labels |
| Label-free accuracy estimation | **Yes** (large, active) | Estimates target accuracy on unlabelled data; validated on corruption/OOD benchmarks | Validation in an **agricultural deployment** with real field imagery |
| GdScore (2024/25) | **Yes** | Gradient-norm label-free estimation | **Transferability to agricultural shift**; behaviour on BatchNorm backbones |
| PPI | **Yes**, incl. **in agriculture** (crop yield, 2025) | Small gold-standard sample + model predictions → valid intervals | Applied to **classifier accuracy** rather than a regression estimand |
| PPI power analysis | **Yes** (2026) | Labels needed scales with R² of predictions | Empirical check where R² ≈ 0, i.e. our regime |
| Cluster/design-based sampling | **Yes** (design-based PPI, 2026) | Exact design variances for non-i.i.d. labelling | Applied to photograph-level agricultural sampling |
| Architecture dependence of estimators | **?** | Not yet verified | Potential gap — **stage 2 must check** |
| Calibration-anchor pathology | **None found** | — | **Currently the most defensible specific contribution** |

## 9. Provisional verdict

**Leaning A, with a materially narrower contribution than assumed at the start.**

- The *application* gap survives: no label-free **classifier-accuracy** estimation in
  agriculture found, across 148 task × domain queries in two databases.
- The *setting* is well populated (147 works), so novelty cannot rest on studying
  lab→field in agriculture.
- The *labelled-budget* result is **not** novel — PPI formalises it, PPI power analysis
  answers the sample-size question in closed form, and PPI has **already been applied in
  agriculture**.
- The *calibration-anchor* finding has no match so far and is currently the strongest
  specific contribution.

## 10. Outstanding

1. **Stage 2**, once the quota resets:
   ```
   python scripts/litsearch.py --databases openalex
   python scripts/litsearch_screen.py
   ```
2. **Retrieve AutoEval (Deng & Zheng, CVPR 2021) by name** — missed by abstract search.
3. **Check whether estimator architecture-dependence is already documented** — the one `?`
   row that could remove a claimed contribution.
4. **Scopus / Web of Science / IEEE Xplore / ACM DL.** Requires institutional access.
   Until then every null is "none found by this search", never "none exists".
5. Re-run the excluded-set audit after stage 2; it has already caught one bug.
