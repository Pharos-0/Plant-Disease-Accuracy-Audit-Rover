# Systematic Literature Search — Interim Results

**Status: INCOMPLETE. Do not freeze the title or any novelty claim on this.**

Stage 1 (arXiv + Crossref) executed 2026-09-09, all 222 pre-registered queries, zero
failed requests. Stage 2 (OpenAlex) is **blocked until midnight UTC** by a daily quota.
A supplementary targeted pass (`scripts/litsearch_followup.py`) has also been run and is
reported separately. Protocol: `docs/litsearch_protocol.md`.

---

## 1. Three bugs found, all of which flattered the thesis

### 1a. A retrieval bug that fabricated the headline finding

The first attempt logged `oa=0` for 112 of 163 queries — read literally, *no published
work applies label-free accuracy estimation to plant disease*, the load-bearing claim of
the whole direction.

**False.** Those were HTTP 429s. OpenAlex allows 1,000 requests/day, and
`search_openalex()` returned `0` whenever a request failed; the retry logic then hit each
429 four more times, burning the remaining budget. Caught because
`plant disease AND domain shift` also returned zero, which is not credible — the search
has since found 147 examples. Fixed in `c8c9313`. No results from the bad run were written.

### 1b. A screening bug found by the protocol's own audit

The mandated hand-sample of the excluded set showed `wrong_sense_of_accuracy` was
discarding **prediction-powered-inference papers**. The guards read
`and not TASK.search(text)`, true exactly when only the **named-method** regex matched — so
a named-estimator hit was thrown out by filters meant to remove noise, because its abstract
also said "state estimation". Fixed; 8 records recovered; included 329 → 346.

### 1c. An arXiv query bug that faked the Q4 null

`search_arxiv` wraps a query containing no `" AND "` in a single `abs:"..."`. The
protocol's block C entries are concept *combinations* written without `AND`, so S6 ran as
four impossible exact-phrase searches:

```
abs:"calibration accuracy estimation artifact"   ->  0, always
```

Proved by searching for a paper known to exist: `abs:"Are Labels Always Necessary for
Classifier Accuracy Evaluation"` returns **0**, while `ti:"Are Labels Always Necessary"`
returns it immediately. **All four S6 zeros were artefacts, and so were the S5 zeros for
"nuclear norm accuracy estimation" and "Frechet distance accuracy estimation".**

Q4 has been re-run correctly in the supplementary pass (`S6c`), with every phrase quoted
individually and joined by `AND`. It found six methods the original pass missed — see
sections 6 and 7.

> **All three bugs pushed the same way: towards a cleaner, emptier-looking gap.** That is
> the direction to distrust, and it is the argument for running the audit steps.

## 2. What has been searched

| database | status | queries | failed |
|---|---|---:|---:|
| **arXiv** | complete | 174 + 27 supplementary | **0** |
| **Crossref** | complete | 222 + 20 supplementary | **0** |
| **OpenAlex** | **blocked, resets midnight UTC** | 0 of 222 | — |
| Scopus, Web of Science, IEEE Xplore, ACM DL | **no institutional access** | — | — |
| DBLP, Semantic Scholar | refused automated access | — | — |

Pre-registered corpus: **22,340 records → 8,617 unique works → 346 included, 153 adjacent,
8,118 excluded.** Supplementary pass: 2,079 further records, kept in separate files and
never merged into the pre-registered corpus.

Bounding caveats: Crossref's `query.bibliographic` is a relevance ranking, not a boolean
filter, so its totals are never quoted as counts; 73% of records arrive without an abstract
and are screened on **title alone**; arXiv counts are genuine boolean searches and are the
most trustworthy numbers here.

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
| *A double copy from twisted (co)homology at genus one* (2023) | string theory; "leaf" is foliation theory. **Rejected.** |
| *Design-Based Prediction-Powered Inference for Spatial Data* (2026) | methodology, not plant disease — but relevant, section 5 |
| **Scalable Vision-Guided Crop Yield Estimation (2025)** | **genuine — see below** |

### The find that narrows the gap

*Scalable Vision-Guided Crop Yield Estimation* (2025) applies **PPI in agriculture**:
supplements slow gold-standard crop cuts with field **photographs**, trains a vision model
to predict yield, learns a spatially-indexed control function to recalibrate, on ~20,000
real crop cuts of rice and maize in sub-Saharan Africa.

**The claim "labelled-sample-assisted estimation has not been done in agriculture" is false
and must not be made.**

What it does *not* do: estimate a **classifier's accuracy** on unlabelled field images. Its
estimand is average crop yield — a regression quantity from a designed survey. That
distinction is the surviving gap.

## 4. Q3 — is the agricultural lab-to-field setting occupied?

**Yes, heavily — but always *with* labels.** 147 records, including *From Benchmark
Accuracy to Field Performance* (2026), *Explainable Transfer Learning for Multi-Crop Plant
Disease Identification Under Real-Field Conditions* (2026), and *A multi-scale supervised
contrastive framework for cross-domain soybean disease classification* (2026). None
estimates a fixed model's field accuracy without labels. Related Work must engage this set
rather than claim the setting is unstudied.

## 5. The PPI family — read, not just counted

102 records. Four read in full:

**Power Analysis for PPI (2026).** Closed-form formulas for *how many labelled samples are
needed* — literally the RQ5 question. Rule of thumb: the reduction in required labels
**scales roughly with the R² between predictions and ground truth.**

> The single most useful sentence the search has produced. Our estimators have R² = 0.026
> (GdScore) and calibration r ≈ 0. Under this rule PPI theory predicts **essentially no
> labelling saving** — exactly what RQ5 measured. RQ5 stops being an isolated observation
> and becomes consistent with established theory, with the measured R² as the reason.

**Statistical Optimality of PPI (2026).** PPI attains the semiparametric efficiency bound
when the predictor is **score-calibrated**. Phase 7b's finding is that ours is not — so the
condition under which PPI is optimal is the condition our estimators fail.

**Design-Based PPI for Spatial Data (2026).** PPI when labels arrive by **survey design
rather than i.i.d.**, with correlated errors, stratified sampling, exact design variances —
the photograph-cluster problem. Notes coverage can deteriorate as labels accumulate,
relevant to the 63–75% coverage measured for naive crop-level intervals.

**Prediction-Powered Risk Monitoring (2026).** Anytime-valid bounds on running risk from
synthetic plus a few true labels, to detect harmful shift in deployment. The closest
published competitor to a deployment-gate framing; must be cited.

**Consequence:** RQ5's crossover is not a novel discovery, and its ratio estimator is a
hand-rolled special case of an established framework. Position RQ5 as a practical,
agricultural, photograph-level measurement *within* PPI's frame.

## 6. Q4 — is the calibration-anchor pathology known?

The original answer ("nothing found") was an artefact of bug 1c. **Corrected re-run: still
no match for the specific pathology, but the neighbourhood is populated.**

Nearest neighbours, all read:

| work | what it does | why it is not the same |
|---|---|---|
| **ALSA: Anchors in Logit Space for OOD Accuracy Estimation** (2025) | learnable **anchor** points in logit space as an estimator | "anchor" = learned reference points in logit space. **Different concept, same word.** Ours is the source-accuracy intercept of a fitted calibration map. Terminology must be distinguished explicitly. |
| **Distance Matters For Improving Performance Estimation Under Covariate Shift** (2023) | flags test samples far from the training distribution, whose confidences are ill-calibrated; 13 tasks, hundreds of models | **the closest work.** Diagnoses a confidence-calibration failure under shift and fixes it. Does not decompose an estimate into intercept vs data contribution. |
| **IW-GAE** (2023) | importance-weighted group accuracy estimation, explicitly for calibration under UDA | improves calibration; does not identify estimates dominated by the calibration anchor |

**Standing:** no work found that decomposes a calibrated estimator's output into
intercept-versus-data share, or that reports an estimator whose apparent accuracy comes
from its source anchor. This remains the **strongest specific candidate contribution** —
but it is a null over two databases with the subscription databases outstanding, and
"Distance Matters" is close enough that it must be read in full and distinguished
carefully.

## 7. Q1 — are the nine estimators representative? **No.**

The corrected searches found methods the original pass missed. Revised inventory:

| method | year | in our nine | note |
|---|---|---|---|
| ATC | 2022 | **yes** | |
| DoC | 2021 | **yes** | |
| COT | 2023 | **yes** | |
| nuclear norm / Confidence-Dispersity | 2023 | **yes** | |
| MaNo | 2024 | **yes** | |
| GdScore | 2024/25 | **yes** | |
| confidence, entropy | — | **yes** | baselines |
| **AutoEval** | 2020/21 | no | regression from dataset feature statistics on a meta-dataset |
| **Projection Norm** | 2022 | no | pseudo-labels, **retrains**, measures parameter divergence |
| **Agreement-on-the-Line** | 2022 | no | needs a **collection of models** — we have 20 |
| **Mandoline** | 2021 | no | importance weighting via user-written slicing functions |
| **DAPPER** | 2021 | no | label-free performance estimation via mutual information |
| **AETTA** | 2024 | no | prediction disagreement against dropout inferences |
| **IW-GAE** | 2023 | no | importance-weighted group accuracy |
| **Distance-Matters** | 2023 | no | distance-gated confidence |
| **ALSA** | 2025 | no | learnable anchors in logit space |

**The nine are a sample of this literature, not the state of the art**, and the thesis must
say so. Per the current decision no estimator #10 is being added — but the *claim about
coverage* has to be corrected accordingly.

Two omissions are more awkward than the others: **Agreement-on-the-Line** applies to a
collection of models, which we have; and **Distance-Matters** targets exactly the
confidence-miscalibration-under-shift phenomenon this thesis reports.

### A second blow to "new application domain"

**DAPPER (2021)** is label-free performance estimation for **heterogeneous mobile sensing
deployment** — the same genre move as this thesis, in a different applied domain, five
years earlier. Together with the medical-imaging paper (arXiv:2507.22776, 2025), the
"take label-free estimation into an applied deployment domain" move is established at
least twice. Agriculture is still unoccupied; the *genre* is not novel.

## 8. Draft literature-gap table

| Area | Existing work? | What it does | What is missing |
|---|---|---|---|
| Plant disease lab→field | **Yes** (147 found) | Measures the gap with labels, or adapts the model | Assessing it **without** field labels |
| Label-free accuracy estimation | **Yes** (≥16 methods) | Estimates target accuracy on unlabelled data; validated on corruption/OOD benchmarks | Validation in an **agricultural deployment** on real field imagery |
| Label-free estimation in an applied domain | **Yes** — mobile sensing (2021), medical imaging (2025) | Same genre move, other domains | Agriculture specifically |
| GdScore | **Yes** | Gradient-norm label-free estimation | **Transferability to agricultural shift**; BatchNorm behaviour |
| PPI | **Yes**, incl. **in agriculture** (crop yield, 2025) | Small gold-standard sample + predictions → valid intervals | Applied to **classifier accuracy**, not a regression estimand |
| PPI power analysis | **Yes** (2026) | Labels needed scale with R² of predictions | Empirical check in the R² ≈ 0 regime |
| Cluster/design-based sampling | **Yes** (2026) | Exact design variances for non-i.i.d. labelling | Applied to photograph-level agricultural sampling |
| Confidence miscalibration under shift | **Yes** (Distance-Matters 2023, IW-GAE 2023) | Diagnoses and corrects it | Decomposition of an estimate into anchor vs data |
| Architecture dependence of estimators | **Not found** | SQ1 returned nothing direct | Potential gap — **stage 2 must confirm** |
| Calibration-anchor pathology | **Not found** | ALSA uses "anchor" for a different thing | **Strongest specific candidate** |

## 9. Provisional verdict

**Leaning A, with a contribution materially narrower than assumed, and narrowing further
with each correction.**

- The *application* gap survives: no label-free **classifier-accuracy** estimation in
  agriculture found, across 148 task × domain queries.
- The *setting* is well populated (147 works) — novelty cannot rest on it.
- The *genre* (label-free estimation in an applied domain) is established in mobile sensing
  and medical imaging.
- The *labelled-budget* result is **not** novel — PPI formalises it, answers the sample-size
  question in closed form, and has already been used in agriculture.
- The *estimator coverage* claim must be weakened: nine of at least sixteen.
- The *calibration-anchor* finding still has no match and is the strongest specific
  contribution — pending a full read of Distance-Matters.

The wording the evidence currently supports is **"remain insufficiently characterised"**,
never "have never been studied".

## 10. Outstanding

1. **Stage 2**, once the quota resets:
   ```
   python scripts/litsearch.py --databases openalex
   python scripts/litsearch_screen.py
   ```
2. **Read *Distance Matters For Improving Performance Estimation Under Covariate Shift*
   (2023) in full** — the nearest neighbour to the Phase-7b contribution.
3. **Confirm architecture-dependence is undocumented** — the one row that could remove a
   claimed contribution.
4. **Scopus / Web of Science / IEEE Xplore / ACM DL.** Requires institutional access.
   Until then every null is "none found by this search", never "none exists".
5. Re-run the excluded-set audit after stage 2; it has already caught one bug.
