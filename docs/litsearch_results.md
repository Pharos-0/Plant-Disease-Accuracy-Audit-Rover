# Systematic Literature Search — Results

**Status: substantially complete. One block (S4) outstanding. Title still not frozen.**

- Stage 1 (arXiv + Crossref): **complete**, 222/222 queries, zero failures.
- Stage 2 (OpenAlex): **174/222** — S1, S2, S3, S5, S6 complete. Only **S4** (agricultural
  context, 48 queries) remains, the least decision-critical block.
- Supplementary targeted pass: complete, reported separately.

Protocol: `docs/litsearch_protocol.md`. Corpus: **31,906 records → 15,980 unique works →
253 included, 275 adjacent.**

---

## 1. Four bugs found, all of which flattered the thesis

**1a. Retrieval.** The first run logged `oa=0` for 112 of 163 queries — read literally, *no
published work applies label-free accuracy estimation to plant disease*. Those were HTTP
429s recorded as zeros. Fixed in `c8c9313`. The query that exposed it,
`plant disease AND domain shift`, now returns **489**.

**1b. Screening.** The protocol's excluded-set audit found `wrong_sense_of_accuracy`
discarding prediction-powered-inference papers: the guards read `and not TASK.search(text)`,
true exactly when only the **named-method** regex matched. Fixed; 8 records recovered.

**1c. arXiv query construction.** `search_arxiv` wrapped any query without `" AND "` in a
single `abs:"..."`, so block C ran as impossible exact phrases. Proved by
`abs:"Are Labels Always Necessary for Classifier Accuracy Evaluation"` → **0** while
`ti:"Are Labels Always Necessary"` returns it. All four Q4 zeros were artefacts.

**1d. Budget accounting.** The advertised `x-ratelimit-limit: 1000` is not the binding
limit; a **monetary budget of ~100 requests/UTC day** is. Two runs stopped at exactly query
100, the second having spent the whole day re-running S1+S2. Fixed with resume support and
`--searches` prioritisation (`53c38c9`).

**Precision regression, found and fixed here.** The 1b fix made a named-method match
override the noise filters — correct in principle, but bare `\bMaNo\b` then admitted *Mano
River*, *Mano Dura*, *mano-thermo-sonication*, NFV *MANO*, grinding stones and a hand
exoskeleton. Included inflated to 551. `MaNo` now requires its own context (matrix norm /
accuracy estimation / distribution shift); included fell to **253** with **no recall loss**.

> **The four bugs all pushed towards a cleaner, emptier gap.** Each correction has shrunk
> the claimable contribution. That is the search working as intended.

## 2. Search reliability, measured

`scripts/litsearch_audit.py` tests the search against a known-answer set of 18 label-free
accuracy-estimation papers, each confirmed by reading it.

| check | result |
|---|---|
| known-relevant **retrieved** by the pre-registered queries | **14 / 18 (78%)** |
| retrieved papers the screener **kept** | **14 / 14 (100%)** |
| screener misses | **0** |

The four missed (Distance-Matters, Mandoline, ALSA, IW-GAE) were found only by the
supplementary pass. **Retrieval, not screening, is the weak link.** A null at ~78% recall
is real evidence, not proof — which is exactly why the supported wording is "insufficiently
characterised", never "never studied".

Known imperfection, recorded not fixed: AETTA and DAPPER are label-free estimation papers
filed `adjacent: adaptation_not_estimation` because their abstracts describe test-time and
domain adaptation. Retained and read, so nothing was lost; the category is wrong.

## 3. Q2 — the load-bearing question

> Has any prior work applied label-free accuracy estimation to plant-disease or
> agricultural imagery?

**Nothing found, on all three databases, with S2 and S3 complete.**

| search | queries | arXiv hits |
|---|---:|---:|
| S2 task × agriculture | 88 | **1** |
| S3 task × agriculture × shift | 60 | **0** |

Every flagged candidate was read in full and rejected: rice **leaf-width** estimation
(2022); a string-theory paper where "leaf" is foliation theory (2023); supervised YOLOv8
detection with labelled mAP (2023); grapevine water status assessed by **cross-validation
against ground data** (2024); LiDAR ranging (2015); and three *MaNo* string collisions
(mano-thermo-sonication, Spanish *mano de obra*, and a plant named Mano).

**The one genuine neighbour** is *Scalable Vision-Guided Crop Yield Estimation* (2025):
PPI in agriculture, supplementing gold-standard crop cuts with field **photographs**,
~20,000 crop cuts of rice and maize. Its estimand is **average yield** — a regression
quantity from a designed survey — not a classifier's accuracy. **"Labelled-sample-assisted
estimation has not been done in agriculture" is false and must not be claimed.**

## 4. Q3 — the agricultural lab-to-field setting

**Heavily occupied, always with labels.** 260 adjacent records, including *From Benchmark
Accuracy to Field Performance* (2026) and *Explainable Transfer Learning for Multi-Crop
Plant Disease Identification Under Real-Field Conditions* (2026). None estimates a fixed
model's field accuracy without labels. Related Work must engage this set.

## 5. PPI — read, not counted

**Power Analysis for PPI (2026)** gives closed-form labelled-sample requirements, the saving
**scaling as the R² between predictions and truth**.

> Our estimators have R² = 0.026 and calibration r ≈ 0, so PPI theory predicts
> **essentially no labelling saving** — exactly what RQ5 measured. RQ5 becomes consistent
> with established theory, with the measured R² as the reason.

**Statistical Optimality of PPI (2026):** PPI is efficient when the predictor is
**score-calibrated** — the condition Phase 7b shows ours fails. **Design-Based PPI for
Spatial Data (2026):** labels arriving by survey design rather than i.i.d. — the
photograph-cluster problem. **Prediction-Powered Risk Monitoring (2026):** anytime-valid
running-risk bounds to detect harmful shift; the closest competitor to a deployment gate.

RQ5's crossover is **not** a novel discovery. Position it as a practical agricultural
measurement *within* PPI's frame.

## 6. Q4 — the calibration-anchor pathology

**S6 ran successfully on OpenAlex this time** (counts 210, 1366, 1503, 88 — not the earlier
artefactual zeros) and produced **zero included records**. The null is now real rather than
a bug.

Nearest neighbours, all read:

| work | what it does | why it is not the same |
|---|---|---|
| **Distance-Matters** (ICCV-W 2023) | flags test samples far from the training distribution, whose confidences are ill-calibrated; 13 tasks, 259 models | **the closest work.** Full text searched: **no match** for `decompos*`, `intercept`, `anchor`, `trivial`, `constant baseline`, or `batchnorm`. Theirs is *which samples* to trust; ours is *how much of the estimate comes from the data at all*. |
| **ALSA** (2025) | learnable **anchor** points in logit space | different concept, same word. Terminology must be distinguished explicitly. |
| **IW-GAE** (2023) | importance-weighted group accuracy for calibration under UDA | improves calibration; does not decompose an estimate |

**Standing: no work found that decomposes a calibrated estimator's output into
intercept-versus-data share.** This remains the **strongest specific candidate
contribution**, now on a genuine null rather than an artefact.

## 7. Q1 — the nine are a sample, and a general benchmark already exists

Two findings from the completed S5, both consequential.

### 7a. ODP-Bench (2025) — a general benchmark already exists

*ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction* provides "most commonly
used OOD datasets and existing practical performance prediction algorithms", supplies
trained models as a testbench, and notes that "evaluation protocols in previous literature
are inconsistent".

> **A general-purpose systematic benchmark of these estimators exists.** The thesis cannot
> claim to be the first systematic evaluation of label-free estimators. Its benchmark claim
> must be scoped to **agricultural laboratory-to-field shift on real field imagery**, which
> ODP-Bench does not cover.

### 7b. The method inventory, final

At least **twenty** methods across **three research communities** (ML/vision, software
testing, statistics/PPI), from **2016 onward**:

| in our nine | not evaluated |
|---|---|
| confidence, entropy, ATC-MC, ATC-NE, DoC, COT, nuclear norm, MaNo, GdScore | AutoEval, Projection Norm, Agreement-on-the-Line, GDE, Mandoline, DAPPER, AETTA, IW-GAE, Distance-Matters, ALSA, Aries, Platanios agreement (2017/18), Steinhardt & Liang (2016), ODP-Bench algorithms |

Selection is justified explicitly in `docs/estimator_selection.md`, which separates
principled exclusions from ignorance. **No estimator is being added.**

### 7c. "New application domain" is established at least five times

| domain | work | year |
|---|---|---|
| mobile sensing | DAPPER | 2021 |
| autonomous driving | Label-Free Robustness Estimation (IJCV) | 2021 |
| medical imaging | arXiv:2507.22776 | 2025 |
| brain–computer interfaces | Unsupervised Accuracy Estimation for BCI (IEEE TBME) | 2023/25 |
| battery health | Condition-aware generative transfer | 2026 |

Agriculture remains unoccupied. **The genre move is not novel.**

## 8. Q5 — architecture dependence, downgraded

Previously "not found". The completed S5 forces a weaker statement.

*Classification Accuracy Estimation Without Labels via **Architecture-Agnostic** Model
Agreement* (2025) explicitly advertises architecture-agnosticism and reports robustness
"across diverse model types — including convolutional networks and Transformers".

> So **architecture sensitivity is a recognised concern** in this literature: at least one
> 2025 method positions itself against it. What still has **no match** is a *systematic
> cross-estimator analysis* of how estimator reliability varies with backbone.

Supporting evidence that the analysis is unoccupied: **Distance-Matters evaluates 259 models
across 14 architecture families and never analyses architecture** — it averages over them.
ODP-Bench's "capability boundary" analyses have not been read and could contain it.

**Downgraded from "gap" to "weak, unverified secondary contribution."** Do not lead with it.

## 9. Literature-gap table

| Area | Existing work? | What it does | What is missing |
|---|---|---|---|
| Plant disease lab→field | **Yes** (260) | Measures with labels, or adapts | Assessing it **without** field labels |
| Label-free accuracy estimation | **Yes** (≥20 methods, 3 communities, since 2016) | Estimates target accuracy on unlabelled data | Validation in an **agricultural deployment** |
| General benchmark of these estimators | **Yes** — ODP-Bench (2025) | Standardised OOD benchmark + testbench models | **Agricultural lab→field on real field imagery** |
| Label-free estimation in an applied domain | **Yes ×5** | Same genre move | Agriculture specifically |
| GdScore | **Yes** | Gradient-norm estimation | **Transferability to agricultural shift** |
| PPI | **Yes**, incl. **in agriculture** (2025) | Small labelled sample + predictions | Applied to **classifier accuracy** |
| PPI power analysis | **Yes** (2026) | Labels needed scale with R² | Empirical check where R² ≈ 0 |
| Confidence miscalibration under shift | **Yes** (Distance-Matters, IW-GAE) | *Which samples* to distrust | Decomposition of the **estimate** into anchor vs data |
| Architecture dependence | **Partly** — a 2025 method claims architecture-agnosticism | Recognised as a concern | Systematic cross-estimator analysis — **weak, unverified** |
| Calibration-anchor pathology | **Not found** (genuine null) | ALSA uses "anchor" differently | **Strongest specific candidate** |

## 10. Verdict

**A, with a contribution materially narrower than at the start.** What survives:

1. **Primary.** Label-free **classifier-accuracy** estimation has not been characterised
   under agricultural laboratory-to-field shift on real field imagery. The setting is
   populated, the genre is established in five other domains, and a general benchmark
   exists — but this specific combination is not occupied.
2. **Strongest specific.** The **calibration-anchor pathology**, on a genuine null, with the
   nearest neighbour read in full and distinguished.
3. **Weak secondary.** Architecture dependence — recognised as a concern, no systematic
   analysis found. Do not lead with it.

What does **not** survive: firstness of the domain move; the labelled-budget crossover as a
discovery; "first systematic evaluation of label-free estimators"; and any claim the nine
represent the state of the art.

Supported wording: **"remain insufficiently characterised."**

## 11. Outstanding

1. **S4 on OpenAlex** (48 queries, next budget day) — agricultural context; 260 adjacent
   records already held, so this is confirmatory:
   ```
   python scripts/litsearch.py --databases openalex --searches S4
   python scripts/litsearch_screen.py && python scripts/litsearch_audit.py
   ```
2. **Read ODP-Bench's "capability boundary" analyses** — the one place architecture
   dependence could already be documented.
3. **Scopus / Web of Science / IEEE Xplore / ACM DL** — requires institutional access.
   Until then every null is "none found by this search".
