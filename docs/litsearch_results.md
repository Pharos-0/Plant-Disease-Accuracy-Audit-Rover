# Systematic Literature Search — Interim Results

**Status: NEARLY COMPLETE. Do not freeze the title yet.**

- Stage 1 (arXiv + Crossref): **complete**, 222/222 queries, zero failures.
- Stage 2 (OpenAlex): **99/222** — **S1 and S2 complete**, S3–S6 pending a quota reset
  (midnight UTC). S2 is the load-bearing block, so the central question is answered on all
  three databases.
- Supplementary targeted pass: complete, reported separately.

Protocol: `docs/litsearch_protocol.md`.

---

## 1. Three bugs found, all of which flattered the thesis

**1a. Retrieval.** The first run logged `oa=0` for 112 of 163 queries — read literally, *no
published work applies label-free accuracy estimation to plant disease*. Those were HTTP
429s: OpenAlex allows 1,000 requests/day and `search_openalex()` returned `0` on failure,
while the retry logic burned the remaining budget. Fixed in `c8c9313`. The same query that
exposed it, `plant disease AND domain shift`, now returns **489**.

**1b. Screening.** The protocol's mandated excluded-set audit found
`wrong_sense_of_accuracy` discarding prediction-powered-inference papers: the guards read
`and not TASK.search(text)`, true exactly when only the **named-method** regex matched.
Fixed; 8 records recovered.

**1c. arXiv query construction.** `search_arxiv` wraps a query containing no `" AND "` in a
single `abs:"..."`, so the protocol's block-C entries ran as impossible exact phrases.
Proved by `abs:"Are Labels Always Necessary for Classifier Accuracy Evaluation"` → **0**
while `ti:"Are Labels Always Necessary"` returns it immediately. All four Q4 zeros and two
S5 zeros were artefacts. Re-run correctly in the supplementary pass.

> **All three pushed the same way: towards a cleaner, emptier gap.** Each correction has
> shrunk the claimable contribution. That is the search working.

## 2. Coverage

| database | status | queries | failed |
|---|---|---:|---:|
| **arXiv** | complete | 174 + 27 supplementary | **0** |
| **Crossref** | complete | 222 + 20 supplementary | **0** |
| **OpenAlex** | **S1+S2 done; S3–S6 pending** | 99 of 222 | **0** |
| Scopus, Web of Science, IEEE Xplore, ACM DL | **no institutional access** | — | — |

**28,715 records → 13,514 unique works → 391 included, 253 adjacent, 12,870 excluded.**

Caveats: Crossref's `query.bibliographic` is a relevance ranking, not a boolean filter, so
its totals are never quoted as counts; 73% of records arrive without an abstract and are
screened on **title alone**; OpenAlex's `title_and_abstract.search` ANDs terms without
enforcing the quoted phrase, so its queries are a **superset** of the protocol's phrase
search — recall higher, precision lower, which makes a null *stronger* evidence.

## 3. Q2 — the load-bearing question

> Has any prior work applied label-free accuracy estimation to plant-disease or
> agricultural imagery?

**Nothing found, now across all three databases with S2 complete.**

| search | queries | arXiv hits |
|---|---:|---:|
| **S2** task × agriculture | 88 | **1** |
| **S3** task × agriculture × shift | 60 | **0** |

Six records were flagged `task_and_agriculture`. **All six were read in full and all six
were rejected:**

| record | verdict |
|---|---|
| *Automatic Detection of Rice Disease in Images of Various Leaf Sizes* (2022) | estimates **rice leaf width**. Rejected. |
| *A double copy from twisted (co)homology at genus one* (2023) | string theory; "leaf" = foliation theory. Rejected. |
| *Plant Disease Detection and Segmentation using End-to-End YOLOv8* (2023) | ordinary **supervised** detection, labelled mAP/F1. Rejected. |
| *Assessing the spatial-temporal performance of ML in predicting grapevine water status* (2024) | regression assessed by **cross-validation against ground data**. Labels used. Rejected. |
| *Development and Field Testing of LiDAR Applications in Agricultural Mobile Vehicles* (2015) | LiDAR ranging; wrong sense of "accuracy". Rejected. |
| *Design-Based PPI for Spatial Data* (2026) | methodology, not plant disease — relevant, section 5. |
| **Scalable Vision-Guided Crop Yield Estimation (2025)** | **genuine — narrows the gap, below.** |

### The find that narrows the gap

*Scalable Vision-Guided Crop Yield Estimation* (2025) applies **PPI in agriculture**:
supplements gold-standard crop cuts with field **photographs**, trains a vision model,
learns a spatially-indexed control function, on ~20,000 crop cuts of rice and maize in
sub-Saharan Africa.

**"Labelled-sample-assisted estimation has not been done in agriculture" is false and must
not be claimed.** What it does not do is estimate a **classifier's accuracy** — its
estimand is average yield, a regression quantity from a designed survey. That distinction
is the surviving gap.

## 4. Q3 — is the agricultural lab-to-field setting occupied?

**Yes, heavily — but always with labels.** 238 adjacent records, including *From Benchmark
Accuracy to Field Performance* (2026) and *Explainable Transfer Learning for Multi-Crop
Plant Disease Identification Under Real-Field Conditions* (2026). None estimates a fixed
model's field accuracy without labels. Related Work must engage this set.

## 5. PPI — read, not counted

**Power Analysis for PPI (2026)** derives closed-form labelled-sample requirements, with
the saving **scaling as the R² between predictions and ground truth**.

> Our estimators have R² = 0.026 and calibration r ≈ 0, so PPI theory predicts
> **essentially no labelling saving** — exactly what RQ5 measured. RQ5 becomes consistent
> with established theory, with the measured R² as the reason.

**Statistical Optimality of PPI (2026):** PPI is efficient when the predictor is
**score-calibrated** — the condition Phase 7b shows ours fails. **Design-Based PPI for
Spatial Data (2026):** labels arriving by survey design rather than i.i.d., exact design
variances — the photograph-cluster problem; notes coverage can decay as labels accumulate.
**Prediction-Powered Risk Monitoring (2026):** anytime-valid running-risk bounds to detect
harmful shift — the closest published competitor to a deployment-gate framing.

RQ5's crossover is therefore **not** a novel discovery, and its ratio estimator is a
hand-rolled special case. Position it as a practical agricultural measurement *within*
PPI's frame.

## 6. Q4 — Distance-Matters read in full

*Distance Matters For Improving Performance Estimation Under Covariate Shift*
(Roschewitz & Glocker, **ICCV-W 2023**, arXiv:2308.07223) is the nearest neighbour to
Phase 7b. Read via full text, searched specifically for the decisive features.

**What it does.** Softmax confidence becomes unreliable for samples far from the training
distribution, so it adds a **distance check** — K-NN distance in embedding space, threshold
= 99th percentile of the ID validation set — to exclude untrustworthy samples before
estimating. Yields ATC-DistCS and GDE-DistCS. Evaluated on 13 tasks and **259 ImageNet
models across 14 architecture families**, beating ATC and COT.

**What it does not do** — each searched for explicitly in the full text:

| feature | present? |
|---|---|
| decomposition of an estimate into components | **no match** for `decompos*`, `intercept`, `anchor` |
| trivial / constant-predictor baseline | **no match** for `trivial`, `constant baseline` |
| per-architecture breakdown or architecture-dependence claim | **none** — architectures are evaluation *coverage*, never an analysis variable |
| BatchNorm / normalisation discussion | **no match at all** |

**Verdict on the user's question: it identifies confidence miscalibration under shift and
fixes it. It does not decompose an estimator into a source-calibration anchor and a
target-data signal.** These are different failure modes: theirs is *which samples* to
trust; ours is *how much of the estimate comes from the data at all*. Phase 7b is **not
pre-empted** — but Distance-Matters is close enough that it must be cited and distinguished
in exactly these terms.

One honest caveat: ATC's threshold is itself fitted so that source error is matched, which
is an anchor in a loose sense. Nobody found so far analyses it that way.

## 7. Q1 — the nine estimators are a sample, not the state of the art

Stage 2 and the corrected searches surfaced a **foundational thread and two further
communities** that the original pass missed entirely:

| method | year | note |
|---|---|---|
| **Estimating Accuracy from Unlabeled Data** (Platanios et al.) | 2017/2018, 51 cites | **agreement between competing classifiers**; explicitly motivated by train→new-distribution transfer. Predates ATC/DoC. A whole family we do not cover. |
| **Unsupervised Risk Estimation via Conditional Independence** (Steinhardt & Liang) | 2016 | method-of-moments test-error estimation |
| **Aries: Labeling-Free Accuracy Estimation** | 2023, 14 cites | from the **software-testing** community |
| **Label-Free Robustness Estimation of Object Detection CNNs for Autonomous Driving** | IJCV 2021, 23 cites | **applied domain** |
| AutoEval (Deng & Zheng) | 2020/21, 78 cites | regression from dataset feature statistics |
| Projection Norm, Agreement-on-the-Line, GDE, Mandoline, DAPPER, AETTA, IW-GAE, Distance-Matters, ALSA | 2021–2025 | see previous revision |

**At least three research communities work on this problem** — ML/vision, software testing,
and statistics (PPI). Our nine cover the confidence / matrix-norm / optimal-transport
branch of one of them. Per the current decision no estimator #10 is being added; the
**coverage claim is corrected instead.**

### "New application domain" is now established at least three times

| domain | work | year |
|---|---|---|
| mobile sensing | DAPPER | 2021 |
| autonomous driving | Label-Free Robustness Estimation (IJCV) | 2021 |
| medical imaging | arXiv:2507.22776 | 2025 |

Agriculture remains unoccupied. **The genre move is not novel.**

## 8. Q5 — architecture dependence

**Still not found, and the strongest available negative evidence now supports that.**

- The supplementary SQ1 block (8 arXiv conjunctions + 1 title search) returned nothing
  analysing estimator behaviour across architectures.
- **Distance-Matters evaluates 259 models across 14 architecture families and never
  reports architecture dependence** — architectures are averaged over, not analysed. That
  is the largest model pool in any paper found, so its silence is informative.

Two honest limits: OpenAlex S5/S6 have not run, and a negative result from figure-free
full-text search cannot exclude an unlabelled plot. This row is **provisionally open as a
secondary contribution**, to be confirmed after the remaining 123 OpenAlex queries.

## 8b. Measured search recall — the audit now quantifies its own reliability

`scripts/litsearch_audit.py` runs two checks: a **recall test** against a known-answer set
of 18 label-free accuracy-estimation papers confirmed by reading them during this search,
and the protocol's seeded excluded-set sample.

| check | result |
|---|---|
| known-relevant papers **retrieved** by the pre-registered 222 | **14 / 18 (78%)** |
| retrieved papers the screener **kept** | **14 / 14 (100%)** |
| screener misses | **0** |

**The four not retrieved** — Distance-Matters, Mandoline, ALSA, IW-GAE — were found only by
the supplementary pass with correctly-constructed conjunction queries. So the
pre-registered search has a **measured recall of roughly 78%** on a known-answer set.

Two things follow, and both belong in the thesis:

1. The screening rules are now trustworthy: nothing known-relevant that was retrieved got
   thrown away. The two bugs that did throw things away are fixed and this test would catch
   a recurrence.
2. **Retrieval, not screening, is the weak link.** A null result from a search with ~78%
   recall is real evidence but not proof — which is exactly why the supported wording is
   "insufficiently characterised", not "never studied".

One classification imperfection, recorded rather than fixed: **AETTA** and **DAPPER** are
label-free estimation papers but were filed `adjacent: adaptation_not_estimation`, because
their abstracts describe test-time adaptation and mobile-sensing domain adaptation. They
were retained and read, so nothing was lost, but the category is wrong.

The excluded-set sample (15 per reason, seed 20260909) was read and found clean:
agronomy, remote sensing, pose/depth estimation and peer-review comments. No new bug.

## 9. Literature-gap table

| Area | Existing work? | What it does | What is missing |
|---|---|---|---|
| Plant disease lab→field | **Yes** (238) | Measures with labels, or adapts | Assessing it **without** field labels |
| Label-free accuracy estimation | **Yes** (≥20 methods, 3 communities, since 2016) | Estimates target accuracy on unlabelled data | Validation in an **agricultural deployment** |
| Label-free estimation in an applied domain | **Yes ×3** — mobile sensing, autonomous driving, medical imaging | Same genre move | Agriculture specifically |
| GdScore | **Yes** | Gradient-norm estimation | **Transferability to agricultural shift** |
| PPI | **Yes**, incl. **in agriculture** (2025) | Small labelled sample + predictions | Applied to **classifier accuracy** |
| PPI power analysis | **Yes** (2026) | Labels needed scale with R² | Empirical check where R² ≈ 0 |
| Confidence miscalibration under shift | **Yes** (Distance-Matters 2023, IW-GAE 2023) | Diagnoses *which samples* to distrust | Decomposition of the **estimate** into anchor vs data |
| Architecture dependence of estimators | **Not found** | Distance-Matters has 259 models, does not analyse it | **Provisional secondary contribution** |
| Calibration-anchor pathology | **Not found** | ALSA uses "anchor" for a different thing | **Strongest specific candidate** |

## 10. Verdict

**A, with a contribution materially narrower than assumed.** What survives:

1. **Primary.** Label-free **classifier-accuracy** estimation has not been characterised
   under agricultural laboratory-to-field shift. The setting is populated, the genre is
   established elsewhere, but this specific combination is not occupied.
2. **Strongest specific.** The **calibration-anchor pathology** — an estimate dominated by
   its source-fitted intercept rather than target-data signal — has no match in any
   database searched, and the nearest neighbour (Distance-Matters) was read in full and
   diagnoses a different failure mode.
3. **Provisional secondary.** **Architecture dependence** of estimator reliability.

What does **not** survive: firstness of applying label-free estimation to a new domain; the
labelled-budget crossover as a discovery; and any claim that the nine estimators represent
the state of the art.

Supported wording: **"remain insufficiently characterised"** — never "have never been
studied".

## 11. Outstanding

1. **Finish stage 2** — 123 OpenAlex queries (S3–S6) after midnight UTC:
   ```
   python scripts/litsearch.py --databases openalex
   python scripts/litsearch_screen.py
   ```
   Retrieval resumes from the start; S1/S2 records already on disk merge automatically.
2. **Re-run the excluded-set audit** on the merged corpus; it has already caught one bug.
3. **Scopus / Web of Science / IEEE Xplore / ACM DL** — requires institutional access.
   Until then every null is "none found by this search", never "none exists".
