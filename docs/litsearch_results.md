# Systematic Literature Search — Interim Results

**Status: INCOMPLETE. Do not cite the verdict yet.**

Stage 1 (arXiv + Crossref) executed 2026-09-09, all 222 pre-registered queries, zero
failed requests. Stage 2 (OpenAlex) is **pending a daily quota reset** and is the
best-coverage database of the three. Protocol: `docs/litsearch_protocol.md`.

---

## 1. First, a bug that would have fabricated the headline finding

The first attempt at this search produced, in its log, exactly the result the thesis
wanted:

```
[68/222] S2 unsupervised model evaluation AND plant disease    oa=0
[69/222] S2 unsupervised model evaluation AND crop disease     oa=0
... 112 of 163 queries reporting oa=0
```

Read literally: *no published work applies label-free accuracy estimation to plant
disease*. That is the load-bearing claim of the whole thesis direction.

**It was false.** Those were not zero results — they were HTTP 429s. OpenAlex allows
1,000 requests/day, and `search_openalex()` returned a count of `0` whenever the request
failed. Worse, the retry logic hit each 429 four more times, so once the quota ran out
every remaining query burned four further requests against a budget of zero.

It was caught because `plant disease AND domain shift` also returned zero, which is not
credible — there is a large agricultural domain-shift literature, and this search has now
found 147 examples of it.

Fixed in `c8c9313`: every backend returns `(records, count_or_None, status)`; a failed
request yields `None` and a status string, never `0`; a quota refusal aborts the run
rather than being retried; per-query status is recorded and summarised; and if nothing
succeeds the script writes nothing at all rather than an empty "result". **No results from
the bad run were ever written** — it was killed before its write phase.

Recorded here because it is the most dangerous class of error this project can make, and
because the fix is what makes the eventual null claim trustworthy: every zero in the
corpus below is now verifiably a real zero.

## 2. What has been searched

| database | status | queries | failures |
|---|---|---:|---:|
| **arXiv** | complete | 174 (48 not run by design) | **0** |
| **Crossref** | complete | 222 | **0** |
| **OpenAlex** | **pending quota reset** | 0 of 222 | — |
| Scopus, Web of Science, IEEE Xplore, ACM DL | **no access** | — | — |
| DBLP, Semantic Scholar | refused automated access | — | — |

Corpus so far: **22,340 records identified → 8,617 unique works → 329 included, 162
adjacent, 8,126 excluded.**

Two caveats that bound everything below:

- **Crossref's `query.bibliographic` is a relevance ranking, not a boolean filter.** It
  ignores the `AND` and reports totals in the millions. Its totals are never quoted as
  counts. Its records are screened like any other, which is why it contributes many
  records and almost no included ones.
- **73% of retrieved records arrive with no abstract** (Crossref deposits them sparsely).
  Those are screened on **title alone**, which has lower recall. OpenAlex has far better
  abstract coverage, so stage 2 will materially improve this.

**arXiv counts, by contrast, are genuine boolean phrase searches over abstracts**, and are
the most trustworthy numbers here.

## 3. Q2 — the load-bearing question

> Has any prior work applied label-free accuracy estimation to plant-disease or
> agricultural imagery?

**Nothing found so far.**

| search | queries | arXiv hits |
|---|---:|---:|
| **S2** task × agriculture | 88 | **1** |
| **S3** task × agriculture × shift | 60 | **0** |

The single S2 hit was read in full: *"Automatic Detection of Rice Disease in Images of
Various Leaf Sizes"* (Kiratiratanapruk et al., 2022). It estimates **rice leaf width** to
size image tiles. Not accuracy estimation. **Rejected.**

Screening flagged two records as "task AND agriculture". Both were read and both are
regex false positives:

| record | why it matched | verdict |
|---|---|---|
| *Design-Based Prediction-Powered Inference for Spatial Data* (2026) | "prediction-powered inference" + spatial/crop vocabulary | statistical methodology, not plant disease — **but see section 5** |
| *A double copy from twisted (co)homology at genus one* (2023) | `MaNo` matched, and "leaf" in the **foliation-theory** sense | string-theory paper. **Rejected.** |

**Provisional and not yet quotable:** on arXiv and Crossref, no work applies dataset-level
label-free accuracy estimation to plant disease or agricultural imagery. This must be
re-checked once OpenAlex runs.

## 4. Q3 — is the agricultural lab-to-field setting itself occupied?

**Yes, heavily — but always *with* labels.** 147 records were classified `adjacent:
agricultural_shift_no_estimation`, including:

- *From Benchmark Accuracy to Field Performance: Hybrid Deep Learning-Based Plant Disease
  Classification* (2026)
- *Explainable Transfer Learning for Multi-Crop Plant Disease Identification Under
  Real-Field Conditions* (2026)
- *Enhancing Generalization of Tomato Leaf Disease Classification via TDR Model and
  Field-Condition* (2026)
- *A multi-scale supervised contrastive framework for cross-domain soybean disease
  classification* (2026)

This is the distinction the thesis rests on, and the corpus supports it: the agricultural
community studies lab→field shift constantly, by **measuring accuracy with labels** or by
**adapting the model**. None of the 147 estimates a fixed model's field accuracy without
labels.

That is a real gap — but it also means Related Work must engage this set seriously rather
than claim the setting is unstudied.

## 5. The substantive finding: an entire adjacent family was missed

**Prediction-powered inference (PPI)** — 86 arXiv hits, 87 records in the corpus, and an
active 2024–2026 literature.

PPI combines a **small gold-standard labelled sample** with **model predictions on
unlabelled data** to produce estimates and confidence intervals that are valid regardless
of how good the model is. That is a formal, published treatment of *precisely* the RQ5
setting, and RQ5's ratio estimator over photograph clusters is a hand-rolled special case
of it.

Directly relevant titles already in the corpus:

- *Power Analysis for Prediction-Powered Inference* (2026) — **this is the RQ5 question**:
  how many labels do you need.
- *Prediction-Powered Risk Monitoring of Deployed Models for Detecting Harmful
  Distribution Shifts* (2026)
- *Design-Based Prediction-Powered Inference for Spatial Data* (2026) — handles labels
  arriving through a **survey design** rather than i.i.d., which is exactly the
  photograph-cluster problem.
- *Statistical Optimality of Prediction-Powered Inference* (2026)

**Consequence for the thesis, stated plainly:** RQ5's crossover result cannot be presented
as a novel discovery. There is an established statistical framework for combining a small
labelled sample with model predictions, and the honest position is to cite PPI, situate
RQ5 within it, and be explicit that RQ5 measures the crossover empirically in an
agricultural deployment setting rather than deriving it.

This also sharpens the earlier objection — *"isn't that universal, smaller labelled will
give better results, how is this novel"*. The answer is now evidenced: correct, it is not
novel on its own, and there is a literature that formalises it.

## 6. Q4 — is the calibration-anchor pathology known?

**Nothing found.** All four S6 queries returned 0 on arXiv, and no included record was
found by S6.

This is **weak** evidence: four queries, one database, and the phrasing of a pathology is
hard to guess in advance. It is consistent with Phase 7b being a genuine contribution, and
is not sufficient to claim priority. Re-check after OpenAlex, and this is one of the
questions most in need of the subscription databases.

## 7. Q1 — are the nine estimators representative?

The named-method sweep (S5) found the expected canon, and no unrepresented major
*accuracy-estimation* family:

| method | arXiv hits | in our nine |
|---|---:|---|
| prediction-powered inference | 86 | **no — see section 5** |
| projection norm | 36 | no |
| agreement-on-the-line | 31 | no |
| difference of confidences | 25 | yes (DoC) |
| AutoEval | 15 | no |
| average thresholded confidence | 4 | yes (ATC) |
| confidence optimal transport | 2 | yes (COT) |
| GdScore | 1 | yes |
| MaNo | 134 → **3 genuine** | yes |

Two screening artefacts worth recording: `MaNo` matches the **MANO parametric hand model**
(ubiquitous in 3D vision) and Spanish *"la mano"*; only 3 of 134 hits are the estimator.

So the nine cover the confidence/matrix-norm/optimal-transport families, and the notable
absences are **projection norm**, **agreement-on-the-line**, and **AutoEval** — all
label-free accuracy estimators we did not evaluate. That is a limitation to state, not
necessarily a gap to fill: per the current decision, no estimator #10 is being added.

## 8. Provisional verdict — A or B?

**Leaning A, with the contribution narrower than hoped, and not yet final.**

- The *application* gap looks real: no label-free accuracy estimation in agriculture found
  so far, in either database, across 148 task × domain queries.
- The *setting* is well populated (147 agricultural lab→field works), so novelty cannot
  rest on "nobody studies lab-to-field in agriculture".
- The *labelled-budget* result is **not** novel as a phenomenon — PPI formalises it.
- The *calibration-anchor* finding has no match so far, and is currently the most
  defensible specific contribution.

**No title or novelty claim is frozen.** Section 9 must complete first.

## 9. Outstanding before any claim is made

1. **Run stage 2** once the OpenAlex quota resets (midnight UTC):
   ```
   python scripts/litsearch.py --databases openalex
   python scripts/litsearch_screen.py
   ```
   Screening merges the stages automatically.
2. **Read the PPI cluster properly**, especially *Power Analysis for Prediction-Powered
   Inference* and *Design-Based PPI for Spatial Data*, and position RQ5 against them.
3. **Scopus / Web of Science / IEEE Xplore / ACM DL** with these same Boolean strings.
   Requires institutional access; cannot be done from here. Until then, every null is
   "none found by this search", never "none exists".
4. **Sample the excluded set by hand** to confirm the screening rules are not discarding
   relevant work — required by protocol section 8 and not yet done.
