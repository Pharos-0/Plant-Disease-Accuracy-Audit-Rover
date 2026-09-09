# Systematic Literature Search — Protocol

**Written and committed before the search runs.** Queries, databases, inclusion and
exclusion criteria and screening rules are fixed in advance so the result cannot be shaped
by what turns up. Supersedes the five ad-hoc web searches in `docs/novelty_verification.md`
(2026-09-06), which that document itself labels "not a systematic review".

**Search executed:** 2026-09-09. Coverage is as indexed on that date.

---

## 1. Why this search happens now

The thesis has enough experimental material (RQ1–RQ5, Phase 7, Phase 7b). What it does not
yet have is a defensible statement of where it sits in the literature. The search must
decide between:

- **A.** Label-free accuracy estimation under agricultural laboratory-to-field shift is
  genuinely underexplored, and this thesis is a first systematic treatment of it; or
- **B.** It is an existing benchmark reproduced on a different dataset, in which case the
  contribution must be narrowed to what survives — most likely the Phase-7b
  calibration-anchor diagnostic rather than the application.

**No title or novelty claim is frozen before this completes.** The outcome is allowed to be
B; that is the point of pre-registering.

## 2. Questions

| # | Question | Decides |
|---|---|---|
| **Q1** | What label-free accuracy-estimation methods exist? | whether our nine are representative, or a major family is missing |
| **Q2** | Has any prior work applied label-free accuracy estimation to plant-disease or agricultural imagery? | the load-bearing claim; A vs B |
| **Q3** | Has any prior work studied laboratory-to-field shift *for accuracy estimation* rather than for classification accuracy? | whether the shift setting is itself occupied |
| **Q4** | Has the calibration-anchor pathology (an estimate dominated by the source-fitted intercept rather than the target signal) been described before? | whether Phase 7b is a contribution or a rediscovery |

## 3. Databases

**Searched programmatically**, with every query string recorded verbatim and re-runnable:

| database | why | access |
|---|---|---|
| **OpenAlex** | ~250M works; indexes IEEE, ACM, Springer, Elsevier, MDPI metadata | open API |
| **Crossref** | publisher-deposited DOI metadata; independent index for cross-checking | open API |
| **arXiv** | preprints; this literature moves through arXiv first | open API |

**Not searched, and this is a real limitation.** Scopus and Web of Science need an
institutional subscription; the IEEE Xplore and ACM DL native APIs need keys. Probed on
2026-09-09 and refused (HTTP 401/403). OpenAlex and Crossref index the *metadata* of IEEE
and ACM publications, so those venues are covered at metadata level — but that is **not
equivalent** to a native search of those platforms, which also index full text and apply
their own thesauri.

> **This search therefore cannot support a claim of exhaustive coverage.** Confirming it in
> Scopus / Web of Science / IEEE Xplore / ACM DL with these same Boolean strings remains a
> task requiring institutional access, and is recorded as outstanding.

DBLP was probed and refused automated access (bot challenge). Semantic Scholar was probed
and rate-limited; it is used opportunistically if it responds, and its absence is recorded
rather than worked around.

## 4. Concept blocks

Searches are cross-products of these blocks. Strings are fixed here and not edited after
seeing results.

**Block T — the task (label-free performance estimation):**
`"unsupervised accuracy estimation"`, `"label-free accuracy estimation"`,
`"automatic model evaluation"`, `"AutoEval"`, `"predicting model performance"`,
`"performance estimation without labels"`, `"estimating classifier accuracy"`,
`"unsupervised model evaluation"`, `"out-of-distribution performance prediction"`,
`"accuracy prediction unlabeled"`, `"unsupervised risk estimation"`

**Block D — the domain (agriculture):**
`"plant disease"`, `"crop disease"`, `"leaf disease"`, `"PlantVillage"`,
`"precision agriculture"`, `"agricultural imagery"`, `"agricultural vision"`,
`"crop monitoring"`

**Block S — the shift:**
`"distribution shift"`, `"domain shift"`, `"laboratory to field"`, `"lab-to-field"`,
`"field conditions"`, `"out-of-distribution"`

**Block M — named methods (for the Q1 inventory):**
`"average thresholded confidence"`, `"difference of confidences"`,
`"confidence optimal transport"`, `"MaNo"`, `"GdScore"`, `"nuclear norm" accuracy`,
`"agreement-on-the-line"`, `"projection norm"`, `"Frechet distance" accuracy estimation`,
`"prediction-powered inference"`, `"confidence-based performance estimation"`

**Block C — the Phase-7b pathology (Q4):**
`"calibration" "accuracy estimation" artifact`, `"intercept" "accuracy estimation"`,
`"source accuracy" baseline "accuracy estimation"`,
`"trivial baseline" performance estimation`

## 5. Searches to run

| # | Composition | Answers |
|---|---|---|
| S1 | T alone | Q1 — field size, venues, years |
| S2 | T × D | **Q2 — the load-bearing question** |
| S3 | T × S × D | Q3 |
| S4 | D × S | context: who studies agricultural lab→field at all, with labels |
| S5 | M, each term alone | Q1 — method inventory completeness |
| S6 | C | Q4 — is the anchor pathology known |

## 6. Inclusion criteria

A record is **included** if it:

1. is a research article, conference paper, or preprint; **and**
2. is in English; **and**
3. either (a) proposes or evaluates a method for estimating a trained model's accuracy or
   performance on an **unlabelled** target set, or (b) applies such a method in an
   agricultural or plant-disease setting.

## 7. Exclusion criteria

**Excluded**, with the reason recorded per record:

- **wrong sense of "accuracy"** — positional/GPS accuracy, sensor measurement accuracy,
  yield estimation, localisation error. This is the dominant false-positive class and is
  screened explicitly.
- **uses target labels** — ordinary supervised evaluation, cross-validation, active
  learning with an annotation loop.
- **domain adaptation or self-supervision without estimation** — changing the model to fit
  the target is a different task from estimating a fixed model's accuracy on it. This
  distinction is the single most important screening rule for Q2, because it is where the
  agricultural literature overwhelmingly sits.
- **per-sample selective prediction / abstention / conformal prediction** — a per-image
  correctness decision, not a dataset-level accuracy estimate. Recorded separately as
  adjacent work rather than silently dropped.
- **not English**, or **no retrievable abstract** to screen on.

## 8. Screening

1. **Deduplicate** across the three databases by DOI, then by normalised title.
2. **Title and abstract screening** against sections 6–7, one recorded decision and reason
   per record.
3. **Adjacent-work set:** records excluded under the selective-prediction or
   domain-adaptation rules are kept in a separate table, because Related Work must
   distinguish them explicitly rather than pretend they do not exist.
4. **PRISMA-style counts** reported: identified → deduplicated → screened → included, with
   exclusion reasons tallied.

Screening is by title and abstract only. **No full texts are read in this pass**, and no
verdict here rests on a claim about a paper's internals.

## 9. Interpretation discipline

Binding on the write-up:

- A null result — no agricultural label-free accuracy-estimation work found — is
  **"none found by this search"**, never "none exists". Section 3 states why the search
  cannot be exhaustive.
- The word "first" is not to be used on the strength of this search alone.
- Counts are reported as returned. Where a query returns an implausible count it is
  reported with the caveat, not silently re-tuned.
- If the outcome is **B**, the contribution is narrowed in the results document rather than
  the search being re-run with different terms.

## 10. Outputs

```
results/litsearch/raw_<database>.csv     every record retrieved, unscreened
results/litsearch/deduplicated.csv       after DOI/title merge
results/litsearch/screened.csv           decision + reason per record
results/litsearch/prisma_counts.csv      the flow numbers
results/litsearch/query_log.csv          every query string, verbatim, with counts
docs/litsearch_results.md                findings and the A/B verdict
```

Re-runnable as `python scripts/litsearch.py`, with a manifest recording the run date,
database endpoints and per-query counts.
