# Scopus Search — Integration Record and Provenance Audit

**2026-09-28.** Scopus was searched on 2026-09-18 from the SRM campus network on a different
machine. The evidence arrived in the `PC-handoff` bundle and is now merged at
`literature/final_audit/`. This document records what the evidence supports, what it does
not, and one provenance discrepancy found while checking it.

This changes the paper. Before this merge both manuscripts stated *"Scopus, Web of Science
and the ACM Digital Library were not searched"*, which was true when written and is now
false.

---

## 1. Access route (reproducible)

Scopus Advanced Search answers **unauthenticated guest** queries from the SRM campus
network. Result URLs can be constructed directly:

```
https://www.scopus.com/results/results.uri?sort=plf-f&src=s&sot=a&sdt=a&origin=searchadvanced&s=<URL-ENCODED QUERY>
```

Counts are read from the page text (`/([\d,]+)\s+documents?\s+found/i`); a zero renders as
*"No documents matching your keywords were found"*. Syntax is `TITLE-ABS-KEY( ... )`, with
`{braces}` for literal hyphens.

**Scope caveat that must travel with every count:** the session was an unauthenticated
guest. Counts and records returned normally, but **no export was performed** and full-text
links were gated. The counts are Scopus's own reported totals, not a downloaded record set.
This is weaker evidence than the IEEE audit, where 641 records were retrieved and screened
individually, and the paper should not imply otherwise.

## 2. Provenance discrepancy found during this check

`literature/final_audit/scopus_urls.json` holds **15** prepared queries: A1, A4, A5, A6, B1,
B2, B3, C1, C2, C3, C4, D1, E1, E2, E3.

`literature/final_audit/SCOPUS_RESULTS.md` reports counts for **14** queries: A1, A2, A3, A4,
A5, B1, B2, B3, C1, C1b, C2, C3, C4, E1.

The two sets are not the same. Four prepared queries (**A6, D1, E2, E3**) have stored URLs
but no reported count. Three reported queries (**A2, A3, C1b**) have counts but no stored
URL. The handoff's summary figure of "15 queries executed" therefore does not match the
executed log.

**Resolution adopted:** the manuscripts claim **"counts recorded for 14 queries"**, which is
what the executed log actually supports. The four prepared-but-unreported queries are
recorded here as **NOT EXECUTED (or executed without the count being recorded)** — and, per
the project's standing rule, are **not** recorded as zeros.

## 3. Results the paper relies on

| ID | Query intent | Hits | Used in the paper |
|---|---|---:|---|
| **A1** | accuracy/performance/model evaluation × unlabeled/label-free × plant disease/crop/leaf/agriculture/PlantVillage/PlantDoc/tomato | **0** | yes — the core agricultural gap |
| **C2** | source × target × accuracy estimation × decompos*/contribut*/anchor | **0** | yes — the decomposition gap |
| **C3** | calibration anchor / source anchor / anchor term × accuracy estimation | **0** | yes — same |
| **C4** | ablation × unsupervised/label-free accuracy estimation | **0** | yes — same |
| C1b | sensitivity analysis/Sobol × accuracy estimation × shift/unlabeled | **2**, both false positives (sepsis PU-learning; EEG domain adaptation) | yes |
| A2 | `"unsupervised accuracy estimation"` | 7, **none agricultural** | yes |
| A3 | `"label-free accuracy estimation"` | 1 (AETTA) | no |
| A4 | automated/automatic model evaluation | 41 | no |
| A5 | OOD performance prediction | 1 (ODP-Bench) | no |
| B1 | PlantVillage × domain/distribution shift/OOD/cross-domain | 51 | yes — established context |
| B2 | plant disease × domain/distribution shift × estimat*/predict*/evaluat* | 34 | no |
| B3 | PlantVillage × PlantDoc | 105 | no |
| C1 | sensitivity analysis/Sobol × accuracy/model/performance estimation | 1,061 | no — dominated by the measurement sense |
| E1 | prediction-powered inference × agriculture/crop/plant disease/image classification | 3 | no |

### The seven `"unsupervised accuracy estimation"` records

GdScore (TMLR 2025) · a BCI application (IEEE TBME 2025) · **GBCE (PAKDD 2025)** · MaNo
(NeurIPS 2024) · domain-adaptive adversarial perturbation (ICCV 2023) · Confidence and
Dispersity (ICML 2023) · self-training ensembles (NeurIPS 2021).

Six of the seven were already in the corpus. The seventh, GBCE, was new and is now
classified — see `docs/fulltext_checks.md`.

## 4. Why this is a net gain for the paper

The gap now has **four independently corroborating indexes** rather than three: IEEE Xplore
(641 screened records), Scopus (A1 = 0), OpenAlex (10 precise queries, all 0) and arXiv (6
queries, all 0). Convergence across different indexes, syntaxes and coverage is
substantially stronger than any single database, and Scopus is the database an IEEE or
Scopus-indexed reviewer is most likely to ask about by name.

It also removes the most easily-raised objection to the manuscript, which was simply that a
major index had not been consulted.

## 5. What remains not searched, and why

**Web of Science — access genuinely unavailable.** The investigation is recorded in
`literature/final_audit/FINAL_NOVELTY_AUDIT.md` §8 and concluded negatively:

- the personal Clarivate account holds only a **free Researcher Profile**; every
  document-search path returns `/wos/error` and the DOM contains no document-search form;
- the campus IP was verified genuine (RDAP `103.4.222.252`, `SRMU-GROUP-IN`, registrant SRM
  University), yet Clarivate returned the sign-in page with `roaming=true`, meaning it does
  **not** recognise that IP as a subscribing address;
- `INFED (Indian Federation)` is present in the institution list, so the route exists:
  *Institutional Sign In → INFED → Go to institution → SRM → SRM credentials.* The discovery
  service became unresponsive and entering institutional credentials was out of scope.

Conclusion: SRM's entitlement is either federated-only, or absent. 17 prepared WoS queries
are ready at `literature/final_audit/WOS_QUERIES.md` should access be obtained.

**ACM Digital Library — Cloudflare bot challenge, not resolved and not bypassed.**

**Methods wording now used in both manuscripts:** databases searched are IEEE Xplore,
Scopus, arXiv, OpenAlex and Crossref; Web of Science and the ACM Digital Library were not
searched, as institutional access was unavailable.

## 6. Three terminology traps, now all in the manuscript

1. **`label-free` is a cross-disciplinary homonym.** All eight OpenAlex records matching
   `"label-free"` with `"plant disease"` are Raman-spectroscopy or biosensor papers, where
   the term means *without a chemical dye*. This is why the paper now leads with
   **unsupervised accuracy estimation** and defines "label-free" once as a synonym.
2. **Agriculture says *domain shift*; machine learning says *distribution shift*.** In IEEE,
   `"plant disease" AND "distribution shift"` returns 0 while `AND "domain shift"` returns
   27. A review querying only the latter undercounts agricultural work.
3. **In IEEE, `"accuracy estimation"` predominantly means *measurement* accuracy** — GNSS,
   radar, indoor positioning, image registration — accounting for 255 of 641 retrieved
   records.

All three are recorded in the limitations section of the full manuscript, and the first two
in the conference version.
