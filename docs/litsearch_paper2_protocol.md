# Literature Search for Paper 2: Protocol

**Written and committed 2026-10-13, before any query in this search was run.** The
hypothesis, databases, search strings, inclusion rules and screening steps are fixed here so
that what turns up cannot reshape them. Results go in `docs/litsearch_paper2_results.md`.

---

## 1. What is being tested

Paper 2 must not assume that nobody has studied its problem. The search tests a **narrow
hypothesis**, not a novelty claim:

> **H.** Using an unvalidated disease classifier, or its confidence, to *withhold* spraying under
> laboratory-to-field domain shift creates a treatment-reliability problem that existing
> selective-spraying approaches have not adequately evaluated.

**Allowed outcomes.**

- **H holds** in the sources searched. Paper 2 may then say so, scoped to those sources and this
  search date.
- **H is partly occupied.** Some study evaluates classifier-gated treatment, uncertainty-aware
  spraying, or field-shift reliability of a sprayer's classifier. Paper 2's claims are then
  narrowed to whatever that study leaves open, and the study is discussed by name.
- **H is occupied.** Paper 2 is reframed as a replication or extension, and says so.

## 2. Questions

| # | Question |
|---|---|
| Q1 | Which selective or targeted spraying systems let a disease classifier (or its confidence) decide whether or where to treat? What sensing, policy, evaluation setting and outcomes? |
| Q2 | Does any spraying study evaluate its classifier's reliability under domain shift, or use uncertainty, confidence thresholds or abstention to decide treatment? |
| Q3 | Outside spraying: is uncertainty-aware, abstaining or reject-option plant-disease classification studied *as a basis for a treatment decision*? |
| Q4 | Are agricultural sprayer or robot controllers verified for safety (state machines, formal methods, runtime monitoring, digital twins, fault injection)? |
| Q5 | Do selective-spraying evaluations report coverage of targets (missed treatment) alongside chemical reduction? |

## 3. Databases

| Database | Access | Query form |
|---|---|---|
| OpenAlex | public API, no key | `search=` on title, abstract and full text, with boolean AND/OR and quoted phrases |
| Crossref | public API | `query.bibliographic`, relevance-ranked, no boolean; top 100 per query |
| arXiv | public API | `search_query` on abstract, boolean, quoted phrases |

**Not searchable from here:** Scopus, Web of Science, IEEE Xplore's native search and the ACM
Digital Library need institutional access. Any null result is therefore "none found in OpenAlex,
Crossref and arXiv on this date", never "none exists". No personal details are sent with any
request.

## 4. Search strings

Blocks:

- **S, spraying:** `selective spraying`, `targeted spraying`, `spot spraying`,
  `precision spraying`, `site-specific spraying`, `variable rate spraying`, `smart sprayer`,
  `robotic sprayer`, `spraying robot`, `patch spraying`
- **D, disease:** `plant disease`, `disease detection`, `leaf disease`, `crop disease`
- **R, reliability:** `uncertainty`, `confidence`, `domain shift`, `distribution shift`,
  `out-of-distribution`, `abstention`, `reject option`

Query families, run verbatim against every database (Crossref receives the same words without
boolean operators):

| Family | Pattern | Queries | Answers |
|---|---|---:|---|
| F1 | each S term AND (`plant disease` OR `disease detection` OR `leaf disease` OR `crop disease`) | 10 | Q1 |
| F2 | each S term AND (`uncertainty` OR `confidence` OR `domain shift` OR `out-of-distribution`) | 10 | Q2 |
| F3 | `plant disease` AND (each of `uncertainty`, `abstention`, `reject option`, `out-of-distribution`, `domain shift`) AND (`spraying` OR `pesticide` OR `treatment`) | 5 | Q3 |
| F4 | (`agricultural robot` OR `spraying robot` OR `agricultural sprayer`) AND (each of `finite state machine`, `formal verification`, `runtime monitoring`, `digital twin`, `fault injection`, `safety`) | 6 | Q4 |
| F5 | (`selective spraying` OR `spot spraying` OR `targeted spraying`) AND (each of `hit rate`, `missed`, `spray coverage`) | 3 | Q5 |

That is 34 queries per database. OpenAlex returns up to 200 records per query, Crossref 100 and
arXiv 100.

## 5. Recall check

Three studies already known to be relevant are a **known-answer set**. The search must retrieve
them, or its recall is reported as limited:

- Oberti et al. 2016, selective spraying of grapevines for disease control (`10.1016/j.biosystemseng.2015.12.004`)
- Gonzalez-de-Soto et al. 2016, a robotised patch sprayer (`10.1016/j.biosystemseng.2015.12.018`)
- Partel et al. 2019, low-cost AI precision weed management (`10.1016/j.compag.2018.12.048`)

## 6. Inclusion and exclusion

**Include** a record if it is a scholarly work and any of these holds:

- **I1.** A spraying or treatment system (ground robot, boom, UAV, or decision system) in which an
  image-based classifier's or detector's output decides whether or where treatment is applied.
- **I2.** A study of classifier uncertainty, confidence, abstention or domain shift explicitly tied
  to a spraying or treatment decision.
- **I3.** Safety verification of an agricultural robot or sprayer controller.

**Exclude:**

- E1: classification or detection with no treatment decision;
- E2: nozzle hydraulics, drift or deposition physics only;
- E3: a different sense of the term (for example "confidence interval" of yield);
- E4: not scholarly (news, catalogues);
- E5: not in English.

## 7. Screening

1. **Automatic first pass.** A record goes on if its title or abstract contains a spraying or
   treatment term and either a vision or classifier term (for I1 and I2) or a safety or
   verification term (for I3). The rule is recorded beside every record.
2. **Manual pass.** Every survivor's title and abstract is read and marked include or exclude
   with the criterion code. Records without an abstract are judged on the title, and marked so.
3. **Closest studies.** Included records that could overlap H are read further: the full text
   where it is open access, otherwise the abstract. **What was read is recorded for each.**
4. **Comparison table** for the closest studies: sensing; decision policy (what decides
   treatment, whether confidence is used, what happens when the classifier is uncertain);
   evaluation setting (laboratory, simulation, greenhouse or field; whether chemical was
   actually applied); outcomes measured (chemical reduction, coverage or missed targets);
   whether domain shift was evaluated; stated limitations.

## 8. What the paper may say afterwards

- **H holds:** "In OpenAlex, Crossref and arXiv, searched on 2026-10-13, we found no selective-
  spraying study that evaluates the reliability of classifier-gated treatment under laboratory-
  to-field shift", together with the closest studies and how they differ.
- **H partly or fully occupied:** claims narrowed to what remains, the overlapping studies cited
  and discussed, and the change recorded in the results document.

Never: "first", "novel" or "no prior work" without that scope.
