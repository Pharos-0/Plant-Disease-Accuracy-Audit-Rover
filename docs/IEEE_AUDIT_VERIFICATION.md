# IEEE Xplore Audit — Verification and Consequences

**2026-09-11.** The IEEE Xplore search was run by the user on separate hardware and
supplied as four files. Before acting on it I verified it independently, because a search I
did not run is exactly the kind of evidence that should not be taken on trust.

Evidence now in the repository: `results/litsearch/ieee/` — 123 queries, 641 unique records,
full screening with per-record classification.

---

## 1. Is it legitimate? Yes.

### 1.1 DOI verification — the decisive test

Twelve DOIs spanning every claimed class were resolved through the Crossref API on
2026-09-11. **All twelve returned exactly the claimed title and year.**

| DOI | resolved title | year |
|---|---|---|
| 10.1109/ICCV51701.2025.00180 | ODP-Bench: Benchmarking Out-Of-Distribution Performance Prediction | 2025 |
| 10.1109/TBME.2025.3542253 | Unsupervised Accuracy Estimation for Brain–Computer Interfaces… | 2025 |
| 10.1109/TPAMI.2021.3136244 | AutoEval: Are Labels Always Necessary for Classifier Accuracy Evaluation? | 2024 |
| 10.1109/CVPR52733.2024.02706 | AETTA: Label-Free Accuracy Estimation for Test-Time Adaptation | 2024 |
| 10.1109/ICCV51070.2023.01507 | Unsupervised Accuracy Estimation … Adversarial Perturbation | 2023 |
| 10.1109/ICCV51070.2023.01841 | CAME: Contrastive Automated Model Evaluation | 2023 |
| 10.1109/ICSE48619.2023.00152 | Aries: Efficient Testing of DNNs via Labeling-Free Accuracy Estimation | 2023 |
| 10.1109/ICCV48922.2021.00117 | Predicting with Confidence on Unseen Distributions | 2021 |
| 10.1109/CVPR.2013.419 | A Lazy Man's Approach to Benchmarking… | 2013 |
| 10.1109/ICDM.2012.161 | Active Evaluation of Classifiers on Large Datasets | 2012 |
| 10.1109/ACCESS.2022.3169279 | Predicting Classification Accuracy of Unlabeled Datasets… | 2022 |
| 10.1109/IJCNN55064.2022.9892649 | Automatic Model Evaluation using Feature Importance Patterns… | 2022 |

**12/12 resolved. 0 fabricated.**

### 1.2 Independent cross-validation

Seven of the Class-A papers had already been found independently by this project's own
arXiv/OpenAlex/Crossref searches — AutoEval, DoC, AETTA, Lee *et al.* perturbation, Aries,
ODP-Bench and the BCI paper. The IEEE audit reproduces them and adds four that our searches
missed (CAME, You *et al.*, Silva & Veloso, and the two labelled-sample papers).

### 1.3 Methodological signals that it is genuine

- **It argues against the thesis in several places.** A fabricated audit flatters; this one
  concludes "**B — IEEE evidence requires narrowing/modifying the thesis**" and states
  plainly that it "is not a clean bill of novelty."
- **It diagnoses its own coverage limit.** Probing with `"the remainder of this paper is
  organized"` — a phrase in the body of tens of thousands of papers — returned 10 hits,
  proving the search is metadata-scope, not full text. That is a good diagnostic and an
  inconvenient finding to report.
- **It records truncation.** Three broad queries hit a 100-result cap and are flagged.
- **It re-tested all 47 zero-result queries**, which is the guard against the exact bug that
  produced a false null earlier in this project.
- **The false-positive analysis matches ours independently**: "accuracy estimation" in IEEE
  most often means GNSS/radar/registration accuracy, not classifier accuracy.

### 1.4 One flaw found

AutoEval appears twice (IEEE-P001, IEEE-P002) with the same DOI, described as the TPAMI and
CVPR versions. The CVPR 2021 paper has a different DOI, so this is a deduplication
imperfection rather than two records. Handled as a single bibliography entry. It is a minor
defect and, if anything, evidence of genuine imperfect dedup rather than fabrication.

**Verdict: the audit is legitimate and its conclusions are actionable.**

## 2. What it changed in the manuscript

| threat | severity | action taken |
|---|---|---|
| **ODP-Bench** (ICCV 2025) occupies the "systematic evaluation" position | HIGH | Now cited and distinguished explicitly in Related Work: it does not cover agricultural lab-to-field imagery or what an estimate responds to. No first-systematic-evaluation claim anywhere. |
| **BCI** (IEEE TBME 2025) confirms domain transfer is a repeated move | HIGH | Related Work now states that carrying label-free estimation into a new applied domain is established — mobile sensing, TTA, BCI — and that the agricultural setting is not presented as a novelty in itself. |
| **Welinder (CVPR 2013)** and **Katariya (ICDM 2012)** pose the labelled-sample question | MEDIUM | **Contribution (vi) rewritten.** It now reads as a quantification for agricultural lab-to-field shift of a question posed over a decade ago, citing both. Related Work credits them as the origin. |
| Architecture-dependence is contested ground (Distance-Matters) | — | No architecture-novelty claim was ever made; unchanged. |
| **Maize label-free pruning** (IEEE TAFE 2026) invites title confusion | pre-emptive | Cited and distinguished: it is network *pruning*, not accuracy estimation. |

Also updated: every "IEEE Xplore was not searched" statement. IEEE **has** now been
searched, at metadata scope. Scopus has since been searched too (2026-09-18, counts for 14
queries; see `litsearch_scopus.md`). Web of Science and the ACM Digital Library have not.

New bibliography entries, all with Crossref-verified DOIs: `yu2025odpbench`, `peng2023came`,
`hu2023aries`, `lopezgordo2025bci`, `welinder2013lazy`, `katariya2012active`,
`you2022predicting`, `silva2022feature`, `maize2026pruning`. Four existing entries upgraded
from preprint records to verified IEEE DOIs.

## 3. A finding worth carrying into the paper

`"plant disease" AND "distribution shift"` returns **0** in IEEE Xplore, while
`"plant disease" AND "domain shift"` returns **27**. The agricultural literature says
*domain shift*; the machine-learning literature says *distribution shift*. This is a real
terminology split, not a search bug, and any review querying only the ML term will
undercount agricultural work. It is now stated in the manuscript's limitations as practical
guidance for future reviews.

## 4. What survives, and what does not

**Survives.** No IEEE-indexed paper applies label-free classifier-accuracy estimation to
plant disease or agriculture. This rests on converging evidence: seven independent
zero-result queries, plus a set intersection of 219 agricultural × 95
performance-estimation records that yielded two records, both false positives (crop *yield*
estimation; *sensor* accuracy).

**Strongest surviving contribution.** The calibration-anchor / target-contribution
decomposition. No IEEE paper decomposes an estimator into a source-anchor term and a
target-data term, nor reports a trivial-constant baseline. **Caveat carried forward:** a
decomposition of this kind would live in an analysis section, and metadata search cannot
see analysis sections. Full-text reading of the Class-A papers remains outstanding.

**Does not survive.** Any framing resting on being first into a new domain, on the word
"systematic" alone, or on originating the labelled-sample comparison.

## 5. Still outstanding

1. **Full-text reading of the 11 Class-A papers**, specifically to check whether any
   performs an anchor-versus-target decomposition. This is the one check that could still
   affect the lead contribution.
2. **Web of Science and the ACM Digital Library** — not searched, access unavailable; the
   WoS investigation concluded negatively for documented reasons. **Scopus has since been
   searched** — see `litsearch_scopus.md`.
3. The IEEE search is metadata-scope. Every zero means "no paper with this phrase in title,
   abstract or index terms", never "no such paper exists".
