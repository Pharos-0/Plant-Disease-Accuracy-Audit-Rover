# Literature Search for Paper 2: Results

**Searched 2026-10-13** under the protocol in `docs/litsearch_paper2_protocol.md`, which was
committed before any query ran. Raw data: `results/litsearch_paper2/`. Scripts:
`scripts/litsearch_paper2.py` (retrieval), `scripts/litsearch_paper2_screen.py` (screening
record), `scripts/litsearch_paper2_supplement.py` (post-hoc pass, §3).

---

## Verdict on the hypothesis

> **H** — letting an unvalidated disease classifier, or its confidence, *withhold* spraying under
> laboratory-to-field domain shift creates a treatment-reliability problem that existing
> selective-spraying approaches have not adequately evaluated.

**Outcome: H is partly occupied.** The closest work already establishes four things Paper 2 must
credit and not claim:

1. **Disease detectors feeding a sprayer collapse from lab to field.** Polder et al. 2025
   (*J. Field Robotics*; full text read) evaluated a disease detector whose output drives a
   decision-support system and spray prescription maps. F1 fell from 66.3% to 34.8% (grape downy
   mildew), 45.1% to 5.5% (apple scab) and 42.1% to 4.2% (carrot Alternaria) between closed-set
   and open-set field images. The authors explicitly expect PlantVillage-trained models to degrade
   in the field. A weed-sprayer prototype (*J. ASABE* 2026) went from a 100% weed hit rate indoors
   to 74.6% in the field. A rice-disease advisory system (Han & Zhou 2025) fell from about 80% to
   about 13% accuracy on an external dataset, and its unknown-input detectors scored AUROC 0.512 and
   0.486.
2. **Uncertainty should not reduce treatment — for weeds, with a guarantee.** Melki et al. 2025
   (arXiv; full text read) built a weed-spraying pipeline on conformal prediction. Its lenient rule
   ("spray whenever *weed* is in the prediction set") is the only one that guarantees treating at
   least 90% of weeds; the strict rule (spray only on a confident singleton) covered about 70%.
   Tested on new farms ("near out-of-distribution", no field test), the guarantee held for most
   farms but fell well short on some, which the authors attribute to visual differences from the
   calibration data.
3. **Reporting treatment success beside chemical saved, and the resolution trade-off, are
   established** in weed spraying. Avent et al. 2025 measured treatment success of a commercial
   targeted sprayer per sensitivity setting: 98–99.6% at the most sensitive setting, falling to
   48–84% at the least, by weed class. Ruigrok et al. 2020 evaluated detection at the level of
   spray decisions. Darbyshire et al. 2023 defined weed coverage rate and area sprayed. Berge et
   al. 2008 and Petelewicz et al. 2024 simulated how spraying resolution trades errors against
   reduction.
4. **Formal safety assurance of agricultural treatment robots exists.** Probabilistic model
   checking and runtime-verification monitors have been applied to UV-light treatment robots (IEEE
   CASE 2023, 2024) and to human injury risk with agricultural collaborative robots (*Computers
   and Electronics in Agriculture* 2024).

**What remains (in OpenAlex, Crossref and arXiv, searched 2026-10-13):** no study found

- evaluates a sprayer in which a **disease** classifier's call may **withhold** treatment,
  **under severe laboratory-to-field shift**, with **missed treatment** as the outcome, against a
  policy that sprays all detected plant tissue on the same frames;
- uses **label-free (unsupervised) accuracy estimation as a sprayer's deployment gate** and
  measures how it behaves when field accuracy falls;
- combines either of these with a real-imagery evaluation of the resulting reduction in
  estimated pesticide use.

This is a **narrower** gap than Paper 2 originally implied, and the paper now says so.

## 1. Retrieval

| Database | Queries | Records retrieved | Failed requests |
|---|---:|---:|---:|
| OpenAlex | 34 | 5,337 | 0 |
| Crossref | 34 | 3,400 | 0 |
| arXiv | 34 | 13 | 0 |

8,750 records reduce to **4,407 unique works** (by DOI, then by normalised title). **946** pass
the automatic first rule (§7.1 of the protocol). Every query is logged verbatim in
`query_log.csv`. The per-query `raw_records.csv` files (12.7 MB) are kept locally and not
committed; `unique_records.csv` holds every retrieved work after deduplication.

## 2. Recall check

The pre-registered queries retrieved **1 of 3** known-answer papers: Oberti et al. 2016 (disease
spraying) was found, but Gonzalez-de-Soto et al. 2016 and Partel et al. 2019 (both **weed**
sprayers) were not. The cause is structural: every query family required a disease or reliability
term, and those two papers contain neither. **Recall on weed-only sprayers is therefore limited**,
and §3 was added to cover the part of that literature that bears on H.

## 3. Supplementary pass — post hoc, not pre-registered

Added after the recall check failed, and reported separately. Six queries of the form
`weed AND (targeted spraying OR spot spraying OR ...) AND <reliability term>` were run on OpenAlex
and arXiv, plus two plain-word probes for the missed papers on Crossref.

- OpenAlex: 6/6 queries succeeded. **arXiv: 5 of 6 requests failed** (HTTP 429 or timeout) and
  were not retried, so arXiv coverage for this pass is weak.
- The two missed papers **were** retrievable with plain words, which confirms the miss was a
  query-design issue.
- 212 records were new to the main corpus. They were screened for the pass's single question:
  does a targeted-spraying study address classifier uncertainty, confidence or shift? Relevant
  finds: the *J. ASABE* 2026 smart-sprayer study (field hit rate 74.6% against 100% indoors); a
  2026 arXiv study of weed-detection transfer under cross-field shift; a 2023 ICCV-workshop
  Bayesian crop/weed segmentation with uncertainty; and a 2025 study of spatio-temporal stability
  of weed detection in tomato fields. Melki et al. 2025 reappeared. **None evaluates classifier-
  gated withholding of treatment against a spatial-only policy, or a label-free accuracy gate.**

## 4. Screening

| Code | Meaning | Records |
|---|---|---:|
| I1-disease | a disease or pest detector decides whether or where to treat | 83 |
| I1-target | a detector decides where to spray weeds, crop or canopy | 96 |
| I1-context | reviews, decision-support and treatment-recommendation systems | 24 |
| I2 | uncertainty, abstention or shift tied to a treatment decision | 6 |
| I3 | safety assurance, verification, supervision or simulation | 13 |
| E1 | detection/classification without a treatment decision; general reviews | 581 |
| E2 | spray physics only | 38 |
| E3 | another sense of the terms (malaria spraying, coatings, medicine) | 96 |
| E4 | not scholarly | 9 |

**222 included of 946.** Each include/exclude decision was made by reading the title and abstract
(the title alone for records without an abstract). **For excluded records the specific E-code was
then assigned by a keyword rule** (`scripts/litsearch_paper2_screen.py`) and approximates the
reason; the decision to exclude is the manual one.

**A pattern worth reporting.** The 83 disease-driven systems are dominated by detect-then-spray
prototypes that report classifier accuracy on held-out images from the training distribution and,
where a sprayer is built, chemical reduction. In the abstracts read, none of the disease-sprayer
prototypes outside the closest set evaluated the classifier on field images from a different
distribution, and none reported what the sprayer does when the classifier is wrong or uncertain.
This is an observation from abstracts, not a full-text audit.

## 5. Closest studies

What was read is recorded for each: **FT** = full text, **A** = abstract only.

| Study | Read | Sensing → decision | Uncertainty / shift handling | Evaluation | Outcomes | Shift evaluated |
|---|---|---|---|---|---|---|
| Melki et al. 2025, arXiv 2501.07185 | FT | RGB → ResNet-50 classifier → conformal sets → spray rule (weeds) | Conformal prediction; lenient rule sprays when weed is plausible | Image-level, old and new farms; no field test | ≥90% weed coverage guaranteed with lenient rule; ~70% with strict | Near-OOD: guarantee fails on some farms |
| Polder et al. 2025, *J. Field Robot.* | FT | RGB smart camera → CNN detector → DSS → prescription map → sprayer | None in the spray decision; open-set evaluation | Commercial vineyards, orchards, fields; 31 h hardware run | F1 drop closed→open set (above) | **Yes**, lab-condition → field |
| Han & Zhou 2025, *JTIE* | A | Image → classifier → advisory with gates | Open-set rejection, quality triage, abstain | Grouped test + external dataset | ~80% → ~13% accuracy external; unknown-detection AUROC ~0.5 | **Yes**, external dataset |
| Oberti et al. 2016 (2013 conf.) | A | Multispectral sensor on robot arm → spray symptomatic canopy (powdery mildew) | — | First experiments | — (not in abstract) | No |
| Tewari et al. 2020, *AI in Agriculture* | A | Webcams → colour segmentation → severity → spray time (paddy) | — | Field, variable vs constant rate | ≥33.9% less chemical | No |
| Anand et al. 2025, *Front. Agric. Sci. Eng.* | A | Spectral sensor → SVM (black rot) → targeted spraying | — | Cauliflower crop | 72.5% less chemical | No |
| Sharma & Raheman 2025, *Cogent Food Agric.* | A | YOLOv3 + CNN (tomato healthy / early / late blight) → variable-rate spray | — | Laboratory trolley | 83.3% dose accuracy; 92.8% validation accuracy | No |
| Chang et al. 2023, *Agriculture* | A | DL detection of weeds and unhealthy crops → spray | Robust navigation under varying light | Field robot | 83% effective weed coverage, 53% reduction | Light varied for navigation only |
| Avent et al. 2025, *Weed Technol.* | A | Commercial targeted sprayer (sensitivity settings) | Sensitivity setting trades coverage for savings | Multi-state field trials | Treatment success 98–99.6% → 48–84% by setting | Across weed species and sizes |
| Ruigrok et al. 2020, *Sensors* | A | Weed detector → plant-specific spraying robot | — | Image-, application- and field-level | 96% weeds controlled, 3% crop hit | No |
| Darbyshire et al. 2023, *Front. Plant Sci.* | A | Detectors → simplified spray model | — | Two datasets, nozzle precision varied | Weed coverage rate vs area sprayed | No |
| Berge et al. 2008, *Acta Agric. Scand. B* | A | Simulated camera decisions → patch spraying | — | Simulation on mapped weeds | Mapping vs spraying error vs control-area size | No |
| Adam et al. 2024; 2023, IEEE CASE | A | UV-C treatment robot | Model checking → runtime monitors (ROS2) | Design and verification | Verified safety requirements | — |

## 6. What changes in Paper 2

- **Related work rewritten** around the four occupied points, citing each study above.
- **The design rule is credited**: "uncertainty must not reduce treatment" is presented as the
  disease-classification counterpart of Melki et al.'s weed rule, tested under a severe shift where
  a calibration-based guarantee cannot be assumed — not as a new principle.
- **Coverage-beside-reduction and the nozzle-resolution trade-off are presented as established
  practice** that Paper 2 follows, not as contributions.
- **The safety work is presented as engineering verification** of a lightweight state machine;
  formal safety assurance of agricultural robots is cited as the stronger existing approach.
- **The contribution is narrowed** to the gap above, scoped to the databases and date searched.

## 7. Limits of this search

Scopus, Web of Science, IEEE Xplore's native search and the ACM Digital Library were not
searchable from here. Crossref ranks by relevance and does not support boolean queries. The
pre-registered queries missed weed-only sprayers (recall 1/3), and the supplement's arXiv coverage
is weak (5 of 6 requests failed). Only two of the closest studies were read in full; the rest are
characterised from abstracts and are labelled so.
