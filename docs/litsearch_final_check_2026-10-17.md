# Final literature check — IEEE Xplore, Scopus, Google Scholar

**2026-10-17.** Run directly in the browser on the three databases that both prior searches
recorded as **not reachable** — `docs/litsearch_paper2_results.md` §7 and
`results/litsearch/TRIGGER_SATURATION_NOVELTY.md` both state IEEE Xplore, Scopus and Web of
Science were not searched. This check closes that gap for the project's **two core novelty
claims**. It does not re-open the trigger-saturation verdict (that was decided on prior art
*found*, which this check does not disturb).

## Claims under test

- **Thesis.** Label-free (unsupervised) **accuracy estimation** of a plant-disease classifier
  under laboratory→field shift, with a **source-vs-target variance decomposition** as the core
  contribution.
- **Paper 2.** A sprayer in which a **disease** classifier's call may **withhold** treatment
  under severe lab→field shift — missed treatment as the outcome, measured against a policy that
  sprays all detected plant tissue — and/or a **label-free accuracy estimate used as the spray
  gate**.

## Verdict

**Both core claims survive.** Neither conjunction returns an occupying study on any of the three
databases. Two new related-work papers were found that must be **cited and distinguished** (below);
both are reliability/robustness work on the same lab→field setting, and neither does accuracy
estimation, a variance decomposition, or a spray/withhold decision.

## Queries and counts

Search mechanism verified live each time (a known-populated query returns results, so the zeros
are genuine, not an auth or syntax failure).

| # | Database | Query (TITLE-ABS-KEY / default AND) | Result |
|---|---|---|---|
| 1 | Scholar | plant disease classifier selective spraying domain shift withhold treatment | surfaced Xiang 2026 (below); rest robustness/adaptation |
| 2 | Scholar | "unsupervised accuracy estimation" OR "label-free accuracy" + domain shift + plant disease | ~nil |
| 3 | Scholar | estimating classifier accuracy unlabeled field images plant disease | all **adaptation / pseudo-label / semi-supervised**, none accuracy *estimation* |
| 4 | IEEE | "plant disease" AND spraying AND "domain shift" | **0** |
| 5 | IEEE | "plant disease" AND "domain shift" (sanity) | 33 — all detection/adaptation/calibration; **none** spraying |
| 6 | IEEE | "accuracy estimation" AND "domain shift" AND (agriculture OR crop OR "plant disease") | **0** |
| 7 | IEEE | (spraying OR pesticide) AND "plant disease" AND (uncertainty OR abstention OR withhold) | 4 — **none** a disease sprayer handling uncertainty/withholding |
| 8 | Scopus | "accuracy estimation" AND ("domain shift" OR "distribution shift") AND (crop OR agriculture OR "plant disease") | **0** |
| 9 | Scopus | "plant disease" AND (spraying OR pesticide OR "spot spray") AND ("domain shift" OR "field condition" OR uncertainty OR confidence OR abstain OR withhold) | **0** |
| 10 | Scopus | "plant disease" AND "domain shift" (sanity) | 44 — all robustness/adaptation/uncertainty **classification** |
| 11 | Scopus | (plant disease/crop/agri) AND ("predicting/estimating accuracy" OR "performance estimation" OR "label-free" …) | 718 but **contaminated**: "label-free" matches biosensing (Biochem 234 / Chem 175); no agri accuracy-estimation paper in the head |

IEEE query 7's four hits: a capacitive spray-volume sensor; an apple-leaf detection app; an
agricultural-enterprise production-planning paper; a tomato disorder-detection survey. None gates
a sprayer on classifier reliability.

## New related work found (must cite — neither occupies a gap)

1. **Xiang, Shi & Zhu (2026)**, *Frontiers in Plant Science*,
   **DOI 10.3389/fpls.2026.1826962** — "Quantifying the reliability gap in cross-domain plant
   disease classification: benchmarking the limited efficacy of standard mitigation techniques
   under controlled-to-field shift." PlantVillage→PlantDoc, parent-image-aware split. Benchmarks
   temperature scaling, **selective prediction**, image-level shift analysis, Grad-CAM,
   target-aware adaptation (AdaBN, feature-moment matching, DANN), frozen-backbone + ensembles.
   Headline: a fine-tuned ResNet-50 loses **67.7 points** of accuracy cross-domain while **mean
   predicted confidence stays at 79.76%**; DINOv2-S/14 best unknown-detection AUROC 0.764.
   - *Closest problem-setting match to the thesis found anywhere.* It **benchmarks mitigation**;
     it does **not** estimate the classifier's field accuracy label-free and does **no**
     source/target variance decomposition. No sprayer. It **corroborates** both of our works:
     confidence stays high while accuracy collapses ⇒ confidence-gated withholding is unsafe.

2. **Shrivastava, Agrawal, Shukla & Dubey (2026)**, *Scientific Reports* **16(1):24348** —
   "An uncertainty-aware evaluation framework based on hierarchical vision transformers for
   robust cross-domain plant leaf disease classification." Hierarchical ViT + **Monte Carlo
   Dropout** uncertainty + **temperature calibration**, bidirectional New Plant Diseases↔PlantDoc,
   97.8% / 93.6% accuracy, ECE 0.032 / 0.041.
   - A better-**calibrated, more robust classifier**. Not label-free accuracy estimation; no
     variance decomposition; no spray decision. Related work, same lab↔field theme.

Beyond these two: ~33 IEEE and ~44 Scopus papers (2025–26) on *plant disease × domain shift*,
all detection/classification robustness, adaptation, calibration or uncertainty-aware
classification — e.g. IEEE "Beyond Accuracy: Robustness and Calibration Evaluation…" (2025),
"Systematic Evaluation of Cross-Dataset Domain Shift… Tomato Disease" (2026), "DDAN-RGAL…
Field-Deployable Plant Disease Detection" (IEEE T-AFE 2026); Scopus "From laboratory to field:
Frozen foundation-model features…" (Smart Agric. Tech. 2026), "From Benchmark Accuracy to Field
Performance…" (Sustainability 2026). **None** estimates accuracy label-free or gates a sprayer.

## Consequences for the writeup

1. **Thesis related work** — add Xiang 2026 and Shrivastava 2026 as the nearest neighbours, and
   state the distinction explicitly: they **mitigate/benchmark** reliability (calibration,
   selective prediction, uncertainty, adaptation) on the same PlantVillage/NPDD→PlantDoc shift;
   this thesis instead **estimates the deployed classifier's field accuracy without labels** and
   **decomposes source vs target variance**. Use Xiang's "confidence 79.76% while accuracy
   collapses 67.7 pts" as external corroboration of the motivating failure.
2. **The thesis may no longer claim to be first to document the controlled-to-field reliability
   gap** with calibration/selective-prediction/OOD on this dataset pair — Xiang 2026 does that.
   The thesis's distinct contribution (label-free accuracy *estimation* + variance decomposition)
   is unaffected.
3. **Paper 2** — the IEEE+Scopus zero for disease-sprayer × shift × withholding confirms the
   narrowed gap already stated in `litsearch_paper2_results.md`. No change to the claim; cite
   this check as the database closure.
4. **Pending**: fetch verified BibTeX for both new DOIs (Frontiers DOI above; Scientific Reports
   16:24348 — resolve the s41598 DOI) and add to the thesis bib + related-work section.

## Rover software/hardware — coverage and status (added 2026-10-17)

The rover's control software and hardware are **not a novelty claim** of this project. The prior
`results/litsearch/TRIGGER_SATURATION_NOVELTY.md` already ruled the spatial (tissue-based)
spot-spray trigger, the nozzle lead-compensation and the FSM control to be **prior art / standard
engineering**, explicitly *not* contributions; the rover is a deployment platform plus a
quantified audit. To close the IEEE/Scholar gap for that earlier arXiv/OpenAlex/Crossref check,
the rover was re-searched directly on **IEEE Xplore and Google Scholar** today:

| Database | Query | Result |
|---|---|---|
| IEEE | ("spot spraying" OR "selective spraying") AND (threshold OR "vegetation index") AND (canopy OR saturation) | 0 (spraying literature is thin on IEEE) |
| IEEE | ("selective spraying" OR "spot spraying" OR "targeted spraying") AND (robot OR rover OR "plant disease") | **21** — mature prior art (e.g. "Autonomous Precision Agricultural Sprayer Robot" 2023; ROS pest-treatment architecture 2021, cited 47; "Tomato Care 5.0… Autonomous Input Application" 2025; "AgriGuard… Autonomous Precision Sprayer" 2026) |
| Scholar | low-cost rover selective spot spraying tomato disease field raspberry pi | abundant — Pi-based tomato-disease spray rovers and 2024–26 systematic reviews; e.g. Javed et al. 2025 (IEEE): Pi + camera + motor driver + spraying, same stack |

**Conclusion.** A low-cost selective-spraying rover for plant disease is firmly established prior
art — consistent with the project never claiming the rover as novel. Critically, **none** of the
rover/sprayer prior art gates spraying on the classifier's *reliability under laboratory-to-field
shift*, nor on a label-free accuracy estimate, so none occupies the project's actual contribution
either. No change to any claim; the rover chapters remain framed as deployment + audit, cited to
this prior art.

## Web of Science — searched decision: SKIPPED (accepted as sufficient without it)

Web of Science was reached today but sits behind an **institutional sign-in** (Clarivate /
webofscience.com); the login was **not** completed — authenticating on the user's account is
outside what may be done automatically. **Decision (2026-10-17, user):** skip WoS and accept
Google Scholar + IEEE Xplore + Scopus as sufficient, on the grounds that (a) WoS Core Collection
overlaps those indices heavily, (b) they converge on the same verdict, and (c) the rover is not a
novelty claim. **The literature search for this project is therefore closed.** If an examiner
later requires WoS specifically, the queries to run are the two core-claim conjunctions (rows 8–9
above) and the rover row; nothing in the three searched indices suggests the result would differ.

## Limits

Searched from the browser on 2026-10-17; counts are live and may drift. Scopus was reached by
institutional IP (search + abstracts; full text not required for a novelty check). IEEE and
Scholar needed no sign-in for titles/abstracts. Web of Science was **not** searched (not opened).
Screening here was by title, keywords and abstract; two papers (Xiang, Shrivastava) were read at
abstract level in full. This check targets the two core claims only.
