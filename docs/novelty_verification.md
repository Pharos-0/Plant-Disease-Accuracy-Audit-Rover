# Novelty Verification — Label-Free Accuracy Estimation for Plant Disease

**Status:** partial verification, 2026-09-06. Five targeted web searches. **This is not a systematic review.**
Abstracts and search summaries only; no full texts read. Treat every verdict below as provisional
until the Scopus/IEEE Xplore search in Step 1 of the protocol is complete.

---

## 1. The load-bearing claim

The thesis direction rests on one claim:

> No published work applies dataset-level, label-free accuracy estimation
> (ATC / DoC / AutoEval / MANO / Confidence-Dispersity) to plant disease classification.

If this is false, the direction collapses and must be re-scoped.

---

## 2. Searches run (2026-09-06)

| # | Query | Outcome |
|---|-------|---------|
| 1 | `"label-free" OR "unsupervised" accuracy estimation unlabeled test set plant disease classification agriculture` | Agricultural hits were **self-supervised pretraining, contrastive learning, unsupervised clustering, domain adaptation** — all different tasks. No accuracy estimation. |
| 2 | `"average thresholded confidence" OR "AutoEval" OR "predicting model performance" unlabeled data crop disease agricultural imagery` | ATC/AutoEval hits are all general-ML. The only agriculture hits were YOLO confidence-threshold tuning — unrelated (per-detection thresholding, not accuracy estimation). |
| 3 | `unsupervised accuracy estimation automated model evaluation remote sensing agriculture crop unlabeled target domain benchmark` | Agriculture hits were **all unsupervised domain adaptation** for crop mapping. Confirms the two tasks are distinct and only UDA is represented in agriculture. |
| 4 | `sample size small batch unsupervised accuracy estimation variance how many unlabeled samples ATC reliable estimate` | No dedicated small-sample study for ATC/DoC found. Sample-size discussion exists in prediction-powered-inference literature at N=500–10,500 — a different regime and different problem. |
| 5 | `label shift ATC average thresholded confidence fails known limitation class distribution shift accuracy estimation` | **Key result — see §4.** |

---

## 3. Verdict on the load-bearing claim

**SUPPORTED, provisionally.** Five searches surfaced zero applications of dataset-level
label-free accuracy estimation to plant disease or crop imagery. Agricultural work in this
neighbourhood is consistently *domain adaptation* (changing the model to fit the target) or
*self-supervised learning* (reducing label needs at training), never *estimating the accuracy
of a fixed model on an unlabeled target set*.

The distinction is explicit in the literature: unsupervised accuracy estimation considers a
trained, **fixed** model and various unlabeled test sets, with the goal of estimating its
performance — a different task from adapting the model.

**Caveat:** absence of evidence from five web searches is weak evidence of absence.
Must be confirmed against Scopus / IEEE Xplore / ACM DL with logged Boolean queries.

---

## 4. The finding that changes the positioning

Search 5 surfaced:

> **"Label-free estimation of clinically relevant performance metrics under distribution shifts"**
> — arXiv:2507.22776 (2025), medical imaging.

This paper does, for **medicine**, exactly the genre move this thesis proposes for **agriculture**:
takes established label-free estimators into an applied domain and benchmarks them under shift.

### Consequences — both directions

**Against us.** "First to apply label-free accuracy estimation to a new application domain" is
**no longer an unclaimed move**. A reviewer who knows this paper will say: *the template exists,
you changed the dataset.* The framing "novel because it's agriculture" is therefore **weak**, and
should not be the headline claim.

**For us.** Three real benefits:
1. It **proves the genre is publishable** — de-risks the whole direction.
2. It supplies a **methodological blueprint** (metrics, protocol, failure analysis) to follow and cite.
3. Its central finding is **directly predictive for our setting**: simulated prevalence
   (label) shifts caused *systematic failure of all performance estimation techniques*, with
   confusion-matrix-derived variants (CM-ATC, CM-DoC) degrading worst.

Point 3 matters operationally. Our source and target class priors differ sharply
(see `dataset_class_mapping.md`), so **label shift is guaranteed, not hypothetical**, in the
PlantVillage→Tomato-Village arm. The medical paper predicts our estimators will fail there.
That gives us a **pre-registered hypothesis** rather than a fishing expedition — a significant
rigor upgrade, and a stronger paper if the prediction holds.

---

## 5. Nearest neighbours (to be distinguished explicitly in Related Work)

| Work | What it does | Why it does not pre-empt us |
|---|---|---|
| Garg et al., **ATC**, ICLR 2022 | Threshold on confidence score, fit on source val | The method we import; never applied to agriculture |
| Deng & Zheng, **AutoEval**, CVPR 2021 | Regression from dataset-level features | Method, not application; meta-training requirements need scrutiny (§24 of brief) |
| Deng et al., **Confidence & Dispersity**, ICML 2023 | Prediction-matrix characterisation (nuclear norm) | Method; general benchmarks only |
| **Label-free clinical metrics**, arXiv 2507.22776 (2025) | Same genre, **medical** imaging | Different domain; **removes "new-domain" novelty**, supplies blueprint + hypothesis |
| Białek et al., arXiv 2401.08348 (2024) | CBPE — performance under covariate shift w/o labels | Tabular/monitoring focus; a baseline to include |
| **Introspective perception**, Daftry 2016; AIJ 2023 | Robot predicts own perception failure | **Per-frame**, learned from auto-labelled data — different granularity from session-level estimation |
| Conformal weeding (2025), SOFS-Rice (2026) | Per-prediction abstention w/ guarantees | **Per-sample selective prediction**, not dataset-level accuracy estimation (see brief §39) |
| 2026 plant-disease reliability-gap benchmark | Measures lab→field gap | Measures it **with labels**; does not estimate it without them |
| FBR-UDA (2026) | Background-recomposition UDA | Claimed to admit it could not report field performance for lack of labels — **NOT YET VERIFIED**, see §6 |

---

## 6. Still unverified — do before freezing the title

- [ ] **FBR-UDA "no labels" admission.** The motivating quote for the entire thesis. Locate the paper, read the results section, quote it exactly with a page reference. If it does not say this, the motivation needs replacing.
- [ ] **Systematic search.** Scopus + IEEE Xplore + ACM DL, Boolean queries logged verbatim, with counts. Target ~25–30 screened papers.
- [ ] **Read arXiv:2507.22776 in full.** It is the closest neighbour and the most likely source of a "this is just X for plants" rejection. We must state the difference precisely.
- [ ] **Small-sample regime.** Confirm no existing study characterises ATC/DoC bias-variance at N=25–400. This is our strongest remaining novelty candidate (§7).

---

## 7. Revised, honest novelty position

Ranked by how well each survives a hostile reviewer.

**Tier 1 — Primary claim (defensible).**
A **shift-stratified** benchmark: the same fixed classifiers evaluated across a deliberately
ordered ladder of shift — none → web/in-the-wild → real Indian field → robot deployment —
under *simultaneous covariate and label shift*. Prior work benchmarks estimators on synthetic
corruptions or standard OOD splits; the ordered, real, application-grounded ladder is ours.

**Tier 2 — Strongest genuinely-fresh contribution.**
The **small-sample / session regime**. Published estimator evaluations use full test sets of
thousands of images. A rover session is 25–400 images. Characterising estimator bias and
variance in that regime — and deriving *the minimum number of unlabeled observations needed
for a target estimation error* — is an operational question we found no answer to, and it is
not agriculture-specific. It is a general contribution demonstrated in agriculture.

**Tier 3 — Framing contribution.**
The **deployment-gate operating characteristic**: treat the estimate as a decision rule, sweep
the threshold, and report correctly-withheld vs. wrongly-withheld sessions. Converts an
estimation metric into a decision metric. Small but, as far as we found, unstated.

**Never claim:**
- ✗ "First robot to assess its own reliability" — introspective perception, 2016.
- ✗ "Novel accuracy estimation method" — we import, we do not invent.
- ✗ "First application of label-free estimation to a new domain" — medical, 2025.
- ✗ Any per-image correctness claim — that is selective prediction (brief §39).
