# §16 Verification — the four items left open at handoff

MASTER_PROMPT §16 lists four unresolved checks, "cheap and one is load-bearing".
This document resolves three of them and records why the fourth is blocked.

**Headline: the load-bearing item is FALSIFIED.** The FBR-UDA paper does not make
the admission the thesis motivation was built on. §16 is explicit about what
follows — *"If it does not say this, the motivation must be replaced."* §1 below
does that, and the replacement is stronger than the original.

Verified 2026-09-08. Access limits are stated per item rather than glossed.

---

## 1. FBR-UDA's "could not report field performance for lack of labels" — **FALSIFIED**

**The paper.** Woosang Jeon, Taehyeong Kim, Sanghyeok Choi, Kyuseok Yang,
*"Bridging the Lab-to-Field Gap in Plant Disease Diagnosis Through Unsupervised
Domain Adaptation Enhanced by Background Recomposition"*, Ecological Informatics
(ScienceDirect `S1574954125005886`; SSRN abstract 5217081).

**Access caveat, stated plainly.** Both the ScienceDirect page and the SSRN PDF
returned HTTP 403, so the full text could not be read directly and **no verbatim
page-cited quotation was obtainable**. The finding below rests on the abstract and
results text as returned consistently across four independent search retrievals.
That is weaker evidence than §16 asked for, and the conclusion is stated with that
limit attached — but the evidence is one-directional and not close.

**What the paper actually does.** It reports quantitative accuracy on real-field
test data throughout:

| Reported figure | Value |
|---|---|
| Lab-trained baseline on real-field data | **47.1 – 89.3%** across crops (apple worst at 47.1%) |
| Improvement from background recomposition (w/ FBR) | **+9.9 – 23.6%** across all crops |
| Grape leaf, baseline → with FBR | **57.4% → 81.0%** |
| CDAN+E with FBR | **91.1%** |
| **Real-to-Real benchmark** | **96.1 – 98.8%** |

**The conclusion.** A paper cannot report a 47.1% baseline, a 91.1% adapted result
and a 96.1–98.8% Real-to-Real upper bound **without labeled real-field test data**.
It plainly had them. The sentence that the handoff read as an admission is almost
certainly the standard UDA framing — adaptation "without requiring labeled
real-field data" — which refers to **training**, not evaluation. Unsupervised
domain adaptation means the *adaptation* is unsupervised. It has never meant the
evaluation is.

### The replacement motivation

The thesis does not need repair, only a correction of which sentence carries the
weight. MASTER_PROMPT §1 already states the true claim:

> Everyone who has measured that collapse used **labeled** field test sets — which
> a deployed machine never has.

FBR-UDA is now **positive evidence for that claim rather than a paper conceding a
gap**. It is a 2025 lab-to-field plant-disease paper that needed labeled field data
at three separate points — to quantify the baseline collapse, to show its method
helped, and to build a Real-to-Real ceiling. Every one of those numbers is
unavailable to a rover in a field.

That is the honest and much sharper framing: **the field is not failing to measure
the collapse; it is measuring the collapse with an instrument the deployed system
does not possess.** Cite FBR-UDA as an exemplar of the dependency, never as a
confession of it.

**Action taken:** the "could not report field performance" claim must not appear in
the thesis. `docs/novelty_verification.md` §5 lists it as *"NOT YET VERIFIED"* in
its nearest-neighbour table; that row is now resolved as **false** and is superseded
by this document.

---

## 2. arXiv:2507.22776 read in full — the difference stated precisely

**The paper.** Tim Flühmann, Alceu Bissoto, Trung-Dung Hoang, Lisa M. Koch,
*"Label-free estimation of clinically relevant performance metrics under
distribution shifts"*. Abstract and full HTML text both accessible.

**What it does.** Benchmarks **CBPE, ATC, DoC** plus their own confusion-matrix
generalisations **CM-ATC** and **CM-DoC**, on chest X-ray classification
(CheXpertPlus, PadChest, ChestX-Ray8) under real-world and simulated covariate and
prevalence shift.

**Its central finding, which corroborates ours.** *"All estimation methods struggled
under prevalence shifts, especially for prevalence-dependent metrics."* Our arm D
carries naturally-occurring prevalence shift (balanced source → 56/31/13 target) and
we observe the same collapse. Two independent domains, same failure mode.

**Where we differ — precisely.**

| | arXiv:2507.22776 | This thesis |
|---|---|---|
| Domain | chest X-ray (medical) | tomato leaf disease (agriculture) |
| Estimators | CBPE, ATC, DoC, CM-ATC, CM-DoC | ATC-MC/NE, DoC, **COT, MaNo, nuclear norm**, confidence, entropy |
| Prevalence shift | **simulated** | **naturally occurring** (PlantVillage → Tomato-Village) |
| Evaluation-set size | **~22,000** images (real shifts); **1,000** per condition (controlled) | **N = 25–400** — a rover session |
| Sample-size analysis | **none.** Only *"We repeat the test set construction 50 times … to reduce sampling variability"* — averaging, not a sample-size study | **the headline analysis**: bias and variance vs N, and whether N\* exists |
| Deployment sessions / batches | **not considered** | the unit of analysis |
| Decision gating | **not considered**; thresholds discussed are classification confidence thresholds only | operate/withhold gate, swept τ, **lift over base rate** |

**Consequence for the novelty position.** Tier 1 ("new domain") remains weak, exactly
as `novelty_verification.md` §7 already concluded — this paper made that move first,
for medicine. But **Tier 2 and Tier 3 are confirmed unoccupied by the nearest
neighbour**: it evaluates on 1,000–22,000 images and never varies that number, and it
never converts an estimate into a deployment decision. Our small-sample/session regime
and our gate operating characteristic do not overlap with it.

---

## 3. Systematic search — partially completed, limits stated

§16 asks for Scopus + IEEE Xplore + ACM DL with Boolean queries logged verbatim and
counts, targeting 25–30 screened papers. **This machine has web search but no
institutional database access, so this is a web-search screen, not a database
search.** It is stronger than the handoff's five searches and weaker than what §16
specified. Treat the load-bearing claim as *well-supported, not systematically
proven*, and run the database search before the viva.

### Queries logged verbatim

| # | Query | Outcome |
|---|---|---|
| 1 | `FBR-UDA foreground background recomposition unsupervised domain adaptation plant disease` | Located the target paper; see §1 |
| 2 | `"background recomposition" unsupervised domain adaptation plant disease lab to field tomato` | Confirmed authorship and results |
| 3 | `"Bridging the Lab-to-Field" plant disease "background recomposition" abstract "real-field" accuracy evaluation test set labeled` | Returned the per-crop accuracy figures used in §1 |
| 4 | `"background recomposition" plant disease "Real-to-Real" benchmark accuracy "Lab-to-Real" CDAN FBR tomato pepper grape apple results` | Confirmed the Real-to-Real labeled benchmark |
| 5 | `label-free unsupervised accuracy estimation unlabeled test set plant disease crop agriculture 2025 2026` | **All agricultural hits are self-supervised learning, semi-supervised learning, domain adaptation, or severity grading. Zero dataset-level label-free accuracy estimation.** |
| 6 | `how many unlabeled samples needed reliable accuracy estimation ATC DoC sample size variance small test set` | No dedicated small-sample study; see §4 |
| 7 | `robot deployment session-level batch accuracy estimation gate withhold operate unlabeled confidence threshold agriculture sprayer` | Surfaced the nearest robotics neighbour; see below |
| 8 | `Jeon Kim Choi Yang "Bridging the Lab-to-Field" …` | Authorship confirmation |

### Verdict on the load-bearing claim

**Still SUPPORTED.** Query 5 is the decisive one: searching directly for label-free
accuracy estimation in agriculture returns self-supervised pretraining, semi-supervised
learning, domain adaptation and severity estimation — the same three neighbouring
tasks the handoff found, and no instance of estimating a fixed model's accuracy on an
unlabeled agricultural target set.

### New nearest neighbour found — and it does not pre-empt us

**"Confidence-Gated Robot Autonomy: When Does Uncertainty Actually Help?"**
(arXiv:2605.18045). This is the closest robotics work to our Tier 3 framing and had
to be checked carefully. It does **not** pre-empt it:

- Its gate is **per-sample** — `π_τ(x) = 1 if u(x) ≤ τ else 0`, applied to each input.
- It uses **per-sample uncertainty to defer**, which is *selective prediction* — the
  exact confusion MASTER_PROMPT §2 forbids, at the exact granularity we refuse.
- It employs **no dataset-level label-free accuracy estimator** of any kind.
- Domain is temporal activity recognition (EGO4D, iAssist, GTEA), not agriculture.

It is nonetheless worth citing as **convergent evidence**, because its central finding
rhymes with ours at a different granularity: it reports *"a dataset-dependent competence
regime below which uncertainty provides a weak and unstable error ranking."* We find the
session-level analogue — once the classifier collapses under shift, confidence-derived
statistics stop ranking sessions (r ≈ 0.07 on arm D). Two literatures, two granularities,
the same underlying failure: **uncertainty stops carrying information precisely when the
base model stops being competent, which is when you most need it.**

Also noted: agricultural sprayer patents describe a "spray operation confidence level
detector" producing per-target confidence. Prior art for confidence-gated spraying
exists in the patent literature, at per-target granularity. Cite it and distinguish it;
do not claim novelty for the idea of gating a sprayer on confidence.

---

## 4. Small-sample regime — **confirmed unoccupied**, our strongest remaining claim

No study was found that characterises ATC/DoC bias and variance at N = 25–400.

- **arXiv:2507.22776** (nearest neighbour) evaluates at 1,000–22,000 and never varies it.
- **"AutoEval Done Right"** (arXiv:2403.07008) is about sample efficiency in evaluation
  but solves a *different problem*: it combines **human-labeled** data with synthetic
  labels to reduce annotation cost. It is not label-free, so it is out of scope by the
  §12 criterion.
- The prediction-powered-inference literature discusses sample size at N ≈ 500–10,500 —
  a different regime and a different problem, as the handoff already noted.
- Query 6 surfaced no dedicated small-sample analysis for ATC or DoC.

**This confirms `novelty_verification.md` §7 Tier 2 as the strongest genuinely-fresh
contribution**, and our result gives it teeth: in the N = 25–400 regime only **1 of 96**
(arm, estimator, backbone) cells under shift ever reaches MAE ≤ 0.05, because the error
is bias-dominated rather than variance-dominated. The operational question *"how many
unlabeled observations do I need?"* has an answer, and the answer is **"more will not
help you"** — which is only visible if you study the regime.

---

## 5. Remaining open item — now half resolved

- [x] **Measured camera height: 24–25 cm** (steel rule to lens centre, 2026-09-08).
      **Arm E is viable.** At that height the camera images the lower canopy, which is
      where late blight and early blight appear first — so the mast height is
      defensible on epidemiological grounds rather than being an accident. The caveat
      is that a 24 cm wide-field frame is a *further* shift beyond the laboratory
      leaf close-up, and `src/rover/leaf_proposal.py` exists to narrow it. See
      `docs/rover_hardware_inventory.md` §3.
- [ ] **One camera frame pointed at a real plant.** Still outstanding — all supplied
      photographs are indoors on tile. `rover/hardware_tests/test_components.py
      --test camera` saves this frame automatically.

## 6. Actions required in the thesis

1. **Delete** the "FBR-UDA could not report field performance for lack of labels"
   claim. It is false.
2. **Replace** it with the framing in §1: prior lab-to-field work measures the collapse
   using labeled field test sets, which a deployed machine does not have. Cite FBR-UDA's
   own reported numbers as the exemplar.
3. **Cite arXiv:2507.22776 prominently** as the nearest neighbour, state the difference
   using the table in §2, and note that its prevalence-shift finding independently
   corroborates ours.
4. **Cite arXiv:2605.18045** to distinguish per-sample confidence gating from
   session-level accuracy estimation, and as convergent evidence.
5. **Run the Scopus/IEEE/ACM search** before the viva to close §3 properly.
