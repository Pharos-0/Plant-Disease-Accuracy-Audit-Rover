# Experiment Protocol — Version Ladder & Factorial Grid

Modelled on the XRAYEARTH ablation workflow (10 sequential variants, one factor per step),
adapted to an estimation benchmark rather than an architecture search.

---

## 0. The structural difference from XRAYEARTH — and why this project is cheap

In XRAYEARTH each variant was a full 12-hour training run: 10 variants ≈ 120 GPU-hours.

Here, **training happens once and evaluation happens thousands of times.**

```
EXPENSIVE, RUN ONCE          CHEAP, RUN MILLIONS OF TIMES
─────────────────────        ────────────────────────────
4 backbones × 5 seeds        session sampling
= 20 training runs     ───►  estimator evaluation      ───► results
~2,900 source images         bootstrap CIs
(minutes each on 5060)       (pure numpy on cached arrays)
```

The enabling move is **Stage 2: cache logits and penultimate features to disk** for every
(model, domain) pair. Every estimator, every session, every N, every seed afterwards is
array arithmetic. This is what lets a single-semester project report
`4 archs × 6 estimators × 4 domains × 5 session sizes × 5 seeds × 200 bootstrap sessions`
with real confidence intervals — a rigor level that a training-bound ablation cannot reach.

**Budget estimate on the RTX 5060 (8 GB):** ~3–5 h total training, ~1 h inference caching,
minutes for the entire estimator grid. Contrast: XRAYEARTH's 120 h. The saved compute is spent
on **repetition and error bars** instead.

---

## 1. Version ladder

Each version adds exactly one factor and answers exactly one question.

| V | Adds | Question it answers | Cost |
|---|---|---|---|
| **V1** | ResNet-50, source-only, **random image split** | What does the literature's protocol report? (deliberately reproduces the inflated number) | 1 run |
| **V2** | **Leaf-grouped split** (`leaf_grouping/`) | How much of PlantVillage accuracy is near-duplicate leakage? | 1 run |
| **V3** | Target evaluation on Tomato-Village | Is there a real lab→field drop? *(gate: brief §18)* | inference only |
| **V4** | Mean max-softmax baseline | Does the trivial estimator work? | free |
| **V5** | Mean entropy baseline | Does a second trivial estimator work? | free |
| **V6** | **ATC** (Garg et al., ICLR 2022) | Does the canonical method transfer? | free |
| **V7** | **DoC** (Difference of Confidences) | Does a bias-corrected variant do better? | free |
| **V8** | MANO / Nuclear-norm / COT | Do non-confidence estimators behave differently? | free |
| **V9** | + MobileNetV3, EfficientNet-B0, ViT-S/16 | Is estimator quality architecture-dependent? | 3 runs |
| **V10** | **Full** — shift ladder × session-size sweep × gate operating characteristic × rover session | The thesis | free |

**V1 → V2 is a genuine mini-contribution.** PlantVillage ships 4 photographs per physical leaf.
A random image split leaks the same leaf across train and test; a leaf-grouped split does not.
The delta is a number nobody has connected to *estimator calibration*: ATC and DoC fit their
thresholds against **source validation accuracy**, so an inflated source accuracy propagates
directly into a biased target estimate. V1 vs V2 measures that propagation. Report it.

---

## 2. Factorial grid (V10)

| Axis | Levels | n |
|---|---|---|
| Backbone | ResNet-50, MobileNetV3-L, EfficientNet-B0, ViT-S/16 | 4 |
| Estimator | Confidence, Entropy, ATC, DoC, MANO/NN, COT | 6 |
| Target domain (shift ladder) | see below | 5 |
| Session size N | 25, 50, 100, 200, 400 | 5 |
| Seed | 5 seeds × 200 bootstrap sessions | — |

### Target domains — the primary independent variable

These represent **different types and magnitudes** of shift. Do **not** assume an increasing
severity ordering: measure shift magnitude empirically (MMD / Fréchet distance on penultimate
features) and report whether an ordering exists. Discovering it is a result; assuming it is an error.

| Arm | Target | Labels | Shift type | Target imgs |
|---|---|---|---|---:|
| **A** | PlantVillage held-out (leaf-grouped) | `SHARED_3` | **none** — calibration control | ~730 leaves |
| **B** | PlantDoc | `SHARED_8` | web / in-the-wild | 744 |
| **C** | PlantDoc | `SHARED_3` | web *(label-space control vs B)* | 262 |
| **D** | Tomato-Village | `SHARED_3` | **real Indian field** | 1,616 |
| **E** | Rover frames | `SHARED_3` | robot deployment | to collect |

Arm A must be near-perfect — if an estimator fails with *no* shift, it is broken, not challenged.
Arm C isolates label-space size from shift type. Arm D is the thesis. Arm E is the demonstration.

---

## 3. Pre-registered hypotheses

State these **before** running, so results are confirmatory rather than post-hoc.

These are **hypotheses to test, not outcomes to produce.** Any may be falsified; a clear
falsification is a good result.

- **H1.** Confidence-based estimators (Confidence, Entropy, ATC, DoC) *may* exhibit positive bias
  (over-estimation) as shift increases. *Rationale:* lab-trained models tend to be overconfident
  under covariate shift.
- **H2.** Estimation error *may* increase under combined covariate and label/prevalence shift —
  relevant to Arm D, where Tomato-Village is 56% LB / 31% EB / 13% H against a near-balanced
  source. *Rationale:* prevalence shift is a documented failure mode of ATC and DoC
  (`novelty_verification.md` §4).
- **H3.** Estimation error and variance are expected to decrease as N increases, potentially
  revealing a practically useful minimum session size **N\***. Characterising that relationship is
  the headline analysis.
- **H4.** Image-level leakage in V1 *may* alter source calibration and consequently affect target
  estimates relative to the leaf-grouped V2 protocol.

**Anti-confirmation rules:** do not tune any estimator, threshold, split or hyperparameter to move
a result toward a hypothesis; fix the protocol before seeing results. If a hypothesis is falsified,
report it prominently and diagnose why — never reword it to fit the data (brief §27).

---

## 4. Leakage firewall (brief §7, §20, §24)

Two physically separate code paths, enforced at module level:

```
src/estimators/   ← MAY see: source val labels, source/target logits+features. NEVER target labels.
src/evaluation/   ← MAY see: target labels. Runs strictly AFTER estimates are written to disk.
```

`src/estimators/` must not import anything that can read a target label. Add a unit test that
asserts this. Estimates are written to `results/raw/estimates/` and are **immutable** before
evaluation opens them.

Splits: PlantVillage by `Leaf #`; Tomato-Village sessions by base timestamp.

---

## 5. Deployment gate — the operating characteristic (RQ3)

The rover decision is **session-level only** (brief §2, §39). For threshold τ:

| | actual ≥ τ | actual < τ |
|---|---|---|
| **est ≥ τ** → operate | correct operate | **unsafe operate** ← the costly error |
| **est < τ** → withhold | wasted withhold | correct withhold |

Sweep τ ∈ [0.5, 0.95] and plot unsafe-operate rate vs. wasted-withhold rate — a ROC-shaped
curve for deployment gating. This converts an estimation metric (MAE) into a decision metric,
and is the framing contribution in `novelty_verification.md` §7 Tier 3.

---

## 6. Rover: prototype status, production software

The claim is **not** that this chassis is a field-ready sprayer. It is:

> The session-gating software is hardware-agnostic and complete; the current chassis is a
> low-fidelity prototype that demonstrates the interface an adequately-provisioned platform
> would use.

| Layer | Status |
|---|---|
| Session buffer, estimator, gate, logging | **Production** — same code as the benchmark, no reimplementation |
| Leaf-region proposal (ExG → connected components → crops) | Production; reuses the delivered rover's honest CV half |
| Chassis, camera mast, enclosure | **Prototype** — cardboard, taped PVC mast, one webcam, one HC-SR04 |

Stated limitations, to be written into the thesis rather than hidden: no IR row sensors, no
battery ADC, no tank sensor, cardboard enclosure unsuitable for actual spray operation. The
thesis must remain complete if this chapter is deleted (brief §3).

---

## 7. Stop conditions checked at each version

| After | Stop if | Then |
|---|---|---|
| V2 | source accuracy < 0.85 on `SHARED_3` | training pipeline is broken, not the science |
| V3 | target accuracy ≈ source accuracy | **STOP COND. C** — no measurable shift; escalate to Arm B/E |
| V3 | target accuracy < 0.40 | near-chance on 3 classes; estimates uninformative — reconsider label space |
| V6 | ATC estimate constant across sessions | **STOP COND. F** — investigate before proceeding |
| any | estimator reads a target label | **STOP COND. G** — halt, audit, rerun |

---

## 8. Outputs (brief §22)

Every version writes `results/raw/<version>/manifest.json` with seed, config hash, package
versions, dataset checksums, and metrics. Figures generated from manifests only — never hand-made.

Headline figures: estimated-vs-actual scatter with `y=x` (Fig 5) · MAE by estimator (Fig 6) ·
**N vs MAE and N vs variance (Fig 7–8) — the sample-efficiency result** · bias by shift level
(Fig 9) · gate operating characteristic (new) · V1-vs-V2 leakage propagation (new).
