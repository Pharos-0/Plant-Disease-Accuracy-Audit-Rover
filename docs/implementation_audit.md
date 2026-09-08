# Implementation Audit — thesis phase, 2026-09-08

Read-only audit of the repository against `PROJECT_STATUS.md`. **No code changed, no
experiment rerun, no result overwritten.** Every figure below was recomputed from the
logged tables during the audit, not copied from the status document.

---

## 1. Verdict

**The status document is accurate.** All eight estimators × four arms of MAE values
reproduce from `results/tables/aggregate_metrics.parquet` to within 6e-4, as do the
accuracies, the N\* counts, the calibration ablation and the capture-day figures.

| Claim in PROJECT_STATUS | Audit result |
|---|---|
| 32 MAE values (8 estimators × 4 arms, N=100) | **all reproduce** |
| Arm accuracies 0.969 / 0.281 / 0.501 / 0.472 | **reproduce** |
| N\* : 1/96 shifted, 41/48 no-shift | **reproduce** |
| Calibration: 21/24 shifted pairs worse | **reproduces** |
| Capture days 0.604 / 0.491 / 0.369 | **reproduce** |
| 30 trained models | **confirmed** |
| 4 backbones × 5 seeds | **confirmed**, with a caveat — §3 |
| 720,000 scored sessions | **confirmed** |
| 72 tests passing | **confirmed** |
| 19 git commits | now **20** (the status doc's own commit) |

Working tree is clean.

---

## 2. Leakage firewall — verified live

Re-tested during the audit, including the negative control:

- baseline: 7/7 firewall tests pass;
- inject `from data.index import build_all` into `src/estimators/methods.py` →
  **the suite fails**;
- revert → passes again.

The firewall is not decorative. It detects a real violation.

---

## 3. Architecture ablation — what is actually there

**Do not assume this was a balanced 4×5 design.** The manifests show:

| version | protocol | label set | backbone | seeds | n_train |
|---|---|---|---|---:|---:|
| V1 | v1_image | SHARED_3 | resnet50 | 5 | 2042 |
| V2 | v2_leaf | SHARED_3 | resnet50 | 5 | 2044 |
| V2S7 | v2_leaf | SHARED_7 | resnet50 | 5 | 6811 |
| V9 | v2_leaf | SHARED_3 | efficientnet_b0 | 5 | 2044 |
| V9 | v2_leaf | SHARED_3 | mobilenetv3_large | 5 | 2044 |
| V9 | v2_leaf | SHARED_3 | vit_small_patch16 | 5 | 2044 |

Consequences:

1. **The clean four-way comparison is V2 + V9** — same protocol (`v2_leaf`), same label
   set (`SHARED_3`), same seeds {0…4}, identical hyperparameters (epochs/lr/batch all
   constant across all 30 runs). This is a fair comparison and covers **arms A, C, D**.
2. **Arm B (SHARED_7) exists for ResNet-50 only.** No architecture claim may be made
   about arm B.
3. ResNet-50 is over-represented (15 of 30 models) because it also carries the V1/V2
   leakage ablation and the SHARED_7 arm. Any statistic pooled over *all* models is
   ResNet-weighted and should not be presented as an architecture average.

### 3.1 Result — the failure persists across architectures

Arm D, N=100, independent unit = **seed** (5 per backbone), signed bias:

| estimator | efficientnet_b0 | mobilenetv3_large | resnet50 | vit_small_patch16 |
|---|---:|---:|---:|---:|
| atc_mc | +0.312 | +0.345 | +0.319 | +0.346 |
| atc_ne | +0.296 | +0.337 | +0.284 | +0.340 |
| confidence | +0.350 | +0.408 | +0.249 | +0.293 |
| cot | +0.235 | +0.195 | +0.041 | +0.031 |
| doc | +0.365 | +0.394 | +0.291 | +0.328 |
| entropy | +0.111 | +0.243 | **−0.060** | +0.077 |
| mano | +0.442 | +0.433 | +0.406 | +0.399 |
| nuclear_norm | +0.327 | +0.265 | +0.261 | +0.373 |

**31 of 32 (backbone, estimator) cells over-estimate.** The single exception is
ResNet-50 with the entropy baseline.

Source accuracy versus arm-D bias:

| backbone | source acc | mean arm-D bias |
|---|---:|---:|
| mobilenetv3_large | 0.9234 | +0.328 |
| efficientnet_b0 | 0.9454 | +0.305 |
| resnet50 | 0.9812 | +0.224 |
| vit_small_patch16 | **0.9968** | +0.273 |

Source accuracy spans 7.3 points with **no monotone relationship to estimator bias**.
The most accurate source model (ViT-S/16, 0.9968) still over-estimates by +0.273.

This is the user-specified **Outcome A** — the failure is not a ResNet artefact.

### 3.2 One genuine architecture dependence, worth reporting

**COT is architecture-dependent.** Its arm-D bias ranges from +0.031 (ViT-S/16) and
+0.041 (ResNet-50) to +0.195 (MobileNetV3-L) and +0.235 (EfficientNet-B0) — a 7-fold
spread, far larger than any other estimator's. Every other method is comparatively flat
across backbones.

This is **Outcome B for one estimator inside an otherwise Outcome-A picture**, and it was
not predicted. It is a finding, not a nuisance: COT is also the only estimator that uses
the class marginal rather than the confidence distribution alone. Whether those two facts
are connected is **untested** — see §6.

---

## 4. Reproducibility gaps found

Two concrete gaps against the section-27 requirements. Both are fixable and neither
invalidates existing results.

### 4.1 No manifest for the estimation or evaluation stages

`write_manifest` is called **only** by `src/training/train.py`. The estimator grid, the
N-sweep, the capture-day diagnostic and the temperature ablation write parquet/CSV
outputs with **no accompanying manifest** — so estimator version, session sampling rule,
bootstrap count, seeds and the git commit at estimation time are not recorded alongside
those results.

The information is partly recoverable (columns inside the parquet files,
`session_overlap.parquet`), but it is not a manifest and it is not automatic.

**Recommendation:** extend `utils/manifest.py` usage to the estimation and evaluation
drivers before any new experiment is added. Cheap, and it makes every future result
traceable in the way the training runs already are.

### 4.2 All 30 training runs were logged with `git_dirty: True`

Across 6 distinct commits. The recorded commit hash therefore does **not** pin the exact
source state used for training.

This is a genuine caveat and should be stated in the thesis rather than hidden. It is
mitigated but not eliminated by: the pipeline being seed-deterministic; the working tree
now being clean; and all headline numbers reproducing from the logged tables.

**Recommendation:** do not retrain to fix this — the cost outweighs the benefit and it
would replace verified results with new ones. Record it as a limitation, and ensure all
*future* runs are made from a clean tree.

---

## 5. Claim classification

### 5.1 PROVEN — directly measured, reproducible from logs

1. The lab-to-field drop (0.972 → 0.472 on real field data).
2. **The no-shift control validates the implementations** — ATC and DoC reach MAE 0.013
   on arm A. This is what makes the negative result a transfer failure rather than a bug,
   and it is the single most important defensive fact in the thesis.
3. All 32 MAE values and their bias counterparts.
4. Correlation collapse (arm A ≈ 0.42–0.53 → arm D ≈ 0.007–0.17).
5. N-sweep: 1/96 shifted cells reach MAE ≤ 0.05 versus 41/48 unshifted.
6. Calibration ablation: source ECE 0.043 → 0.007, yet 21/24 shifted pairs worse.
7. Capture-day gradient: 0.604 / 0.491 / 0.369 true accuracy against near-flat estimates.
8. Architecture persistence: 31/32 cells over-estimate.
9. H3 and H4 falsified; leakage delta −0.0008 with 0/8 estimators differing.
10. Gate lift = 1.00× for confidence-based estimators at τ=0.60.
11. ATC/DoC/COT equivalence to the official implementations (L3).
12. `web_rover.py` SHA-256 match and every audit defect confirmed, including zero logging.

### 5.2 SUGGESTED — interpretation, consistent with evidence but not directly established

1. **"Confidence becomes decoupled from correctness under shift."** This is the headline
   *explanation*, and it is currently an **inference from symptoms** — flat estimates,
   near-zero correlation, collapsed estimate variance. The mechanism has **not** been
   measured directly. This is exactly what RQ4 must fix, and until it does the wording
   must stay interpretive.
2. That COT survives better *because* it uses the class marginal. Plausible, and
   consistent with the architecture spread in §3.2 — but untested.
3. That the 24–25 cm camera height is appropriate because blight appears on lower
   foliage. Reasoning from agronomy, not measured here.

### 5.3 MISSING

| Gap | Blocks |
|---|---|
| **RQ4 diagnostics** — the mechanism behind the failure | The strongest available scientific upgrade |
| **RQ5 small-labelled-sample comparison** | Turning the negative result into practical guidance |
| Arm E (rover frames) | Physical access |
| CM-ATC / CM-DoC | Comparison to the nearest neighbour |
| Second source dataset | The "single source domain" limitation |
| Scopus/IEEE/ACM search | Any completeness claim about the literature |
| Estimation-stage manifests | Section-27 compliance for non-training runs |
| Rover functional tests | Chapter 6 beyond a design description |

---

## 6. Is the negative result strong enough for an M.Tech thesis?

**Yes — with one addition.**

What already makes it defensible:

- a **validated control** proving the methods are correctly implemented;
- **scale**: 30 models, 720k scored sessions, 8 estimators, 4 arms;
- **architecture-independence**, 31/32 cells;
- **the obvious objection pre-empted** — calibration was tested and does not rescue it;
- **a real-world diagnostic** (capture days) rather than only bootstrap resampling;
- **two falsified hypotheses reported honestly**;
- **independent corroboration** from an unrelated domain (arXiv:2507.22776).

What is missing to make it a *strong* thesis rather than a solid one: the thesis
currently establishes **that** the estimators fail and **that** three obvious fixes do
not help, but explains **why** only by inference. A viva will ask "why?", and
"confidence decouples from correctness" is presently an interpretation, not a
measurement.

**RQ4 is therefore the highest-value remaining work**, and it is cheap: it operates on
cached logits and features, needs no retraining and no GPU.

---

## 7. Prioritised recommendation

| Priority | Work | Value | Cost | Risk |
|---|---|---|---|---|
| **1** | **RQ4 diagnostics** — measure the confidence/correctness decoupling directly; separate deployable unlabelled signals from post-hoc labelled ones | **High** — converts the central explanation from interpretation to measurement | Low (numpy on cached arrays) | Low |
| **2** | **RQ5 small-labelled-sample** — how many labelled field images beat the best label-free estimate? | **High** — turns a negative result into actionable guidance | Low–medium | Low |
| **3** | Estimation-stage manifests | Medium (compliance) | Very low | None |
| **4** | CM-ATC / CM-DoC, after an eligibility check | Medium | Medium | Medium — must not become count-padding |
| **5** | Arm E | Medium | Blocked on hardware | High |
| **6** | Second source dataset | High for generality | High | Medium |
| **7** | Database literature search | Required for any completeness claim | Low, but needs access | None |

### What should NOT be done

- **Do not retrain** to clear the `git_dirty` flag. It would replace verified results to
  fix a bookkeeping issue.
- **Do not add estimators to raise the count.** Each needs an eligibility record first.
- **Do not attempt a new method yet.** Section 16 of the brief is explicit and correct:
  diagnostics first. There is currently no measured failure mode to correct.
- **Do not pool statistics across all 30 models** and call it an architecture average —
  it is ResNet-weighted (§3).
- **Do not make any architecture claim about arm B.** It is ResNet-50 only.
- **Do not treat Tomato-Village's 1,616 crops as independent observations.** The
  effective pool is 391 photographs.
- **Do not soften the negative result.**
