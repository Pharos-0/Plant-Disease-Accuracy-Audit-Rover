# Why These Nine Estimators — and Why Not the Others

Written 2026-09-09, after the systematic search revealed that the evaluated set is a
**sample** of this literature rather than its state of the art. A benchmark that calls
itself systematic has to justify its selection explicitly, so this document does that —
including where the honest answer is "we did not know about it".

---

## 1. What was actually decided, and when

**The nine were fixed at the design stage, before the systematic literature search.** They
were chosen from prior knowledge of the confidence / matrix-norm / optimal-transport branch
of the field. The search then found at least eleven further methods and two further
research communities.

This document therefore does **not** claim the nine were selected by applying the criteria
below. It states the criteria the nine do in fact satisfy, and records for every omitted
method whether it **fails a criterion** or was simply **not known at design time**. Those
are different admissions and are kept apart, because retrofitting a rationale onto a
decision that was made another way would be dishonest.

## 2. The nine

| estimator | family | reference |
|---|---|---|
| confidence | baseline | mean max-softmax |
| entropy | baseline | mean normalised entropy |
| ATC-MC | confidence threshold | Garg et al., ICLR 2022 |
| ATC-NE | confidence threshold | Garg et al., ICLR 2022 |
| DoC | confidence difference | Guillory et al., 2021 |
| COT | optimal transport | Lu et al., ICLR 2023 |
| nuclear norm | prediction-matrix | Deng et al., ICML 2023 |
| MaNo | matrix norm | Xie et al., NeurIPS 2024 |
| GdScore | classifier-head gradient | Xie et al., TMLR 2024/25 |

## 3. Operational criteria

These follow from the deployment scenario — a frozen classifier scoring a rover session —
and from the Stage-2 caching design ("expensive once, cheap millions of times"):

| # | criterion | why it matters here |
|---|---|---|
| **C1** | **Post-hoc on a frozen model.** No retraining, no auxiliary training task, no modification of the deployed classifier. | The deployed model is fixed; a method that retrains cannot run on a rover. |
| **C2** | **Single model.** No second model or ensemble required. | We deploy one classifier per configuration. |
| **C3** | **Computable from cached logits and penultimate features.** | The whole experimental design depends on caching activations once and scoring cheaply. |
| **C4** | **No target labels, ever.** | The leakage firewall. Non-negotiable. |
| **C5** | **Dataset/session-level estimand.** | Per-image correctness is selective prediction, a different problem. |

All nine satisfy C1–C5.

## 4. The omitted methods, one by one

| method | year | fails a criterion? | honest status |
|---|---|---|---|
| **Projection Norm** | 2022 | **Yes — C1, C3.** Pseudo-labels the test set and **trains a new model**, then measures parameter divergence. | Principled exclusion. Cannot run post-hoc from a cache. |
| **AutoEval** (Deng & Zheng) | 2020/21 | **Yes — C1, C3.** Requires building a synthetic meta-dataset and training a regressor over dataset-level features. | Principled exclusion, though a meta-dataset could in principle be built. |
| **Mandoline** | 2021 | **Yes — C3.** Requires user-written "slicing functions" encoding prior knowledge of the shift axes. | Principled exclusion; also a different input contract. |
| **ALSA** | 2025 | **Yes — C1.** Learnable anchors must be fitted. | Principled exclusion, and published after our design was fixed. |
| **IW-GAE** | 2023 | **Partly — C3.** Needs source/target density-ratio estimation. | Borderline. Not known at design time. |
| **Agreement-on-the-Line / GDE** | 2021/22 | **Fails C2 as designed** — needs a *collection* of models. | **But we have 20 models across 4 architectures.** C2 reflects the deployment story, not a limitation of our data. **This is an omission of opportunity, not of principle.** |
| **Platanios, Estimating Accuracy from Unlabeled Data** | 2017/18 | Fails C2 — needs multiple competing classifiers. | Same as above: our 20 models could support it. Not known at design time. |
| **Distance-Matters** (ATC-DistCS) | 2023 | **Fails nothing.** Needs training-set embeddings and a K-NN distance check — **we cache penultimate features already.** | **Applicable, and omitted only because it was unknown.** The most awkward omission: it targets the same confidence-unreliability-under-shift phenomenon this thesis reports. |
| **DAPPER** | 2021 | **Fails nothing** — mutual information between inputs and outputs. | Applicable. Not known at design time. |
| **AETTA** | 2024 | Needs dropout inference at test time; satisfies C1 for models with dropout. | Borderline-applicable. Not known at design time. |
| **Aries** | 2023 | Unassessed — from the software-testing literature. | Not known at design time. |
| **PPI family** | 2023– | **Fails C4 by construction** — PPI *requires* a small gold-standard labelled sample. | Not a competitor to the label-free estimators; it is the framework RQ5 sits inside. Cited, not benchmarked. |

## 5. The honest summary

Of eleven omitted label-free methods:

- **four are principled exclusions** (Projection Norm, AutoEval, Mandoline, ALSA) — they
  violate C1 or C3 and could not be run in this design without changing it;
- **one is out of scope by definition** (PPI — it uses labels, and is the frame for RQ5);
- **six were omitted because they were not known when the design was fixed**, and of those
  **at least three (Distance-Matters, DAPPER, Agreement-on-the-Line) are applicable to the
  cached data and the 20 trained models we already have.**

> **This must be stated as a limitation in the thesis, not defended as a design choice.**
> The correct claim is: *nine estimators spanning the confidence, prediction-matrix,
> optimal-transport and gradient families were evaluated*. The incorrect claim is that they
> represent the state of the art, or that the benchmark is exhaustive.

## 6. Does the omission threaten any conclusion?

| conclusion | threatened? |
|---|---|
| "None of the **nine evaluated** estimators beat an oracle constant here" | **No** — scoped to the nine by construction. |
| "Label-free accuracy estimation fails under agricultural shift" | **Yes** — and this claim is already barred in `PROJECT_STATUS.md` section 7 for exactly this reason. |
| Architecture dependence of the **evaluated** estimators | **No** — an observation about the nine. Generalising to all estimators would need the omitted ones. |
| The **calibration-anchor** finding (Phase 7b) | **No** — it is a measurement of GdScore's own behaviour under our calibration, not a comparative claim. |

The omissions bound the *breadth* of the benchmark. They do not undermine the specific
measured results, provided every claim stays scoped to the nine.

## 7. If breadth is later required

The cheapest additions, given what is already cached, would be **Distance-Matters**
(features are cached; needs a K-NN index over training embeddings) and
**Agreement-on-the-Line** (needs only the 20 existing models' predictions). Both are
recorded here as the obvious next step **if and only if** the thesis is later judged to
need wider coverage.

**No estimator is being added at present.** The current decision is to stop expanding
horizontally and to characterise what is already measured.
