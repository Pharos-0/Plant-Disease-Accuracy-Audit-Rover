# Phase 7b Pre-Registration — the BatchNorm hypothesis

**Committed before the experiment is run.** Predictions are fixed in advance so the
result is confirmatory, and so a null result is as reportable as a positive one.

---

## 1. The observation this arises from

Phase 7 found GdScore and COT to be almost exactly complementary
(`docs/rq7_gdscore_results.md` §4):

| backbone | BatchNorm | GdScore MAE | COT MAE |
|---|---|---:|---:|
| efficientnet_b0 | yes | **0.0736** ① | 0.2479 |
| mobilenetv3_large | yes | **0.1021** ① | 0.2016 |
| resnet50 | yes | 0.0520 ② | **0.0475** ① |
| vit_small_patch16 | **no** | **0.4120** ⑨ | **0.0388** ① |

GdScore ranks first on two of the three BatchNorm backbones and **last of nine** on the
one architecture without BatchNorm.

## 2. Hypothesis

> **H-BN.** GdScore's relative advantage on the BatchNorm backbones depends on the
> `model.train()` forward pass, in which BatchNorm uses **batch** statistics rather than
> running statistics.

Rationale: the official implementation scores under `train()`. For a BatchNorm network
that recomputes every normalisation against the current batch — a substantive change,
which Phase 7 measured as disagreeing with eval mode on 41–62% of predictions. ViT-S/16
has no BatchNorm, so `train()` is a no-op there (verified: 100% prediction agreement),
and the mechanism is simply absent.

## 3. The experiment

A single controlled change. Everything else held identical to Phase 7:

| held fixed | value |
|---|---|
| models | the same 20 frozen checkpoints (V2 + V9, `SHARED_3`) |
| target | the same 1,616 Arm-D crops |
| pseudo-label rule | argmax, with random substitution below `threshold` |
| threshold | 0.5 |
| norm_type | 0.3 (fractional, not Frobenius) |
| batch size for scoring | 128 |
| `gd_seed` | 1 |
| calibration | the same source-validation bootstrap linear fit |
| estimand | the same true eval-mode pool accuracy |

**Changed:** activations come from the existing **`eval()`-mode** Stage-2 cache instead of
the train-mode cache. Nothing else.

Cost: seconds. No GPU, no retraining, no new caching — the eval cache already exists.

## 4. Pre-registered predictions

**If H-BN holds:**
- GdScore's MAE on `efficientnet_b0`, `mobilenetv3_large` and `resnet50` **degrades
  substantially** under eval mode, and it loses its first-place rankings;
- the GdScore/COT complementarity weakens or disappears.

**If H-BN is false:**
- the per-backbone pattern is **essentially unchanged**, and the architecture dependence
  must be sought elsewhere (calibration, representation geometry, class-prior shift).

**Built-in control — this validates the experiment itself:**
- **ViT-S/16 must be essentially unchanged** between the two runs, because train and eval
  are the same computation for it (up to TF32 noise). If ViT moves materially, the
  comparison is broken and no conclusion may be drawn from the other three.

## 5. Metrics

Reported per backbone, train-mode versus eval-mode: MAE, rank among the nine estimators,
calibration correlation, within-backbone Spearman of raw score against true accuracy, and
raw-score magnitude.

## 6. Interpretation discipline

Binding on the write-up:

- The result may be described as **consistent with** or **inconsistent with** H-BN.
- It may **not** be described as proving that BatchNorm *causes* the pattern. This is one
  intervention on one architecture family with n = 5 seeds; the train/eval switch changes
  the forward pass globally, not BatchNorm in isolation.
- A stronger causal test — e.g. swapping normalisation layers and retraining — is **out of
  scope** and is not proposed.
- If the control fails, the experiment is reported as invalid and nothing else is claimed.

## 7. What this cannot establish

- Anything about GdScore beyond this agricultural setting; it was validated on
  corruption-style benchmarks.
- Anything about the other estimators' architecture dependence — only GdScore is varied.
- Anything per-image.
