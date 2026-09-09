# Phase 7b Results — the BatchNorm hypothesis

**Experiment `rq7b_batchnorm`, 2026-09-09.** Pre-registration:
`docs/rq7b_batchnorm_plan.md`, committed as `0c0bebe` **before** the experiment ran.

One controlled change against Phase 7: eval-mode activations instead of train-mode.
No retraining, no GPU, no existing result modified.

---

## 1. Answer

**The result is consistent with H-BN**, and the built-in control validates the
comparison.

But the post-hoc analysis in section 4 shows the mechanism is **not** the one the
hypothesis implicitly assumed. GdScore's advantage on the BatchNorm backbones does not
come from train-mode statistics making its *score* informative. The score barely moves
between modes. What moves is the **calibration anchor**, and the consequence is sharper
than the original hypothesis:

> GdScore's estimate in this setting is, to within 1.6–6.6%, just the model's
> source-validation accuracy recomputed in the same forward mode. Under `train()` on a
> BatchNorm backbone that number collapses from ≈0.94 to ≈0.52 — which happens to land
> near the true field accuracy of 0.44–0.53. **The gradient signal is almost not being
> used.**

## 2. The control passed

ViT-S/16 has no BatchNorm, so `train()` and `eval()` are the same computation for it and
it must be unchanged. It is:

| control quantity | value | threshold |
|---|---:|---:|
| max \|MAE difference\| across the 5 seeds | **0.00110** | < 0.02 |
| max relative raw-score difference | **1.45e-02** | numerical |

The comparison is therefore valid, and the pre-registered reading of the other three
backbones may proceed.

## 3. Pre-registered result

GdScore MAE on the full 1,616-crop unlabelled pool, per backbone:

| backbone | BatchNorm | train mode | eval mode | change | rank train → eval |
|---|---|---:|---:|---:|---|
| efficientnet_b0 | yes | **0.0736** | 0.4878 | **+0.414** | ① → ⑨ |
| mobilenetv3_large | yes | **0.1021** | 0.4715 | **+0.369** | ① → ⑨ |
| resnet50 | yes | 0.0520 | 0.4456 | **+0.394** | ② → ⑨ |
| vit_small_patch16 | **no** | 0.4120 | 0.4122 | +0.0003 | ⑨ → ⑨ |

Against the pre-registered predictions:

- **"MAE degrades substantially on the three BatchNorm backbones and it loses its
  first-place rankings"** — yes. All three fall to **last of nine**.
- **"the GdScore/COT complementarity weakens or disappears"** — yes. Under eval mode
  GdScore is last on every backbone, so there is no complementary pattern left.
- **ViT unchanged** — yes, to 0.0011 MAE.

This is the pattern H-BN predicted, and it is not the pattern the "H-BN is false" branch
predicted. **The result is consistent with H-BN.**

The calibration correlation stays near zero and negative in *both* modes (−0.030 to
−0.121), so the switch did not repair the calibration; it changed where the map lands.

## 4. Post-hoc: what actually moved — **not pre-registered**

Everything in this section is exploratory, run after seeing the confirmatory result
(`scripts/rq7b_mechanism.py`). It is measurement, not modelling, but it was not
pre-specified and is reported at that lower evidential weight.

It exists because the confirmatory numbers contain an oddity. The **target raw score is
essentially unchanged** between the two modes:

| backbone | raw score, eval | raw score, train | ratio |
|---|---:|---:|---:|
| efficientnet_b0 | 2.720e+09 | 2.713e+09 | **1.00** |
| mobilenetv3_large | 2.734e+09 | 2.705e+09 | **1.00** |
| resnet50 | 1.731e+10 | 1.879e+10 | **1.09** |
| vit_small_patch16 | 6.783e+08 | 6.794e+08 | **1.00** |

A 0.25% change in the score cannot produce a 0.41 change in the estimate through an
honest calibration. So the switch did not act on GdScore's field score at all.

**(a) The estimate is essentially the intercept.** Decomposing
`estimate = intercept + slope × score`, the share carried by the target data is:

| backbone | eval | train |
|---|---:|---:|
| efficientnet_b0 | 2.4% | 6.6% |
| mobilenetv3_large | 1.6% | 1.8% |
| resnet50 | 2.2% | 3.1% |
| vit_small_patch16 | 2.9% | 3.0% |

The fitted slope is 9e-13 to 3e-11 against a score of ~1e9 to 1e10, so **93–98% of the
estimate is the intercept** — a constant that never saw the field data.

**(b) The intercept is the source-validation accuracy, recomputed in the same mode.**
That is what moved:

| backbone | source-val acc, EVAL | source-val acc, TRAIN | true field acc |
|---|---:|---:|---:|
| efficientnet_b0 | 0.9426 | **0.5166** | 0.4447 |
| mobilenetv3_large | 0.9248 | **0.5540** | 0.4553 |
| resnet50 | 0.9877 | **0.5676** | 0.5297 |
| vit_small_patch16 | 0.9959 | 0.9959 | 0.5698 |

Train-mode BatchNorm degrades the model's *own source* accuracy by ~0.40 on the three BN
backbones, and leaves ViT untouched. Across all 20 models, GdScore's train-mode estimate
correlates with train-mode source accuracy at **r = 0.999**.

**(c) The two are interchangeable.** A rule that ignores GdScore entirely and simply
predicts the source-validation accuracy in the matching forward mode reproduces GdScore's
error to within 0.003 everywhere:

| backbone | GdScore, train | source-acc rule, train | GdScore, eval | source-acc rule, eval |
|---|---:|---:|---:|---:|
| efficientnet_b0 | 0.0736 | **0.0726** | 0.4878 | 0.4979 |
| mobilenetv3_large | 0.1021 | **0.0987** | 0.4715 | 0.4695 |
| resnet50 | 0.0520 | **0.0543** | 0.4456 | 0.4580 |
| vit_small_patch16 | 0.4120 | 0.4261 | 0.4122 | 0.4261 |

**(d) Why the target sits outside the calibration range under eval mode.** The source
bootstrap scores roughly double when scored in train mode, moving them up to meet the
target score. Distance of the target from the source bootstrap mean, in source SDs:

| backbone | eval | train |
|---|---:|---:|
| efficientnet_b0 | 5.3 σ | **0.1 σ** |
| mobilenetv3_large | 2.8 σ | **0.1 σ** |
| resnet50 | 14.8 σ | 4.5 σ |
| vit_small_patch16 | 23.1 σ | 23.2 σ |

Under train mode the target score falls **inside** the source bootstrap range on 100% of
EfficientNet and MobileNet seeds; under eval mode it is a 3–15 σ extrapolation. ViT is a
23 σ extrapolation in both — which is why it is rank 9 either way.

**(e) Scope: this is specific to GdScore.** The other two source-calibrated estimators do
not collapse to their intercept, because their scores are O(1) rather than O(1e9):

| estimator | share of the estimate carried by the data |
|---|---:|
| nuclear_norm | 89.2% |
| mano | 36.2% |
| **gdscore** | **1.6–6.6%** |

So this is not an artefact of our linear calibration in general. It is the interaction
between our calibration and GdScore's unbounded, very large-magnitude score.

## 5. What this means for the thesis

1. **Phase 7's headline is unchanged and is now better supported.** GdScore does not
   transfer to this shift. Phase 7 already declined to call its third-place MAE a success,
   on the strength of three independent checks; Phase 7b identifies what the third place
   actually was.

2. **The "GdScore is COT's mirror image" pattern should be retired as a finding about
   estimators.** Phase 7 section 4 reported it as a complementary architecture pattern with
   a mechanistic hypothesis attached. It survives only in train mode, and section 4 above
   shows train mode reaches it through the source-accuracy anchor rather than through the
   gradient score. It is better described as an artefact of scoring under `train()` than as
   a property of the method. *(Phase 7's file is left unmodified; this supersedes its
   section 4 reading.)*

3. **A reproduction-fidelity tension is now explicit, and it is the honest place to leave
   it.** The official implementation scores under `train()`. We reproduced that faithfully
   (L3), and it is what produced the favourable Phase 7 numbers. But on a BatchNorm
   backbone `train()` is not the deployed classifier — it is a *different model*, ~40
   accuracy points worse on its own source data. Whether the published method intends this
   on BatchNorm architectures is not something this thesis can determine, and we do not
   claim it is a flaw in the method: it may simply be an interaction with our small
   calibration sets and 3-class heads that does not arise on the corruption benchmarks
   GdScore was designed for.

4. **The RQ5 conclusion is untouched and slightly strengthened.** Nine estimators, none
   beating an oracle constant. The one that came closest did so through its calibration
   anchor.

## 6. Interpretation discipline

Binding, per the pre-registration section 6:

- The pre-registered result **is consistent with H-BN**: the train/eval switch removes
  GdScore's advantage on exactly the three BatchNorm backbones and leaves the one
  non-BatchNorm backbone unchanged.
- It is **not** proof that BatchNorm *causes* the pattern. This is one intervention, on
  one architecture family, with n = 5 seeds. The `train()`/`eval()` switch changes the
  forward pass globally — dropout and any other mode-dependent layer included — not
  BatchNorm in isolation. ViT-S/16 differs from the other three in more than the presence
  of BatchNorm.
- A stronger causal test (swapping normalisation layers and retraining) remains **out of
  scope** and is not proposed.
- Section 4 is **post-hoc and was not pre-registered**. Its measurements are exact and
  reproducible, but it was written after seeing the result and is reported as exploratory.

## 7. What this does not establish

- That GdScore is a poor method. It was designed and validated for corruption-style
  benchmarks; this is a different shift, one source dataset, one field dataset, three
  classes, 20 models. **Failure here is not a general verdict on the method.**
- That the source-accuracy rule in section 4(c) is a usable estimator. It is a diagnostic
  showing what GdScore reduces to here. It has no principled justification, it fails
  completely on ViT (MAE 0.426), it beats the oracle constant on none of the four
  backbones, and **it is not proposed as a method.**
- Anything about the other estimators' architecture dependence — only GdScore was varied.
- Anything about GdScore at small N, or per-image.

## 8. Reproduce

```bash
python scripts/rq7b_batchnorm_test.py   # pre-registered confirmatory comparison
python scripts/rq7b_mechanism.py        # post-hoc mechanism analysis
```

Outputs in `results/rq7b_batchnorm/`, with manifests recording inputs, parameters and
cache fingerprints.
