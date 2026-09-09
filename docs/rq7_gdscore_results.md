# Phase 7 Results — does a later, stronger estimator change the conclusion?

**Experiment `rq7_gdscore`, 2026-09-09.** Option C: GdScore evaluated on the complete
1,616-crop unlabelled Arm-D pool only, not inside the N-sweep. Eligibility and
reproduction record: `docs/gdscore_eligibility.md`.

No retraining. No existing result modified. No N-sweep.

---

## 1. Answer

**No. The conclusions are unchanged — and now harder to dismiss.**

GdScore (Xie et al., TMLR 2025) does not transfer to this agricultural shift. It places
third of nine by MAE, but that ranking is **not evidence that it works**: its
score-to-accuracy calibration has correlation ≈ 0, its within-backbone correlation with
true accuracy is inconsistent in sign, and it is beaten by a constant.

The important new finding is broader than GdScore:

> **Not one of the nine estimators beats an oracle constant.** The true accuracies of our
> 20 models span 0.407–0.636 (sd **0.075**), while every estimator's MAE is **≥ 0.134** —
> larger than the entire spread it is trying to explain.

---

## 2. Reproduction fidelity — L3

Implemented from the official `algs/gdscore.py`, preserving `threshold=0.5`,
`norm_type=0.3` (the fractional p-quantity, not Frobenius), `batch_size=128`, seed
control, and the `train()`-mode forward.

- **Closed-form gradient verified against autograd** across four norm types (0.3, 0.5, 1,
  2), relative tolerance 1e-6. 13 tests in `tests/test_gdscore_reference.py`.
- **Train-mode cache built for all 20 models** with full state-dict snapshot/restore,
  verified tensor-by-tensor so the frozen classifiers are returned unchanged.
- **Sanity check passed:** ViT-S/16 has no BatchNorm, so train and eval must agree —
  **100% prediction agreement**, max relative logit difference 4.4e-03 (cuDNN TF32 noise
  at a different batch size, not a semantic difference).

Train vs eval divergence for the BatchNorm backbones, confirming the cache was necessary:

| backbone | relative logit difference | prediction agreement |
|---|---:|---:|
| resnet50 | 1.29 | 0.586 |
| efficientnet_b0 | 1.09 | 0.380 |
| mobilenetv3_large | 1.07 | 0.465 |
| vit_small_patch16 | 0.003 | **1.000** |

As with ATC and nuclear-norm/MaNo, the **score** is a faithful reproduction; the
**score-to-accuracy map is ours** (linear fit on source-validation bootstraps), and is
recorded as such.

## 3. Result on the full unlabelled pool

| estimator | MAE | median | sd |
|---|---:|---:|---:|
| entropy | 0.1337 | 0.0946 | 0.108 |
| cot | 0.1339 | 0.1077 | 0.103 |
| **gdscore** | **0.1599** | 0.1072 | **0.161** |
| nuclear_norm | 0.3089 | 0.3034 | 0.097 |
| atc_ne | 0.3109 | 0.3202 | 0.086 |
| confidence | 0.3214 | 0.3204 | 0.095 |
| atc_mc | 0.3268 | 0.3230 | 0.075 |
| doc | 0.3407 | 0.3437 | 0.077 |
| mano | 0.4165 | 0.4119 | 0.068 |

Third place, and by far the **highest variance** (sd 0.161) — which the per-backbone
breakdown explains.

## 4. The architecture pattern — GdScore is COT's mirror image

Per-backbone MAE:

| estimator | efficientnet_b0 | mobilenetv3_large | resnet50 | vit_small_patch16 |
|---|---:|---:|---:|---:|
| **gdscore** | **0.0736** ①| **0.1021** ①| 0.0520 ② | **0.4120** ⑨ |
| cot | 0.2479 | 0.2016 | **0.0475** ①| **0.0388** ①|
| entropy | 0.1115 | 0.2382 | 0.1003 | 0.0847 |

GdScore ranks **first on the two backbones where COT fails**, and **last of nine on ViT**,
where COT is best. The two methods are almost exactly complementary.

**A mechanistic reading, offered as a hypothesis rather than a result:** GdScore scores
under `model.train()`, so for a BatchNorm backbone the forward pass is recomputed against
*batch* statistics — which is precisely the operation that produces its signal. ViT-S/16
has no BatchNorm, so train mode is a no-op and that mechanism is absent. This would be
worth testing directly, and is not established here.

## 5. Why the ranking is not a success — the calibration carries no signal

Three independent checks say GdScore's low MAE on three backbones is **not** evidence
that its score tracks accuracy in this setting.

**(a) The score-to-accuracy map is fitting noise.** Correlation of the source-fitted
calibration:

| backbone | calibration r |
|---|---:|
| efficientnet_b0 | −0.109 |
| mobilenetv3_large | −0.061 |
| resnet50 | −0.038 |
| vit_small_patch16 | −0.121 |

All essentially zero, and all *negative*.

**(b) Within-backbone correlation with true accuracy is inconsistent in sign**
(n = 5 seeds each, so these are indicative only):

| backbone | Spearman | Pearson |
|---|---:|---:|
| efficientnet_b0 | −0.300 | −0.119 |
| mobilenetv3_large | −0.667 | −0.551 |
| resnet50 | **+0.600** | +0.317 |
| vit_small_patch16 | −0.100 | +0.023 |

Pooled across all 20 models: Spearman **−0.165** — the wrong direction. (Reported
within-backbone as well as pooled because RQ4 found a Simpson reversal in exactly this
kind of pooled statistic.)

**(c) On the authors' own metric it also fails.** GdScore is evaluated in the original
paper by correlation with accuracy, not MAE, so we report that too — it should not be
judged solely by a mapping we fitted. Across our 20 models, R² = 0.026.

**The interpretation:** with a calibration slope fitted to noise, the estimate is close to
a fixed value obtained by extrapolating the source-fitted line to the target score range.
For the three BatchNorm backbones that value happens to land near the true accuracy
(≈0.52–0.56 against a truth of 0.44–0.53). For ViT it lands at **0.98 against a truth of
0.57**. Landing near the answer is not the same as tracking it.

## 6. The finding that matters most: nothing beats a constant

Across the 20 models, true pool accuracy has mean 0.500 and **standard deviation 0.075**
(range 0.407–0.636). A predictor that ignores the data entirely and always returns the
median (0.4678) achieves **MAE 0.0610**.

| estimator | MAE | vs constant |
|---|---:|---|
| *oracle constant (median)* | **0.0610** | — |
| entropy | 0.1337 | worse |
| cot | 0.1339 | worse |
| gdscore | 0.1599 | worse |
| …all others | 0.31–0.42 | worse |

Per backbone the gap is wider still — the constant achieves 0.008–0.057, and **GdScore
beats it on none of the four**.

**This constant is an oracle and is not deployable** — it is chosen using the true
accuracies a deployed system does not have. It is a diagnostic ceiling, not a baseline a
practitioner could use. But as a diagnostic it is sharp: **the estimators are not merely
biased, they are less accurate than ignoring the input entirely and guessing the central
tendency.**

It also sharpens RQ5. The labelled sample reaches MAE 0.0795 at 20 photographs and
**0.0513 at 50** — so roughly **50 labelled photographs beat even the oracle constant**,
while no label-free estimator does at any budget.

## 7. Consequences for the thesis

1. **The Phase-5 and RQ5 conclusions stand**, and are strengthened. The failure is not an
   artefact of testing weak or dated estimators: a 2025 TMLR method, faithfully
   reproduced, also fails.
2. **The architecture-dependence finding is reinforced and becomes richer.** It is no
   longer "COT works on some backbones" but a complementary pattern across two methods,
   with a testable mechanistic hypothesis (BatchNorm).
3. **A new and sharper way to state the negative result** is now available: every
   evaluated estimator has error larger than the spread of the quantity it estimates.
4. **The examiner's question is answered.** "Why didn't you test a stronger, more recent
   method?" — we did, at L3 fidelity, and it did not change the conclusion.

## 8. What this does not establish

- That GdScore is a poor method. It was designed and validated for corruption-style
  benchmarks (CIFAR-10-C, ImageNet-C, Office-Home); this is a different shift, one source
  dataset, one field dataset, three classes, 20 models. **Failure here is not a general
  verdict on the method.**
- That the BatchNorm mechanism explains the architecture pattern. That is a hypothesis
  arising from the result, not a tested claim.
- Anything about GdScore's behaviour at small N. It was deliberately kept out of the
  N-sweep because its score is batch-size dependent (3.2× across batch sizes).
- Anything per-image.
- That a constant predictor is a usable deployment strategy. It is an oracle.

## 9. Recommended next step

**Do not enter the N-sweep.** The pre-registered condition for doing so was that GdScore
prove substantially better on the full pool. It did not, so the batch-size protocol
problem does not need solving.

The most valuable follow-up is now the **BatchNorm hypothesis** in §4: it is cheap
(re-score GdScore for the BN backbones under `eval()` mode and see whether its advantage
disappears), it is a direct test of a mechanism rather than another benchmark entry, and
it would turn the complementary GdScore/COT pattern from an observation into an
explanation.
