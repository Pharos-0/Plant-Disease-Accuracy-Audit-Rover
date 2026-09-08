# RQ4 Diagnostic Results

**Experiment `rq4`, 2026-09-08.** Pre-registered in `docs/rq4_diagnostic_plan.md`, which
was committed before any diagnostic ran. Tables in `results/rq4/`, manifest alongside
them, figures `rq4_fig01`–`rq4_fig10`.

No training, no rerun of the estimator grid, no new estimator, no existing table
overwritten. 20 models (V2 + V9, `SHARED_3`, four backbones × five seeds). Arm B is
excluded throughout: it exists for ResNet-50 only.

---

## Headline

**The "confidence decouples from correctness" interpretation is now measured, and it is
correct — but it is two distinct failures, not one.**

| | Arm A (no shift) | Arm D (real field) |
|---|---:|---:|
| **Overconfidence** — confidence − accuracy | −0.035 … +0.016 | **+0.243 … +0.404** |
| **Discrimination** — AUROC of confidence for correctness | **0.942** | **0.606** |
| Confidence separation (correct − incorrect) | 0.179 … 0.283 | 0.017 … 0.076 |
| Expected calibration error | 0.025 … 0.050 | 0.244 … 0.405 |

The model is **well calibrated with no shift** (ECE 0.025–0.050, confidence − accuracy
≈ 0) and becomes both **overconfident** and **uninformative** under field shift. Those are
separable defects and they break different estimators.

**Decision-gate outcome: OUTCOME 2** — a clear mechanism exists, but measuring it
requires target labels. See §8.

---

## 1. Diagnostic A — confidence stops ranking errors `POST-HOC`

AUROC of confidence for predicting per-prediction correctness, mean over 5 seeds; CI is a
**cluster bootstrap over photographs**, not images.

| backbone | arm A | arm C | arm D |
|---|---:|---:|---:|
| resnet50 | 0.978 | 0.747 | 0.647 |
| vit_small_patch16 | 0.985* | 0.711 | 0.674 |
| efficientnet_b0 | 0.915 | 0.620 | 0.571 |
| mobilenetv3_large | 0.908 | 0.603 | **0.533** |
| **pooled** | **0.942** CI [0.902, 0.972] | 0.670 CI [0.603, 0.733] | **0.606** CI [0.562, 0.649] |

\* ViT-S/16 reaches **100.0% accuracy on arm A for 2 of 5 seeds**, where AUROC is
undefined (no errors to rank). Its arm-A figure uses the 3 valid seeds, and the paired
test for that backbone is therefore n = 3. This is a data limitation, not a code fault.

Paired t over seeds, arm A vs arm D, Holm-corrected: every backbone collapses,
all Holm-adjusted p ≤ 3.9e-5. Against chance (0.5), one-sample t over 5 seeds:
resnet50 p = 0.0001, vit p = 0.012, efficientnet p = 0.008, **mobilenetv3_large
p = 0.092 — not distinguishable from chance.**

**For MobileNetV3-Large on real field data, confidence carries no usable information
about whether a prediction is right.** Figure 1 shows this directly: on arm A the correct
and incorrect confidence distributions are separated; on arm D they lie on top of each
other, both massed near 1.0.

## 2. Diagnostic B — overconfidence, and it is large `POST-HOC`

Confidence − accuracy is ≈ 0 on arm A and **+0.24 to +0.40** on arms C and D. ECE rises
roughly tenfold; Brier goes from 0.008–0.117 to 0.665–0.903.

**The estimator bias *is* this calibration gap, for some estimators but not all.** Across
the four backbones on arm D:

| estimator | corr(bias, confidence − accuracy) | mean abs difference |
|---|---:|---:|
| confidence | +0.999 | 0.004 |
| doc | **+0.995** | **0.028** |
| mano | +0.801 | 0.099 |
| atc_ne | +0.473 | 0.054 |
| atc_mc | +0.281 | 0.058 |
| nuclear_norm | −0.138 | 0.067 |

The `confidence` row is definitional — that estimator *is* mean confidence, so its bias
equals the gap by construction. It is reported as a consistency check on the pipeline,
not as a finding. **DoC is the informative row:** it is not definitionally the target gap,
yet it reproduces it to 0.028. ATC does not — so **ATC and DoC fail for different
reasons**, which a single "overconfidence" story would have hidden.

## 3. Diagnostic C — the estimators lose between-session sensitivity `POST-HOC`

Slope of estimated on true session accuracy (1.0 = tracks perfectly, 0 = constant), N=100:

| estimator | arm A | arm C | arm D |
|---|---:|---:|---:|
| atc_mc | 0.523 | 0.107 | **0.047** |
| atc_ne | 0.459 | 0.227 | 0.072 |
| doc | 0.369 | 0.010 | **0.015** |
| cot | 0.025 | 0.231 | 0.117 |
| mano | 0.157 | 0.075 | **−0.013** |

On arm D a 20-point change in true session accuracy moves the ATC-MC estimate by about
**one point**. Variance ratios confirm it: var(estimate)/var(true) is 0.115 for DoC and
0.082 for MaNo — these estimators are close to constants.

**This is a second, independent failure.** An estimator that were merely offset could be
corrected by subtracting a constant; one with slope ≈ 0 cannot, because it is not
responding to the session at all.

Note that slopes are below 1 even on arm A (0.16–0.52). MAE is nevertheless tiny there
because true session accuracy barely varies (sd ≈ 0.016). Good arm-A MAE therefore does
**not** demonstrate good session tracking, and should not be cited as if it did.

## 4. Diagnostic I — "bias-dominated" is now earned `POST-HOC`

| quantity | arm A | arm C | arm D |
|---|---:|---:|---:|
| slope of log(variance) on log(N) (−1 = ideal 1/N) | −0.885 | −0.317 | −0.394 |
| bias² share of MSE at N=400 | 0.377 | 0.923 | **0.850** |

Mean change in \|bias\| from N=25 to N=400 on arm D: **+0.0038** — nothing.

So variance does fall with N, \|bias\| does not, and at N=400 roughly **85% of MSE is
bias²**. The phrase *bias-dominated* is now measured rather than asserted.

A secondary observation: variance falls **more slowly than 1/N under shift** (−0.39 vs
−0.89 on arm A). Two mechanisms plausibly contribute — sessions overlap at large N
(Jaccard 0.009 at N=25 rising to 0.142 at N=400) and Tomato-Village supplies only 391
independent photographs — but this analysis does not separate them.

## 5. Diagnostic F — the failure is strongly class-conditional `POST-HOC`

Arm D, averaged over 20 models:

| class | true share | predicted share | recall | precision | mean confidence when predicted | confidence − precision |
|---|---:|---:|---:|---:|---:|---:|
| healthy | 0.134 | 0.049 | **0.079** | 0.519 | 0.629 | +0.110 |
| early_blight | 0.307 | **0.594** | 0.741 | **0.385** | **0.827** | **+0.442** |
| late_blight | 0.559 | 0.357 | 0.468 | 0.750 | 0.816 | +0.066 |

On arm A every recall is 0.95–0.97 and confidence − precision is ≈ −0.02.

**This is the concrete mechanism.** In the field the model funnels predictions into
*early blight* — predicting it for 59% of images when it is true for 31% — at 0.83
confidence and **0.385 precision**. A confidence-averaging estimator sees 0.83 and reports
accuracy near 0.83 while the truth is 0.47.

**Healthy recall collapses to 0.079.** The classifier almost never recognises a healthy
leaf in the field. That is also the safety-relevant direction: a gate driven by this model
would authorise spraying healthy plants.

## 6. Diagnostics D + E — deployable signals, and a Simpson trap `DEPLOYABLE`

Spearman correlation between each unlabelled signal and true session accuracy, **within
arm D**, N=100:

| signal | pooled ρ | per-backbone range | sign consistent? |
|---|---:|---|---|
| MMD (features vs source) | **+0.221** | **−0.400 … −0.108** | ✅ all negative |
| dominant class share | −0.089 | −0.455 … −0.027 | ✅ all negative |
| marginal TV vs source | +0.125 | −0.462 … +0.104 | ❌ flips |
| marginal entropy | −0.153 | −0.133 … +0.463 | ❌ flips |
| mean confidence | +0.063 | −0.061 … +0.188 | ❌ flips |

**The pooled MMD correlation is positive while every within-backbone correlation is
negative — a Simpson reversal.** Its cause is visible in the data: backbones with higher
mean MMD also have higher mean accuracy (ResNet 0.201/0.524, ViT 0.215/0.564 versus
EfficientNet 0.135/0.446, MobileNet 0.115/0.451), so the between-backbone trend inverts
the within-backbone one.

**Any future correction built on a shift statistic must operate within a fixed backbone.**
A pooled fit would learn the wrong sign.

Two signals — MMD and dominant-class-share — are directionally consistent across all four
backbones, in the physically sensible direction (more feature shift, or more collapsed
predictions, means lower accuracy). But magnitudes are weak (\|ρ\| 0.03–0.46) and vary
several-fold by backbone.

Across arms the same signals separate domains easily (mean confidence ρ = +0.663,
marginal TV −0.649, MMD −0.582). **Distinguishing lab from field is easy; ranking sessions
within the field is what fails** — and only the latter is the deployment question, since a
deployed rover is always in arm D.

## 7. Diagnostic G — the capture-day test, and why it is decisive `DEPLOYABLE` + `POST-HOC`

Three real acquisition days. True accuracy falls 21 points; the deployable signals barely
move:

| capture date | photographs | true accuracy | mean conf | mean entropy | MMD | dominant share |
|---|---:|---:|---:|---:|---:|---:|
| 2022-03-23 | 88 | **0.592** | 0.826 | 0.402 | 0.175 | 0.592 |
| 2022-03-24 | 154 | **0.484** | 0.823 | 0.407 | 0.169 | 0.614 |
| 2022-03-25 | 149 | **0.378** | 0.811 | 0.433 | 0.155 | 0.618 |
| **range** | | **0.214** | **0.015** | 0.031 | 0.020 | 0.026 |

- Mean confidence and entropy move in the **right direction** — but confidence moves
  0.015 while accuracy moves 0.214, roughly **14× too small** to be useful.
- **MMD moves the wrong way**: the hardest day has the *least* measured feature shift,
  contradicting its within-session negative correlation.
- The accuracy ordering is consistent across all four backbones; the MMD ordering is not.

n = 3 days, so this is direction and magnitude only, with no significance claim. But it is
the most operationally realistic test available, and no deployable signal passes it.

## 8. Diagnostic E — the COT hypothesis is **weakened** `DESCRIPTIVE, n = 4`

The pre-registered question was whether COT's architecture-dependent bias relates to its
use of the class marginal. Across four backbones the association looks strong — COT bias
versus marginal TV r = −0.978, versus marginal entropy r = +0.979.

**But the control refutes specificity.** MaNo shows r = −0.967 and DoC −0.784 against the
same quantity, and neither uses a class marginal. Only ATC-MC breaks the pattern (+0.335).

**Conclusion: the pattern is a general property of how each backbone's predictions
collapse under shift, not evidence that COT's marginal is responsible.** The hypothesis
is weakened, not supported. At n = 4 architectures this is hypothesis-generating at best,
exactly as pre-registered.

## 9. Diagnostic H — architecture `POST-HOC` + `DEPLOYABLE`

| backbone | arm-D accuracy | AUROC arm A → D | confidence − accuracy | mean estimator bias |
|---|---:|---|---:|---:|
| mobilenetv3_large | 0.455 | 0.908 → 0.533 | +0.404 | +0.328 |
| efficientnet_b0 | 0.445 | 0.915 → 0.571 | +0.351 | +0.305 |
| resnet50 | 0.530 | 0.978 → 0.647 | +0.243 | +0.224 |
| vit_small_patch16 | 0.570 | 0.985 → 0.674 | +0.287 | +0.273 |

The failure is **architecture-independent in direction** — all four collapse, all four
over-estimate — while its **magnitude is architecture-dependent**, and stronger backbones
degrade less. There is no monotone relation between source accuracy and estimator bias.

Classification: **architecture-independent in kind, architecture-dependent in degree, with
one estimator-specific interaction (COT).**

---

## 10. Answers to the Phase-5 questions

**A. Is "confidence decouples from correctness" directly supported?** **Yes**, and it is
now measured rather than inferred.

**B. Which quantities demonstrate it?** AUROC of confidence for correctness (0.942 →
0.606); confidence separation (0.22 → 0.05); confidence − accuracy (≈0 → +0.24…+0.40);
ECE (0.03 → 0.24…0.41); slope of estimate on truth (0.37…0.52 → 0.01…0.07).

**C. What is the failure, primarily?** **A combination of three measured components**, and
the thesis should say so rather than picking one:
1. **overconfidence** — sets the size of the bias, and accounts for DoC's bias almost
   exactly (to 0.028);
2. **loss of confidence-correctness discrimination** — removes the ability to rank
   sessions, giving slopes ≈ 0;
3. **class-conditional collapse** — the concrete driver, with prediction mass moving to
   early blight at 0.385 precision and 0.83 confidence, and healthy recall at 0.079.

**D. Does the mechanism differ by architecture?** In degree, not in kind.

**E. Is COT's architecture dependence explained by class marginals?** **Not supported.**
The same association appears for estimators that use no marginal.

**F. Does N reduce variance without reducing bias?** **Yes.** Variance falls (log-log
slope −0.39), \|bias\| changes by +0.004, and bias² is ~85% of MSE at N=400.

**G. What is genuinely available without labels?** Mean confidence, entropy, predicted
class marginal and its distances from source, dominant-class share, MMD and Fréchet
feature distances. **All of them are weak within-domain predictors** (\|ρ\| ≤ 0.46), only
two are sign-consistent across backbones, and none survives the capture-day test.

**H. What remains hypothesis?** Why some backbones degrade less; whether session overlap
or photograph clustering causes the sub-1/N variance decay; whether MMD's sign reversal
between session and day level is an artefact of scale or something structural.

---

## 11. Decision gate: **OUTCOME 2**

> *A clear failure mechanism exists but cannot be measured without labels.*
> → **Do not invent a correction. Frame it as a limitation of label-free estimation.**

The reasoning:

- The mechanism **is** clear and measured (§10-B).
- **Every quantity that measures it needs labels.** Confidence − accuracy needs accuracy;
  AUROC needs correctness; the class-conditional collapse needs the confusion matrix.
- The **deployable proxies are too weak to substitute**: within-domain \|ρ\| ≤ 0.46,
  inconsistent in magnitude across backbones, and the two most operationally realistic
  tests fail — the capture-day signals move ~14× too little, and MMD reverses sign
  between the session and day levels.

**This is a substantive finding, not a disappointing one.** It says something sharp about
the ceiling on label-free estimation in this setting: *the very thing that would tell you
the estimate is wrong — the gap between confidence and correctness — is itself
unobservable without the labels you do not have.*

**Recommendation: do not proceed to Phase 9 method development.** There is no measured,
deployable failure signal to correct with. The honest next step is **RQ5** — quantify how
few labelled field images would beat every label-free estimator, converting this into
practical guidance.

Two weak leads are recorded for completeness, neither strong enough to build on:
MMD and dominant-class-share are sign-consistent across all four backbones. Any future
attempt must operate **within a fixed backbone** (§6) and must be validated against the
capture-day test, which both currently fail.

---

## 12. Limitations of this phase

1. **Observational, not causal.** 30 fixed models, nothing intervened upon. "Explains"
   means "accounts for variance in", never "causes".
2. **n = 4 architectures and n = 3 capture days.** Diagnostics E and G are descriptive.
3. **One source dataset, one target field dataset.**
4. **Arm B excluded** — ResNet-50 only.
5. **ViT arm-A AUROC uses 3 of 5 seeds** (two seeds attain 100% accuracy).
6. **Sessions overlap** (Jaccard up to 0.142 at N=400) and Arm D has 391 independent
   photographs. Session-level correlations are not independent replicates.
7. **Image-level statistics are aggregated to explain session-level failure.** This is
   not a per-image correctness claim and must not be written as one.
