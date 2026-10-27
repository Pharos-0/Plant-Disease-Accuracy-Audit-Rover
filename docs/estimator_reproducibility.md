# Estimator Reproducibility

One row per estimator, with an honest reproduction level (study protocol).

| Level | Meaning |
|---|---|
| **L0** | paper read |
| **L1** | code runs |
| **L2** | works on example data |
| **L3** | reproduces a published number / the reference implementation within tolerance |
| **L4** | transfers to our experiment |

Every level below is backed by a test in `tests/test_atc_reference.py` or
`tests/test_leakage_firewall.py`, not by assertion. Reference repositories were
cloned to `C:\ds\_refrepos\` and are read directly by the tests.

---

## 1. Summary table

| Estimator | Version | Kind | Level | Official code used | Evidence |
|---|---|---|---|---|---|
| Mean max-softmax confidence | V4 | **baseline** | L4 | n/a — one line | `test_estimates_are_probabilities` |
| Mean normalised entropy | V5 | **baseline** | L4 | n/a — one line | `test_estimates_are_probabilities` |
| **ATC-MC** | V6 | method | **L3** | ✅ `saurabhgarg1996/ATC_code` | threshold identical to `find_ATC_threshold` on 8 random seeds; end-to-end estimate matches `get_ATC_acc` to 1e-12 |
| **ATC-NE** | V6 | method | **L3** | ✅ same | plus `negative_entropy` matches `ATC_helper.get_entropy` to 1e-9 |
| **DoC** | V7 | method | **L3** | ✅ same repo, `predict_acc_helper.get_doc` | formula match to 1e-12 |
| **COT** | V8 | method | **L3** | ✅ `luyuzhe111/COT` | matches official `ot.emd2` formulation to <1e-3 (see §3) |
| Nuclear norm (Confidence & Dispersity) | V8 | method | **L2** | ❌ not obtained | runs and behaves sensibly on synthetic shift; calibration is ours (§4) |
| MaNo | V8 | method | **L2** | ❌ repo not found at the URL tried | runs and behaves sensibly on synthetic shift; calibration is ours (§4) |

---

## 2. ATC and DoC — what was verified

The official ATC release was cloned and is imported directly by the test suite.

**Threshold selection.** `src/estimators/methods.py::find_atc_threshold` is a
vectorised transcription of `ATC_helper.find_ATC_threshold`. The reference sorts
the scores and sweeps, tracking `fp` (mistakes still above the threshold) and
`fn` (correct predictions already below it), choosing the threshold that
minimises `|fp - fn|`. Our version computes the same quantity with cumulative
sums and reproduces both the returned `min_fp_fn` and the threshold exactly
across eight randomised trials spanning error rates from 1% to 50%.

An earlier draft of our code used `np.quantile(scores, source_error)` instead.
That is the same idea but not the same number, and it was replaced once the
reference was in hand.

**DoC.** The official usage is `v1acc + get_doc(conf_source, conf_target)` where
`get_doc(a, b) = mean(b) - mean(a)`, i.e. source accuracy minus the drop in mean
confidence. Ours matches to 1e-12.

**One documented divergence — temperature scaling.** The ATC paper specifies
calibrated probabilities. The released `get_acc.py` fits a `TempScaling`
calibrator and then computes `calib_probs = softmax(inverse_softmax(probs))`,
which is the identity — so the released pipeline effectively runs *uncalibrated*.
We follow the released code (no temperature scaling) so that our ATC is
comparable to the numbers that code produces, and record the discrepancy here
rather than silently picking one. Adding temperature scaling fitted on source
validation would remain label-free with respect to the target and is a
legitimate future ablation.

---

## 3. COT — equivalence argument

The official implementation builds `n` one-hot vectors sampled from the source
label distribution and solves a balanced `n x n` earth-mover problem with cost
`||e_k - p||_1 / 2`. For a one-hot `e_k`,

```
||e_k - p||_1 = (1 - p_k) + sum_{j != k} p_j = 2 (1 - p_k)
```

so that cost is exactly `1 - p_k`, which is the cost our implementation uses.
We solve the equivalent **semi-discrete** problem — `n` samples against `K`
classes with the exact source marginal — instead of drawing a finite one-hot
sample. That removes the sampling noise; it is the same estimator with lower
variance. Verified numerically against `ot.emd2` (POT) to better than 1e-3, with
the residual attributable to integer rounding when realising the marginal.

The official code returns an **error**; we return `1 - error` so that every
estimator in the suite is on an accuracy scale.

---

## 4. Nuclear norm and MaNo — what is ours and why

Both methods produce a **score that correlates with accuracy**, not an accuracy.
Turning a score into an accuracy needs a mapping. Ours is fitted on 200 bootstrap
subsets of the *source validation* set, each contributing a (score, accuracy)
pair, then a linear fit — implemented in `_SourceCalibratedScore`.

This uses source labels only, so the target stays unlabeled and the method
remains eligible. But the limitation is real and must be stated in the thesis:
**a mapping fitted on source cannot know how the score behaves once the
distribution moves.** On synthetic covariate shift both over-estimate badly
(+0.21 and +0.25), which is consistent with that limitation rather than evidence
of a coding error.

These are marked **L2, not L3**: no official implementation was obtained, so
there is no reference number to match. They are reported as *our implementation
of the published score*, never as the authors' method reproduced.

---

## 5. Eligibility review — required before inclusion

Every candidate is first classified **eligible / ineligible** for this setting:
a **frozen** classifier evaluated on an **unlabeled deployment session**.

| Candidate | Verdict | Reasoning |
|---|---|---|
| Confidence, Entropy | **eligible** | pure functions of target logits; no fitting at all |
| ATC-MC / ATC-NE | **eligible** | threshold fitted on *source* validation labels; target unlabeled |
| DoC | **eligible** | needs source accuracy and source confidence only |
| COT | **eligible** *(with a caveat)* | needs the *target* label marginal, which is unobservable; the source marginal substitutes for it. Under prevalence shift that substitution is wrong by construction — which is precisely hypothesis H2, so COT is kept as an informative failure case rather than excluded |
| Nuclear norm, MaNo | **eligible** | score computed on target logits; calibration uses source labels only |
| **AETTA** | **INELIGIBLE — excluded** | see §5.1 |
| **AutoEval** | **INELIGIBLE as published — excluded** | see §5.2 |

### 5.1 AETTA (label-free accuracy estimation for test-time adaptation)

**Excluded, and the exclusion is the finding.**

AETTA estimates accuracy for a model that is *adapting at test time*. Its
estimate is built from the disagreement between the model's prediction and
predictions obtained under dropout-perturbed variants of the **adapting** model,
and its error analysis is framed around adaptation failure. The setting assumed
is a model whose parameters change as the test stream arrives.

Our setting is the opposite by design: the classifier is **frozen** at
deployment. A sprayer rover carries fixed weights; nothing adapts between
sessions. Applying AETTA here would mean either (a) introducing test-time
adaptation purely so that AETTA becomes applicable, which changes the system
under study rather than measuring it, or (b) running AETTA against a frozen
model, where its central assumption simply does not hold and any resulting
number would be uninterpretable.

Per section 8 — *do not adapt it merely to increase the estimator count*. An
argued exclusion is the stronger contribution. Recorded here, reported in the
thesis.

### 5.2 AutoEval

**Excluded as published.**

AutoEval (Deng & Zheng, CVPR 2021) regresses accuracy from dataset-level
features, and the regressor is **meta-trained on a population of sample sets
whose accuracies are known**. Producing that training signal requires labeled
sets drawn from the relevant distribution. In our setting the only labeled data
is the laboratory source domain, so a meta-regressor could be trained on source
sample sets alone — which is exactly the source-only calibration already
represented by the nuclear-norm and MaNo rows, and which we know cannot
anticipate target-side behaviour.

Constructing the meta-training population from *labeled target-like* data would
break the label-free premise and trips the section 12 stop condition: *an
estimator that requires labeled target data to fit is not label-free here.*

AutoEval is therefore documented and excluded, with the nuclear-norm and MaNo
rows standing as the source-calibrated-regression family in the benchmark.

---

## 6. Baselines are labelled baselines

Mean confidence and mean entropy are **baselines**. They are reported as such in
every table and figure and are never presented as a contribution. They are in the
suite to answer a real question — *does the trivial estimator already work?* —
which is worth answering before importing anything more elaborate.
