# Stop Condition F — investigation

Study protocol:

> | After | Stop if |
> |---|---|
> | V6 | ATC estimate constant across sessions → investigate before continuing |

**The condition was triggered. This document is the required investigation.**
Verdict: **not a bug — a real and reportable property of the estimators under
field shift.** The ladder continues, with the finding promoted to a headline
result rather than a footnote.

All numbers below are averaged over **10 trained models** (V1 and V2 protocols ×
5 seeds), not a single run.

---

## 1. What triggered it

The capture-day diagnostic (`scripts/run_capture_day_diagnostic.py`) treats each
Tomato-Village capture day as one session — a real acquisition outing rather than
a constructed subset. True accuracy varies enormously across the three days. The
estimates barely move.

| capture date | images | true accuracy | atc_mc | atc_ne | confidence | cot | doc | entropy | mano | nuclear_norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-03-23 | 620 | **0.655** | 0.836 | 0.799 | 0.775 | 0.584 | 0.816 | 0.468 | 0.933 | 0.808 |
| 2022-03-24 | 609 | **0.515** | 0.837 | 0.811 | 0.769 | 0.577 | 0.811 | 0.457 | 0.930 | 0.806 |
| 2022-03-25 | 387 | **0.345** | 0.808 | 0.752 | 0.759 | 0.568 | 0.800 | 0.439 | 0.928 | 0.802 |
| **spread** | | **0.309** | 0.029 | 0.059 | 0.016 | 0.017 | 0.016 | 0.029 | 0.004 | 0.006 |

True accuracy spans **31 points**. Every estimate spans **6 points or less**.
MaNo moves by 0.004 while the thing it is estimating moves by 0.309.

On the hardest day the classifier achieves **0.345** on a three-class problem —
barely above the 0.333 chance level — while DoC reports 0.800 and MaNo 0.928.

## 2. Is it a bug?

No. Three independent checks say the machinery is sound.

**(a) The same code is accurate when there is no shift.** On arm A (PlantVillage
held-out, leaf-grouped) DoC reaches MAE 0.0104 and ATC-MC 0.0113. A broken
estimator does not do that.

**(b) Correlation with the truth collapses only under shift.** Pearson r between
estimated and true session accuracy at N=100, over 200 bootstrap sessions × 10
models:

| estimator | arm A (no shift) | arm C (web) | arm D (real field) | sd of estimate (D) |
|---|---:|---:|---:|---:|
| atc_mc | 0.509 | 0.379 | **0.034** | 0.098 |
| atc_ne | 0.369 | 0.374 | **0.012** | 0.132 |
| confidence | 0.596 | 0.127 | **−0.044** | 0.036 |
| doc | 0.560 | 0.224 | **−0.016** | 0.036 |
| entropy | 0.496 | 0.168 | **−0.008** | 0.058 |
| mano | 0.244 | 0.495 | **0.260** | 0.029 |
| nuclear_norm | 0.273 | 0.519 | **0.266** | 0.092 |
| cot | 0.049 | 0.508 | **0.429** | 0.044 |

**Five of eight estimators have essentially zero or negative correlation with true
session accuracy on the real-field arm** — including ATC, the canonical method,
and DoC. Only COT (0.43), nuclear norm (0.27) and MaNo (0.26) retain any signal.

**(c) The estimate is far less variable than the truth.** On arm D the true
session accuracy has standard deviation **0.112**, but the estimates have 0.036
(confidence, DoC) to 0.132. The low-variance ones have collapsed toward a
constant; the higher-variance ones (ATC-NE at 0.132) vary *without tracking*,
which the near-zero correlation confirms is noise rather than signal.

## 3. Why it happens

Every estimator that failed here is a functional of the model's **confidence
distribution**: ATC thresholds it, DoC takes its mean, MaNo and nuclear norm take
matrix norms of it. That is sound only while confidence remains coupled to
correctness.

Under strong covariate shift the lab-trained network stays confident on images it
is getting wrong. Its confidence distribution is then nearly the same on an easy
day and on a hard one, even though accuracy has halved — so any statistic of that
distribution is nearly the same too. The estimator is not failing to compute its
statistic; the statistic has stopped carrying the information.

This is consistent with the prediction carried into the project from
arXiv:2507.22776 (see `docs/novelty_verification.md` §4), which found that
prevalence shift systematically breaks these estimators in medical imaging. Here
the same collapse appears under combined covariate and prevalence shift in
agriculture, and it is measurable at the level of an individual field session.

## 4. Consequence for the thesis

The framing changes from *how accurate is each estimator* to something sharper and
more useful:

> Under real lab-to-field shift, confidence-based accuracy estimators do not merely
> mis-estimate the session accuracy — they lose the ability to **rank** sessions at
> all, and they fail in the unsafe direction.

For a deployment gate this is the worst possible failure mode. A gate needs to
separate a good session from a bad one. An estimator with r ≈ 0 cannot do that at
any threshold: sweeping τ moves the operate/withhold split essentially at random
with respect to true session quality. This is why the gate operating characteristic
(Fig 10) is reported alongside MAE — MAE alone would understate how bad this is,
because a well-chosen constant can have a modest MAE while being useless as a
decision rule.

## 5. Smallest valid alternative (§12 requires one)

The ladder is **not** bypassed. What changes:

1. **Report correlation and estimate-variance alongside MAE everywhere.** MAE
   cannot distinguish "biased but informative" from "constant". `pearson_r`,
   `std_estimate` and `std_actual` are emitted in
   `results/tables/aggregate_metrics.parquet` and now appear in the Phase 1 report.
2. **Keep COT and nuclear norm prominent.** They retain the most session-level
   signal on arm D (r = 0.43 and 0.27). That is a genuine comparative result and it
   was not predictable in advance.
3. **Do not tune anything to rescue ATC.** §7's anti-confirmation rule is binding:
   the protocol was fixed before the results were seen, and a clear falsification
   is a good result.
4. **Treat the capture-day diagnostic as a primary result.** Three real acquisition
   days with a 31-point true accuracy spread is a far more convincing demonstration
   than bootstrap resampling, because the sessions are real outings.
5. **Carry the near-chance caveat.** Day 3 sits at 0.345 against a 0.333 chance
   level. The §12 condition "target accuracy < 0.40" is *not* tripped for arm D
   pooled (0.51) but the hardest day is close to uninformative on its own terms,
   and any per-day claim must say so.

## 6. Status

Investigated, explained, documented. **Ladder continues.** No estimator, threshold,
split or hyperparameter was changed in response to these results.
