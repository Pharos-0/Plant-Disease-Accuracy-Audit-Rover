# Stop Condition F — investigation

MASTER_PROMPT §12:

> | After | Stop if |
> |---|---|
> | V6 | ATC estimate constant across sessions → investigate before continuing |

**The condition was triggered. This document is the required investigation.**
Verdict: **not a bug — a real and reportable property of the estimators under
field shift.** The ladder continues, with the finding promoted to a headline
result rather than a footnote.

---

## 1. What triggered it

The capture-day diagnostic (`scripts/run_capture_day_diagnostic.py`) treats each
Tomato-Village capture day as one session. True accuracy varies enormously across
the three days; the estimates barely move.

| capture date | images | **true accuracy** | ATC-MC | DoC | MaNo | entropy | COT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022-03-23 | 620 | **0.635** | 0.750 | 0.810 | 0.892 | 0.461 | 0.586 |
| 2022-03-24 | 609 | **0.493** | 0.783 | 0.818 | 0.893 | 0.473 | 0.584 |
| 2022-03-25 | 387 | **0.306** | 0.758 | 0.804 | 0.886 | 0.440 | 0.546 |
| **spread** | | **0.329** | 0.033 | 0.014 | 0.007 | 0.033 | 0.040 |

True accuracy spans **33 points**. Every estimate spans **4 points or less**, and
none of them moves in the right direction reliably. DoC reports ~0.81 on the day
the classifier actually achieves **0.306 — below the 0.333 chance level for three
classes**.

## 2. Is it a bug?

No. Three independent checks say the machinery is sound.

**(a) The same code is accurate when there is no shift.** On arm A (PlantVillage
held-out, leaf-grouped) ATC-MC reaches MAE 0.0126 and DoC 0.0118, and the spread
of the estimates matches the spread of the truth almost exactly
(`std_estimate` 0.0147 vs `std_actual` 0.0142). A broken estimator does not do that.

**(b) Correlation with the truth collapses only under shift.** Pearson r between
estimated and true session accuracy, at N=100, over 100 bootstrap sessions:

| estimator | arm A (no shift) | arm C (web) | arm D (real field) |
|---|---:|---:|---:|
| atc_mc | 0.656 | 0.214 | **0.008** |
| atc_ne | 0.521 | 0.353 | 0.060 |
| confidence | 0.629 | 0.413 | 0.138 |
| doc | 0.629 | 0.413 | 0.138 |
| mano | 0.381 | 0.497 | 0.225 |
| nuclear_norm | 0.487 | 0.401 | 0.378 |
| entropy | 0.535 | 0.500 | 0.175 |
| cot | 0.109 | 0.122 | 0.424 |

**On the real-field arm, ATC — the canonical method — has essentially zero
correlation with true session accuracy (r = 0.008).**

**(c) The variance of the estimate collapses relative to the truth.** Arm D,
N=100: `std_actual` = 0.106, but `std_estimate` = 0.026 for DoC and confidence.
The estimate is roughly four times less variable than the quantity it is
estimating. It has become close to a constant.

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
separate a good session from a bad one. An estimator with r = 0.008 cannot do that
at any threshold: sweeping τ moves the operate/withhold split essentially at
random with respect to the true session quality. This is why the gate operating
characteristic (Fig 10) is reported alongside MAE — MAE alone would understate how
bad this is, because a well-chosen constant can have a modest MAE while being
useless as a decision rule.

## 5. Smallest valid alternative (§12 requires one)

The ladder is **not** bypassed. What changes:

1. **Report correlation and estimate-variance alongside MAE everywhere.** MAE
   cannot distinguish "biased but informative" from "constant". `pearson_r`,
   `std_estimate` and `std_actual` are already emitted in
   `results/tables/aggregate_metrics.parquet`.
2. **Keep COT and nuclear norm prominent.** They retain the most session-level
   signal on arm D (r = 0.42 and 0.38). That is a genuine comparative result and
   it was not predictable in advance.
3. **Do not tune anything to rescue ATC.** §7's anti-confirmation rule is binding:
   the protocol was fixed before the results were seen, and a clear falsification
   is a good result.
4. **Treat the capture-day diagnostic as a primary result**, not a supplementary
   one. Three real acquisition days with a 33-point true accuracy spread is a far
   more convincing demonstration than bootstrap resampling, because the sessions
   are real outings rather than constructed subsets.
5. **Record that day 3 falls below chance.** At 0.306 on a three-class problem the
   classifier is worse than random on that day. Per §12 the "target accuracy < 0.40"
   condition is *not* tripped for arm D pooled (0.50), but it **is** breached for
   that day considered alone. Any per-day analysis must carry that caveat, since
   accuracy estimates for a below-chance session are not meaningfully interpretable.

## 6. Status

Investigated, explained, documented. **Ladder continues.** No estimator, threshold,
split or hyperparameter was changed in response to these results.
