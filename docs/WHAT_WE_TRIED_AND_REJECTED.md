# What We Tried and Rejected

**2026-09-11.** Five substantial ideas were pursued and abandoned on their own evidence.
They are recorded in full because a thesis that shows only what worked is not showing its
work.

---

## 1. A new target-information estimator

**Why it seemed reasonable.** Phase 7b measured GdScore's output as 93–98% source-anchor
and 1.6–6.6% target signal. If an estimator is barely using the target, correcting for that
ought to help.

**Experiment.** Six candidate families (A–F) analysed mathematically before implementation
(`docs/NEW_ESTIMATOR_NOVELTY_AUDIT.md`).

**Result.** Every family reduced to empirical-Bayes shrinkage or ensemble weighting; Family
F is James–Stein outright.

**Why rejected.** All six respond to *weak* target information by leaning **harder** on the
anchor — they formalise the pathology rather than repair it.

**Lesson.** An intuition about a failure mode does not imply a correction for it. Deriving
the candidate before coding it saved implementing five dead ends.

## 2. TI-adaptive shrinkage (Family F, implemented anyway)

**Why it seemed reasonable.** Rejecting the strongest candidate on paper is weaker evidence
than testing it.

**Experiment.** `scripts/ti_candidate.py`: `estimate = w·base(T) + (1−w)·source_accuracy`,
`w` = measured TI. Ablations: fixed `w = 0.5`, and `w = 0` (anchor only). 20 models.

**Result.** **0 of 9 bases improved. 7 significantly harmed** (paired Wilcoxon, Holm),
largest +0.3099 on COT. `anchor_only` MAE 0.4629.

**Why rejected.** The source anchor is a very poor predictor under this shift (source
accuracy 0.963, target 0.500), so shrinking toward it can only hurt — and the method shrinks
hardest exactly when TI is low.

**Lesson.** The derivation predicted the direction of failure, and the experiment confirmed
it. Reported in full rather than quietly dropped.

## 3. The deployment gate as an autonomy enabler

**Why it seemed reasonable.** `session_gate.py` already produced OPERATE / REVIEW /
WITHHOLD. Gating autonomy on estimated reliability is the obvious application.

**Experiment.** `scripts/gate_evaluation.py`: 75,240 session-signal pairs, full operating
characteristic swept, no threshold tuned; baselines always-operate, never-operate,
confidence, nine estimators, oracle.

**Result.** At τ_safe = 0.80 only **0.2%** (PlantDoc) and **0.1%** (Tomato-Village) of
sessions are genuinely safe. Within-field ranking: best ρ = 0.22.

**Why rejected.** The classifier is ≈0.50 in the field. No gate can find safe sessions that
do not exist. The bottleneck is the classifier, not the gate.

**Lesson.** Evaluating an obvious combination is a contribution; proposing it would not have
been. The negative result is the deployment recommendation.

## 4. Pooled-model session ranking

**Why it seemed reasonable.** Pooling all 20 models gives more data and tighter statistics.

**Experiment.** Spearman(signal, true accuracy) pooled across models, then repeated paired
per model.

**Result.** Pooled, GdScore looked like the **best** ranker (ρ = 0.696 on PlantDoc). Paired
per model it is the **worst** (0.076), and entropy beats it significantly (−0.148, 95% CI
[−0.215, −0.082], p = 8.3e-05, n = 40).

**Why rejected.** Pooled, GdScore was tracking *between-model* accuracy differences. A rover
has one model, so the within-model paired analysis is the one that answers the question.

**Lesson.** Third Simpson-type reversal in this project. The favourable analysis was the
wrong one, again.

## 5. The calibration-versus-discrimination inversion hypothesis

**Why it seemed reasonable.** The pooled numbers suggested that the estimators with the
worst MAE were the best rankers — an attractive story about calibration and discrimination
measuring different things.

**Experiment.** Spearman(MAE, rank ability) across the nine, using the correct per-model
paired ranking.

**Result.** **−0.218, p = 0.574.** No significant relationship.

**Why rejected.** The apparent inversion was an artefact of the same pooling error as #4.

**Lesson.** A finding that survives only under the wrong analysis is not a finding. Caught
before it reached the thesis.

---

## Also caught and corrected during the work

| # | Error | Direction it pushed | How it was caught |
|---|---|---|---|
| 1 | HTTP 429s recorded as zero literature hits | towards an emptier gap | `plant disease AND domain shift` returning 0 is not credible |
| 2 | Screening filters discarding PPI papers | towards an emptier gap | the protocol's mandated excluded-set audit |
| 3 | arXiv queries built as impossible exact phrases | towards an emptier gap | searching for a paper known to exist |
| 4 | Bare `\bMaNo\b` matching *Mano River*, *Mano Dura*, NFV *MANO* | towards an inflated corpus | reading the top hits |
| 5 | Degenerate synthetic world (source accuracy 1.000, bootstrap sd 0.0000) drove TI to 1.0 for every probe | towards a broken measure | the known-α validation failed |
| 6 | Camera probe passing a blind camera on `frames > 0` | towards a false all-clear | 13 byte-identical black frames |

**Every one of the first four pushed towards a cleaner-looking gap.** That is the direction
to distrust, and it is why the audit steps exist.
