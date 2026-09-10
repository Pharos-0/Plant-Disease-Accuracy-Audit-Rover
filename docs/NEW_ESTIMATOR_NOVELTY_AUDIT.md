# Novelty and Equivalence Audit — Target Informativeness

**Phases 3–4.** Written before any candidate estimator is implemented, so the equivalence
verdicts cannot be shaped by whether a candidate later performs well.

Evidence status is marked throughout: **VERIFIED** (read from a file or executed),
**SOURCE-SUPPORTED** (a named paper), **DERIVATION** (mathematics stated here),
**INFERENCE**, or **NOT VERIFIED**.

---

## 1. Formal setting

Let `f` be a frozen classifier, `S` the source-side labelled information available at fit
time (source-validation logits, features, labels), and `T` an unlabelled target sample. A
label-free accuracy estimator is a map

```
    A : (S, T)  ->  [0, 1]
```

We want to quantify how much of the variation in `A` is driven by `T`.

## 2. Why the additive decomposition cannot be the definition

Phase 7b used `estimate = intercept + slope·score(T)` and reported an intercept share of
93–98% for GdScore. **DERIVATION + VERIFIED (`docs/NEW_ESTIMATOR_REPOSITORY_AUDIT.md` §2):**
that form exists for only three of nine estimators — `nuclear_norm`, `MaNo`, `GdScore` —
and it comes from `_SourceCalibratedScore`, **our own calibration wrapper**, not from the
published methods.

- `confidence`, `entropy`: `needs_source = False`. No anchor exists, so the target share is
  1 **by construction**. A definitional artefact, not a finding.
- `ATC`, `COT`: source information enters as a **parameter** (a threshold; a class
  marginal), not an additive term. The decomposition is undefined, not merely small.
- `DoC`: `source_acc − (source_conf − conf(T))`. Here the anchor **is** intrinsic to the
  published method.

**Conclusion (DERIVATION):** an additive split is estimator-specific and partly an artefact
of our own design choice. It must be replaced by a definition that is well-defined for any
`A`, whatever its internal form.

## 3. Definition adopted: perturbation-based Target Informativeness

Place distributions over the two inputs — `𝒮` over plausible source calibration sets
(bootstrap resamples of source-validation), `𝒯` over plausible target samples (bootstrap
resamples of the unlabelled pool). Then define

```
    V_T      = Var_T ( E_S [ A(S, T) ] )        variation attributable to the target
    V_S      = Var_S ( E_T [ A(S, T) ] )        variation attributable to the source anchor
    TI       = V_T / (V_T + V_S)
```

`TI ∈ [0, 1]`. `TI = 0` when the estimator ignores the target entirely; `TI = 1` when the
source anchor is irrelevant.

Two companion diagnostics, both used below:

```
    anchor sensitivity  σ_S = sd_S ( A(S, T₀) )   target held at the full pool
    target sensitivity  σ_T = sd_T ( A(S₀, T) )   source held at the full calibration set
```

and a **target-removal** check: replace `T` with an uninformative surrogate (shuffled,
duplicated, distribution-matched noise, a fixed reference set) and measure whether `A`
moves at all.

## 4. Equivalence analysis — the metric

This is the part the direction most likely fails on, so it is stated first and plainly.

| existing framework | relationship | verdict |
|---|---|---|
| **Sobol' indices / variance-based global sensitivity analysis** (Sobol' 1993) | `TI` as defined is precisely a **first-order Sobol' index** for the input group `T`, normalised by the two-group total. | **MATHEMATICALLY EQUIVALENT.** The metric is *not* new mathematics. |
| **Functional ANOVA decomposition** | The same variance decomposition, expressed as a functional expansion. | **EQUIVALENT.** |
| **Permutation importance** (Breiman 2001) | The target-removal check is group-level permutation importance applied to an entire input block. | **EQUIVALENT** for that component. |
| **Influence functions / leave-one-out** | Measures per-sample influence; `TI` is a variance ratio over input groups. Related, not the same. | **DISTINCT but adjacent.** |
| **Sobol-based explanation of neural networks** (*Look at the Variance!*, 2021) | Applies Sobol' indices to explain a network's **predictions**. | **SOURCE-SUPPORTED.** Same tool, different object. |

**Verdict on the metric: NOT NOVEL AS MATHEMATICS.** Any claim of a "new metric" must be
withdrawn. What is defensible is the *object of study*: applying variance-based sensitivity
analysis to **label-free accuracy estimators**, treating source-calibration and
target-sample as the two input groups.

**Targeted verification (VERIFIED, executed 2026-09-10, arXiv API):**

| query | arXiv total |
|---|---:|
| `abs:"sensitivity analysis" AND abs:"accuracy estimation" AND abs:"distribution shift"` | **0** |
| `abs:"permutation importance" AND abs:"model evaluation"` | **0** |
| `abs:"how much" AND abs:"unlabeled" AND abs:"accuracy estimation"` | **0** |
| `abs:"Sobol" AND abs:"neural network" AND abs:"sensitivity"` | 10 (prediction explanation, not estimator auditing) |

Wording this supports: *"No prior work applying variance-based sensitivity analysis to
label-free accuracy estimators was identified in the searched sources."* It does **not**
support "first" or "nobody has done this".

## 5. Equivalence analysis — the candidate estimators

Each family from the brief, with the strongest argument *against* it constructed
deliberately, per the brief's §38.

### Family A — target-information-weighted calibration
`estimate = anchor + λ(T) · correction`, `λ` from target reliability.
**Argument against:** this is a weighted average of a prior (anchor) and a data-driven term,
with the weight set by estimated reliability. That is **empirical-Bayes shrinkage**.
→ **DERIVATIVE.**

### Family B — anchor/target decomposition, combined by a reliability rule
Structurally identical to A once the combination rule is written down.
→ **DERIVATIVE.**

### Family C — target-sensitivity-calibrated estimator
Shrink the estimate by how responsive it is to target changes. The *source* of the weight
(measured sensitivity) is unusual; the *operation* is still shrinkage toward a prior.
→ **DERIVATIVE**, with a non-standard weight.

### Family D — robust target correction with source fallback
Shrinkage with a fallback; the fallback is the prior.
→ **DERIVATIVE.**

### Family E — reliability-weighted combination of existing estimators
Ensemble weighting. The brief itself rules this out as not automatically new.
→ **DERIVATIVE / COMBINATION.**

### Family F — adaptive shrinkage toward the source anchor
Explicitly **James–Stein / empirical Bayes**.
→ **MATHEMATICALLY EQUIVALENT to a known estimator class.**

### 5.1 The structural objection that applies to all six

**DERIVATION.** Every family above responds to *weak target information* by leaning **more**
heavily on the source anchor. But the motivating observation is that anchor dominance is
the pathology. So these candidates do not repair the failure — **they formalise it.**
Shrinkage lowers variance and can lower MAE while making `TI` *worse*, which is the exact
confusion the audit exists to expose.

**Consequence:** the "new estimator" route is, on present analysis, derivative *and*
directionally wrong. This is recorded before any implementation, and it is the reason the
empirical work below is aimed at the audit rather than at a leaderboard win.

### 5.2 The one use that is not shrinkage

If `TI` is low, the honest action is not to produce a better number but to **decline to
report one** — a validity gate on the estimator's output.

| nearest existing work | relationship |
|---|---|
| selective prediction / reject option | applied to a **classifier's per-sample** output; here it would apply to an **accuracy estimator's dataset-level** output |
| conformal prediction | provides coverage guarantees for predictions, not a validity gate on an estimator |
| Prediction-Powered Risk Monitoring (2026) | monitors deployed risk using a few true labels; **uses labels**, `TI` does not |

→ **POSSIBLY NOVEL — NEEDS HUMAN REVIEW.** Not claimed as novel here.

## 6. Classification summary

| candidate | verdict |
|---|---|
| `TI` as a metric | **MATHEMATICALLY EQUIVALENT** to a first-order Sobol' index |
| `TI` applied to label-free accuracy estimators | **POSSIBLY NOVEL — NEEDS HUMAN REVIEW** (no prior work identified; absence is not proof) |
| Family A, B, C, D | **DERIVATIVE** (empirical-Bayes shrinkage) |
| Family E | **DERIVATIVE / COMBINATION** (ensemble weighting) |
| Family F | **MATHEMATICALLY EQUIVALENT** (James–Stein) |
| `TI` as a validity gate | **POSSIBLY NOVEL — NEEDS HUMAN REVIEW** |

## 7. The experiment that decides the direction

Everything above is analysis. One empirical question determines whether any of it matters:

> **Does `TI` predict estimator error?** If estimators with low `TI` are systematically less
> reliable, `TI` is a useful diagnostic. If `TI` is uncorrelated with `|estimate − truth|`,
> the direction is dead regardless of its mathematical standing.

**Status: NOT RUN.** No prediction is offered here. The experiment is specified in
`docs/NEW_ESTIMATOR_FEASIBILITY.md` and executed in `scripts/ti_audit.py`.
