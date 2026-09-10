# Rover Research Integration — Evaluated, Not Assumed

**2026-09-11.** `src/rover/session_gate.py` already implements a session-level
OPERATE / REVIEW / WITHHOLD gate. Its thresholds were admitted guesses and it had never
been scored against the truth. This document reports that evaluation.

Experiment: `scripts/gate_evaluation.py` — 75,240 (signal, session) pairs; 3 target arms ×
20 models × N ∈ {25,50,100,200} × 40 repetitions × 9 signals; photograph-level sampling
without replacement. **No threshold was tuned**: the full operating characteristic is swept.
Outputs in `results/gate_evaluation/`.

---

## 1. The gate cannot enable autonomy here, and the reason is not the gate

True session accuracy, measured (`gate_sessions.csv`):

| arm | mean | sd | min | max |
|---|---:|---:|---:|---:|
| source_test | 0.9622 | 0.033 | 0.84 | 1.00 |
| plantdoc | 0.4974 | 0.083 | 0.16 | 0.84 |
| tomatovillage | 0.4997 | 0.095 | 0.057 | 0.853 |

Fraction of sessions where the classifier is genuinely good enough to act on:

| τ_safe | source_test | plantdoc | tomatovillage |
|---:|---:|---:|---:|
| 0.55 | 1.000 | 0.260 | 0.290 |
| 0.65 | 1.000 | 0.039 | 0.067 |
| 0.70 | 1.000 | 0.011 | 0.018 |
| 0.80 | 1.000 | **0.002** | **0.001** |

> **At any operating threshold a sprayer would plausibly require, essentially no field
> session is safe.** The correct decision is "do not operate autonomously" — and reaching it
> needs no estimator at all. The bottleneck is the classifier (≈0.50 in the field), not the
> gate.

## 2. Two different questions, two different answers

**(a) "Has the domain shifted?"** — pooled across arms, AUC for separating safe from unsafe
sessions:

| signal | τ=0.60 | τ=0.65 | τ=0.70 |
|---|---:|---:|---:|
| cot | 0.866 | 0.943 | **0.981** |
| doc | 0.894 | 0.944 | 0.968 |
| confidence | 0.842 | 0.914 | 0.949 |
| gdscore | 0.650 | 0.588 | 0.563 |

Easy — 0.80–0.98 for most signals. **But this is essentially OOD detection**: it separates
laboratory from field. It does not need an accuracy estimator, and it must not be reported
as one working.

**(b) "Which of my field sessions is my classifier worse on?"** — the question a rover
already in a field actually asks. Per-model paired Spearman(score, true accuracy), averaged
over 40 model×arm pairs:

| signal | rank ability |
|---|---:|
| entropy | **0.224** |
| mano | 0.214 |
| confidence / doc | 0.206 |
| nuclear_norm | 0.200 |
| cot | 0.156 |
| atc_ne / atc_mc | 0.124 / 0.123 |
| gdscore | **0.076** |

**Nothing ranks well.** The best signal reaches ρ = 0.22.

### A Simpson reversal, caught

Pooling all 20 models made GdScore look like the **best** ranker (ρ = 0.696 on PlantDoc).
The per-model paired analysis shows it is the **worst** (0.076), and entropy beats it
significantly (difference −0.148, 95% CI [−0.215, −0.082], Wilcoxon p = 8.3e-05, n = 40).
Pooled, GdScore was tracking *between-model* accuracy differences, not within-model session
differences. A rover has one model, so the paired analysis is the correct one. This is the
third Simpson-type reversal found in this project; it is recorded rather than exploited.

Correspondingly, Spearman(MAE, rank ability) across the nine = **−0.218 (p = 0.574)** —
**no significant inversion**, and no basis for a "calibration vs discrimination" claim.

## 3. Consequence for the architecture

The proposed chain — camera → classifier → reliability assessment → TRUST / CONSERVATIVE /
HUMAN — is **not supported by evidence** as an autonomy enabler. What is supported:

- A **shift detector** would correctly refuse autonomy in the field. It would also refuse it
  everywhere, because the classifier is genuinely not good enough there.
- **TI must not gate anything.** Confirmed independently: TI does not predict error, and the
  sign of its relationship reverses with shift severity.

## 4. What the rover should be

**A deployment case study and demonstration**, retaining the existing two-mode design:

- **NORMAL MODE** — classifier → downstream autonomous action. Works with the research
  module disabled.
- **RESEARCH MODE** — classifier → session gate → operate / review / withhold, used to
  *demonstrate and instrument* the finding, not to claim safe autonomy.

The gate's real demonstrated value is **negative and still useful**: it correctly withholds
in the field. That is the honest deployment recommendation — under this shift, autonomous
spraying should not be enabled, and a small labelled sample (RQ5: ~20–50 photographs) is the
practical route to knowing field accuracy.

## 5. Never claim

- ✗ that the estimator knows the correct disease;
- ✗ that it knows whether an individual leaf prediction is correct — `SessionGate` raises
  `PerImageQueryError` precisely to prevent this;
- ✗ that a predicted disease maps to a validated chemical treatment — downstream action
  stays abstract;
- ✗ that any signal makes autonomous spraying safe under this shift.
