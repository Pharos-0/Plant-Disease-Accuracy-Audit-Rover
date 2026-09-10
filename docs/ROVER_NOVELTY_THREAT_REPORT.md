# Rover Novelty Threat Report

**2026-09-11.** The strongest arguments *against* the rover integration being novel, stated
as an examiner would.

## Threat 1 — "This is runtime perception monitoring, which automotive solved years ago"

**Strength: HIGH.** arXiv returns 15 records for runtime monitoring of perception in
autonomous systems, including *PerceMon* (2021), *Certified Control* (2021), *Monitoring of
Perception Systems* (2022) and runtime validation for automated driving (2024) and trains
(2024).

**Honest response.** The *concept* is established and we do not claim it. What differs is
the mechanism — those works use formal specifications, redundant sensing or verifiable
checkers; ours uses dataset-level label-free accuracy estimation — and the domain. **This
reduces the rover's contribution to an application, not an idea.**

## Threat 2 — "It's an obvious combination"

**Strength: MEDIUM-HIGH.** "Take runtime perception monitoring, apply it to an agricultural
robot, use a label-free estimator as the signal" is describable in one sentence.

**Honest response.** We cannot rebut this on novelty. We can only rebut it on **evidence**:
we *evaluated* the combination and found it **does not work** — no signal ranks field
sessions usefully (best ρ = 0.22), and at any plausible threshold ~0.1% of field sessions
are safe. A negative evaluation of an obvious combination is a legitimate contribution;
proposing the combination would not have been.

## Threat 3 — "The vineyard paper already did agricultural cross-domain reliability"

**Strength: MEDIUM.** *From vineyard to vision* (2025) evaluates six detectors across three
countries, argues a new reliability metric, and gives a deployment decision matrix.

**Honest response.** Every metric it uses requires **labels**. It is labelled cross-domain
evaluation. Our question is what you can know **without** target labels. Level 2 overlap.

## Threat 4 — "ODP-Bench already benchmarks these estimators"

**Strength: MEDIUM.** True, and we must not claim a first systematic benchmark.

**Honest response.** ODP-Bench covers corruption/OOD benchmarks, not agricultural
laboratory-to-field imagery, and does not measure target responsiveness.

## Threat 5 — "TI is just a Sobol index"

**Strength: HIGH, and conceded.** It is. We state the equivalence rather than defend against
it, and claim only the application.

## Residual defensible position

> No prior work gating agricultural autonomous action on label-free accuracy estimation was
> identified in the sources searched — and we show, by measurement, that it would not work
> under this shift.

Not "first". Not "novel concept". **An evaluated negative.**
