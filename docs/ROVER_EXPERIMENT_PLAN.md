# Rover Experiment Plan

**2026-09-11.** Deliberately short: the gate evaluation removed the case for a large rover
experiment programme.

## Status of field access

**No validated rover field data exists.** `docs/rover_hardware_test_report.md` records the
camera as `NOT RUN`, and the camera currently returns a blank stream (60 frames, per-frame
std 0.0000). **Nothing is fabricated.** Experimental validation therefore rests on
PlantVillage → {source_test, PlantDoc, Tomato-Village}, and the rover is a **deployment
case study and demonstration**.

## What is already done

| experiment | status |
|---|---|
| Zero / severe shift | **done** — three arms |
| Multiple sample sizes | **done** — N = 25/50/100/200 |
| Multiple architectures, seeds | **done** — 4 × 5 |
| Multiple target domains | **done** — 3 |
| Baselines (Part 10) | **done** — always-operate, never-operate, confidence, 9 estimators, oracle |
| Statistical testing | **done** — paired Wilcoxon, bootstrap CIs, Holm |
| Failure analysis | **done** — Phase 7b, TI audit, gate evaluation |
| Reproducibility | **done** — manifests, seeds, 126 tests |

## What remains — minimal

1. **Demonstration run** (OPTIONAL): replay a cached Tomato-Village session through
   `SessionGate` in RESEARCH MODE and log the verdict trace, to show the pipeline end to end.
   No new science; useful for the viva.
2. **Fix the camera probe** (RECOMMENDED, deferred by the user): make the check reject a
   uniform frame, and make frame-saving opt-in.
3. **Scopus / WoS / IEEE / ACM search** (STRONGLY RECOMMENDED): requires institutional
   access. IEEE matters most — it is where agricultural robotics and safety engineering
   publish.

## What NOT to run

- A larger rover experiment programme. The gate cannot enable autonomy here; more rover
  experiments would measure the classifier's field accuracy, which is already measured.
- Threshold tuning on the test arms.
- A second candidate estimator.
