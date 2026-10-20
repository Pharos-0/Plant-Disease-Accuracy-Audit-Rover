# First real session: one-plant pilot — results (2026-10-21)

Pre-registered as a pilot the night before (`docs/camera_experiment_protocol.md` §11). One nursery
tomato plant, photographed through the **rover's own camera** (`/raw`, banner-free) with the
decision rule unchanged and both policies logged per frame. Labels are the operator's visual
judgement, recorded as *visual only (nursery stock)*. One plant ⇒ no interval is estimable.

| Capture | Folder | Frames | Used for |
|---|---|---:|---|
| attempt 1 | `camera_sessions/20261021_pilot_P01/` | 10 | label-free observations only — its labels are inconsistent (see its `NOTE.md`) |
| attempt 2 | `camera_sessions/20261021_pilot_P01_attempt2/` | 22 | label-free observations; label-dependent figures shown but not offered as evidence |

## Against the pre-registered quantities (§11.2)

| # | Quantity | Result |
|---|---|---|
| P1 | Capture chain end to end on a real plant, rover camera | **Yes** — 32 frames, both policies on every frame, clean `/raw` frames |
| P2 | False alarms on leaves judged healthy | The one leaf judged healthy in room light (attempt 2, L01): **5/5 frames called *early blight*, at 0.95–0.98** — above τ = 0.888, so not abstentions. Wasted spray 5/5 under both policies |
| P3 | Tissue detection; too-dark deferral | Spatial-only found tissue on **23/23** frames not too dark; **9/9** truly dark frames (median brightness 3–7) deferred. A wooden door behind the plant was counted as tissue |
| P4 | Confidence under shift | Mean max-softmax 0.90 in room light, 0.74 lights-off — on different leaves; the only same-leaf pair (L05) had the lights on in both, so the dry run's "more confident in low light" was **not testable** |
| P5 | §16 artefacts (plant frame + height) | **Done** — frames of the real plant; lens height 24.5 cm (`results/rover_artifacts/20261021_041635/`) |
| P6 | Rover AUTO on the real plant | **Not yet run** |

## Across all 32 frames (needs no disease label)

| Observation | Frames |
|---|---:|
| Classifier predicted *healthy* | 1/32, never confidently |
| Confident disease calls (≥ τ) | 16/32 |
| Gated and spatial-only took the same decision | **32/32** |
| Label-free session gate open (estimate up to 1.00) | 17/32 |

**Reading.** The classifier almost never called a leaf healthy, so the gated policy — which leaves
a visible leaf unsprayed only on a confident *healthy* call — withheld nothing, and the two
policies agreed on every frame: gating would have saved no chemical. This answers the dry run's
open question (protocol §9: whether the classifier's reluctance to call a leaf healthy holds for
real leaves) — on this plant, it does.

**Is the plant diseased, and was the classifier right?** Unknown. After the session the operator
noted brown spots on some leaves, and the frames show them (L03 most clearly), so not every
disease call need be wrong. But the classifier's confidence did not follow the visible damage:
0.95–0.98 on L01, which looks clean, and 0.57–0.64 on L03, the most visibly spotted. A classifier
that calls almost everything diseased is right on a diseased leaf by default; detection would show
as confident *healthy* calls on healthy leaves, and there were none in 32 frames. The label-free
estimate (up to 1.00) cannot know the diagnosis either. Settling it needs (1) a qualified diagnosis
of the plant — which would also turn this plant into valid labelled data — and (2) a leaf known to
be healthy shown to the camera.

**Not offered as evidence.** Against the operator's visual labels (attempt 2): accuracy 40% in
room light and 58% lights-off, AUROC 0.504, and no frame labelled diseased missed by either
policy — with one plant the exact one-sided 95% upper bound on the share of diseased plants with
a miss is 95%. The primary outcome still needs the protocol's sample (≥ 6 diseased + ≥ 4 healthy
plants, confirmed).

## Found during the session, and fixed before attempt 2 or before use

- The OpenCV in the project environment has no window support, so the protocol's window capture
  could not run → a browser capture with the same pipeline and record (`demo/web_capture.py`).
- Attempt 1 saved one leaf under two diagnoses and one plant as two → the page now refuses a
  second diagnosis for a leaf and confirms a new plant; a bright "low light" frame is warned.
- The rover camera's sensor stuck twice after a reopen (static stripes read as 86% leaf) → a
  camera-fault deferral (protocol §11.4), tested on the 1,800 dry-run frames (none tripped) and
  confirmed live when it caught the second episode.
