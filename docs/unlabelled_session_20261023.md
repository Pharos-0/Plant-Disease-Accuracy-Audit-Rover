# An unlabelled session on the rover — 2026-10-23

The thesis's question put to the robot, with **no human label anywhere**: given unlabelled field
images, can the trained system tell how well its classifier is doing? Tool:
`scripts/unlabelled_rover_session.py`. Frames through the rover camera (banner-free), decided by
the deployed rule; then the **eight label-free estimators** of the thesis's standard set, fitted
on the laboratory source-validation split exactly as in the thesis, estimate the classifier's
accuracy on the frames the system judged. (GdScore, the ninth, needs a forward pass in `train()`
mode — `docs/gdscore_eligibility.md` §4 — which a live sprayer does not make.)

| | Live capture, 23 Oct | Pilot frames, 21 Oct (labels ignored) |
|---|---|---|
| Frames (judged) | 60 (60) — one static view of the whole plant | 32 (23) — leaves close to the lens, room light and dark |
| Classifier calls | early blight 54, late blight 6, healthy **0** | early blight 30, late blight 2, healthy **0** |
| Confident (≥ τ) calls | 0 — every frame abstained | 0 confident *healthy* |
| Spatial-only / gated sprays | 60 / 60 — same decision on all 60 | 23 / 23 — same on all 32 |
| confidence | 0.691 | 0.857 |
| entropy | 0.448 | 0.695 |
| ATC-MC | 0.600 | 0.783 |
| ATC-NE | **1.000** | 0.870 |
| DoC | 0.726 | 0.891 |
| nuclear norm | 0.585 | 0.713 |
| MaNo | 0.903 | **0.919** |
| COT | **0.433** | **0.472** |
| **Range of the eight estimates** | **0.43 – 1.00** | **0.47 – 0.92** |

**Reading.** On the very same frames the eight estimators put the classifier's accuracy anywhere
from 43% to 100%. They cannot all be right, and without labels nothing says which is. Where a
reference exists — the operator's visual labels on the 21 Oct capture — agreement was 50%. The
thesis measured why: under laboratory-to-field shift these estimates respond mainly to the
laboratory calibration (the best erred by 0.1337 on average; none beat a constant guess), and the
classifier's own confidence does not track its errors. On the robot that means a label-free
estimate cannot be what lets a classifier withhold spray. The sprays themselves here came from
the spatial policy; the classifier, never confidently calling a leaf healthy, withheld nothing.

The live capture is one static viewpoint seen 60 times (frame-to-frame change 4.2 = sensor
noise), and the plant is small in the frame against curtain, wall and door — a whole-image
classifier is judging mostly the room. Its figures describe this view, not the plant's health.
