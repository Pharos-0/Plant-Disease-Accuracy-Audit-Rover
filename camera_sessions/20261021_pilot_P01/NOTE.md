# Pilot capture, attempt 1 — 2026-10-21 03:05–03:07 (rover camera, web capture)

10 frames of the one nursery tomato plant, through the rover's camera (`/raw`), with the
pre-registered decision rule. `log.jsonl` and `frames/` are kept **unmodified**.

## Why its labelled results are not reported

Found by inspecting the labels and frames — reasons about the labels themselves, not about
any outcome:

1. **One leaf, two diagnoses.** `P01L01` was saved as *healthy* (#1) and as *late blight*
   (#2); `P01L04` as *early blight* (#5) and as *healthy* (#6). A leaf has one diagnosis.
2. **A second plant that is the same plant.** Frames #8–10 are labelled plant `P02`; only one
   plant was bought, and they show the same pot. With two plant ids the analysis resamples
   two "plants" and prints intervals that one plant cannot support.
3. **Disease labels are visual guesses.** Some leaves have brown, curled margins (#2, #3, #8)
   and were labelled *late blight*; browning on a young nursery plant has several possible
   causes and no one has confirmed a disease. The protocol (§2, runbook step 6) says not to
   guess a disease.
4. **The condition tag does not match the light.** All 10 are tagged *low light*, but their
   median brightness is 69–142 of 255 — ordinary indoor light. The operator explained
   afterwards that it was night, so the room light was taken for "low light"; the protocol
   defines the conditions by the room (§3: *normal* = ordinary room light; *low light* = room
   lights off, indirect light only), so these were *normal*-condition frames. The raw tags are
   left as recorded.
5. **Too few frames:** 1–2 per leaf, against the protocol's 5 per leaf per condition; no
   *normal* frames, so nothing can be paired against normal.

So accuracy, AUROC, missed treatment, wasted spray, the gated-vs-spatial comparison and any
per-condition figure from this session are **not reported**. (The analysis output beside this
note — `analysis_by_condition.csv`, `policy_comparison.csv`, `pilot_report.csv`,
`analysis_status.json` — is kept as produced, for the record.)

## What it does show (needs no disease label)

- **P1 — the chain works on a real plant through the rover camera:** 10 frames logged, both
  policies recorded on every one, clean `/raw` frames.
- **P3 — tissue detected:** the spatial-only policy fired on 10/10 frames, none deferred.
  Frame #1 is mostly the wooden door, which the any-colour tissue test counts as tissue — the
  background matters.
- **The classifier calls this plant diseased:** 9 of 10 frames (8 *early blight*, 1 *late
  blight*); 5 at or above τ = 0.888 (0.92–0.97), including clean green leaves (#4, #10).
- **The label-free gate is fooled:** the session estimate reached **1.00** and the gate was open
  on 4 of 10 frames while the classifier was calling green leaves blighted.
- Because the classifier never confidently called a leaf healthy, the gated policy withheld
  nothing: both policies sprayed 10/10. This session cannot separate them.

A second attempt, with the capture page's new guards (one diagnosis per leaf; *next plant*
confirmed), follows in its own folder.
