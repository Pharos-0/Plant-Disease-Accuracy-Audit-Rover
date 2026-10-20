# Pilot capture, attempt 2 — 2026-10-21 (rover camera, web capture with guards)

22 frames, 6 leaves, one nursery tomato plant (`P01`), through the rover's camera (`/raw`), with
the pre-registered decision rule and both policies logged. Diagnosis source on every frame:
*visual only (nursery stock)*. `log.jsonl` and `frames/` are unmodified.

The capture page's new guards held: one diagnosis per leaf, one plant.

## Labels and conditions — what to bear in mind

- The operator labelled two leaves *late blight* (L03, L04: yellowing with brown/red margins) and
  two *early blight* (L05, L06), by eye. No one has confirmed a disease; the protocol's pilot
  plan (§11) assumed a healthy plant. **No reported figure depends on a disease label.**
- *Low light* means room lights off. 9 of the 12 frames tagged low light were taken that way
  (median brightness 3–7 of 255) and all 9 were deferred as too dark. The other 3 (L05, #14–16)
  have median brightness ~104 — the lights were on — so L05's low-light/normal pair does not
  test lighting.
- Normal and low light were mostly captured on *different* leaves, so the within-leaf pairing
  covers only L05, which (above) is not a lighting contrast.

## What it shows

Pooled with attempt 1 for the label-free observations (32 frames; see `docs/pilot_results_20261021.md`):

- The classifier predicted *healthy* on 0 of these 22 frames; 11 confident disease calls.
- The leaf judged healthy (L01), in room light: 5/5 frames called *early blight* at 0.95–0.98,
  above τ = 0.888.
- Gated and spatial-only took the same decision on 22/22 frames; the label-free gate was open on
  13/22 (estimate 0.28–1.00).
- Spatial-only found tissue on 13/13 frames that were not too dark; 9/9 truly dark frames
  deferred.

`analyse_camera_session.py` output is beside this note. Its accuracy (40% normal, 58% low light),
AUROC (0.504) and missed treatment (0% for both policies; exact 95% upper bound with one plant:
95%) are against visual labels on one plant, and are not offered as evidence.
