# AUTO on the real plant — 2026-10-21

Rover in control mode from boot (camera opened once, healthy). Tank empty, pump outlet
disconnected, nobody on the drive buttons — confirmed by the operator immediately before each run.
Relay actuation only; no liquid.

## Run 1 — room background (laptop clock 04:51:00–04:52:31; `telemetry.jsonl`, `relay_pin_log.txt`)
- AUTO fired the relay **29 times in ~91 s** (~3.1 s apart: 1 s pulse + 3 s cooldown).
- The relay pin, read on the Pi ten times a second, shows **exactly 29 pulses** (released → energised):
  the software's fire count and the hardware agree.
- **Lens covered → deferred (too dark), firing stopped**, and resumed when uncovered.
- **Plant removed → AUTO kept firing.** The rightmost zone stayed at 71–75% "tissue": the wooden
  door. With the door covered the dark door frame (~27%) and the curtain (13–18%) would still pass
  the 15% threshold. The any-colour tissue test, which must accept browns and yellows to see
  diseased leaves, counts wood as leaf: against this background the spatial policy sprays the
  furniture.

## Run 2 — white sheet filling the view (Pi clock 00:26:51–00:27:42)
- Operator: clicked with a leaf in view, **stopped with the leaf removed**, resumed when it returned.
- Not logged frame by frame (the recorders had stopped). The rover's cumulative counter went
  29 → 42: **13 fires in 51 s**, against ~16 for uninterrupted firing — consistent with a ~10 s pause.

## Reading
AUTO works as designed on the robot: it fires on leaf tissue, defers when it cannot see, and the
relay does what the software records. Its weakness is the tissue test's background: it needs a
plain background here, and in the field the learned crop detector (thesis Chapter 8).
