# Rover Code Audit — `web_rover.py`

**Artefact:** `rover/original/web_rover.py`, 241 lines, stored read-only.
**SHA-256:** `8801697bd77185ba8a544586ed64233fa65d54a29c759ab22c9b6b856104d3ba`
**Provenance:** delivered by the third party. User confirms this is the **only** file.
**Audited:** 2026-09-06, static analysis + numerical verification of the gamma LUT.

Every claim below is verified against the file, not inferred.

---

## 1. Hardware actually addressed

| Component | Pins | Role in code |
|---|---|---|
| Motor driver | IN1-4 = 17/27/22/23, ENA=18, ENB=19 | 2 PWM channels @1 kHz, `speed` 20–100 |
| Relay (pump) | 24 | Binary on/off, no timing or dosage |
| Ultrasonic HC-SR04 | TRIG=5, ECHO=6 | One reading/frame; only used for a 20 cm stop in AUTO mode |
| Camera | `VideoCapture(0)` @320×240 | Sole perception sensor |

## 2. Subsystems claimed in the delivered paper — grep counts in the delivered file

| Claimed | Occurrences |
|---|---|
| IR line sensors | **0** |
| Battery / ADC / voltage monitoring | **0** / **0** / **0** |
| Pesticide tank sensor | **0** |
| FSM / state machine / states | **0** / **0** / **0** |
| Return-to-home / homing | **0** / **0** |
| Threading | **0** |

The paper's Table I (92.4% navigation accuracy, 89.1% spraying precision, 41.7% pesticide
reduction, 18.6 Wh/run, <2% fault rate) has **no code path that could have produced any of it**.
There is no logging of any kind in this file.

---

## 3. Defects, verified

### D1 — Gamma correction is inverted *(new; not previously identified)*
Lines 110–113. Measured on the actual LUT (`adjust_gamma` convention: γ>1 brightens, γ<1 darkens):

| Branch | Trigger | Mean 0..255 maps to | Intended |
|---|---|---|---|
| `adjust_gamma(frame, 2.0)` | `brightness > 170` (too bright) | 127.5 → **169.3** ↑ | darken |
| `adjust_gamma(frame, 0.7)` | `brightness < 70` (too dark) | 127.5 → **104.6** ↓ | brighten |

Both branches **amplify** the exposure problem. Fix: swap the two constants. Field-critical —
in strong sun the frame is already blown out, this pushes it further, saturation collapses,
and the `S≥30` HSV floor then admits more non-plant material.

### D2 — Three stacked fail-open safety behaviours *(new)*
1. `SPRAY_ENABLED = True` at line 27 — **the sprayer is armed at startup**.
2. `get_distance()` returns `None` on timeout (lines 78–86) — i.e. on sensor failure or unplug.
3. `if dist and dist < 20` (line 185) — `None` is falsy, so obstacle-stop **silently disables**.

Net: if the ultrasonic fails, the rover keeps driving *and* keeps spraying, with no indication.

### D3 — No `try/finally` around the main loop
Cleanup (lines 237–241) is unreachable on any exception. Motors and the **spray relay stay live**.
`cap.read()` failure `break`s cleanly, but nothing else does.

### D4 — Keyboard control is coupled to the preview windows *(new; constrains the refactor)*
`cv2.waitKey(1)` (line 174) only delivers events while an OpenCV window has focus. The two
`imshow` calls (lines 230–231) are the largest performance cost over VNC — **but they cannot
simply be deleted**, because doing so removes all input handling. Any performance refactor must
replace the input mechanism at the same time.

### D5 — No web server, despite the filename
Zero occurrences of `flask`, `http`, `server`, `socket`. This is a local OpenCV GUI application
requiring a display or VNC session. There is no browser control path — which is why the VNC
lockout was total rather than merely inconvenient.

### D6 — Relay pin reconfigured every frame
`relay_on()`/`relay_off()` call `GPIO.setup()` (lines 61, 65) on every iteration regardless of
state change. Should fire only on transition.

### D7 — Ultrasonic polled unconditionally
`dist = get_distance()` (line 163) runs every frame even in manual mode where the value is
cosmetic. Two busy-wait loops, 20 ms timeout each.

### D8 — Bang-bang steering
Lines 188–198 snap to a full turn on any left/right imbalance. This is the reported
"automated is still not effective" zig-zagging. Needs proportional control.

### D9 — Uncalibrated spray threshold
`c > 1500` (line 166). The centre region is ~106×240 = 25,440 px, so the trigger is ≈5.9% green
coverage. No calibration procedure exists anywhere in the code.

This is **inherited prototype logic**. It is **not an agronomically validated spray threshold and
must never be presented as one.** Retain it for engineering demonstration only, label it as
inherited and uncalibrated in the thesis, and either calibrate it against annotated frames or
replace it — but do not defend the number as given. "Why 1500?" is an obvious viva question with
no defensible answer today.

### D10 — Loose HSV bounds
`lower=[25,30,30]`, `upper=[95,255,255]` (lines 121–122). Saturation floor of 30/255 admits
grey and desaturated surfaces. Not crop-specific, contradicting the paper's claim of thresholds
"empirically set to tomato and brinjal leaf morphology".

---

## 4. What is genuinely sound

Not everything here is a defect, and the thesis should say so:

- The **float-relay trick** (`GPIO.setup(RELAY, GPIO.IN)` to release, `OUT`+`LOW` to assert) is a
  correct and neat way to keep an active-low relay module de-asserted at boot.
- CLAHE on the LAB L-channel is a reasonable illumination-normalisation choice.
- Morphological close-then-open on the mask is standard and correctly ordered.
- The code is **honest about what it is** — a classical CV script. It contains no stub, no fake
  model, no placeholder pretending to be learned. The dishonesty is in the paper, not the code.

---

## 5. Role in this thesis

`web_rover.py` is the **V0 baseline**, frozen. It is never edited in place. Two additions only:

1. session frame logging (~5 lines);
2. allowing the gate module to set the **existing** `SPRAY_ENABLED` flag (~2 lines).

Everything else — leaf-crop proposal, classifier, session buffer, estimator, gate — lands in
`rover/` as new modules sharing the benchmark's estimator code. D1 and D3 must be fixed before
any unattended field run. D4/D6/D7/D8 form the V0→V1 engineering comparison, measured in FPS
and steering oscillation, not asserted.
