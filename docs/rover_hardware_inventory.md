# Rover Hardware Inventory

**Status: NOT STARTED — blocked on physical access to the rover.**

MASTER_PROMPT §9 is explicit: *the delivered third-party paper is not evidence that
any component exists or works*, and *a component visible in a photograph is present,
not working*. Nothing in this file may be filled in from the paper, from a photo, or
from the delivered source code. Only from hands-on inspection and an isolation test.

Until then every row stays `UNVERIFIED`. That is the honest state, and §9 says an
unverifiable component is **recorded as unverified, never assumed working**.

---

## How to complete this file

1. Run `rover/hardware_tests/test_components.py` on the Pi, one subsystem at a time.
   Read its SAFETY notes first — wheels off the ground, tank empty.
2. Paste each `RESULT` line and its evidence into
   `docs/rover_hardware_test_report.md`.
3. Fill the table below from that evidence, not from recollection.
4. Measure the camera height with a tape measure and photograph the mounting.

---

## 1. Component table

For **every** component §9 requires: physically present? · working? · independently
tested? · interface/GPIO? · power requirement? · software driver? · evidence?

| # | Component | Present? | Working? | Independently tested? | Interface / GPIO | Power | Driver | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Raspberry Pi (model) | `UNVERIFIED` | `UNVERIFIED` | no | — | — | — | run `--test platform` |
| 2 | OS version | `UNVERIFIED` | — | no | — | — | — | `/etc/os-release` |
| 3 | Camera (model, resolution, driver) | `UNVERIFIED` | `UNVERIFIED` | no | `VideoCapture(0)` per code — USB/V4L2, **not** CSI | — | — | run `--test camera` |
| 4 | Camera mounting and **measured height** | `UNVERIFIED` | — | no | — | — | — | tape measure + photo |
| 5 | Motor driver (model) | `UNVERIFIED` | `UNVERIFIED` | no | IN 17/27/22/23, EN 18/19 | — | — | run `--test motors` |
| 6 | Motors (x2) | `UNVERIFIED` | `UNVERIFIED` | no | via driver | — | — | run `--test motors` |
| 7 | Chassis | `UNVERIFIED` | — | no | — | — | — | photo + description |
| 8 | Ultrasonic HC-SR04 | `UNVERIFIED` | `UNVERIFIED` | no | TRIG 5, ECHO 6 | 5 V | — | run `--test ultrasonic` |
| 9 | Relay module | `UNVERIFIED` | `UNVERIFIED` | no | pin 24, active-low | — | — | run `--test relay` |
| 10 | Pump | `UNVERIFIED` | `UNVERIFIED` | no | via relay | — | — | run `--test relay` |
| 11 | Tank (capacity) | `UNVERIFIED` | — | no | no sensor in code | — | — | measure |
| 12 | Battery (chemistry, capacity) | `UNVERIFIED` | `UNVERIFIED` | no | **no ADC in code** | — | — | measure with a multimeter |
| 13 | Voltage regulator / converter | `UNVERIFIED` | `UNVERIFIED` | no | — | — | — | measure output under load |
| 14 | Wiring | `UNVERIFIED` | `UNVERIFIED` | no | — | — | — | continuity check + photo |
| 15 | Existing controls (keyboard via OpenCV window) | `UNVERIFIED` | `UNVERIFIED` | no | `cv2.waitKey` | — | — | see defect D4 |

## 2. Claimed but absent from the delivered software

The audit (§2 of `docs/rover_code_audit.md`) counted occurrences in the delivered
source. These subsystems appear **zero** times:

| Claimed in the delivered paper | Occurrences in code |
|---|---|
| IR line sensors | 0 |
| Battery / ADC / voltage monitoring | 0 |
| Pesticide tank sensor | 0 |
| FSM / state machine | 0 |
| Return-to-home / homing | 0 |
| Threading | 0 |

If any of this hardware is physically bolted to the chassis, it is **unwired as far as
the software is concerned**. Record presence separately from function; do not let a
present-but-unused component imply a working subsystem.

## 3. Power budget

`UNVERIFIED` — requires measurement. Note that the delivered paper's 18.6 Wh/run
figure has **no code path that could have produced it** (there is no logging of any
kind in the delivered file), so it cannot be reused. Measure or omit.

## 4. Decision this file feeds

Arm E (rover frames) is viable only if the camera produces usable leaf imagery at the
mounted height. §16 requires **one sample frame from the actual camera pointed at a
real plant, plus the measured camera height**. `--test camera` saves that frame.
Until it exists, arm E stays out of the results and the thesis stands on arms A–D —
which is the intended design: *the thesis must remain complete if this entire chapter
is deleted.*
