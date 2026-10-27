# Rover Hardware Inventory

**Status: PARTIAL — populated from photographs 2026-09-08. No component has been
functionally tested.**

The study protocol is explicit and the distinction governs this whole document:

> A component visible in a photograph is *present*, not *working*.

So every row below separates **present?** (photographs can settle this) from
**working?** (only an isolation test can). Nothing here may be cited as a working
subsystem until `rover/hardware_tests/test_components.py` has produced evidence.

**Evidence base:** five photographs supplied by the user — chassis underside, top
deck, front bumper, whole rover, and a ruler measurement of camera height.

---

## 1. Component table

| # | Component | Present? | Working? | Tested? | Interface / GPIO | Evidence |
|---|---|---|---|---|---|---|
| 1 | **Raspberry Pi 4 Model B** | **YES** | `UNVERIFIED` | no | — | top-deck photo: 2× micro-HDMI, USB-C power, 2× USB3 (blue) + 2× USB2, gigabit Ethernet — the Pi 4B port layout |
| 2 | OS version | `UNVERIFIED` | — | no | — | run `--test platform` |
| 3 | **USB webcam** (small red/black cube) | **YES** | `UNVERIFIED` | no | `VideoCapture(0)` — USB, **not** the CSI ribbon camera, as the audit inferred | whole-rover photo; USB lead runs down the mast to a Pi USB port |
| 4 | **Camera mast + height** | **YES** | — | **MEASURED** | PVC pipe, green-taped at base | **24–25 cm** lens centre above ground (ruler photo). See §3 |
| 5 | **DC gearmotors ×4** | **YES** | `UNVERIFIED` | no | via driver | chassis photo: four motors, one legible as "VEGA ROBO KIT". **Four, not two** — see §2.1 |
| 6 | **Wheels ×4** | **YES** | `UNVERIFIED` | no | — | chassis and whole-rover photos |
| 7 | **Motor driver (H-bridge)** | **NOT VISIBLE** | `UNVERIFIED` | no | code needs IN1-4 = 17/27/22/23, ENA=18, ENB=19 | **no L298N/L293D in any photograph** — see §2.2 |
| 8 | **Relay module** | **YES** | `UNVERIFIED` | no | pin 24, active-low | top-deck photo: blue single-channel module, **SRD-05VDC-SL-C**, 10 A 250 VAC / 10 A 30 VDC, opto-isolated (red + green LEDs) |
| 9 | **HC-SR04 ultrasonic** | **YES** | `UNVERIFIED` | no | TRIG 5, ECHO 6 | front-bumper photo: legible silkscreen `Vcc Trig Echo Gnd`, panel-mounted |
| 10 | **Buck converter** | **YES** | `UNVERIFIED` | no | — | top-deck photo: blue LM2596-style step-down, adjustable trimmer, 220 µF 35 V caps, `IN+` / `OUT+` |
| 11 | **Charge module** | **YES** | `UNVERIFIED` | no | — | top-deck photo: small green PCB, micro-USB in + JST connector — TP4056-class Li-ion charger |
| 12 | **Breadboard + jumper wiring** | **YES** | `UNVERIFIED` | no | ~10-way ribbon to Pi GPIO | top-deck photo |
| 13 | **Chassis** | **YES** | — | no | cardboard box + perforated steel motor mounts + L-brackets | all photos |
| 14 | **Pump** | **NOT VISIBLE** | `UNVERIFIED` | no | via relay | absent from all five photographs |
| 15 | **Tank** | **NOT VISIBLE** | `UNVERIFIED` | no | no sensor in code | absent from all five photographs |
| 16 | **Battery pack** | **NOT VISIBLE** | `UNVERIFIED` | no | **no ADC in code** | absent; a USB lead leaves the Pi off-frame, so the rover appears **tethered** in these photos — see §2.3 |
| 17 | IR line sensors | **NOT VISIBLE** | — | — | 0 occurrences in code | claimed by the delivered paper; no hardware, no code |

---

## 2. Three findings that need resolving before any field run

### 2.1 Four motors, two PWM channels

The chassis carries **four** gearmotors and four driven wheels. `web_rover.py`
drives exactly **two** channels — `IN1/IN2 + ENA` and `IN3/IN4 + ENB`. The motors
must therefore be **paired** (left pair on one channel, right pair on the other),
which is electrically normal skid-steer but has consequences the thesis should state:

- Current draw per channel is roughly **doubled** versus the 2-motor design the pin
  map implies. An L298N is rated ~2 A per channel and drops ~2 V; four small
  gearmotors stalling together can exceed that.
- Defect **D8** (bang-bang steering) gets worse with four driven wheels, because
  skid-steer scrub increases. The "automated is still not effective" zig-zagging
  reported by the user is consistent with this.

**To confirm:** which motors are wired to which channel, and whether they are
parallel-paired. Not determinable from the photographs.

### 2.2 No motor driver is visible — and the code cannot work without one

The Pi cannot drive these motors directly; GPIO pins source ~16 mA. The code's six
control pins (`IN1-4`, `ENA`, `ENB`) are the standard **L298N** interface, so a
driver board must exist for the rover to have ever moved. It does not appear in any
of the five photographs — the top deck shows only the Pi, relay, buck converter,
charge module and breadboard.

Most likely it is mounted **under the top deck** or on the chassis underside outside
the framed area. **This is the single most important thing to photograph or confirm
next**, because §9 forbids assuming it: without an identified driver we cannot state
its current rating, its logic-voltage compatibility, or whether the `ENA`/`ENB` pins
drive real enable inputs.

### 2.3 No battery, and the rover appears tethered

No battery pack is visible, and a USB lead runs from the Pi off-frame in two photos.
Combined with the TP4056-class charge module and the buck converter, the intended
design is clearly battery-powered — but **as photographed the rover is not a
free-roaming platform.**

This matters for the thesis rather than for the software: a session-gating experiment
needs the rover to complete an outing under its own power. Until a battery is fitted
and its runtime measured, Arm E sessions would have to be collected on a tether,
which must be disclosed if so.

It also confirms audit §2: there is **no ADC and no voltage monitoring anywhere in
the code**, so the delivered paper's "18.6 Wh/run" has neither hardware nor software
behind it. That figure must not be reused.

---

## 3. Camera height — 24–25 cm — and what it means for Arm E

**Measured** with a steel rule against the lens centre: **24–25 cm above ground**.
This is the §16 measurement that decides Arm E viability.

**Assessment: Arm E is viable, with one honest caveat about what it images.**

- At 24–25 cm the camera sees the **lower canopy and soil line**, not the whole
  plant. For a mature tomato plant (typically 60–200 cm) this is the bottom of the
  crop.
- That is **not** a bad thing for this particular disease set. Late blight and early
  blight characteristically appear on **lower, older foliage first**, so a
  low-mounted camera looks exactly where the earliest symptoms are.
- The source domains are close-up, leaf-filling images. A 24–25 cm mast pointed at
  the lower canopy will produce **wider frames containing several leaves plus soil
  and background**. That is the gap `src/rover/leaf_proposal.py` exists to close: it
  proposes leaf-sized crops from a wide frame so the classifier sees something
  comparable to its training distribution.
- **Still needed:** one frame from this camera pointed at a **real plant**. All five
  supplied photographs are indoors on tile. `--test camera` saves such a frame.

**Consequence for the thesis:** the mast height is defensible and can be stated as a
design choice justified by blight epidemiology, not an accident. But the domain gap
between a 24 cm wide-field frame and a laboratory leaf close-up is *another* shift on
top of the one already measured, and Arm E should be reported as such rather than
being treated as equivalent to Arm D.

---

## 4. Subsystems the delivered paper claims that have neither hardware nor code

Re-verified against the now-available `rover/original/web_rover.py` by
`tests/test_frozen_baseline.py` — every count is **zero**:

| Claimed | Occurrences in code | Visible in photos |
|---|---:|---|
| IR line sensors | 0 | no |
| Battery / ADC / voltage monitoring | 0 | no |
| Pesticide tank sensor | 0 | no |
| FSM / state machine | 0 | n/a |
| Return-to-home | 0 | n/a |
| Threading | 0 | n/a |
| Web server (despite the filename) | 0 | n/a |
| **Logging of any kind** | **0** | n/a |

The last row is decisive: the file writes nothing to disk, so **no metric in the
delivered paper's Table I could have been produced by this software.** None of those
figures may appear in the thesis, including as a baseline to improve on.

---

## 5. What to do next

1. **Photograph or identify the motor driver** (§2.2). Highest priority — the pin map
   is unverifiable without it.
2. Run `python3 rover/hardware_tests/test_components.py --all --json report.json` on
   the Pi and paste the output into `docs/rover_hardware_test_report.md`. Wheels off
   the ground, tank empty.
3. **Capture one camera frame pointed at a real tomato plant** at the measured 24–25 cm.
   This is the last §16 item and it opens Arm E.
4. Confirm whether a battery is fitted, and measure runtime under load (§2.3).
5. Fix **D1** (swap the gamma constants) and **D3** (wrap the main loop in
   `try/finally`) before any unattended run. Both are one-line-class changes and both
   are field-safety issues.
