# Rover Hardware Test Report

**Status (2026-10-16): FIRST HARDWARE SESSION RUN.** Platform and camera PASS;
ultrasonic FAIL (0/15 valid readings); relay and motors still NOT RUN, held on their
safety conditions. Run over SSH from the laptop to the Pi on the laptop hotspot
(`pi@192.168.137.167`), using `rover/hardware_tests/test_components.py` copied to the Pi.
See the session note at the end for the full command output.

This file records the outcome of **per-component isolation tests**, run *before* any
ML integration (MASTER_PROMPT §9). The point of testing in isolation is that a
failing relay or a dead camera must be found here, not inside a session-gating
experiment where its symptoms would be misread as an estimator problem.

Fill this in by running, on the Pi:

```bash
python3 rover/hardware_tests/test_components.py --all --json rover_hw_report.json
```

then pasting each `RESULT` line and its evidence below. **Do not summarise from
memory** — paste the actual output.

---

## Test log

| Date | Component | Status | Evidence (command output / measurement / photo) | Tester |
|---|---|---|---|---|
| 2026-10-16 | platform | `PASS` | `RESULT platform: PASS - Raspberry Pi 5 Model B Rev 1.0` | SSH laptop→Pi |
| 2026-10-16 | camera | `PASS` | `RESULT camera: PASS - 50 frames in 3 s at 640x480, frame std 57.009`; USB "Generic HD camera" (349c:2317) at `/dev/video0`; live feed, not blank; **no frame saved, no plant imaged** | SSH laptop→Pi |
| 2026-10-16 | ultrasonic | `FAIL` | `RESULT ultrasonic: FAIL - 0/15 valid readings, median None cm`; all 15 readings `null`; `ECHO` (BCM 6) read a constant `0` over 20 samples, never pulsed high | SSH laptop→Pi |
| — | relay | `NOT RUN` | **safety hold** — requires tank empty or pump outlet disconnected | |
| — | motors | `NOT RUN` | **safety hold** — requires chassis on blocks, wheels clear of the ground | |

---

## Mandatory checks that are not simple pass/fail

### 1. Ultrasonic failure mode (defect D2)

Testing that the sensor *works* is not sufficient. The delivered code has three
stacked fail-open behaviours:

1. `SPRAY_ENABLED = True` at line 27 — the sprayer is **armed at startup**;
2. `get_distance()` returns `None` on timeout — i.e. on sensor failure or unplug;
3. `if dist and dist < 20` — `None` is falsy, so the obstacle stop **silently
   disappears**.

**Required test:** unplug the HC-SR04 and confirm what the rover does. Expected on
the delivered code: it keeps driving *and* keeps spraying, with no indication. Record
the observed behaviour verbatim.

Result (2026-10-16): the basic read was run **but the sensor itself failed** —
`0/15 valid readings`, every reading `null`, and `ECHO` (BCM 6) resting at a constant `0`.
So the obstacle stop has no working sensor behind it at all. The deliberate unplug /
fault-injection step is `NOT RUN` and is moot until the sensor reads at all: a sensor that
already returns `None` is, for the obstacle-stop logic, indistinguishable from the unplugged
case. **Cause not diagnosed** (wiring, the pin map TRIG=BCM5/ECHO=BCM6, a missing ECHO voltage
divider, or a dead sensor are all possible); to be checked with the Pi powered off.

### 2. Relay boot state

Confirm the relay is **de-asserted at power-on**, before any software runs. The float
trick (`GPIO.setup(RELAY, GPIO.IN)`) is designed for exactly this, and the audit
judged it correct — but it must be observed on the actual module, since active-high
modules would behave oppositely.

Result: `NOT RUN` — held until the tank is empty or the pump outlet is disconnected.

### 3. Camera sample frame and measured height (§16)

Two artefacts are required, and together they decide whether **arm E is viable at all**:

- one frame from the actual camera pointed at a **real plant** (saved automatically by
  `--test camera`);
- the camera height above ground, **measured with a tape measure**, not estimated.

Result (2026-10-16): the camera **hardware** is verified working (live feed, 50 frames,
frame std 57.0 — not blank), but **neither §16 artefact exists yet**: the test was run with
frame-saving off, so no plant frame was captured, and the height was not re-measured this
session. Both remain `NOT RUN`; arm E stays out of the results until they exist.

### 4. Gamma correction after the D1 fix

The delivered `adjust_gamma` branches were inverted: with the convention
`out = (i/255)^(1/gamma) * 255`, gamma > 1 brightens and gamma < 1 darkens, yet the
code applied `2.0` when the frame was too bright and `0.7` when too dark — both
branches amplifying the exposure problem. `src/rover/leaf_proposal.py` swaps them.

**Required test:** capture frames in direct sun and in shade, before and after the
fix, and record mean brightness for each. This is measured, not asserted.

Result: `NOT RUN`

---

## Components that cannot be verified

Any component that cannot be tested is listed here and recorded as **unverified** in
`docs/rover_hardware_inventory.md`. Per §9 it is **never assumed working**, and no
thesis claim may rest on it.

| Component | Why it could not be verified |
|---|---|
| HC-SR04 ultrasonic | tested 2026-10-16, returned 0/15 valid readings — present but **not working** as wired; cause not yet diagnosed |
| relay / pump | `NOT RUN` — held on the empty-tank / disconnected-pump safety condition |
| motors / driver | `NOT RUN` — held on the wheels-off-the-ground safety condition |

---

## Numbers that must not be reused

The delivered paper's Table I (92.4% navigation accuracy, 89.1% spraying precision,
41.7% pesticide reduction, 18.6 Wh/run, <2% fault rate) has **no code path that could
have produced any of it** — the delivered file contains no logging of any kind. None
of these figures may appear in the thesis, including as a baseline to improve on. Any
performance number in the rover chapter must come from a run logged by our own code.

---

## Note added 2026-09-28 — the digital twin does not change this report

A digital twin now exercises the rover control software end to end
(`docs/digital_twin.md`, `rover/digital_twin.py`). It simulates GPIO, camera and ultrasonic
sensor so the real control code can be run and its safety invariants checked.

**Every row above still reads NOT RUN and remains correct.** The twin establishes properties of
the software. It establishes nothing about the Raspberry Pi, the camera, the motor driver, the
ultrasonic sensor, the relay, the pump, the battery or power draw, none of which has been
powered on. A simulated actuator firing proves a decision was taken, not that a pump works.

---

## Session note 2026-10-16 — first hardware session, verbatim

Access: the Pi's account password was unknown and SSH was off. With the owner's
go-ahead, the laptop's SSH public key was installed for user `pi` via a one-time
`firstrun.sh` on the boot partition (`rover/sdcard_backup_20261016/` holds the originals
and the undo steps); the boot command line restored itself on first boot. Tests were then
run over SSH from the laptop (`pi@192.168.137.167`, Pi on the laptop hotspot), with
`rover/hardware_tests/test_components.py` copied to `/home/pi/`.

Environment observed on the Pi: Raspberry Pi 5 Model B Rev 1.0; Debian GNU/Linux 13
(trixie); Python 3.13.5; OpenCV 4.10.0; NumPy 2.2.4; `lgpio`, `gpiozero`, `RPi.GPIO`
importable; **PyTorch not installed**. USB camera present (`349c:2317 Generic HD camera`).

```
RESULT platform: PASS - Raspberry Pi 5 Model B Rev 1.0

RESULT camera: PASS - 50 frames in 3 s at 640x480, frame std 57.009
# (OpenCV GStreamer "Cannot query video position" warning printed; harmless)

RESULT ultrasonic: FAIL - 0/15 valid readings, median None cm
# readings: 15x null; ECHO (BCM6) resting state over 20 samples: all 0
```

What this session establishes: the **Pi boots and runs**, and the **USB camera delivers a
live, non-blank feed** — the two things the camera experiment most depends on. It does not
establish the ultrasonic sensor (failed), the relay, the pump, the motors, the driver or the
battery. Nothing was actuated. The §16 camera artefacts (a saved frame of a real plant, a
tape-measured height) were not produced and remain NOT RUN.

Still outstanding, in order of safety:
1. Ultrasonic — diagnose with the Pi **powered off**: check the HC-SR04 wiring against
   TRIG=BCM5 / ECHO=BCM6 and whether an ECHO voltage divider is present, then re-test.
2. Relay — only after the tank is empty or the pump outlet is disconnected.
3. Motors — only after the chassis is on blocks with all wheels clear of the ground.
