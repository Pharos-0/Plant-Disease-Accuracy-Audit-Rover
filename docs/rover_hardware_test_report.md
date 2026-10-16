# Rover Hardware Test Report

**Status (2026-10-16): FIRST HARDWARE SESSION RUN.** Platform, camera and ultrasonic PASS
(the ultrasonic failed on the first read because the sensor was not switched on; it passed
once powered, with a reliability caveat below); relay and motors still NOT RUN, held on their
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
| 2026-10-16 | ultrasonic | `FAIL` then `PASS` | **First read, sensor not powered:** `FAIL - 0/15 valid, median None`. **After switching the sensor on:** `PASS - 8/15 valid, median 23.4 cm`, readings `[34.1, None, 3.9, None, 25.5, None, 25.5, None, 23.4, None, 5.2, None, 5.2, None, 4.9]` cm — varying real distances confirm it measures, but **every second reading drops to `None`** (reliability caveat, see §1) | SSH laptop→Pi |
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

Result (2026-10-16): the first read returned `0/15 valid readings` — **because the sensor
was not switched on**. Once powered, a re-run gave `8/15 valid readings, median 23.4 cm`, with
distances varying from 3.9 to 34.1 cm, so the HC-SR04 **is present and measuring**. Two caveats
stand:

1. **Reliability:** in both powered runs roughly half the readings dropped to `None`, in a
   regular every-other-reading pattern. That is most likely too little settling time between
   pulses in the test loop (an echo from the previous ping still in flight), but it has not been
   isolated; a sensor that intermittently returns `None` still trips the fail-open behaviour below.
2. **The fault-injection step is still `NOT RUN`:** the deliberate unplug test — confirming the
   rover keeps driving and spraying when the sensor returns `None` — was not performed, and
   matters precisely because `None` readings already occur in normal operation.

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

RESULT ultrasonic: FAIL - 0/15 valid readings, median None cm     # sensor not switched on
# after switching the sensor on and re-running:
RESULT ultrasonic: PASS - 8/15 valid readings, median 5.22 cm / 23.4 cm (two runs)
# readings run 2: [34.1, None, 3.9, None, 25.5, None, 25.5, None, 23.4, None, 5.2, None, 5.2, None, 4.9] cm
```

What this session establishes: the **Pi boots and runs**, the **USB camera delivers a
live, non-blank feed**, and the **ultrasonic sensor measures distance once powered** (though
about half its readings drop to `None` — see §1). It does not establish the relay, the pump,
the motors, the driver or the battery. Nothing was actuated. The §16 camera artefacts (a saved frame of a real plant, a
tape-measured height) were not produced and remain NOT RUN.

Correction within the session: the first ultrasonic FAIL was because the sensor was not
switched on, not a wiring fault. After powering it, the re-run PASSED. The earlier "diagnose
the wiring" note is therefore withdrawn; what remains for the ultrasonic is the reliability
check (the `None`-every-other-reading pattern) and the deliberate unplug fault-injection test.

Still outstanding, in order of safety:
1. Ultrasonic — characterise the dropped readings (settling time between pulses), then run the
   deliberate unplug / fault-injection step.
2. Relay — only after the tank is empty or the pump outlet is disconnected.
3. Motors — only after the chassis is on blocks with all wheels clear of the ground.
