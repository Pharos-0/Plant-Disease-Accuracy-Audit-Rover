# Rover Hardware Test Report

**Status: NO TESTS RUN — blocked on physical access to the rover.**

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
| — | platform | `NOT RUN` | | |
| — | camera | `NOT RUN` | | |
| — | ultrasonic | `NOT RUN` | | |
| — | relay | `NOT RUN` | | |
| — | motors | `NOT RUN` | | |

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

Result: `NOT RUN`

### 2. Relay boot state

Confirm the relay is **de-asserted at power-on**, before any software runs. The float
trick (`GPIO.setup(RELAY, GPIO.IN)`) is designed for exactly this, and the audit
judged it correct — but it must be observed on the actual module, since active-high
modules would behave oppositely.

Result: `NOT RUN`

### 3. Camera sample frame and measured height (§16)

Two artefacts are required, and together they decide whether **arm E is viable at all**:

- one frame from the actual camera pointed at a **real plant** (saved automatically by
  `--test camera`);
- the camera height above ground, **measured with a tape measure**, not estimated.

Result: `NOT RUN`. Arm E remains out of the results until both exist.

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
| — | — |

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
