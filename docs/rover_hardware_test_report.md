# Rover Hardware Test Report

**Current status (2026-10-23).**

| Item | Status |
|---|---|
| Isolation tests: platform, camera, ultrasonic, relay, motors | **PASS** (2026-10-16) |
| Teleop drive + relay actuation, end to end | **PASS** (2026-10-17, operator-confirmed) |
| §2 relay state at power-on | **PASS** (2026-10-21, verified cold boot) |
| §1 ultrasonic fault injection | **PASS** (2026-10-23, sensor loss injected in software — the sensor shares the motors' power switch and is soldered in): forward refused 1.09 s after the fault, reverse still allowed, forward restored 0.14 s after recovery — read off the motor-driver pins (`results/rover_tests/20261021_fault_injection/FAULT_INJECTION_SUMMARY.md`) |
| §3 plant frame | **Done** — 32 rover-camera frames of the real plant (pilot, `camera_sessions/20261021_pilot_P01*/frames/`, on disk, not in git) |
| §3 camera height | **Done** — lens 24.5 cm above the ground (operator: "24.5 cm is the height of the camera in the rover"); artefact `results/rover_artifacts/20261021_041635/` |
| AUTO on the real plant | **Done** — 29 fires in 91 s, all 29 seen on the relay pin; stops when the lens is covered; with the plant removed it stops against a plain white background but keeps firing at a wooden door (`results/rover_tests/20261021_auto_on_plant/SUMMARY.md`) |
| Raspberry Pi's own power | **Measured** (2026-10-23, from the Pi 5's power chip, nothing moving): about 1.7–2 W with the camera server running, about 4.8 W with all four cores busy; 5 V input ≥ 4.95 V, never throttled (`results/rover_tests/20261023_pi_power/SUMMARY.md`) |
| Pump with water | **Done** (2026-10-23, operator-confirmed: "pump ran, water out") — four 1 s relay pulses with the pump on water, nozzle into a cup; pulse length 1.00–1.01 s on the relay pin. Dose per pulse **not measured** |
| Motor / relay / pump load on the Pi's 5 V input | **Measured** (2026-10-23, wheels in the air, 35 % duty): no sag beyond 0.01 V. The motor and pump current does not reach the Pi's input, so a motor start or pump pulse did not brown out the Pi (`results/rover_tests/20261023_pi_power/SUMMARY.md`) |
| Motor / pump / camera current; battery capacity; dose per pulse; loaded drive on the floor | **NOT RUN** — needs an inline meter or a current sensor |
| Camera reliability | after a reopen the camera has stuck into stripes three times (the third on 2026-10-23, switching back to view-only after the load test) and once lost its blue channel; each cleared only by a power cycle; all are detected as camera faults (no spray). Not every reopen fails: the switch to control on 2026-10-23 came up clean |
| Obstacle stop | works, but only saw one obstacle at ~4 cm (rover contacted it, 2026-10-21) — a backstop, not a guarantee |

Rover left in its safe default: view-only server running and enabled at boot; control server
stopped and disabled; motor pins low, relay released.

**Status (2026-10-16): ALL FIVE ISOLATION TESTS RUN — all PASS.** Platform, camera and
ultrasonic PASS from the harness directly; relay and motors PASS by operator observation of
the physical rover (the owner confirmed the empty-tank and wheels-off-the-ground safety
conditions, then watched the actuation). The ultrasonic carries a reliability caveat (about
half its readings drop to `None`, §1) and two mandatory follow-ups remain NOT RUN: the
ultrasonic unplug/fault-injection step (§1) and the §16 camera artefacts (a saved plant frame
and a tape-measured height). Run over SSH from the laptop to the Pi on the laptop hotspot
(`pi@192.168.137.167`), using `rover/hardware_tests/test_components.py` copied to the Pi.
See the session note at the end for the full command output.

**Update (2026-10-21): plant session — hardware results and findings.** (Pi clock in UTC.)

- **§2 relay boot state: PASS.** On a cold boot verified by the Pi's uptime (booted 21:56:09),
  the operator watched the relay LED through boot: it stayed off. A first attempt cycled only
  the motor/relay supply while the Pi stayed up (uptime 23 min) and is **not** counted.
- **§1 fault injection: partly run, NOT complete.** With the sensor switched off by the operator,
  telemetry showed `sensor_lost: true` and forward `blocked` within about a second — the
  fail-safe flag works on the robot. But the rover was disarmed at that moment (the page bug
  below), so the wheels stopping *under a forward command* was not observed. A second run
  (laptop holding forward, 22:25:48–22:27:06, 404 drive commands) ended with the operator
  reporting that the rover **"hit an obstacle and stopped; it did not roll over"**. **The wheels
  were on the floor**: the "wheels off the ground" confirmation had been given before a
  power-cycle of the Pi, during which the operator handled the rover, and it was not
  re-confirmed before the run — a process error on the laptop side. The rover drove forward,
  contacted the obstacle, and stopped with the sensor reading 3.7 cm: **the obstacle stop engaged
  only at contact**, so the HC-SR04 did not see that obstacle in time (low, narrow, soft or angled
  objects can fall outside or absorb its beam). No damage reported. The run's own log was lost
  when it was stopped; the Pi's request log is the record.
  **Consequences:** (1) the ultrasonic obstacle stop is a backstop, not a guarantee — forward
  motion needs the operator watching; (2) every motor run is preceded by a fresh confirmation of
  the physical setup, and runs are kept short.
- **Camera: a stuck-sensor failure mode.** Twice, after the camera was closed and reopened (a
  server restart or a mode switch), its sensor delivered static uniform-row stripes — s.d. ≈ 80,
  so the old blank-frame check passed them — which the tissue test read as 86% leaf. A raw grab
  from the driver showed the same, so it is the camera, not OpenCV. A USB reset did not clear it;
  **a power cycle of the Pi did**. The servers, the demo and `test_components.py --test camera`
  now treat such frames as a camera fault (no spray); the deployed guard caught the second episode
  live. For the rest of a session the camera should be opened once (rover-control set as the boot
  service for this session; view-only to be restored as the boot default afterwards).
- **Control page bugs, fixed and redeployed:** releasing a drive button posted `/stop`, which
  disarms, so every press after the first was ignored (and merely moving the mouse off a button
  disarmed); and the camera `<img>` never reconnected after a server restart. Release now stops
  the wheels and stays armed; the view loads frames one at a time onto a fixed canvas.
- **Still not run:** §3 artefacts (plant frame + tape-measured height), pump with liquid,
  battery / power draw, AUTO on the real plant.

**Update (2026-10-17): integrated teleop drive + spray confirmed.** With the control server
(`rover-control.service`, `rover/pi_rover_control.py`) running and driven from a browser, the
operator confirmed the rover **drove under teleoperation and the spray actuated** — i.e. the
whole path (browser → Pi → GPIO loop → motor driver / relay) works end to end, not just the
isolated subsystems of 2026-10-16. The ultrasonic was now reading reliably (~94 cm) during this
run. Caveat: whether the pump moved **liquid** (tank loaded) was not separately logged, so the
pump-with-fluid and battery/power-draw items below remain as stated; and the §1 fault-injection,
§2 relay boot-state and §16 camera-artefact follow-ups are still NOT RUN.

This file records the outcome of **per-component isolation tests**, run *before* any
ML integration (study protocol). The point of testing in isolation is that a
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
| 2026-10-16 | relay | `PASS` | software cycled released→asserted(LOW)→released; **operator confirmed** (tank empty/pump disconnected) the indicator LED came on for ~1 s then off, with an audible click on and off. Harness logged `UNVERIFIED` because non-interactive SSH has no tty for the confirmation prompt; the PASS is the operator's observation | SSH + operator |
| 2026-10-16 | motors | `PASS` | drove left fwd / left rev / right fwd / right rev, 1 s each at 40% duty; **operator confirmed** (wheels off the ground) all four directions turned correctly, none the wrong way, no straining. Two channels drive the four motors in a left pair / right pair, as the code implies. Harness logged `UNVERIFIED` (no tty); the PASS is the operator's observation | SSH + operator |

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

**Update (2026-10-20): the teleop server had the same defect, now fixed in the repo — not yet on
the rover.** `rover/pi_rover_control.py` treated a stale or missing distance as *unknown* and
unknown as a clear path, so with the sensor off, forward drove with no obstacle stop — D2 again.
It now **fails safe**: no fresh reading counts as an obstacle, forward is refused, the UI shows
`SENSOR LOST`, and reverse and turns stay available for backing out. This is tested on the
control code (`tests/test_rover_control_server.py`: sensor removed → forward refused → back-out
allowed → recovers when it returns), and a mutation check confirms the test fails on the old
logic. **It is a property of the code until it is deployed and the fault injection below is
run on the rover.**

Fault-injection procedure (after deploying): drive forward at low speed with the wheels off the
ground, then **switch the sensor off with its power switch** — do not pull jumpers from a
powered board — and record verbatim what the wheels do and what the UI shows. Expected: wheels
stop within about a second, `SENSOR LOST (forward refused)`; reverse still turns the wheels.

### 2. Relay boot state

Confirm the relay is **de-asserted at power-on**, before any software runs. The float
trick (`GPIO.setup(RELAY, GPIO.IN)`) is designed for exactly this, and the audit
judged it correct — but it must be observed on the actual module, since active-high
modules would behave oppositely.

Result (2026-10-16): the relay **cycle** test passed — the operator saw the LED come on
for ~1 s and off, with a click each way, under the correct pin convention (released = pin as
input; asserted = OUT + LOW, i.e. active-low). But the *specific* boot-state observation —
that the relay is de-asserted at power-on **before any software runs** — was `NOT RUN`: it
requires watching the module's LED at the moment the Pi powers up, which this session did not
do. The float trick remains the design; it is confirmed correct in operation, not yet at boot.

### 3. Camera sample frame and measured height (§16)

Two artefacts are required, and together they decide whether **arm E is viable at all**:

- one frame from the actual camera pointed at a **real plant**;
- the camera height above ground, **measured with a tape measure**, not estimated.

**Correction (2026-10-20).** This section used to say the frame is "saved automatically by
`--test camera`". It was not: the harness's frame saving is opt-in and had **no command-line
flag**, so the documented step could not be carried out - which is why no frame existed. Two
ways now work:

```bash
# from the laptop, with a rover camera server running (records the tape height with the frame)
.venv/Scripts/python.exe scripts/capture_rover_artifact.py --height-cm <tape reading> --note "..."
# or on the Pi, after stopping the camera services (only one process can hold the camera)
python3 test_components.py --test camera --save-frame
```

Result (2026-10-16): the camera **hardware** is verified working (live feed, 50 frames,
frame std 57.0 — not blank), but **neither §16 artefact exists yet**: the test was run with
frame-saving off, so no plant frame was captured, and the height was not re-measured this
session. Both remain `NOT RUN`; arm E stays out of the results until they exist.

Result (2026-10-21): **the frame exists** — the pilot captured 32 frames of the real plant through
the rover camera (clean `/raw` frames, `camera_sessions/20261021_pilot_P01*/frames/`; kept on disk,
not in git under the repo's `.gitignore`). **The height**: 24.5 cm, given by the operator as the
height of the camera on the rover; recorded with a fresh frame in
`results/rover_artifacts/20261021_041635/artifact.json` (the frame itself is kept on disk, not in git).
Both §16 artefacts now exist.

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
| battery / power draw | not measured on 2026-10-16. On 2026-10-23 only the Pi's own power was measured, from its power chip (`results/rover_tests/20261023_pi_power/SUMMARY.md`). The motors, pump and camera have no current figure, and the battery has no Wh or run-time figure |
| pump (as distinct from its relay) | not run on 2026-10-16 (outlet disconnected). **Run with water on 2026-10-23** (operator-confirmed); its dose per pulse and current remain unmeasured |

All five isolation-test subsystems (platform, camera, ultrasonic, relay, motors) were verified
this session; the rows above are the parts that remain unverified.

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

RESULT relay: UNVERIFIED - no operator confirmation - unverified
# operator confirmed physically: LED on ~1 s then off, audible click on and off -> PASS

RESULT motors: UNVERIFIED - drove 4 directions at 40% duty
# operator confirmed physically: all four directions correct, no wrong way, no straining -> PASS
```

What this session establishes: all five per-component isolation tests pass. The **Pi boots
and runs**; the **USB camera delivers a live, non-blank feed**; the **ultrasonic measures
distance** once powered (with the `None`-every-other-reading caveat, §1); the **relay switches**
(LED + click, operator-confirmed, active-low); and the **motor driver turns all four wheels** in
both directions on each side (operator-confirmed, wheels off the ground). The relay and motor
tests actuated hardware under the operator-confirmed safety conditions (tank empty / pump
disconnected; wheels off the ground). Not established: the **pump itself** (outlet was
disconnected), the **battery / power draw** (not measured), and the relay's **boot-time** state (§2).

Correction within the session: the first ultrasonic FAIL was because the sensor was not
switched on, not a wiring fault. After powering it, the re-run PASSED. The earlier "diagnose
the wiring" note is therefore withdrawn; what remains for the ultrasonic is the reliability
check (the `None`-every-other-reading pattern) and the deliberate unplug fault-injection test.

Still outstanding (no longer blocked on the basic hardware, which now works):
1. Ultrasonic — characterise the dropped readings (likely settling time between pulses), then
   run the deliberate unplug / fault-injection step (does the rover keep driving/spraying when
   the sensor returns `None`?).
2. §16 camera artefacts — a saved frame of a real plant, and a tape-measured camera height.
3. Relay boot-state (§2) — watch the relay LED at the instant the Pi powers on.
4. Pump, battery / power draw — not yet measured. *(2026-10-23: the Pi's own power is now measured; the motors, pump and battery are not.)*
