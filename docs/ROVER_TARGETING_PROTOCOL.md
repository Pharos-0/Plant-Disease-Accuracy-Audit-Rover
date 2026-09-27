# Rover Targeting Experiment — Bench Protocol

**Written 2026-09-28, before any hardware was powered on.** Written in advance
deliberately: a protocol decided during a session is a protocol shaped by what
the hardware happened to do that day.

The software is already built and tested — 80 tests over the latency budget, the
compensator, the speed model and the deposition scoring, all passing against
fakes. The session below produces numbers, not code.

---

## 0. The claim this experiment tests

> A spot sprayer that fires the instant it detects a target sprays *behind* the
> target, by a distance that grows linearly with travel speed. Firing early by
> the measured perception-to-actuation delay removes most of that error.

Falsifiable both ways. If the uncompensated offset does not grow with speed, the
premise is wrong. If compensation does not reduce it, the fix is wrong. Either
outcome is a result; neither is a failure of the session.

**Predicted, before measuring:** total latency 150–400 ms on a Pi 4 with a USB
webcam, dominated by camera buffering. At 0.3 m/s that is a 4.5–12 cm miss.
Recording the prediction now so it cannot be quietly adjusted afterwards.

---

## 1. Safety, which comes first and is not negotiable

1. **The spray line stays disconnected until Step 5.** Every probe up to that
   point runs with the pump physically unplugged from the relay's output
   terminals. The relay may click; nothing may spray.
2. **Water only.** No pesticide, no surfactant, no dye that is not food-safe. The
   experiment measures *where* spray lands, and water stains the paper exactly
   as well as anything else would.
3. **Wheels off the ground for every stationary test.** The rover is on a stand
   for Steps 2–4. A four-motor skid-steer rover that lurches off a bench lands
   on its camera mast.
4. **Kill switch within reach.** Battery disconnect, not a software stop.
5. Anything electrical that gets hot, smells, or behaves unexpectedly: disconnect
   the battery first and diagnose afterwards.

The four-motor / two-channel finding in `rover_hardware_inventory.md` §2.1 is the
specific concern. Four gearmotors stalling together can exceed an L298N's ~2 A
per-channel rating. **Never hold the rover against a wall under power.**

---

## 2. Inventory and isolation probes — 30 minutes

Nothing is assumed working. Run `rover/hardware_tests/test_components.py`
component by component, in this order, and record the result of each before
moving on:

| # | probe | pass condition | if it fails |
|---|---|---|---|
| 1 | platform | Pi boots, OS and Python version recorded | stop; everything downstream needs this |
| 2 | camera | frames arrive AND `frame_is_blank()` is False | check lens cap, USB port, `v4l2-ctl --list-devices` |
| 3 | ultrasonic | plausible distance, changes when a hand moves | check 5 V, and that ECHO is level-shifted |
| 4 | relay | audible click, indicator LED changes, **pump still unplugged** | check active-low wiring on pin 24 |
| 5 | motors | each channel turns its pair, both directions | **this is the one most likely to fail** — the driver was never visible in any photograph |

**Record the motor driver's part number.** The pin map assumes IN1–4 = 17/27/22/23
and ENA/ENB = 18/19. If the board is not an L298N-compatible H-bridge, the pin map
changes and so does the deadband.

> The camera probe already caught a blind camera once: 60 frames delivered, pixel
> standard deviation 0.0000, every pixel valued 2. `ok = frames > 0` passed it.
> That is why the blank check exists and why it runs before anything else.

---

## 3. Speed calibration — 40 minutes

Feeds `rover.speed.fit_speed_model`.

**Setup.** A straight run of at least 3 m on the surface the experiment will use.
Mark a start line and a finish line, measure the gap with a tape, write it down.
Surface matters — carpet and tile give different deadbands — so use one surface
and say which in the thesis.

**Procedure.** For each duty in **20, 30, 40, 50, 60, 70, 80**:

1. Place the rover behind the start line, wheels straight.
2. Command forward at that duty. Start timing as the front axle crosses the start
   line, stop as it crosses the finish line.
3. Three repetitions per duty.
4. Record the battery voltage before the first and after the last run at each
   duty.

**Include the duties where it does not move.** Those runs define the deadband, and
the model needs them. Record them as a very long duration rather than skipping
them — the fitter identifies a stall by speed, not by a flag.

```python
from rover.speed import SpeedRun, fit_speed_model
runs = [SpeedRun(duty=40, distance_m=3.0, duration_s=11.2, battery_v=7.9), ...]
model = fit_speed_model(runs)
print(model.summary())
```

**Expect the fit to refuse if something is wrong.** It raises on a non-positive
duty–speed correlation, which is what reversed motor wiring looks like. Take the
refusal seriously rather than working around it.

---

## 4. Latency budget — 45 minutes

Feeds `rover.measure`.

### 4a. Capture latency, automatic

Wire an LED to a spare GPIO, point it into the lens, room lights low.

```python
from rover.measure import measure_capture_latency
cap = measure_capture_latency(camera, led_on, led_off, repeats=20)
```

`drain_reads` defaults to 10 and must exceed the camera's buffer depth. If the
probe reports failures, raise it to 20 — a buffer deeper than the drain is the
one failure mode the fakes could not rule out.

### 4b. Processing latency, automatic

```python
from rover.measure import measure_processing_latency
proc = measure_processing_latency(camera, pipeline, repeats=50, warmup=5)
```

Run this **on the Pi**, not on a laptop. The whole point is the Pi's speed.

### 4c. GPIO to relay, automatic

```python
from rover.measure import measure_gpio_to_relay
gpio = measure_gpio_to_relay(rover, repeats=50)
```

This is software only and will be small — a lower bound, labelled as one.

### 4d. Relay to droplet, phone camera

The Pi cannot time this: the mechanical close and the pump spin-up both happen
after the GPIO write returns.

1. Phone at **240 fps**, framing the relay indicator LED **and** the nozzle tip in
   one shot.
2. Pump connected, tank has water, nozzle pointed into a bucket.
3. Trigger the relay ten times with a few seconds between.
4. Step through frame by frame. Count frames from the indicator changing to the
   first visible droplet leaving the nozzle.

```python
from rover.measure import ExternalStage
ext = ExternalStage.from_frame_counts("relay_to_droplet", [28, 31, 29, ...], fps=240)
```

4.2 ms resolution against a budget of hundreds of ms. The quantisation is
recorded, not hidden.

### 4e. End-to-end, phone camera — do this one even if time is short

Film the **stimulus LED** and the **relay indicator** together, let the rover's
own detection logic fire the relay, count frames between. This is the most
valuable single number in the session, because it captures correlations between
stages that summing means cannot.

```python
from rover.measure import build_budget, save_budget
total = ExternalStage.from_frame_counts("total", [96, 103, 99, ...], fps=240)
budget = build_budget(cap, proc, gpio, external=[ext], measured_total=total)
save_budget(budget, Path("results/rover_latency/budget.json"),
            external=[ext, total], notes="bench, tethered, carpet")
```

---

## 5. The targeting experiment — 90 minutes

This is the headline result. **Spray line connected from here on. Water only.**

### Track

- Straight run, 4 m minimum.
- **Nine target cards**, water-sensitive paper (or plain paper plus a food-safe
  dye tracer), at 40 cm intervals starting 1 m from the start line.
- Cards flat, long axis **along** the direction of travel — this fixes
  `travel_axis=1`.
- Measure each card. A5 is about 0.21 × 0.148 m. Whatever it is, measure it and
  pass it to `CardGeometry`.
- Number every card in pencil before the run. A card whose position is unknown
  afterwards is a discarded data point.

### Conditions

**2 arms × 4 speeds × 3 repetitions = 24 runs.** Nine cards per run gives up to
216 scored cards, which is ample.

| arm | trigger |
|---|---|
| **naive** | `LeadCompensator.naive_decide` — the V0 behaviour: fire on detection at a fixed 25 cm range |
| **compensated** | `LeadCompensator.decide` — fire at `v x t_total` using the Step 4 budget and the Step 3 speed model |

Speeds: the duties from Step 3 giving roughly **0.15, 0.25, 0.35, 0.45 m/s**, or
whatever four separated speeds the rover can actually hold.

### Order

**Alternate the arms and randomise speed order within each block.** Do not run all
naive runs then all compensated runs. The battery droops, the surface gets damp,
and both drift in the same direction as the effect — a blocked design would let
that drift masquerade as the result.

Record battery voltage before every run. `SpeedModel.stale_for` flags when it has
drifted more than 0.4 V from calibration; recalibrate rather than push on.

### Scoring

Photograph each card flat, even lighting, whole card in frame, from directly
above.

```python
from rover.deposition import CardGeometry, score_card, summarise_runs
geom = CardGeometry(width_m=0.21, height_m=0.148, target_frac=0.5, travel_axis=1)
results = [score_card(imread(p), geom) for p in card_photos]
print(summarise_runs(results))
```

**Sign convention, pinned by test and stated here so nobody re-derives it in the
thesis: negative offset means the spray landed BEHIND the target.** An
uncompensated rover should be consistently negative, growing more negative with
speed.

---

## 6. What gets reported, including the ways it can fail

**Figure 1 — the latency budget.** Stacked bar, one segment per stage, error bars
from the measured spread. Names the dominant stage.

**Figure 2 — the headline.** Mean offset against travel speed, two lines with
confidence intervals. Prediction: naive slopes downward at a gradient equal to the
total latency; compensated stays flat near zero. *The naive slope should equal the
measured latency* — that is a genuine cross-check between two independent
measurements, and it is worth stating whether it holds.

**Figure 3 — hit rate against speed**, two lines. The practical statement: the
speed at which spraying stops working, with and without compensation.

**Table 1 — per condition:** n, hit rate, mean offset, sd, mean absolute offset.

### Outcomes that are not the expected one

| observation | what it means | still reportable? |
|---|---|---|
| naive offset does not grow with speed | latency is small relative to the spray cone, or speeds are too close together | **yes** — report the budget and the null, and say the effect is below the resolution of this rig |
| compensation overshoots to positive | the budget overestimates the delay, probably `drain_reads` too low | **yes** — an overcorrection is evidence the model is directionally right |
| hit rate near 1.0 everywhere | the spray cone is wide enough to cover the error | **yes** — then report *offset*, not hit rate, and say the rig cannot resolve hit rate |
| motors cannot hold a steady speed | skid-steer scrub, documented defect D8 | run fewer speeds with more repetitions, and report the speed spread |

**Only one outcome makes the session a loss: not recording something.** Write down
every run, every voltage, every card number, including the runs that go wrong.

---

## 7. Time budget

| step | duration | needs |
|---|---|---|
| 2 — inventory and probes | 30 min | rover, no spray |
| 3 — speed calibration | 40 min | 3 m run, tape, stopwatch |
| 4 — latency budget | 45 min | LED, phone at 240 fps |
| 5 — targeting runs | 90 min | 4 m track, 9 cards, water |
| — card photography | 20 min | phone, even light |
| **total** | **≈3.5 hours** | |

Steps 2–4 need no spray and can run on a bench. **Step 5 is the only one needing
the full track**, so if time runs short, Steps 2–4 alone still yield the latency
budget — publishable on its own, and nobody has it for this hardware class.

---

## 8. Before you start

- [ ] Battery charged, voltmeter to hand
- [ ] Spray line **disconnected**
- [ ] Tape measure, stopwatch, masking tape, pencil
- [ ] Phone set to 240 fps and confirmed — check, do not assume
- [ ] Target cards cut, measured, numbered
- [ ] 3 m calibration run and 4 m track marked out
- [ ] `git pull` on the Pi, then `pytest` — 216 tests should pass before anything
      is plugged in
