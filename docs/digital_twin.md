# Digital Twin — What It Establishes, and What It Does Not

**2026-09-28.** A simulated rover, so the real control software can be exercised end to end
without hardware. Written because the hardware is unavailable and the alternative — asserting
in prose that the software works — is not evidence.

---

## 1. The boundary, stated first

The code under test is the **real** `rover_v1.RoverHardware`. Only the peripherals are
simulated: GPIO, camera, ultrasonic sensor. Classifier outputs are **real cached activations**
from the frozen experiments, so the decisions taken are the decisions the software would take
on that data. No imagery is synthesised.

**It establishes:**

- the full stack runs end to end: capture → classify → accumulate session → gate → actuate;
- the safety invariants hold, including under injected faults, because those are properties of
  the control code;
- what the software decides, on real model outputs.

**It establishes nothing whatever about:** the Raspberry Pi, the camera, the motor driver, the
ultrasonic sensor, the relay, the pump, the battery, power draw, traction, lens, mounting,
illumination or field behaviour. **A simulated actuator that fires proves a decision was
taken. It proves nothing about a pump.**

`docs/rover_hardware_test_report.md` still has **nine rows reading NOT RUN**, and this document
does not change one of them. Every artefact is stamped `hardware: SIMULATED`, and the writer
refuses to emit a record without that field.

Timing is reported twice and never merged: **virtual seconds**, a property of the scenario, and
**wall seconds**, a property of the workstation that ran it. Neither is a Raspberry Pi figure.

## 2. A defect the twin found in the rover code

`RoverHardware.get_distance()` read `time.time()` directly. Driven by a simulated sensor on a
virtual clock, it measured a pulse of roughly zero seconds, computed **0.0 cm**, and stopped
the rover on every single frame — the obstacle path could not be exercised at all.

That is a testability defect in the rover code, not in the twin: a driver whose only clock is
the wall clock cannot be tested against a simulated sensor, and the pulse-to-distance
conversion is exactly the part worth testing, because a mis-scaled constant is how an obstacle
check fails silently.

Fixed by injecting the time source, defaulting to `time.time` so nothing on the Pi changes. A
test asserts the default is still the wall clock.

## 3. A conceptual error in the first version of the twin

The gate originally computed the session's running accuracy **from the true labels** and gated
on that. Deployment has no labels — that is the premise of the entire project — so it was
demonstrating a gate that cannot exist, and it made the rover look far safer than it is.

The gate now runs a real label-free estimator (ATC-MC), fitted on the source validation split
and applied to the accumulated session logits, exactly as the offline experiments do. True
accuracy is still computed but only as a contrast, labelled `oracle_*` throughout, and nothing
in the control path reads it.

## 4. Three revisions to the invariant checker, and why they are listed

The checker reported **false** violations twice before it was right:

1. It keyed gate states by a float timestamp. A fault record appended without the virtual
   clock advancing collided with the frame that caused it and erased it, so a correctly
   authorised actuation was reported as a violation.
2. It sequenced the relay event before the decision that authorised it, so it read the
   *previous* frame's state and flagged a violation at every boundary where the gate first
   reached OPERATE.

Both are fixed by ordering on a monotonic execution counter rather than on time. A checker
revised until it stops complaining is worthless unless it can still complain, so
`tests/test_digital_twin.py` carries **five negative controls** that inject a specific fault
and assert that exact fault is caught — relay armed under WITHHOLD, relay left armed at the
end, actuation with no OPERATE state at all, an unrejected blank frame, an unstopped obstacle
— plus a positive control and a regression test for the timestamp-tie bug. 14 tests.

## 5. Scenarios and results

| | scenario | arm | result |
|---|---|---|---|
| S1 | zero-shift control | `source_test` | 96 OPERATE, 96 actuations, **0 unsafe** |
| S2 | severe field shift | `tomatovillage` | 46 OPERATE, 45 actuations, **15 unsafe** |
| S3 | camera fault | `tomatovillage` | blank frames rejected; 33 actuations, **11 unsafe** |
| S4 | crash mid-traversal | `source_test` | relay released; invariants held |

All safety invariants held in every scenario. That is a statement about the control software.

### The result worth reporting

On the severe-shift arm the session's true accuracy falls to **0.533** while the label-free
estimator holds at **0.783** — a persistent overestimate of roughly $0.25$.

| | OPERATE | REVIEW | WITHHOLD |
|---|---:|---:|---:|
| label-free gate | 46 | 50 | **0** |
| oracle gate | 31 | 12 | **53** |

**The label-free gate never once reaches WITHHOLD**, while the oracle would withhold for 53 of
120 frames. Fifteen actuations occur that the oracle would have refused.

This is the thesis finding made operational: a rover gated on a label-free estimate keeps
spraying through a session in which it should have stopped, because the estimate is anchored to
source calibration and does not fall with true accuracy.

**One caveat on the ordering.** Frames are served in dataset order, not in a randomised or
physically realistic traversal order. Tomato-Village's early frames happen to be classified
well, which is why estimator and oracle agree at the start. A real traversal order is unknown,
so the *number* of unsafe actuations is scenario-specific; the *direction* — the estimator
overestimates and the gate under-withholds — is the part that follows from the offline results.

## 6. Reproduce

```bash
python scripts/run_digital_twin.py --frames 120
python -m pytest tests/test_digital_twin.py -q
```

Artefacts in `results/digital_twin/`, one JSON per scenario with the full frame trace, the
relay transition log, and the invariant result.
