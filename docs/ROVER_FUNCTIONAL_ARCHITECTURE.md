# Rover Functional Architecture

**2026-09-11.** What exists, from reading the code. Nothing here is aspirational.

## 1. Components present

| component | file | lines | state |
|---|---|---:|---|
| Hardware abstraction, GPIO, PWM | `rover/rover_v1.py` | 401 | working; `RecordingGPIO` allows hardware-free tests |
| Adaptive vision (CLAHE + gamma) | `rover/rover_v1.py:176-224` | — | working, unit-tested |
| Original delivered rover (web UI) | `rover/original/web_rover.py` | 241 | preserved, unmodified |
| Hardware probes | `rover/hardware_tests/test_components.py` | 342 | runs; camera probe currently blind (see §4) |
| Leaf proposal (green-region) | `src/rover/leaf_proposal.py` | 133 | working, unit-tested |
| **Session gate** | `src/rover/session_gate.py` | 165 | working; **evaluated 2026-09-11, see `ROVER_RESEARCH_INTEGRATION.md`** |

Tests: `tests/test_rover_v1.py`, `test_rover_gate.py`, `test_leaf_proposal.py`,
`test_components.py` — inside the 126-test suite.

## 2. The two modes, and why the split matters

```
NORMAL MODE                          RESEARCH MODE
camera                               camera
  -> adaptive vision                   -> adaptive vision
  -> leaf proposal                      -> leaf proposal
  -> frozen classifier                  -> frozen classifier
  -> downstream autonomous action       -> SessionGate.observe(logits)
                                        -> verdict(): OPERATE / REVIEW /
                                           WITHHOLD / INSUFFICIENT
                                        -> action or fallback
```

The rover remains fully functional with the research module disabled. This was a design
requirement and it is met: `SessionGate` is additive and holds no hardware state.

## 3. Safety properties actually enforced in code

- `SessionGate.is_image_correct()` — and its aliases `estimate_for_image`,
  `should_spray_this_leaf` — **raise `PerImageQueryError`**. A dataset-level estimator
  cannot be asked about one leaf, and the code refuses rather than relying on discipline.
- `GateDecision.INSUFFICIENT` below `min_session_size`: no verdict from too few
  observations.
- Only `OPERATE` arms the sprayer; `REVIEW` and `WITHHOLD` do not.
- Sensor failure must map to **stop / spray off / require attention**, never to continue.

## 4. Known defects, unfixed and recorded

| defect | severity | status |
|---|---|---|
| Camera probe passes on `frames > 0`, so a **blind camera reports healthy** — measured: 60 frames, per-frame std 0.0000, every pixel = 2 | **high** (a sensor failure reported as health) | **open**, deferred by the user 2026-09-10 |
| `test_camera` writes a JPEG to the repo root on every `pytest` run | low, but a privacy issue once the camera works | **open**, deferred |
| Hardware probes are collected by pytest and return dicts rather than asserting | low | **open** |
| `GateConfig` thresholds (0.80 / 0.65) are guesses | **superseded** — §Rover integration shows no threshold makes autonomy safe here |

## 5. Never claim

- ✗ that a predicted disease implies a validated chemical treatment — downstream action is
  represented abstractly;
- ✗ per-image correctness;
- ✗ demonstrated autonomous safety. No field trial has been run.
