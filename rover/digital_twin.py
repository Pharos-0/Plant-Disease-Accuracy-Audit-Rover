"""A digital twin of the rover, for exercising the software stack end to end.

WHAT A TWIN CAN AND CANNOT ESTABLISH
------------------------------------
This file simulates the rover's *peripherals* so the real control software can
be run against them. Read that sentence carefully, because it bounds everything
below.

It CAN establish:

  * that the full stack runs end to end without error - camera capture through
    classification, session accumulation, gate decision and actuation;
  * that the safety invariants hold, including under injected faults, because
    those are properties of the control code and the code under test is the
    real `rover_v1.RoverHardware`, not a stand-in;
  * what the software decides on real cached model outputs.

It CANNOT establish anything whatsoever about:

  * whether the Raspberry Pi, camera, motor driver, ultrasonic sensor, relay,
    pump or battery work, or exist;
  * timing on the Pi - any duration measured here is a workstation figure;
  * lens, mounting, illumination, traction, power draw or field behaviour.

A simulated actuator that "fires" proves the *decision* was taken. It proves
nothing about a pump. Every artefact this module writes is stamped
`hardware: SIMULATED`, and `scripts/run_digital_twin.py` refuses to emit a
record without that field.

WHY THE RELAY IS MODELLED AS AN ASSERTION TARGET RATHER THAN AN OUTPUT
----------------------------------------------------------------------
The interesting question is never "did the relay pin go low". It is "could the
relay ever be asserted at a moment when the gate had not authorised it". So the
twin's GPIO records a full timestamped transition log, and the scenario checks
run over that log as invariants. A violation is a test failure, not a warning.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rover"))

from rover_v1 import ECHO, RELAY, TRIG, RecordingGPIO  # noqa: E402

#: Stamped into every artefact. The string is checked by the writer, so an
#: artefact cannot be produced that omits it.
HARDWARE_TAG = "SIMULATED"


# --------------------------------------------------------------------------
# Deterministic clock
# --------------------------------------------------------------------------
class _Seq:
    """Monotonic execution counter, shared by the GPIO log and the frame trace.

    Ordering by this rather than by timestamp is what makes the invariant check
    sound: virtual time can tie, execution order cannot.
    """

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


@dataclass
class TwinClock:
    """Virtual time, so a traversal is reproducible and runs in milliseconds.

    Wall-clock durations are measured separately and reported separately; the
    two are never mixed, because one is a property of the scenario and the other
    of the machine that happened to run it.
    """

    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time does not run backwards")
        self.t += seconds


# --------------------------------------------------------------------------
# GPIO with a transition log
# --------------------------------------------------------------------------
class TwinGPIO(RecordingGPIO):
    """RecordingGPIO plus timestamps, an echo model, and a relay transition log.

    Subclasses the class the project already tests against, so the twin cannot
    drift from the stand-in used by the unit tests.
    """

    def __init__(self, clock: TwinClock, distance_source: Callable[[], float] | None = None,
                 seq: "_Seq | None" = None):
        super().__init__()
        self.clock = clock
        self.seq = seq or _Seq()
        self.transitions: list[dict] = []
        self._distance_source = distance_source or (lambda: 120.0)
        self._echo_high_until = -1.0

    # -- transition logging -------------------------------------------------
    def output(self, pin: Any, value: Any) -> None:  # noqa: N802 - RPi.GPIO API
        super().output(pin, value)
        self.transitions.append({"seq": self.seq.next(), "t": self.clock.now(),
                                 "pin": pin, "event": "output",
                                 "value": bool(value)})
        if pin == TRIG and value:
            # A real HC-SR04 replies after 2 * distance / speed-of-sound.
            d_cm = self._distance_source()
            self._echo_high_until = self.clock.now() + (2 * d_cm / 34300.0)

    def setup(self, pin: Any, mode: Any) -> None:
        super().setup(pin, mode)
        pins = pin if isinstance(pin, (list, tuple)) else [pin]
        for p in pins:
            if p == RELAY:
                self.transitions.append({"seq": self.seq.next(),
                                         "t": self.clock.now(), "pin": p,
                                         "event": "setup",
                                         "value": mode is self.OUT})

    def input(self, pin: Any) -> int:  # noqa: N802 - RPi.GPIO API
        if pin == ECHO:
            # The driver busy-waits on this pin; advance virtual time so the
            # loop terminates instead of spinning.
            self.clock.advance(1e-5)
            return 1 if self.clock.now() < self._echo_high_until else 0
        return super().input(pin)

    # -- derived history ----------------------------------------------------
    def relay_history(self) -> list[dict]:
        return [x for x in self.transitions if x["pin"] == RELAY]


# --------------------------------------------------------------------------
# Peripherals
# --------------------------------------------------------------------------
@dataclass
class SimulatedCamera:
    """Serves real cached frames in capture order, as a camera would.

    It does NOT synthesise imagery. Frames are the real images behind cached
    model outputs, so what the classifier sees is exactly what it saw in the
    offline experiments. The simulation is of the *camera*, not of the scene.
    """

    paths: list[str]
    clock: TwinClock
    frame_interval_s: float = 0.5
    blank_every: int | None = None          # inject a dead-camera fault
    _i: int = field(default=0, init=False)

    def __len__(self) -> int:
        return len(self.paths)

    def read(self) -> tuple[bool, np.ndarray | None, str | None]:
        if self._i >= len(self.paths):
            return False, None, None
        p = self.paths[self._i]
        self._i += 1
        self.clock.advance(self.frame_interval_s)
        if self.blank_every and self._i % self.blank_every == 0:
            # The exact failure the camera probe was fixed to catch: a frame
            # with zero variance, which a naive "did we get bytes" check passes.
            return True, np.full((240, 320, 3), 2, dtype=np.uint8), p
        return True, None, p          # None => caller uses the cached activation


@dataclass
class ObstacleProfile:
    """Scripted distances in cm, indexed by frame. Beyond the script, clear."""

    distances: dict[int, float] = field(default_factory=dict)
    default_cm: float = 120.0
    _frame: int = field(default=0, init=False)

    def step(self) -> None:
        self._frame += 1

    def current(self) -> float:
        return self.distances.get(self._frame, self.default_cm)


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------
@dataclass
class TraversalResult:
    frames: list[dict]
    relay_history: list[dict]
    virtual_seconds: float
    wall_seconds: float
    violations: list[str]
    hardware: str = HARDWARE_TAG


def check_invariants(frames: list[dict], relay_history: list[dict]) -> list[str]:
    """Safety properties that must hold over the whole run.

    Ordering is by execution sequence, never by timestamp: a fault record and
    the frame that caused it share a virtual instant, and an earlier version of
    this function silently dropped the frame because it keyed a dict by the
    float. Equal timestamps must not be allowed to reorder or erase evidence.
    """
    v: list[str] = []

    # gate state in force when each relay event happened, by execution order
    states = sorted(((f.get("seq", 0), f["gate_state"]) for f in frames))

    def state_at(seq: int) -> str:
        prior = [s for q, s in states if q <= seq]
        return prior[-1] if prior else "UNSET"

    for ev in relay_history:
        if ev["event"] == "setup" and ev["value"]:            # armed as OUTPUT
            s = state_at(ev.get("seq", 0))
            if s != "OPERATE":
                v.append(f"relay armed at t={ev['t']:.3f}s while the gate was "
                         f"{s}, not OPERATE")

    asserted = [e for e in relay_history if e["event"] == "output" and not e["value"]]
    if asserted and not any(f["gate_state"] == "OPERATE" for f in frames):
        v.append("relay asserted although the gate never returned OPERATE")

    # Whatever happened, the relay must end released. This is the property the
    # thesis's safety claim actually rests on, and the old checker never tested it.
    arming = [e for e in relay_history if e["event"] == "setup"]
    if arming and arming[-1]["value"]:
        v.append("run ended with the relay still armed as an output")

    for f in frames:
        if f.get("blank_frame") and f["decision"] != "SKIP_BLANK_FRAME":
            v.append(f"blank frame at t={f['t']:.3f}s was not rejected")
        if f.get("obstacle_cm", 999) < 20 and f["decision"] not in ("STOP", "SKIP_BLANK_FRAME"):
            v.append(f"obstacle at {f['obstacle_cm']}cm did not stop the rover "
                     f"at t={f['t']:.3f}s")
    return v
