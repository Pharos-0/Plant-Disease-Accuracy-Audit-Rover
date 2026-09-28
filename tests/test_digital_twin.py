"""Tests for the digital twin, including negative controls for its own checker.

The invariant checker was revised three times while building the twin: it keyed
gate states by a float timestamp and lost records to collisions, and it
sequenced the relay event before the decision that authorised it. Both produced
FALSE violations, and both were fixed.

A checker revised until it reports no violations is worthless unless it can
still report one. So the tests below are mostly negative controls: they inject
a specific fault and assert the checker catches that exact fault. If any of
these stop failing, the checker has stopped working.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rover"))
sys.path.insert(0, str(ROOT / "src"))

from digital_twin import (HARDWARE_TAG, ObstacleProfile, SimulatedCamera,  # noqa: E402
                          TwinClock, TwinGPIO, _Seq, check_invariants)
from rover_v1 import RELAY, RoverHardware  # noqa: E402


# --------------------------------------------------------------- clock
def test_clock_is_monotonic():
    c = TwinClock()
    c.advance(1.0)
    assert c.now() == 1.0
    with pytest.raises(ValueError):
        c.advance(-1.0)


def test_injected_clock_makes_distance_measurable():
    """The defect the twin found: get_distance read the wall clock directly, so
    a simulated sensor measured a ~0 s pulse and reported 0 cm."""
    clock = TwinClock()
    gpio = TwinGPIO(clock, distance_source=lambda: 100.0)
    with RoverHardware(gpio=gpio, clock=clock.now) as rover:
        d = rover.get_distance()
    assert d is not None
    assert 80.0 < d < 120.0, f"expected ~100 cm from the echo model, got {d}"


def test_distance_default_is_still_the_wall_clock():
    """The injected clock must not change behaviour on the Pi."""
    import time as _t
    assert RoverHardware.clock is _t.time


# --------------------------------------------------------------- camera
def test_blank_frame_detection_matches_the_observed_failure():
    clock = TwinClock()
    cam = SimulatedCamera(["a.jpg", "b.jpg"], clock, blank_every=2)
    cam.read()
    _ok, frame, _p = cam.read()
    assert frame is not None
    assert float(np.asarray(frame).std()) < 0.5, "injected frame must be constant"


# --------------------------------------------------------- NEGATIVE CONTROLS
def _trace(states: list[str]) -> list[dict]:
    return [{"seq": i + 1, "t": float(i), "gate_state": s,
             "decision": "X", "blank_frame": False, "obstacle_cm": 999}
            for i, s in enumerate(states)]


def test_checker_catches_relay_armed_under_withhold():
    """The fault that matters: actuating when the gate said do not."""
    frames = _trace(["OPERATE", "WITHHOLD"])
    relay = [{"seq": 3, "t": 1.5, "pin": RELAY, "event": "setup", "value": True},
             {"seq": 4, "t": 1.6, "pin": RELAY, "event": "setup", "value": False}]
    v = check_invariants(frames, relay)
    assert any("not OPERATE" in x for x in v), v


def test_checker_catches_relay_left_armed_at_the_end():
    frames = _trace(["OPERATE"])
    relay = [{"seq": 2, "t": 1.0, "pin": RELAY, "event": "setup", "value": True}]
    v = check_invariants(frames, relay)
    assert any("still armed" in x for x in v), v


def test_checker_catches_actuation_with_no_operate_state_at_all():
    frames = _trace(["WITHHOLD", "WITHHOLD"])
    relay = [{"seq": 3, "t": 1.0, "pin": RELAY, "event": "output", "value": False}]
    v = check_invariants(frames, relay)
    assert any("never returned OPERATE" in x for x in v), v


def test_checker_catches_unrejected_blank_frame():
    frames = [{"seq": 1, "t": 0.0, "gate_state": "OPERATE", "decision": "ACTUATE",
               "blank_frame": True, "obstacle_cm": 999}]
    v = check_invariants(frames, [])
    assert any("blank frame" in x for x in v), v


def test_checker_catches_obstacle_not_stopped():
    frames = [{"seq": 1, "t": 0.0, "gate_state": "OPERATE", "decision": "ACTUATE",
               "blank_frame": False, "obstacle_cm": 5.0}]
    v = check_invariants(frames, [])
    assert any("did not stop" in x for x in v), v


def test_checker_passes_a_correct_trace():
    """The positive control, so the negatives above are not passing vacuously."""
    frames = _trace(["INSUFFICIENT", "OPERATE"])
    relay = [{"seq": 3, "t": 1.5, "pin": RELAY, "event": "setup", "value": True},
             {"seq": 4, "t": 1.6, "pin": RELAY, "event": "output", "value": False},
             {"seq": 5, "t": 1.7, "pin": RELAY, "event": "setup", "value": False}]
    assert check_invariants(frames, relay) == []


def test_equal_timestamps_do_not_erase_a_decision():
    """The exact bug that produced the first false violation: a fault record
    sharing a virtual instant with the frame that caused it."""
    frames = [{"seq": 1, "t": 5.0, "gate_state": "OPERATE", "decision": "ACTUATE",
               "blank_frame": False, "obstacle_cm": 999},
              {"seq": 3, "t": 5.0, "gate_state": "FAULT", "decision": "ABORT",
               "blank_frame": False, "obstacle_cm": 999}]
    relay = [{"seq": 2, "t": 5.0, "pin": RELAY, "event": "setup", "value": True},
             {"seq": 4, "t": 5.0, "pin": RELAY, "event": "setup", "value": False}]
    assert check_invariants(frames, relay) == [], "tie on t must not reorder"


# --------------------------------------------------------------- honesty
def test_hardware_tag_is_simulated():
    assert HARDWARE_TAG == "SIMULATED"


def test_twin_gpio_is_the_tested_stand_in():
    """The twin must subclass the GPIO the unit tests already use, so it cannot
    drift from the behaviour those tests pin down."""
    from rover_v1 import RecordingGPIO
    assert issubclass(TwinGPIO, RecordingGPIO)


def test_relay_released_after_injected_fault():
    """The property the thesis safety claim rests on, end to end."""
    clock = TwinClock()
    gpio = TwinGPIO(clock, distance_source=lambda: 100.0, seq=_Seq())
    with pytest.raises(RuntimeError):
        with RoverHardware(gpio=gpio, clock=clock.now) as rover:
            rover.forward()
            rover.relay_on()
            raise RuntimeError("injected")
    arming = [e for e in gpio.relay_history() if e["event"] == "setup"]
    assert arming and arming[-1]["value"] is False, "relay left armed after a fault"
