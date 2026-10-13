"""Tests for the sprayer FSM: structural safety, fault handling, autonomous return.

The headline property is that the pump can be energised in exactly one state.
It is tested three ways - by inspecting the table, by driving every fault, and
by a randomized fuzz over fused readings - because a safety property that is
only checked one way is usually only true one way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rover"))

from sprayer_fsm import (SPRAY_OK, TRANSITIONS, FusedReading,  # noqa: E402
                         SprayerController, SprayPolicy, State,
                         nozzle_decision, unreachable_spray_states)


def reading(**kw) -> FusedReading:
    base = dict(range_cm=150.0, range_age=0, row_offset=0.0, row_age=0,
                canopy=(0.5, 0.5, 0.5, 0.5), disease_confidence=0.9,
                vision_age=0, tank_fraction=1.0, battery_fraction=1.0,
                resource_age=0, at_dock=False)
    base.update(kw)
    return FusedReading(**base)


def started(policy=SprayPolicy.SELECTIVE) -> SprayerController:
    c = SprayerController(policy=policy)
    c.step(reading(), gate_open=True)          # IDLE -> TRAVERSING
    assert c.state is State.TRAVERSING
    return c


# ---------------------------------------------------------------- structure
def test_spraying_is_the_only_sprayable_state():
    assert [s for s, ok in SPRAY_OK.items() if ok] == [State.SPRAYING]
    assert unreachable_spray_states() == set(State) - {State.SPRAYING}


def test_every_state_has_a_transition_table_entry():
    assert set(TRANSITIONS) == set(State)


def test_fault_is_reachable_from_every_non_terminal_state():
    for s in State:
        if s in (State.FAULT_SAFE_STOP,):
            continue
        assert State.FAULT_SAFE_STOP in TRANSITIONS[s], s


def test_fault_state_only_exits_to_dock():
    assert TRANSITIONS[State.FAULT_SAFE_STOP] == {State.DOCKED}


def test_illegal_transition_raises():
    c = SprayerController()
    with pytest.raises(ValueError):
        c._go(State.SPRAYING)                  # IDLE cannot jump to SPRAYING


# ---------------------------------------------------------------- behaviour
def test_sprays_on_diseased_canopy_with_gate_open():
    c = started()
    cmd = c.step(reading(), gate_open=True)
    assert c.state is State.SPRAYING and cmd.spraying


def test_bare_ground_is_not_sprayed():
    c = started()
    cmd = c.step(reading(canopy=(0.0, 0.0, 0.0, 0.0)), gate_open=True)
    assert not cmd.spraying


def test_only_zones_with_canopy_fire():
    c = started(SprayPolicy.VISION_ONLY)
    cmd = c.step(reading(canopy=(0.6, 0.0, 0.6, 0.0)), gate_open=True)
    assert cmd.nozzles == (True, False, True, False)


def test_healthy_plant_is_skipped_by_selective_policy():
    c = started()
    cmd = c.step(reading(disease_confidence=0.1), gate_open=True)
    assert not cmd.spraying


def test_shut_gate_degrades_to_vision_only_not_blanket():
    canopy = (0.6, 0.0, 0.6, 0.0)
    shut = nozzle_decision(SprayPolicy.SELECTIVE, canopy, 0.1, gate_open=False)
    assert shut == (True, False, True, False)
    assert shut != (True, True, True, True)


def test_stale_vision_stops_spraying_rather_than_guessing():
    c = started()
    cmd = c.step(reading(vision_age=10), gate_open=True)
    assert not cmd.spraying and cmd.drive == "forward"


def test_obstacle_halts_motion_and_pump_then_clears():
    c = started()
    c.step(reading(), gate_open=True)
    cmd = c.step(reading(range_cm=10.0), gate_open=True)
    assert c.state is State.OBSTACLE_STOP
    assert cmd.drive == "stop" and not cmd.spraying
    cmd = c.step(reading(), gate_open=True)
    assert c.state is State.TRAVERSING and cmd.drive == "forward"


# ------------------------------------------------------- autonomous return
@pytest.mark.parametrize("field,value", [("tank_fraction", 0.05),
                                         ("battery_fraction", 0.10)])
def test_low_resource_returns_and_docks(field, value):
    c = started()
    cmd = c.step(reading(**{field: value}), gate_open=True)
    assert c.state is State.RETURNING and cmd.drive == "return"
    assert not cmd.spraying
    cmd = c.step(reading(**{field: value}), gate_open=True)
    assert c.state is State.RETURNING
    cmd = c.step(reading(**{field: value}, at_dock=True), gate_open=True)
    assert c.state is State.DOCKED and cmd.drive == "stop"


def test_no_spraying_while_returning():
    c = started()
    c.step(reading(tank_fraction=0.05), gate_open=True)
    for _ in range(5):
        cmd = c.step(reading(tank_fraction=0.05), gate_open=True)
        assert not cmd.spraying


# ------------------------------------------------------------ fault handling
FAULTS = {
    "stale ultrasonic": dict(range_age=10),
    "stale resource telemetry": dict(resource_age=10),
    "wrong zone count": dict(canopy=(0.5, 0.5)),
}


@pytest.mark.parametrize("name", list(FAULTS))
def test_each_fault_reaches_safe_stop_with_pump_off(name):
    c = started()
    c.step(reading(), gate_open=True)          # spraying when the fault hits
    cmd = c.step(reading(**FAULTS[name]), gate_open=True)
    assert c.state is State.FAULT_SAFE_STOP, name
    assert not cmd.spraying and cmd.drive == "stop"


def test_fault_latches_until_human_reset():
    c = started()
    c.step(reading(range_age=10), gate_open=True)
    for _ in range(5):
        cmd = c.step(reading(), gate_open=True)          # perfectly good input
        assert c.state is State.FAULT_SAFE_STOP and not cmd.spraying


# ------------------------------------------------------------------ fuzzing
def test_fuzz_never_sprays_outside_spraying_state():
    """1,000 runs of 60 random fused readings each. If any command energises a
    nozzle while the controller is not in SPRAYING, the safety claim is false."""
    rng = np.random.default_rng(20261013)
    for run in range(1000):
        c = SprayerController(policy=rng.choice(list(SprayPolicy)))
        for _ in range(60):
            r = reading(
                range_cm=float(rng.choice([5.0, 30.0, 150.0])),
                range_age=int(rng.choice([0, 0, 0, 5])),
                row_age=int(rng.choice([0, 0, 5])),
                canopy=tuple(float(x) for x in rng.random(4)),
                disease_confidence=float(rng.random()),
                vision_age=int(rng.choice([0, 0, 5])),
                tank_fraction=float(rng.choice([1.0, 0.5, 0.05])),
                battery_fraction=float(rng.choice([1.0, 0.5, 0.1])),
                resource_age=int(rng.choice([0, 0, 0, 5])),
                at_dock=bool(rng.random() < 0.1))
            cmd = c.step(r, gate_open=bool(rng.random() < 0.5))
            if cmd.spraying:
                assert c.state is State.SPRAYING, f"run {run}: sprayed in {c.state}"
                assert r.range_cm >= 25.0, f"run {run}: sprayed at an obstacle"


# --------------------------------------------- crop detector path (open field)
def test_weed_is_sprayed_by_vision_only_but_not_by_crop_targeting():
    """The open-field failure of a green detector: a weed is green."""
    green_everywhere = (0.6, 0.6, 0.6, 0.6)
    crop_only_in_zone_0 = (0.9, 0.1, 0.1, 0.1)
    vo = nozzle_decision(SprayPolicy.VISION_ONLY, green_everywhere, 0.9, True,
                         crop_only_in_zone_0)
    sel = nozzle_decision(SprayPolicy.SELECTIVE, green_everywhere, 0.9, True,
                          crop_only_in_zone_0)
    assert vo == (True, True, True, True)
    assert sel == (True, False, False, False)


def test_shut_gate_falls_back_to_crop_only_not_green_or_blanket():
    sel = nozzle_decision(SprayPolicy.SELECTIVE, (0.6, 0.6, 0.6, 0.6), 0.1,
                          gate_open=False, crop_prob=(0.9, 0.1, 0.9, 0.1))
    assert sel == (True, False, True, False)


def test_per_zone_disease_is_respected():
    sel = nozzle_decision(SprayPolicy.SELECTIVE, (0.6,) * 4, 0.0, True,
                          crop_prob=(0.9,) * 4, disease_by_zone=(0.9, 0.1, 0.9, 0.1))
    assert sel == (True, False, True, False)


def test_without_a_detector_behaviour_is_unchanged():
    old = nozzle_decision(SprayPolicy.SELECTIVE, (0.6, 0.0, 0.6, 0.0), 0.9, True)
    assert old == (True, False, True, False)


@pytest.mark.parametrize("field", ["crop_prob", "disease_by_zone"])
def test_wrong_length_zone_vector_faults(field):
    c = started()
    cmd = c.step(reading(**{field: (0.9, 0.9)}), gate_open=True)
    assert c.state is State.FAULT_SAFE_STOP and not cmd.spraying
