"""V1 rover runtime: proof that D1 and D3 are actually fixed.

The point of these tests is that the D3 guarantee is *demonstrated*, not asserted.
A comment saying "cleanup is guaranteed" is worth nothing; a test that throws an
exception mid-drive and then checks the spray relay was released is worth
something, because it is the exact scenario that leaves a sprayer running in a
field.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_rover_v1():
    name = "rover_v1_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "rover" / "rover_v1.py")
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations via sys.modules[cls.__module__], so the
    # module must be registered before it is executed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rv1 = _load_rover_v1()


@pytest.fixture
def hw():
    """A rover on a recording GPIO backend, always shut down afterwards."""
    gpio = rv1.RecordingGPIO()
    rover = rv1.RoverHardware(gpio=gpio)
    yield rover, gpio
    rover.shutdown()


# ==========================================================================
# D1 — the exposure correction now points the right way
# ==========================================================================


def test_d1_gamma_choice_is_the_right_way_round() -> None:
    """The fix in isolation: which gamma is selected at each brightness.

    V0 returned 2.0 where this returns 0.7 and vice versa. That inversion is
    defect D1, and this is the cleanest statement of the fix.
    """
    assert rv1.gamma_for(235) == 0.7, "over-exposed -> darken"
    assert rv1.gamma_for(25) == 2.0, "under-exposed -> brighten"
    assert rv1.gamma_for(128) is None, "normal exposure -> no correction"


def test_d1_gamma_stage_darkens_an_over_exposed_frame() -> None:
    """Measured against the post-CLAHE frame, which is what the gamma sees.

    Comparing against the raw input would conflate two stages: CLAHE *itself*
    brightens an over-exposed frame (235 -> 241), so the gamma's job is to pull
    that back. It does.
    """
    pytest.importorskip("cv2")
    adjust_gamma = rv1._load_adjust_gamma()

    raw = np.full((64, 64, 3), 235, dtype=np.uint8)
    after_clahe = rv1.clahe_normalise(raw)
    assert after_clahe.mean() > raw.mean(), "CLAHE brightens an over-exposed frame"

    out, brightness = rv1.adaptive_vision(raw)
    assert brightness > rv1.BRIGHT_THRESHOLD
    assert out.mean() < after_clahe.mean(), "D1: the gamma stage must darken it"

    # and V0, on the same post-CLAHE frame, would have gone the other way
    v0 = adjust_gamma(after_clahe, 2.0)
    assert v0.mean() > after_clahe.mean(), "V0 brightens it further"


def test_d1_gamma_stage_brightens_an_under_exposed_frame() -> None:
    pytest.importorskip("cv2")
    adjust_gamma = rv1._load_adjust_gamma()

    raw = np.full((64, 64, 3), 25, dtype=np.uint8)
    after_clahe = rv1.clahe_normalise(raw)
    out, brightness = rv1.adaptive_vision(raw)
    assert brightness < rv1.DARK_THRESHOLD
    assert out.mean() > after_clahe.mean(), "D1: the gamma stage must brighten it"

    v0 = adjust_gamma(after_clahe, 0.7)
    assert v0.mean() < after_clahe.mean(), "V0 darkens it further"


def test_d1_v1_moves_the_opposite_way_to_v0() -> None:
    """The V0 constants, applied to the same frame, go the wrong way.

    This is the before/after that justifies calling D1 a fix rather than a
    preference.
    """
    pytest.importorskip("cv2")
    adjust_gamma = rv1._load_adjust_gamma()

    bright = np.full((64, 64, 3), 235, dtype=np.uint8)
    v0_result = adjust_gamma(bright, 2.0)   # what V0 does to a too-bright frame
    v1_result = adjust_gamma(bright, 0.7)   # what V1 does
    assert v0_result.mean() > bright.mean(), "V0 brightens an already-bright frame"
    assert v1_result.mean() < bright.mean(), "V1 darkens it"


def test_d1_returns_v0_signature() -> None:
    """(frame, brightness) — a drop-in replacement for V0's adaptive_vision."""
    pytest.importorskip("cv2")
    out = rv1.adaptive_vision(np.full((32, 32, 3), 128, dtype=np.uint8))
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[1], float)


# ==========================================================================
# D3 — shutdown is guaranteed, which is the whole point
# ==========================================================================


def test_d3_cleanup_runs_when_the_loop_raises(hw) -> None:
    """The scenario that matters: a crash mid-drive with the sprayer on."""
    rover, gpio = hw

    with pytest.raises(RuntimeError, match="camera exploded"):
        with rover:
            rover.forward()
            rover.relay_on()
            assert rover.relay_asserted, "precondition: the sprayer is running"
            raise RuntimeError("camera exploded")

    assert not rover.relay_asserted, "D3: the spray relay stayed asserted after a crash"
    assert gpio.pin_mode[rv1.RELAY] == gpio.IN, "relay must be released by floating the pin"
    assert gpio.cleaned, "GPIO.cleanup() was never reached"
    for pin in (rv1.IN1, rv1.IN2, rv1.IN3, rv1.IN4):
        assert gpio.pin_value[pin] == 0, f"motor pin {pin} left energised"


def test_d3_exception_is_not_swallowed(hw) -> None:
    """Cleanup must not hide the fault that caused it."""
    rover, _ = hw
    with pytest.raises(ValueError, match="the real problem"):
        with rover:
            raise ValueError("the real problem")


def test_d3_cleanup_runs_on_keyboard_interrupt(hw) -> None:
    rover, gpio = hw
    with pytest.raises(KeyboardInterrupt):
        with rover:
            rover.forward()
            rover.relay_on()
            raise KeyboardInterrupt
    assert not rover.relay_asserted
    assert gpio.cleaned


def test_d3_cleanup_runs_on_normal_exit(hw) -> None:
    rover, gpio = hw
    with rover:
        rover.forward()
    assert gpio.cleaned
    assert not rover.relay_asserted


def test_d3_shutdown_is_idempotent(hw) -> None:
    rover, gpio = hw
    rover.start()
    rover.shutdown()
    n = len(gpio.calls)
    rover.shutdown()
    rover.shutdown()
    assert len(gpio.calls) == n, "repeated shutdown must be a no-op"


def test_d3_shutdown_never_raises_even_if_a_step_fails(hw) -> None:
    """A shutdown path that can throw is not a shutdown path.

    If stopping the motors fails, the relay must still be released.
    """
    rover, gpio = hw
    rover.start()

    def boom() -> None:
        raise RuntimeError("motor driver unreachable")

    rover._stop_motors = boom
    rover.shutdown()  # must not raise
    assert gpio.pin_mode[rv1.RELAY] == gpio.IN, "relay released despite the motor failure"
    assert gpio.cleaned


def test_d3_shutdown_before_start_is_safe() -> None:
    rover = rv1.RoverHardware(gpio=rv1.RecordingGPIO())
    rover.shutdown()  # must not raise


def test_d3_motors_are_stopped_before_the_relay_is_released(hw) -> None:
    """Order matters: stop moving, then stop spraying, then drop the pins."""
    rover, gpio = hw
    rover.start()
    rover.forward()
    rover.relay_on()
    gpio.calls.clear()
    rover.shutdown()

    kinds = [c[0] for c in gpio.calls]
    motor_stop = next(
        i for i, c in enumerate(gpio.calls)
        if c[0] == "output" and c[1] == [rv1.IN1, rv1.IN2, rv1.IN3, rv1.IN4]
    )
    relay_release = next(
        i for i, c in enumerate(gpio.calls)
        if c[0] == "setup" and c[1] == rv1.RELAY and c[2] == gpio.IN
    )
    assert motor_stop < relay_release < kinds.index("cleanup")


# ==========================================================================
# what V1 deliberately does NOT change
# ==========================================================================


def test_pin_map_is_identical_to_v0() -> None:
    """V1 must not quietly move a pin; the comparison depends on it."""
    baseline = ROOT / "rover" / "original" / "web_rover.py"
    if not baseline.exists():
        pytest.skip("frozen baseline not present")
    src = baseline.read_text(encoding="utf-8")
    assert f"IN1, IN2, IN3, IN4 = {rv1.IN1}, {rv1.IN2}, {rv1.IN3}, {rv1.IN4}" in src
    assert f"ENA, ENB = {rv1.ENA}, {rv1.ENB}" in src
    assert f"RELAY = {rv1.RELAY}" in src
    assert f"TRIG, ECHO = {rv1.TRIG}, {rv1.ECHO}" in src


def test_relay_float_convention_preserved(hw) -> None:
    """The audit judged V0's float trick correct; V1 must keep it."""
    rover, gpio = hw
    rover.start()
    rover.relay_on()
    assert gpio.pin_mode[rv1.RELAY] == gpio.OUT
    assert gpio.pin_value[rv1.RELAY] == gpio.LOW, "active-low assert"
    rover.relay_off()
    assert gpio.pin_mode[rv1.RELAY] == gpio.IN, "released by floating, not driven HIGH"


def test_remaining_defects_are_declared_not_hidden() -> None:
    remaining = rv1.remaining_defects()
    for d in ("D2", "D4", "D6", "D7", "D8", "D9", "D10"):
        assert d in remaining and remaining[d]
    assert "D1" not in remaining and "D3" not in remaining, "D1 and D3 are fixed in V1"
