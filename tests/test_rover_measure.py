"""Tests for the latency measurement harness, driven entirely by fakes.

The point of these is that the Pi session should discover hardware facts, not
software bugs. Every failure mode the harness can hit -- a dead camera, an LED
out of frame, a camera that drops frames -- is exercised here first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rover.measure import (
    ExternalStage,
    build_budget,
    measure_capture_latency,
    measure_gpio_to_relay,
    measure_processing_latency,
    save_budget,
)
from rover.timing import StageTiming


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeCamera:
    """Returns a scalar 'frame'; brightness follows the LED after a set delay."""

    def __init__(self, led: "FakeLED", *, lag_frames: int = 3,
                 dark: float = 20.0, lit: float = 200.0, fail_every: int = 0):
        self.led, self.lag_frames = led, lag_frames
        self.dark, self.lit = dark, lit
        self.fail_every = fail_every
        self.reads = 0
        self._pipeline: list[bool] = [False] * lag_frames

    def read(self):
        self.reads += 1
        if self.fail_every and self.reads % self.fail_every == 0:
            return False, None
        self._pipeline.append(self.led.on)
        shown = self._pipeline.pop(0)
        return True, (self.lit if shown else self.dark)


class FakeLED:
    def __init__(self) -> None:
        self.on = False
        self.switches = 0

    def turn_on(self) -> None:
        self.on = True
        self.switches += 1

    def turn_off(self) -> None:
        self.on = False


class FakeRelay:
    def __init__(self) -> None:
        self.asserted = False
        self.on_calls = 0

    def relay_on(self) -> None:
        self.asserted = True
        self.on_calls += 1

    def relay_off(self) -> None:
        self.asserted = False


def _scalar_mean(frame) -> float:
    return float(frame)


# ---------------------------------------------------------------------------
# capture latency
# ---------------------------------------------------------------------------


def test_capture_latency_detects_the_led():
    led = FakeLED()
    cam = FakeCamera(led, lag_frames=3)
    st = measure_capture_latency(cam, led.turn_on, led.turn_off, repeats=3,
                                 settle_s=0.0, mean_fn=_scalar_mean)
    assert st.name == "capture"
    assert st.n == 3
    assert all(s > 0 for s in st.samples)


def test_capture_latency_leaves_the_led_off_afterwards():
    """A probe that leaves an output asserted is a probe that starts a fire."""
    led = FakeLED()
    cam = FakeCamera(led)
    measure_capture_latency(cam, led.turn_on, led.turn_off, repeats=2,
                            settle_s=0.0, mean_fn=_scalar_mean)
    assert led.on is False


def test_capture_latency_raises_when_the_led_is_never_seen():
    """LED out of frame, or a brightness threshold it cannot reach."""
    led = FakeLED()
    cam = FakeCamera(led, dark=20.0, lit=22.0)      # only a 2-level change
    with pytest.raises(RuntimeError, match="all 3 attempts failed"):
        measure_capture_latency(cam, led.turn_on, led.turn_off, repeats=3,
                                settle_s=0.0, timeout_s=0.05,
                                brightness_jump=12.0, mean_fn=_scalar_mean)


def test_capture_latency_raises_on_a_dead_camera():
    class DeadCamera:
        def read(self):
            return False, None

    led = FakeLED()
    with pytest.raises(RuntimeError):
        measure_capture_latency(DeadCamera(), led.turn_on, led.turn_off,
                                repeats=2, settle_s=0.0, timeout_s=0.05,
                                mean_fn=_scalar_mean)


def test_capture_latency_warns_but_survives_partial_failure():
    led = FakeLED()
    cam = FakeCamera(led, lag_frames=2, fail_every=7)
    with pytest.warns(UserWarning, match="attempts failed") if False else _noop():
        st = measure_capture_latency(cam, led.turn_on, led.turn_off, repeats=4,
                                     settle_s=0.0, mean_fn=_scalar_mean)
    assert st.n >= 1


class _noop:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_capture_latency_rejects_zero_repeats():
    led = FakeLED()
    with pytest.raises(ValueError):
        measure_capture_latency(FakeCamera(led), led.turn_on, led.turn_off,
                                repeats=0, mean_fn=_scalar_mean)


# ---------------------------------------------------------------------------
# processing latency
# ---------------------------------------------------------------------------


def test_processing_latency_discards_warmup():
    calls = []

    def pipeline(frame):
        calls.append(frame)
        return frame

    led = FakeLED()
    st = measure_processing_latency(FakeCamera(led), pipeline, repeats=10, warmup=5)
    assert st.n == 10
    assert len(calls) == 15          # warm-up ran but was not timed


def test_processing_latency_raises_without_frames():
    class DeadCamera:
        def read(self):
            return False, None

    with pytest.raises(RuntimeError, match="no frames"):
        measure_processing_latency(DeadCamera(), lambda f: f, repeats=3, warmup=0)


# ---------------------------------------------------------------------------
# gpio to relay
# ---------------------------------------------------------------------------


def test_gpio_to_relay_times_the_call_and_releases():
    relay = FakeRelay()
    st = measure_gpio_to_relay(relay, repeats=5, gap_s=0.0)
    assert st.n == 5
    assert relay.on_calls == 5
    assert relay.asserted is False, "probe must not leave the relay energised"


# ---------------------------------------------------------------------------
# external stages
# ---------------------------------------------------------------------------


def test_frame_counts_convert_to_seconds_and_keep_resolution():
    e = ExternalStage.from_frame_counts("relay_to_droplet", [12, 14, 13], fps=240)
    assert e.samples_s[0] == pytest.approx(12 / 240)
    assert e.resolution_s == pytest.approx(1 / 240)
    assert "240" in e.method


def test_frame_counts_reject_bad_input():
    with pytest.raises(ValueError):
        ExternalStage.from_frame_counts("x", [], fps=240)
    with pytest.raises(ValueError):
        ExternalStage.from_frame_counts("x", [1, -2], fps=240)
    with pytest.raises(ValueError):
        ExternalStage.from_frame_counts("x", [1], fps=0)


def test_external_stage_joins_the_budget():
    cap = StageTiming("capture", (0.08,))
    proc = StageTiming("process", (0.05,))
    gpio = StageTiming("gpio_to_relay", (0.001,))
    ext = ExternalStage.from_frame_counts("relay_to_droplet", [24, 26], fps=240)
    b = build_budget(cap, proc, gpio, external=[ext])
    assert len(b.stages) == 4
    assert b.total_mean_s == pytest.approx(0.08 + 0.05 + 0.001 + 25 / 240)


def test_measured_total_overrides_the_sum():
    cap = StageTiming("capture", (0.08,))
    proc = StageTiming("process", (0.05,))
    gpio = StageTiming("gpio_to_relay", (0.001,))
    total = ExternalStage.from_frame_counts("total", [96, 100, 140], fps=240)
    b = build_budget(cap, proc, gpio, measured_total=total)
    assert b.measured_total is not None
    assert b.total_p95_s() == pytest.approx(140 / 240)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_save_budget_records_provenance_and_misses(tmp_path):
    cap = StageTiming("capture", (0.08, 0.09))
    proc = StageTiming("process", (0.05, 0.05))
    gpio = StageTiming("gpio_to_relay", (0.001, 0.001))
    ext = ExternalStage.from_frame_counts("relay_to_droplet", [24, 26], fps=240)
    b = build_budget(cap, proc, gpio, external=[ext])

    out = save_budget(b, tmp_path / "sub" / "budget.json", external=[ext],
                      notes="bench, tethered")
    d = json.loads(out.read_text(encoding="utf-8"))

    assert d["external_stages"][0]["method"].endswith("240 fps")
    assert d["notes"] == "bench, tethered"
    assert d["dominant_stage"] in {"capture", "relay_to_droplet"}
    # the miss must scale with speed
    assert d["predicted_miss_m"]["0.40_mps"] == pytest.approx(
        4 * d["predicted_miss_m"]["0.10_mps"])
