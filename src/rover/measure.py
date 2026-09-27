"""Measuring the latency budget on the actual rover.

Three of the four stages can be timed by the Pi itself. The fourth cannot, and
this module says so rather than inventing a number for it.

    stage              how it is measured                      needs
    -----------------  --------------------------------------  --------------
    capture            LED loopback: flash a GPIO-driven LED    camera + LED
                       into the lens, timestamp the frame in
                       which the pixels change
    process            wall clock around the vision pipeline    camera
    gpio_to_relay      wall clock around the relay call         relay
    relay_to_droplet   NOT measurable from the Pi -- the        phone at 240fps
                       mechanical click and the pump spin-up
                       happen after the GPIO write returns

`relay_to_droplet` is therefore measured externally, with a phone camera at 240
frames per second filming the relay indicator LED and the nozzle in one frame,
and counting frames between them. 240 fps gives 4.2 ms resolution against a
budget of a few hundred milliseconds, which is ample. `ExternalStage` exists to
carry those hand-counted numbers into the same budget as the automatic ones,
tagged with how they were obtained so the provenance survives into the thesis.

The same phone-camera method also gives a *direct* end-to-end total -- film the
stimulus LED and the relay LED together -- and `LatencyBudget.measured_total`
prefers it over summing stages, because it captures whatever correlation exists
between them.

Nothing here runs without hardware, so every function takes an injected rover
and camera. `tests/test_rover_measure.py` drives all of it with fakes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from rover.timing import LatencyBudget, StageTiming

__all__ = [
    "CameraLike",
    "RelayLike",
    "ExternalStage",
    "measure_capture_latency",
    "measure_processing_latency",
    "measure_gpio_to_relay",
    "build_budget",
    "save_budget",
]


class CameraLike(Protocol):
    """The slice of `cv2.VideoCapture` used here."""

    def read(self) -> tuple[bool, Any]: ...


class RelayLike(Protocol):
    """The slice of `RoverHardware` used here."""

    def relay_on(self) -> None: ...
    def relay_off(self) -> None: ...


# ---------------------------------------------------------------------------
# externally measured stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalStage:
    """A stage timed off-Pi, with its method recorded.

    Carrying the method into the data structure is deliberate. A number that
    reaches the thesis without saying how it was obtained is a number a reviewer
    has to take on trust, and "we filmed it at 240 fps and counted frames" is
    both honest and checkable.
    """

    name: str
    samples_s: tuple[float, ...]
    method: str
    resolution_s: float

    def to_stage(self) -> StageTiming:
        return StageTiming(self.name, self.samples_s)

    @staticmethod
    def from_frame_counts(name: str, counts: Sequence[int], fps: float,
                          method: str = "phone camera, frame counting") -> "ExternalStage":
        """Convert counted video frames to seconds.

        The quantisation error is +/- one frame, which at 240 fps is 4.2 ms.
        That is recorded as `resolution_s` rather than being silently dropped.
        """
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not counts:
            raise ValueError(f"stage {name!r}: no frame counts given")
        if any(c < 0 for c in counts):
            raise ValueError(f"stage {name!r}: negative frame count")
        return ExternalStage(name, tuple(c / fps for c in counts),
                             f"{method} at {fps:g} fps", 1.0 / fps)


# ---------------------------------------------------------------------------
# stage 1 -- capture latency by LED loopback
# ---------------------------------------------------------------------------


def measure_capture_latency(camera: CameraLike, led_on: Callable[[], None],
                            led_off: Callable[[], None], *, repeats: int = 20,
                            settle_s: float = 0.30, timeout_s: float = 1.0,
                            brightness_jump: float = 12.0, drain_reads: int = 10,
                            mean_fn: Callable[[Any], float] | None = None
                            ) -> StageTiming:
    """Time from switching an LED on to seeing it in a frame.

    This measures the whole capture path -- sensor integration, USB transfer,
    driver buffering, and any frames already queued inside OpenCV. Buffered
    frames are the dominant term on a USB webcam and are invisible to any
    software-only timing, which is why the loopback exists.

    `brightness_jump` is in grey levels. It must exceed the frame-to-frame noise
    of the camera or the first noisy frame will be mistaken for the LED.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    _mean = mean_fn if mean_fn is not None else _frame_mean
    samples: list[float] = []
    failures = 0

    for _ in range(repeats):
        led_off()
        time.sleep(settle_s)

        # Drain the queue until the dark frames come through.
        #
        # A single read here is not enough, and the reason is the very thing
        # this probe measures: the driver still holds frames captured while the
        # LED was lit during the *previous* repeat. Taking the first frame after
        # led_off() as the baseline therefore captures a lit frame, and the
        # brightness jump it is waiting for can never occur against it. The
        # symptom is that the first repeat succeeds and every later one times
        # out.
        baseline = _settle_baseline(camera, _mean, brightness_jump, timeout_s,
                                    drain_reads=drain_reads)
        if baseline is None:
            failures += 1
            continue

        t0 = time.perf_counter()
        led_on()
        seen = None
        while time.perf_counter() - t0 < timeout_s:
            ok, frame = camera.read()
            if not ok:
                continue
            if _mean(frame) - baseline >= brightness_jump:
                seen = time.perf_counter() - t0
                break
        led_off()

        if seen is None:
            failures += 1
        else:
            samples.append(seen)

    if not samples:
        raise RuntimeError(
            f"capture latency: all {repeats} attempts failed. Either the LED is "
            f"not in frame, the camera is not delivering frames, or "
            f"brightness_jump={brightness_jump} exceeds the LED's effect.")
    if failures:
        import warnings
        warnings.warn(f"capture latency: {failures}/{repeats} attempts failed",
                      stacklevel=2)
    return StageTiming("capture", tuple(samples))


def _settle_baseline(camera: CameraLike, mean_fn: Callable[[Any], float],
                     brightness_jump: float, timeout_s: float,
                     drain_reads: int = 10, stable_reads: int = 3
                     ) -> float | None:
    """Flush buffered frames, then return the settled dark level.

    Two earlier versions of this were wrong, and both failures are worth
    keeping in mind because they would have surfaced as nonsense numbers from
    the Pi rather than as an error.

    The first read a single frame as the baseline. But the driver still holds
    frames captured while the LED was lit during the previous repeat, so the
    baseline *was* a lit frame and the jump it waited for could never occur.
    Symptom: the first repeat succeeds and every later one times out.

    The second drained until three consecutive frames were stable. Three
    consecutive leftover lit frames are perfectly stable, so it latched onto the
    lit level instead. Same symptom.

    What actually works is to flush a bounded number of frames unconditionally
    before believing anything -- `drain_reads` must exceed the camera's buffer
    depth, which is why it is a named parameter and not a constant -- and only
    then require a stable run. The value returned is the minimum over the stable
    run: with the LED off, stale frames can only pull brightness up.

    Returns None if it cannot settle within the timeout, meaning the camera is
    not delivering frames or the LED has not actually gone out.
    """
    t0 = time.perf_counter()

    flushed = 0
    while flushed < drain_reads:
        if time.perf_counter() - t0 >= timeout_s:
            return None
        ok, _ = camera.read()
        if ok:
            flushed += 1

    recent: list[float] = []
    while time.perf_counter() - t0 < timeout_s:
        ok, frame = camera.read()
        if not ok:
            continue
        recent.append(mean_fn(frame))
        if len(recent) > stable_reads:
            recent.pop(0)
        if len(recent) == stable_reads and max(recent) - min(recent) < brightness_jump / 2:
            return min(recent)
    return None


def _frame_mean(frame: Any) -> float:
    """Mean grey level. Imported lazily so this module works without numpy."""
    import numpy as np

    return float(np.asarray(frame).mean())


# ---------------------------------------------------------------------------
# stage 2 -- processing
# ---------------------------------------------------------------------------


def measure_processing_latency(camera: CameraLike,
                               pipeline: Callable[[Any], Any], *,
                               repeats: int = 50,
                               warmup: int = 5) -> StageTiming:
    """Time the vision pipeline on real frames.

    Warm-up runs are discarded. The first calls pay for lazy imports, CPU
    frequency ramp and cache population, and including them would overstate the
    steady-state cost the compensator actually experiences.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    for _ in range(max(0, warmup)):
        ok, frame = camera.read()
        if ok:
            pipeline(frame)

    samples: list[float] = []
    for _ in range(repeats):
        ok, frame = camera.read()
        if not ok:
            continue
        t0 = time.perf_counter()
        pipeline(frame)
        samples.append(time.perf_counter() - t0)

    if not samples:
        raise RuntimeError("processing latency: the camera returned no frames")
    return StageTiming("process", tuple(samples))


# ---------------------------------------------------------------------------
# stage 3 -- GPIO write to relay call returning
# ---------------------------------------------------------------------------


def measure_gpio_to_relay(relay: RelayLike, *, repeats: int = 50,
                          gap_s: float = 0.05) -> StageTiming:
    """Time the `relay_on()` call itself.

    This is a lower bound on the actuation stage and is labelled as one. It
    measures software: pin mode change and level write. It does NOT include the
    electromechanical delay of the relay closing, nor pump spin-up, both of
    which happen after the call returns and must be measured externally.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        relay.relay_on()
        samples.append(time.perf_counter() - t0)
        relay.relay_off()
        time.sleep(gap_s)
    return StageTiming("gpio_to_relay", tuple(samples))


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def build_budget(capture: StageTiming, process: StageTiming,
                 gpio: StageTiming,
                 external: Sequence[ExternalStage] = (),
                 measured_total: ExternalStage | None = None) -> LatencyBudget:
    """Combine automatic and external stages into one budget."""
    stages = [capture, process, gpio] + [e.to_stage() for e in external]
    return LatencyBudget(
        stages=tuple(stages),
        measured_total=measured_total.to_stage() if measured_total else None,
    )


def save_budget(budget: LatencyBudget, path: Path, *,
                external: Sequence[ExternalStage] = (),
                notes: str = "") -> Path:
    """Write the budget, its provenance and the derived misses to JSON."""
    payload = {
        "stages": budget.table(),
        "total_mean_s": budget.total_mean_s,
        "total_sd_s": budget.total_sd_s,
        "total_p95_s": budget.total_p95_s(),
        "dominant_stage": budget.dominant_stage().name,
        "external_stages": [
            {"name": e.name, "method": e.method, "resolution_s": e.resolution_s,
             "n": len(e.samples_s)} for e in external
        ],
        "predicted_miss_m": {
            f"{v:.2f}_mps": v * budget.total_mean_s
            for v in (0.10, 0.20, 0.30, 0.40, 0.50)
        },
        "notes": notes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
