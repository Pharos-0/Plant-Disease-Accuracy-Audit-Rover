"""Rover V1 runtime — fixes defects D1 and D3 from the audit.

V0 is `rover/original/web_rover.py`, frozen and never edited. This module is the
V1 engineering version, and it changes **exactly two things** so the V0 -> V1
comparison stays single-factor:

* **D1 — inverted gamma.** V0 applied `gamma=2.0` when the frame was already too
  bright and `gamma=0.7` when it was already too dark. With this LUT convention
  gamma > 1 *brightens*, so both branches amplified the exposure problem. Field
  -critical: in strong sun the frame is already blown out and V0 pushes it
  further, saturation collapses, and the loose HSV saturation floor then admits
  more non-plant material. **The two constants are swapped.**

* **D3 — no `try/finally` around the main loop.** V0's cleanup at lines 237-241
  is unreachable on any exception, so **motors and the spray relay stay live**
  when the program crashes. Here shutdown is guaranteed four ways: a `finally`
  block, a context manager, an `atexit` hook, and SIGINT/SIGTERM handlers.
  `shutdown()` is idempotent, so running all four is harmless.

Everything else is deliberately left alone — same pins, same PWM frequency, same
control logic, same `c > 1500` trigger. D2, D4, D6, D7, D8 and D10 are **not**
fixed here; see `remaining_defects()` for why and how to enable the D2 fix.

Runs on the Pi with `RPi.GPIO`. For offline testing, inject a fake backend:

    rover = RoverHardware(gpio=RecordingGPIO())
"""
from __future__ import annotations

import atexit
import importlib.util
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_ROOT = Path(__file__).resolve().parents[1]


def _load_adjust_gamma():
    """Load the LUT from `src/rover/leaf_proposal.py` by path.

    Deliberately not `from rover.leaf_proposal import ...`: this file lives in
    `rover/` and that module lives in `src/rover/`, so the package name `rover`
    is ambiguous and which one wins would depend on sys.path ordering. Loading
    by file path removes the ambiguity, and keeps one implementation of the LUT
    rather than a second copy that could drift.
    """
    path = _ROOT / "src" / "rover" / "leaf_proposal.py"
    if not path.is_file():
        raise ImportError(f"cannot find the gamma LUT at {path}")
    name = "_v1_leaf_proposal"
    if name in sys.modules:
        return sys.modules[name].adjust_gamma
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # leaf_proposal uses @dataclass, which resolves annotations through
    # sys.modules[cls.__module__]; register before executing.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.adjust_gamma

# ---------------------------------------------------------------------------
# pin map — identical to V0 (web_rover.py lines 9-12), verified by
# tests/test_frozen_baseline.py
# ---------------------------------------------------------------------------
IN1, IN2, IN3, IN4 = 17, 27, 22, 23
ENA, ENB = 18, 19
RELAY = 24
TRIG, ECHO = 5, 6

PWM_HZ = 1000
DEFAULT_SPEED = 40
MIN_SPEED, MAX_SPEED = 20, 100

# V0's exposure thresholds, unchanged
BRIGHT_THRESHOLD = 170
DARK_THRESHOLD = 70


class GPIOBackend(Protocol):
    """The subset of RPi.GPIO this module uses."""

    BCM: Any
    OUT: Any
    IN: Any
    LOW: Any
    HIGH: Any

    def setmode(self, mode: Any) -> None: ...
    def setwarnings(self, flag: bool) -> None: ...
    def setup(self, pin: Any, mode: Any) -> None: ...
    def output(self, pin: Any, value: Any) -> None: ...
    def input(self, pin: Any) -> int: ...
    def PWM(self, pin: int, hz: int) -> Any: ...
    def cleanup(self) -> None: ...


def _real_gpio() -> GPIOBackend:
    import RPi.GPIO as GPIO  # type: ignore

    return GPIO


# ---------------------------------------------------------------------------
# offline test double
# ---------------------------------------------------------------------------


class _RecordingPWM:
    def __init__(self, pin: int, hz: int, log: list[tuple]) -> None:
        self.pin, self.hz, self._log = pin, hz, log
        self.duty = 0.0
        self.stopped = False

    def start(self, duty: float) -> None:
        self.duty = duty
        self._log.append(("pwm_start", self.pin, duty))

    def ChangeDutyCycle(self, duty: float) -> None:  # noqa: N802 - RPi.GPIO API
        self.duty = duty
        self._log.append(("pwm_duty", self.pin, duty))

    def stop(self) -> None:
        self.stopped = True
        self.duty = 0.0
        self._log.append(("pwm_stop", self.pin))


class RecordingGPIO:
    """In-memory stand-in for RPi.GPIO, so the D3 guarantee can be tested."""

    BCM, OUT, IN, LOW, HIGH = "BCM", "OUT", "IN", 0, 1

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.pin_mode: dict[int, Any] = {}
        self.pin_value: dict[int, Any] = {}
        self.cleaned = False
        self.echo_sequence: list[int] = []

    def setmode(self, mode: Any) -> None:
        self.calls.append(("setmode", mode))

    def setwarnings(self, flag: bool) -> None:
        self.calls.append(("setwarnings", flag))

    def setup(self, pin: Any, mode: Any) -> None:
        for p in pin if isinstance(pin, (list, tuple)) else [pin]:
            self.pin_mode[p] = mode
        self.calls.append(("setup", pin, mode))

    def output(self, pin: Any, value: Any) -> None:
        for p in pin if isinstance(pin, (list, tuple)) else [pin]:
            self.pin_value[p] = value
        self.calls.append(("output", pin, value))

    def input(self, pin: Any) -> int:
        return self.echo_sequence.pop(0) if self.echo_sequence else 0

    def PWM(self, pin: int, hz: int) -> _RecordingPWM:  # noqa: N802 - RPi.GPIO API
        return _RecordingPWM(pin, hz, self.calls)

    def cleanup(self) -> None:
        self.cleaned = True
        self.calls.append(("cleanup",))


# ---------------------------------------------------------------------------
# D1 — the corrected exposure correction
# ---------------------------------------------------------------------------


def clahe_normalise(frame):
    """CLAHE on the LAB L-channel. Identical to V0; split out to be testable.

    Worth knowing before reading the gamma stage: on an already over-exposed
    frame **CLAHE brightens it further** (a uniform 235 comes out at 241). So V0
    was doubly wrong in strong sun — CLAHE pushed the frame up, and then V0's
    `gamma=2.0` pushed it up again. The D1 fix is what pulls it back.
    """
    import cv2

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def gamma_for(brightness: float, bright_threshold: int = BRIGHT_THRESHOLD,
              dark_threshold: int = DARK_THRESHOLD) -> float | None:
    """The D1 fix, isolated: which gamma to apply at this brightness.

    Returns None when no correction is called for. V0 returned 2.0 where this
    returns 0.7 and vice versa — that inversion *is* defect D1.
    """
    if brightness > bright_threshold:
        return 0.7   # darken. V0 used 2.0, which brightens.
    if brightness < dark_threshold:
        return 2.0   # brighten. V0 used 0.7, which darkens.
    return None


def adaptive_vision(frame, bright_threshold: int = BRIGHT_THRESHOLD,
                    dark_threshold: int = DARK_THRESHOLD):
    """CLAHE, then a **correctly signed** gamma pass.

    Returns ``(frame, brightness)`` exactly as V0 did, so this is a drop-in
    replacement for `web_rover.adaptive_vision`.
    """
    import numpy as np

    adjust_gamma = _load_adjust_gamma()

    frame = clahe_normalise(frame)
    brightness = float(np.mean(frame))
    gamma = gamma_for(brightness, bright_threshold, dark_threshold)
    if gamma is not None:
        frame = adjust_gamma(frame, gamma)
    return frame, brightness


# ---------------------------------------------------------------------------
# D3 — hardware ownership with a guaranteed shutdown
# ---------------------------------------------------------------------------


@dataclass
class RoverHardware:
    """Owns every actuator, and guarantees they are released.

    Use as a context manager. `shutdown()` is idempotent and must never raise —
    a shutdown path that can itself throw is not a shutdown path.
    """

    gpio: GPIOBackend | None = None
    speed: int = DEFAULT_SPEED
    _pwm_left: Any = field(default=None, init=False, repr=False)
    _pwm_right: Any = field(default=None, init=False, repr=False)
    _relay_asserted: bool = field(default=False, init=False)
    _shutdown_done: bool = field(default=False, init=False)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.gpio is None:
            self.gpio = _real_gpio()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "RoverHardware":
        g = self.gpio
        g.setmode(g.BCM)
        g.setwarnings(False)
        g.setup([IN1, IN2, IN3, IN4, ENA, ENB, TRIG], g.OUT)
        g.setup(ECHO, g.IN)
        # relay released by floating the pin - V0's trick, judged correct by the
        # audit, and preserved deliberately: driving it HIGH would energise some
        # active-low modules at boot.
        g.setup(RELAY, g.IN)

        self._pwm_left = g.PWM(ENA, PWM_HZ)
        self._pwm_right = g.PWM(ENB, PWM_HZ)
        self._pwm_left.start(0)
        self._pwm_right.start(0)
        self._started = True

        atexit.register(self.shutdown)
        self._install_signal_handlers()
        return self

    def __enter__(self) -> "RoverHardware":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.shutdown()
        return False  # never swallow the exception; just guarantee cleanup

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.shutdown()
            raise KeyboardInterrupt(f"signal {signum}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not on the main thread; the finally block still covers us

    def shutdown(self) -> None:
        """Stop motors, release the relay, release the camera-independent GPIO.

        Idempotent, and swallows its own errors: during a crash this must not
        raise and mask the original traceback.
        """
        if self._shutdown_done or not self._started:
            self._shutdown_done = True
            return
        self._shutdown_done = True

        # order matters: motors first, then the spray relay, then the pins.
        for step in (self._stop_motors, self.relay_off, self._cleanup_pins):
            try:
                step()
            except Exception:  # noqa: BLE001 - a failing step must not block the rest
                pass

    def _stop_motors(self) -> None:
        g = self.gpio
        g.output([IN1, IN2, IN3, IN4], 0)
        for pwm in (self._pwm_left, self._pwm_right):
            if pwm is not None:
                pwm.ChangeDutyCycle(0)
                pwm.stop()

    def _cleanup_pins(self) -> None:
        self.gpio.cleanup()

    # -- motors ------------------------------------------------------------
    def stop(self) -> None:
        g = self.gpio
        g.output([IN1, IN2, IN3, IN4], 0)
        self._pwm_left.ChangeDutyCycle(0)
        self._pwm_right.ChangeDutyCycle(0)

    def _drive(self, a: int, b: int, c: int, d: int,
               left: float, right: float) -> None:
        g = self.gpio
        g.output(IN1, a); g.output(IN2, b)
        g.output(IN3, c); g.output(IN4, d)
        self._pwm_left.ChangeDutyCycle(left)
        self._pwm_right.ChangeDutyCycle(right)

    def forward(self) -> None:
        self._drive(1, 0, 1, 0, self.speed, self.speed)

    def backward(self) -> None:
        self._drive(0, 1, 0, 1, self.speed, self.speed)

    def left_turn(self) -> None:
        self._drive(0, 1, 1, 0, self.speed * 0.6, self.speed)

    def right_turn(self) -> None:
        self._drive(1, 0, 0, 1, self.speed, self.speed * 0.6)

    # -- relay (float convention preserved from V0) ------------------------
    def relay_on(self) -> None:
        g = self.gpio
        g.setup(RELAY, g.OUT)
        g.output(RELAY, g.LOW)
        self._relay_asserted = True

    def relay_off(self) -> None:
        g = self.gpio
        g.setup(RELAY, g.IN)
        self._relay_asserted = False

    @property
    def relay_asserted(self) -> bool:
        return self._relay_asserted

    # -- ultrasonic (unchanged from V0; D2 deliberately not fixed here) -----
    def get_distance(self, timeout_s: float = 0.02) -> float | None:
        g = self.gpio
        g.output(TRIG, False)
        time.sleep(0.0002)
        g.output(TRIG, True)
        time.sleep(0.00001)
        g.output(TRIG, False)

        start = time.time()
        while g.input(ECHO) == 0:
            if time.time() - start > timeout_s:
                return None
        pulse_start = time.time()
        while g.input(ECHO) == 1:
            if time.time() - pulse_start > timeout_s:
                return None
        return (time.time() - pulse_start) * 17150


def remaining_defects() -> dict[str, str]:
    """Audit defects deliberately NOT fixed in V1, and why.

    V1 changes one thing per defect so the before/after comparison means
    something. These are listed rather than silently carried.
    """
    return {
        "D2": (
            "three stacked fail-open behaviours. NOT fixed: it changes the "
            "safety semantics, not just the implementation, so it belongs to a "
            "V2 with its own before/after. Pass spray_armed_at_start=False to "
            "run_session() for the first of the three."
        ),
        "D4": "keyboard input is coupled to the imshow windows; removing them removes control",
        "D6": "relay pin reconfigured every frame rather than on transition",
        "D7": "ultrasonic polled every frame even in manual mode",
        "D8": "bang-bang steering; needs proportional control",
        "D9": "c > 1500 is inherited, uncalibrated prototype logic - never an agronomic threshold",
        "D10": "loose HSV bounds admit desaturated non-plant surfaces",
    }
