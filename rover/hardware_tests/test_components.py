"""Per-component isolation tests for the rover (MASTER_PROMPT section 9).

Run each subsystem **on its own, before any ML integration**. A failing relay or a
dead camera must be found here, not inside a session-gating experiment.

Run on the Raspberry Pi:

    python3 rover/hardware_tests/test_components.py --list
    python3 rover/hardware_tests/test_components.py --test camera
    python3 rover/hardware_tests/test_components.py --test ultrasonic
    python3 rover/hardware_tests/test_components.py --test relay      # tank EMPTY
    python3 rover/hardware_tests/test_components.py --test motors     # wheels OFF ground
    python3 rover/hardware_tests/test_components.py --all --json report.json

Every test prints a RESULT line and records evidence, so
``docs/rover_hardware_test_report.md`` can be assembled from output rather than
from memory. A component that cannot be verified is recorded **unverified** -
never assumed working.

SAFETY
------
* ``motors``  - put the chassis on blocks so the wheels are off the ground.
* ``relay``   - run with the tank empty or the pump outlet disconnected.
* The ultrasonic test includes an **unplug** step, because defect D2 means a
  failed sensor silently disables the obstacle stop.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

# Pin map from docs/rover_gpio_map.md - derived from code, NOT measured.
IN1, IN2, IN3, IN4 = 17, 27, 22, 23
ENA, ENB = 18, 19
RELAY = 24
TRIG, ECHO = 5, 6

PWM_HZ = 1000


def _result(name: str, ok: bool | None, detail: str, evidence: dict | None = None) -> dict:
    state = {True: "PASS", False: "FAIL", None: "UNVERIFIED"}[ok]
    print(f"RESULT {name}: {state} - {detail}")
    return {
        "component": name,
        "status": state,
        "detail": detail,
        "evidence": evidence or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _gpio():
    try:
        import RPi.GPIO as GPIO  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"RPi.GPIO unavailable ({exc}); are you on the Pi?") from None
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    return GPIO


# ---------------------------------------------------------------- platform


def test_platform() -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        with open("/proc/device-tree/model", "rb") as fh:
            info["pi_model"] = fh.read().decode(errors="replace").strip("\x00").strip()
    except Exception:  # noqa: BLE001
        info["pi_model"] = "unknown"
    try:
        info["os_release"] = subprocess.run(
            ["cat", "/etc/os-release"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    ok = info["pi_model"] != "unknown"
    return _result("platform", ok or None, info["pi_model"], info)


# ---------------------------------------------------------------- camera

#: A frame whose spatial standard deviation is below this carries no image. A
#: real sensor always produces some noise; measured blind-camera value: 0.0000.
UNIFORM_FRAME_STD = 0.5


def frame_is_blank(frame, threshold: float = UNIFORM_FRAME_STD) -> bool:
    """True when a frame has no spatial variation, i.e. carries no image.

    Separated from ``test_camera`` so the rule can be tested without hardware.
    """
    import numpy as np

    return bool(np.asarray(frame).std() < threshold)


def test_camera(index: int = 0, warmup: int = 5, save_frame: bool = False) -> dict:
    """Open the camera, grab frames, and check they contain an actual image.

    Two behaviours changed after a measured failure on 2026-09-10: the camera on
    the development machine streamed 60 frames at 640x480 in which **every pixel
    was the constant value 2** - per-frame standard deviation exactly 0.0000. The
    old success rule was ``frames > 0``, so a completely blind camera reported
    HEALTHY. For a rover whose safety rule is "sensor failure -> stop, spray off,
    require attention", that is the wrong way to fail.

    * A frame with no spatial variation is now a FAILURE, not a pass. A real
      sensor - even capped, even in the dark - produces noise; a perfectly
      uniform frame means no sensor data is arriving.
    * ``save_frame`` is **opt-in**. Previously every run wrote a JPEG of whatever
      the camera saw into the repository root, which is both clutter and, once
      the camera works, a privacy problem.
    """
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        return _result("camera", None, f"OpenCV unavailable: {exc}")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return _result("camera", False, f"VideoCapture({index}) would not open")

    for _ in range(warmup):
        cap.read()

    t0, frames, last = time.time(), 0, None
    while time.time() - t0 < 3.0:
        ok, frame = cap.read()
        if not ok:
            break
        last, frames = frame, frames + 1

    evidence: dict = {"frames_captured": frames}
    blank = None
    if last is not None:
        h, w = last.shape[:2]
        spatial_std = float(last.std())
        blank = spatial_std < UNIFORM_FRAME_STD
        evidence.update(
            {
                "width": int(w),
                "height": int(h),
                "measured_fps": round(frames / 3.0, 2),
                "mean_brightness": round(float(last.mean()), 1),
                "frame_std": round(spatial_std, 4),
                "uniform_frame": blank,
            }
        )
        if save_frame:
            path = f"rover_camera_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(path, last)
            evidence["sample_frame"] = path
            print(f"  saved sample frame -> {path}")
    cap.release()

    if frames == 0:
        return _result("camera", False, "camera opened but delivered no frames", evidence)
    if blank:
        return _result(
            "camera", False,
            f"{frames} frames delivered but every pixel is identical "
            f"(std={evidence['frame_std']}) - the camera is blind, not dark. "
            "Check the privacy shutter and the OS camera permission.",
            evidence,
        )
    return _result(
        "camera", True,
        f"{frames} frames in 3 s at {evidence.get('width')}x{evidence.get('height')}, "
        f"frame std {evidence['frame_std']}",
        evidence,
    )


# ---------------------------------------------------------------- ultrasonic


def _distance_cm(GPIO, timeout_s: float = 0.02) -> float | None:
    GPIO.output(TRIG, False)
    time.sleep(0.002)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    t0 = time.time()
    while GPIO.input(ECHO) == 0:
        if time.time() - t0 > timeout_s:
            return None
    start = time.time()
    while GPIO.input(ECHO) == 1:
        if time.time() - start > timeout_s:
            return None
    return round((time.time() - start) * 17150, 2)


def test_ultrasonic(n: int = 15) -> dict:
    try:
        GPIO = _gpio()
    except RuntimeError as exc:
        return _result("ultrasonic", None, str(exc))

    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    readings = [_distance_cm(GPIO) for _ in range(n)]
    time.sleep(0.06)
    valid = [r for r in readings if r is not None]

    evidence = {
        "readings": readings,
        "valid": len(valid),
        "total": n,
        "median_cm": sorted(valid)[len(valid) // 2] if valid else None,
    }
    print("  ACTION REQUIRED: unplug the HC-SR04 and re-run this test.")
    print("  Defect D2: a failed sensor returns None, and `if dist and dist < 20`")
    print("  treats None as falsy - the obstacle stop silently disappears while the")
    print("  rover keeps driving and spraying. Confirm that behaviour explicitly.")

    ok = len(valid) >= n // 2
    return _result(
        "ultrasonic", ok, f"{len(valid)}/{n} valid readings, median {evidence['median_cm']} cm",
        evidence,
    )


# ---------------------------------------------------------------- relay


def test_relay(dwell_s: float = 1.0) -> dict:
    """Assert and release the relay. RUN WITH THE TANK EMPTY."""
    try:
        GPIO = _gpio()
    except RuntimeError as exc:
        return _result("relay", None, str(exc))

    print("  SAFETY: tank must be empty or the pump outlet disconnected.")
    print("  Listen for the relay click and watch its indicator LED.")

    # released = pin floating as input (the delivered code's correct trick)
    GPIO.setup(RELAY, GPIO.IN)
    time.sleep(dwell_s)
    print("  relay RELEASED (pin as input) - LED should be off, pump silent")

    GPIO.setup(RELAY, GPIO.OUT)
    GPIO.output(RELAY, GPIO.LOW)  # active-low assert
    time.sleep(dwell_s)
    print("  relay ASSERTED (OUT + LOW) - LED should be on, pump should run")

    GPIO.setup(RELAY, GPIO.IN)
    print("  relay RELEASED again")

    answer = "unknown"
    if sys.stdin and sys.stdin.isatty():
        answer = input("  Did the relay click and the pump run? [y/N/skip] ").strip().lower()

    ok = True if answer.startswith("y") else (False if answer.startswith("n") else None)
    return _result(
        "relay",
        ok,
        "operator confirmed" if ok else ("operator reported failure" if ok is False
                                         else "no operator confirmation - unverified"),
        {"operator_answer": answer, "dwell_s": dwell_s},
    )


# ---------------------------------------------------------------- motors


def test_motors(duty: int = 40, dwell_s: float = 1.0) -> dict:
    """Drive each side briefly. PUT THE CHASSIS ON BLOCKS FIRST."""
    try:
        GPIO = _gpio()
    except RuntimeError as exc:
        return _result("motors", None, str(exc))

    print("  SAFETY: wheels must be off the ground.")
    for p in (IN1, IN2, IN3, IN4, ENA, ENB):
        GPIO.setup(p, GPIO.OUT)
    pwm_a, pwm_b = GPIO.PWM(ENA, PWM_HZ), GPIO.PWM(ENB, PWM_HZ)
    pwm_a.start(0)
    pwm_b.start(0)

    observed = []
    try:
        for label, pins in (
            ("left forward", (IN1, IN2)),
            ("left reverse", (IN2, IN1)),
            ("right forward", (IN3, IN4)),
            ("right reverse", (IN4, IN3)),
        ):
            hi, lo = pins
            GPIO.output(hi, True)
            GPIO.output(lo, False)
            (pwm_a if "left" in label else pwm_b).ChangeDutyCycle(duty)
            print(f"  {label} at {duty}% for {dwell_s}s")
            time.sleep(dwell_s)
            (pwm_a if "left" in label else pwm_b).ChangeDutyCycle(0)
            GPIO.output(hi, False)
            observed.append(label)
            time.sleep(0.3)
    finally:
        pwm_a.stop()
        pwm_b.stop()
        for p in (IN1, IN2, IN3, IN4):
            GPIO.output(p, False)

    answer = "unknown"
    if sys.stdin and sys.stdin.isatty():
        answer = input("  Did all four directions turn correctly? [y/N/skip] ").strip().lower()

    ok = True if answer.startswith("y") else (False if answer.startswith("n") else None)
    return _result(
        "motors", ok,
        f"drove {len(observed)} directions at {duty}% duty",
        {"sequences": observed, "duty": duty, "operator_answer": answer},
    )


TESTS = {
    "platform": test_platform,
    "camera": test_camera,
    "ultrasonic": test_ultrasonic,
    "relay": test_relay,
    "motors": test_motors,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", choices=sorted(TESTS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", help="write results to this JSON file")
    args = ap.parse_args()

    if args.list or not (args.test or args.all):
        print("available tests:")
        for k in sorted(TESTS):
            print(f"  {k}")
        print("\nRun one at a time. Read the SAFETY notes in the module docstring first.")
        return 0

    names = sorted(TESTS) if args.all else [args.test]
    results = []
    for name in names:
        print(f"\n===== {name} =====")
        try:
            results.append(TESTS[name]())
        except Exception as exc:  # noqa: BLE001
            results.append(_result(name, None, f"test raised {type(exc).__name__}: {exc}"))
        finally:
            try:
                import RPi.GPIO as GPIO  # type: ignore

                GPIO.cleanup()
            except Exception:  # noqa: BLE001
                pass

    print("\n===== summary =====")
    for r in results:
        print(f"  {r['component']:12s} {r['status']}")
    unverified = [r["component"] for r in results if r["status"] != "PASS"]
    if unverified:
        print(f"\nNOT verified working: {', '.join(unverified)}")
        print("Record these as UNVERIFIED in docs/rover_hardware_test_report.md.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
