"""The rover control server's own threads, run against a recording GPIO and a scripted camera.

rover/pi_rover_control.py drives real motors and the real spray relay. Its loops are
run here unmodified - the camera thread and the control thread - with
rover_v1.RecordingGPIO standing in for RPi.GPIO and a scripted camera standing in
for the USB camera. Flask is replaced by a minimal stand-in only so the module
imports; the route handlers are called directly. Like the digital twin, this
establishes properties of the control CODE, and nothing about the hardware.

Checked: AUTO sprays a plant that fills one nozzle zone (which the first, whole-frame
AUTO missed); AUTO never sprays a frame too dark to judge; AUTO and manual spray are
refused without the tank gate; the dead-man stops the motors; /raw is banner-free;
and when the control loop dies, the relay is released and the motors are off.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rover"))

from rover_v1 import RecordingGPIO  # noqa: E402

BENCH = np.full((480, 640, 3), (128, 128, 128), np.uint8)       # grey: no tissue
LEAF = (40, 140, 50)


def one_zone_plant():
    f = BENCH.copy()
    f[:, :80] = LEAF            # half of nozzle zone 0; 12.5% of the whole frame
    return f


def dark_plant():
    return (np.full((480, 640, 3), LEAF, np.float32) * 0.1).astype(np.uint8)


class _Stop(Exception):
    pass


def _fake_flask():
    m = types.ModuleType("flask")

    class Flask:
        def __init__(self, name):
            pass

        def _route(self, path):
            return lambda f: f
        get = post = route = _route

        def run(self, *a, **k):
            pass

    class Response:
        def __init__(self, data=None, mimetype=None):
            self.data, self.mimetype = data, mimetype

    class _Request:
        payload: dict = {}

        def get_json(self, force=False):
            return self.payload

    m.Flask, m.Response, m.request, m.jsonify = Flask, Response, _Request(), (lambda d: d)
    return m


@pytest.fixture(scope="module")
def rover():
    gpio = RecordingGPIO()
    rpi = types.ModuleType("RPi")
    rpi.GPIO = gpio
    fake = {"flask": _fake_flask(), "RPi": rpi, "RPi.GPIO": gpio}
    saved = {k: sys.modules.get(k) for k in fake}
    sys.modules.update(fake)

    spec = importlib.util.spec_from_file_location("prc_under_test",
                                                  ROOT / "rover" / "pi_rover_control.py")
    prc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prc)

    scene = {"frame": BENCH.copy()}
    stop = threading.Event()
    # The HC-SR04, simulated at the level the control loop reads it: a fresh
    # distance every 60 ms while "plugged in", nothing at all once unplugged.
    us = {"on": True, "cm": 100.0}

    def ultrasonic():
        while not stop.is_set():
            if us["on"]:
                with prc._lock:
                    prc._state["distance_cm"] = us["cm"]
                    prc._state["distance_at"] = time.time()
            time.sleep(0.06)

    class ScriptedCamera:
        def __init__(self, index):
            pass

        def isOpened(self):     # noqa: N802
            return True

        def read(self):
            return True, scene["frame"].copy()

    def sleep(s):
        if stop.is_set():
            raise _Stop
        time.sleep(s)

    real_capture = cv2.VideoCapture
    cv2.VideoCapture = ScriptedCamera
    prc.time = types.SimpleNamespace(time=time.time, strftime=time.strftime, sleep=sleep)
    hook = threading.excepthook
    threading.excepthook = lambda a: None if a.exc_type is _Stop else hook(a)
    threads = [threading.Thread(target=prc._camera_loop, args=(0,), daemon=True),
               threading.Thread(target=prc._control_loop, daemon=True),
               threading.Thread(target=ultrasonic, daemon=True)]
    for t in threads:
        t.start()

    def post(handler, payload=None):
        prc.request.payload = payload or {}
        return handler()

    def wait_until(pred, timeout=3.0):
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                return True
            time.sleep(0.02)
        return False

    def show(frame):
        scene["frame"] = frame
        # the camera thread must have judged this scene before anything acts on it
        sd = prc.spatial_decision(frame)
        assert wait_until(lambda: prc._state["deferred"] == sd["deferred"]
                          and prc._state["tissue_frac"] == pytest.approx(sd["tissue_frac"]))

    def relay_fired_since(i):
        return any(c == ("output", prc.RELAY, gpio.LOW) for c in gpio.calls[i:])

    def reset():
        post(prc.stop)
        post(prc.spray_enable, {"on": False})
        us.update(on=True, cm=100.0)
        assert wait_until(lambda: prc._state["relay_until"] == 0.0
                          and prc._state["sensor_lost"] is False)

    def duty_after(i):
        """The last duty cycle the control loop set on the left motor since call i."""
        d = [c[2] for c in gpio.calls[i:] if c[0] == "pwm_duty" and c[1] == prc.ENA]
        return d[-1] if d else None

    def hold(direction, seconds):
        """Hold a drive button: re-send every 150 ms, as the browser UI does."""
        end = time.time() + seconds
        while time.time() < end:
            post(prc.drive, {"dir": direction})
            time.sleep(0.15)

    assert wait_until(lambda: prc._raw is not None and prc._frame is not None)
    yield types.SimpleNamespace(prc=prc, gpio=gpio, post=post, show=show, reset=reset,
                                wait_until=wait_until, relay_fired_since=relay_fired_since,
                                duty_after=duty_after, hold=hold, us=us,
                                stop=stop, threads=threads)

    stop.set()
    for t in threads:
        t.join(timeout=3)
    cv2.VideoCapture = real_capture
    threading.excepthook = hook
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def test_auto_sprays_a_plant_that_fills_one_nozzle_zone(rover):
    r, prc = rover, rover.prc
    r.reset()
    r.show(one_zone_plant())
    assert prc._state["tissue_frac"] == pytest.approx(0.125)   # under the old 15% test
    assert prc._state["spatial_fire"] is True
    i = len(r.gpio.calls)
    r.post(prc.spray_enable, {"on": True})
    r.post(prc.auto, {"on": True})
    assert r.wait_until(lambda: r.relay_fired_since(i))
    assert prc._state["auto_fires"] >= 1


def test_auto_never_sprays_a_frame_too_dark_to_judge(rover):
    r, prc = rover, rover.prc
    r.reset()
    r.show(dark_plant())
    assert prc._state["deferred"] is True and prc._state["spatial_fire"] is False
    fires = prc._state["auto_fires"]
    i = len(r.gpio.calls)
    r.post(prc.spray_enable, {"on": True})
    r.post(prc.auto, {"on": True})
    assert prc._state["auto"] is True
    time.sleep(1.0)                                             # 20 control-loop ticks
    assert not r.relay_fired_since(i)
    assert prc._state["auto_fires"] == fires


def test_spray_is_refused_without_the_tank_gate(rover):
    r, prc = rover, rover.prc
    r.reset()
    r.show(one_zone_plant())
    i = len(r.gpio.calls)
    r.post(prc.auto, {"on": True})                              # tank box not ticked
    r.post(prc.spray)
    assert prc._state["auto"] is False
    time.sleep(0.6)
    assert not r.relay_fired_since(i)


def test_dead_man_stops_the_motors(rover):
    r, prc = rover, rover.prc
    r.reset()
    r.show(BENCH)
    r.post(prc.arm, {"armed": True})
    i = len(r.gpio.calls)
    r.post(prc.drive, {"dir": "forward"})
    assert r.wait_until(lambda: ("pwm_duty", prc.ENA, prc.DUTY) in r.gpio.calls[i:])
    j = len(r.gpio.calls)
    time.sleep(prc.HOLD_S + 0.3)                                # no re-send: link "lost"
    tail = [c for c in r.gpio.calls[j:] if c[0] == "pwm_duty" and c[1] == prc.ENA]
    assert tail and tail[-1][2] == 0
    assert all(r.gpio.pin_value[p] is False
               for p in (prc.IN1, prc.IN2, prc.IN3, prc.IN4))


def test_unplugged_ultrasonic_refuses_forward_but_allows_backing_out(rover):
    """The fault-injection test of the hardware report (section 1), on the control
    code: with the sensor gone, forward must stop - not carry on unprotected."""
    r, prc = rover, rover.prc
    r.reset()
    r.show(BENCH)
    r.post(prc.arm, {"armed": True})
    i = len(r.gpio.calls)
    r.hold("forward", 0.5)
    assert r.duty_after(i) == prc.DUTY                          # sensor fine: drives
    r.us["on"] = False                                          # unplug it
    assert r.wait_until(lambda: prc._state["sensor_lost"], timeout=prc.DIST_STALE_S + 1)
    assert prc.telemetry()["sensor_lost"] is True
    i = len(r.gpio.calls)
    r.hold("forward", 0.6)                                      # operator keeps pushing
    assert r.duty_after(i) == 0                                 # refused
    i = len(r.gpio.calls)
    r.hold("back", 0.5)
    assert r.duty_after(i) == prc.DUTY                          # can still back out
    r.us["on"] = True                                           # plug it back in
    assert r.wait_until(lambda: not prc._state["sensor_lost"])
    i = len(r.gpio.calls)
    r.hold("forward", 0.5)
    assert r.duty_after(i) == prc.DUTY                          # recovered
    r.post(prc.stop)


def test_obstacle_within_range_refuses_forward_only(rover):
    r, prc = rover, rover.prc
    r.reset()
    r.us["cm"] = prc.OBSTACLE_STOP_CM - 15                      # 10 cm ahead
    assert r.wait_until(lambda: prc._state["blocked"] and not prc._state["sensor_lost"])
    r.post(prc.arm, {"armed": True})
    i = len(r.gpio.calls)
    r.hold("forward", 0.5)
    assert r.duty_after(i) == 0
    i = len(r.gpio.calls)
    r.hold("left", 0.5)
    assert r.duty_after(i) == prc.DUTY                          # turning away is allowed
    r.post(prc.stop)


def test_raw_frame_is_clean_and_snapshot_is_annotated(rover):
    r, prc = rover, rover.prc
    r.show(BENCH)
    assert r.wait_until(lambda: True)
    raw = cv2.imdecode(np.frombuffer(prc._raw, np.uint8), cv2.IMREAD_COLOR).astype(int)
    ann = cv2.imdecode(np.frombuffer(prc._frame, np.uint8), cv2.IMREAD_COLOR).astype(int)
    assert np.abs(raw[:25] - 128).max() <= 4                    # no banner on /raw
    assert ann[:25].mean() < 40                                 # banner on /snapshot


def test_telemetry_reports_the_spatial_decision(rover):
    t = rover.prc.telemetry()
    for k in ("tissue_frac", "tissue_zones", "deferred", "spatial_fire", "auto_fires"):
        assert k in t
    assert len(t["tissue_zones"]) == 4


def test_a_dying_control_loop_releases_the_relay_and_stops_the_motors(rover):
    r, prc = rover, rover.prc
    r.reset()
    r.show(one_zone_plant())
    r.post(prc.spray_enable, {"on": True})
    r.post(prc.auto, {"on": True})
    r.post(prc.arm, {"armed": True})
    r.post(prc.drive, {"dir": "forward"})
    i = len(r.gpio.calls)
    assert r.wait_until(lambda: r.relay_fired_since(i))         # relay is ON right now
    r.stop.set()                                                # the loop dies mid-pulse
    for t in r.threads:
        t.join(timeout=3)
    assert not r.threads[1].is_alive()
    assert r.gpio.pin_mode[prc.RELAY] == r.gpio.IN              # released (float trick)
    assert all(r.gpio.pin_value[p] is False
               for p in (prc.IN1, prc.IN2, prc.IN3, prc.IN4))
