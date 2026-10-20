"""Tests for capturing the section-16 artefacts, and for the harness's --save-frame flag.

The artefact is evidence, so the capture must refuse what would make it misleading:
a height that was not measured, and a frame from a dead camera.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "rover" / "hardware_tests")]

import capture_rover_artifact as cra  # noqa: E402


def _rover_serving(frame):
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    body = jpg.tobytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path != "/raw":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"127.0.0.1:{srv.server_address[1]}"


def _plant():
    rng = np.random.default_rng(0)
    f = np.full((480, 640, 3), (120, 120, 120), np.uint8)
    f[100:400, 150:500] = (40, 140, 50)
    return (f.astype(int) + rng.integers(-8, 9, f.shape)).clip(0, 255).astype(np.uint8)


def test_captures_a_clean_plant_frame_with_its_tape_height(tmp_path):
    srv, addr = _rover_serving(_plant())
    try:
        out = cra.capture(addr, 24.5, "tomato on the floor", tmp_path, plant_distance_cm=35)
    finally:
        srv.shutdown()
    meta = json.loads((out / "artifact.json").read_text())
    assert (out / "plant_frame.jpg").is_file()
    assert meta["camera_height_above_ground_cm"] == 24.5 and meta["lens_to_plant_cm"] == 35
    assert meta["height_source"].startswith("tape")
    assert meta["endpoint"] == "raw" and (meta["width"], meta["height"]) == (640, 480)
    assert meta["spatial_policy_would_spray"] is True and meta["tissue_frac"] > 0.3


@pytest.mark.parametrize("height", [0, -3])
def test_refuses_a_height_that_was_not_measured(tmp_path, height):
    with pytest.raises(SystemExit, match="tape"):
        cra.capture("127.0.0.1:9", height, "", tmp_path)
    with pytest.raises(SystemExit, match="tape"):
        cra.capture("127.0.0.1:9", 24.5, "", tmp_path, plant_distance_cm=height)


def test_refuses_a_dead_camera_and_saves_nothing(tmp_path):
    srv, addr = _rover_serving(np.full((480, 640, 3), 2, np.uint8))   # the 2026-09-10 blind camera
    try:
        with pytest.raises(SystemExit, match="uniform"):
            cra.capture(addr, 40, "", tmp_path)
    finally:
        srv.shutdown()
    assert not any(tmp_path.iterdir())


def test_refuses_a_stuck_camera_and_saves_nothing(tmp_path):
    stuck = cv2.imread(str(ROOT / "tests" / "fixtures" / "rover_stuck_sensor_20261021.jpg"))
    srv, addr = _rover_serving(stuck)
    try:
        with pytest.raises(SystemExit, match="stuck"):
            cra.capture(addr, 24.5, "", tmp_path)
    finally:
        srv.shutdown()
    assert not any(tmp_path.iterdir())


def test_harness_fails_a_stuck_camera_that_the_blank_check_passed():
    import test_components as tc
    stuck = cv2.imread(str(ROOT / "tests" / "fixtures" / "rover_stuck_sensor_20261021.jpg"))
    assert not tc.frame_is_blank(stuck)                  # why it slipped through before
    assert tc.frame_is_stuck_stripes(stuck)
    assert tc.frames_are_frozen(stuck, stuck.copy())
    real = _plant()
    noisy = (real.astype(int) + np.random.default_rng(1).integers(-3, 4, real.shape)
             ).clip(0, 255).astype(np.uint8)
    assert not tc.frame_is_stuck_stripes(real)
    assert not tc.frame_is_stuck_stripes(np.full((480, 640, 3), 255, np.uint8))  # blown out
    assert not tc.frames_are_frozen(real, noisy)          # sensor noise: a live camera


def test_harness_fails_a_camera_that_lost_a_colour_channel():
    import test_components as tc
    dead = cv2.imread(str(ROOT / "tests" / "fixtures" / "rover_dead_blue_channel_20261021.jpg"))
    assert not tc.frame_is_blank(dead) and not tc.frame_is_stuck_stripes(dead)
    assert tc.frame_has_dead_channel(dead)
    assert not tc.frame_has_dead_channel(_plant())


def test_harness_save_frame_flag_reaches_the_camera_test(monkeypatch):
    import test_components as tc
    seen = {}

    def fake_camera(**kw):
        seen.update(kw)
        return tc._result("camera", True, "fake")
    monkeypatch.setitem(tc.TESTS, "camera", fake_camera)
    monkeypatch.setattr(sys, "argv", ["test_components.py", "--test", "camera", "--save-frame"])
    tc.main()
    assert seen == {"save_frame": True}
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["test_components.py", "--test", "camera"])
    tc.main()
    assert seen == {}                       # opt-in, as the harness documents
