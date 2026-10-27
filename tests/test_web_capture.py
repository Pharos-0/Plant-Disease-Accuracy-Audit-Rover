"""The browser capture must record exactly what the window capture records.

Run end to end against a stand-in rover camera (a local server with /raw), through
the real HTTP page endpoints: label, condition, save, next leaf, finish. Also pins
the two guards the window left to the operator: no save without a true label, and
no save of a stuck-sensor frame.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

PV = Path(r"C:/ds/plantvillage/raw/color/Tomato___healthy")
STUCK = ROOT / "tests" / "fixtures" / "rover_stuck_sensor_20261021.jpg"
#: trained weights the demo loads; too large for git, so a fresh clone lacks them
WEIGHTS = ROOT / "results" / "raw" / "V9" / "mobilenetv3_large_seed0.pt"


def _jpeg(bgr):
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from web_capture import make_server
    leaf = cv2.resize(cv2.imread(str(sorted(PV.glob("*.JPG"))[0])), (640, 480))
    serving = {"jpg": _jpeg(leaf)}

    class Rover(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.split("?")[0] != "/raw":
                self.send_error(404)
                return
            b = serving["jpg"]
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    rover = ThreadingHTTPServer(("127.0.0.1", 0), Rover)
    threading.Thread(target=rover.serve_forever, daemon=True).start()
    out = tmp_path_factory.mktemp("web_session")
    srv, cap = make_server(f"127.0.0.1:{rover.server_address[1]}", out,
                           "visual only (nursery stock)", "127.0.0.1", 0)
    threading.Thread(target=cap.run, daemon=True).start()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    end = time.time() + 30
    while cap.frames_seen == 0 and time.time() < end:
        time.sleep(0.1)
    yield base, cap, out, serving
    cap.stop.set()
    srv.shutdown()
    rover.shutdown()


def _act(base, what, value=""):
    req = urllib.request.Request(base + "/act", method="POST",
                                 data=json.dumps({"what": what, "value": value}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


pytestmark = pytest.mark.skipif(not (PV.exists() and WEIGHTS.exists()),
                                reason="PlantVillage photographs or trained weights absent")


def test_page_and_frame_are_served(setup):
    base, cap, out, _ = setup
    assert cap.frames_seen > 0
    assert b"<canvas id=cam" in urllib.request.urlopen(base + "/", timeout=5).read()
    assert urllib.request.urlopen(base + "/frame.jpg?t=1", timeout=5).read()[:2] == b"\xff\xd8"


def test_save_is_refused_without_a_true_label(setup):
    base, cap, out, _ = setup
    s = _act(base, "save")
    assert s["saved"] == 0 and "label" in s["message"]


def test_saved_frames_carry_the_same_record_as_the_window_capture(setup):
    base, cap, out, _ = setup
    _act(base, "truth", "healthy")
    _act(base, "cond", "low_light")
    for _ in range(2):
        s = _act(base, "save")
    assert s["saved"] == 2 and s["this_cell"] == 2
    s = _act(base, "next_leaf")
    assert s["leaf"] == "P01L02" and s["this_cell"] == 0
    recs = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    r = recs[-1]
    assert (r["plant_id"], r["leaf_id"], r["condition"], r["truth"]) == \
        ("P01", "P01L01", "low_light", "healthy")
    assert r["diagnosis_source"] == "visual only (nursery stock)"
    assert r["source"].endswith("(raw) via web capture")
    for k in ("pred", "msp", "abstain", "spray", "spray_spatial", "deferred",
              "camera_fault", "canopy", "nozzles", "correct", "tau", "model"):
        assert k in r, k                       # the fields the analysis reads
    assert (out / r["frame_file"]).is_file()


def test_one_leaf_cannot_be_saved_under_two_diagnoses(setup):
    """Seen in the first real pilot: P01L01 saved as healthy, then as late blight."""
    base, cap, out, _ = setup
    _act(base, "next_leaf")
    _act(base, "truth", "healthy")
    assert _act(base, "save")["this_cell"] == 1
    n = cap.log.n
    _act(base, "truth", "late_blight")
    s = _act(base, "save")
    assert s["saved"] == n and "one leaf has one diagnosis" in s["message"]
    _act(base, "truth", "healthy")                       # set back: saving resumes
    assert _act(base, "save")["saved"] == n + 1


def test_low_light_that_looks_bright_is_warned_not_refused(setup):
    base, cap, out, _ = setup
    _act(base, "next_leaf")
    _act(base, "truth", "healthy")
    _act(base, "cond", "low_light")
    n = cap.log.n
    s = _act(base, "save")                               # the stand-in frame is well lit
    assert s["saved"] == n + 1 and "brightly lit" in s["message"]


def test_next_plant_asks_for_confirmation_in_the_page():
    import web_capture
    assert "Start a NEW plant?" in web_capture.PAGE and "go('next_plant')" in web_capture.PAGE


def test_a_stuck_camera_frame_cannot_be_saved(setup):
    base, cap, out, serving = setup
    serving["jpg"] = STUCK.read_bytes()
    end = time.time() + 15
    while not (cap.d and cap.d.camera_fault) and time.time() < end:
        time.sleep(0.1)
    n = cap.log.n
    s = _act(base, "save")
    assert s["camera_fault"] is True and s["saved"] == n and "camera fault" in s["message"]
