"""Tests for reading the rover's camera from the laptop without its status banner.

The rover servers stamp a banner on /snapshot. If it reached the classifier or the
tissue measurement it would bias both, so the reader must prefer the clean /raw
frame and, against an older server, crop the banner and its JPEG ringing away.
Run against real local HTTP servers - the same transport as the hotspot.
"""
from __future__ import annotations

import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "rover"))

from rover_camera import BANNER_ROWS, CROP_ROWS, RoverCamera, base_url  # noqa: E402

SCENE = (40, 160, 60)   # a leaf-green BGR


def _annotated_jpeg() -> tuple[bytes, bytes]:
    """(annotated, clean) JPEGs of one scene, the banner drawn exactly as the servers do."""
    clean = np.full((480, 640, 3), SCENE, np.uint8)
    ok, raw = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, 90])
    f = clean.copy()
    cv2.rectangle(f, (0, 0), (640, 24), (0, 0, 0), -1)
    cv2.putText(f, "AUTO  tissue  52%  dist 94cm  18:13:46", (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    ok, ann = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return ann.tobytes(), raw.tobytes()


def _serve(routes: dict):
    """A local server; routes maps path -> bytes, an int status, or a callable."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            r = routes.get(self.path, 404)
            r = r() if callable(r) else r
            if isinstance(r, int):
                self.send_error(r)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(r)))
            self.end_headers()
            self.wfile.write(r)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def server():
    made = []

    def start(routes):
        srv, addr = _serve(routes)
        made.append(srv)
        return addr
    yield start
    for s in made:
        s.shutdown()


def test_base_url_normalises_every_form_the_operator_might_type():
    for u in ("192.168.137.167:8080", "http://192.168.137.167:8080/",
              "http://192.168.137.167:8080/stream", "192.168.137.167:8080/snapshot",
              "http://192.168.137.167:8080/raw"):
        assert base_url(u) == "http://192.168.137.167:8080"


def test_crop_covers_the_banner_and_its_jpeg_ringing():
    assert CROP_ROWS >= BANNER_ROWS and CROP_ROWS % 8 == 0
    ann, _ = _annotated_jpeg()
    d = cv2.imdecode(np.frombuffer(ann, np.uint8), cv2.IMREAD_COLOR).astype(int)
    assert np.abs(d[CROP_ROWS:] - SCENE).max() <= 4           # clean below the crop
    assert np.abs(d[:BANNER_ROWS] - SCENE).max() > 100       # and the banner is real


def test_old_server_without_raw_falls_back_to_a_cropped_snapshot(server):
    ann, _ = _annotated_jpeg()
    cam = RoverCamera(server({"/snapshot": ann}))
    ok, frame = cam.read()
    assert ok and cam.mode == "snapshot-cropped"
    assert frame.shape == (480 - CROP_ROWS, 640, 3)
    assert np.abs(frame.astype(int) - SCENE).max() <= 4      # no banner pixel survives


def test_updated_server_serves_the_clean_raw_frame(server):
    ann, raw = _annotated_jpeg()
    cam = RoverCamera(server({"/snapshot": ann, "/raw": raw}))
    ok, frame = cam.read()
    assert ok and cam.mode == "raw"
    assert frame.shape == (480, 640, 3)                       # full frame, nothing cropped
    assert np.abs(frame.astype(int) - SCENE).max() <= 4


def test_raw_without_a_frame_yet_is_still_raw_and_then_recovers(server):
    _, raw = _annotated_jpeg()
    state = {"ready": False}
    cam = RoverCamera(server({"/raw": lambda: raw if state["ready"] else 503}))
    assert cam.read() == (False, None) and cam.mode == "raw"
    state["ready"] = True
    ok, frame = cam.read()
    assert ok and frame.shape == (480, 640, 3)


def test_unreachable_rover_is_a_missed_frame_not_a_crash():
    cam = RoverCamera("127.0.0.1:9", timeout=0.5)            # discard port: nothing listens
    assert cam.isOpened() is False
    assert cam.read() == (False, None)


def test_view_only_server_raw_route_serves_the_clean_frame():
    import pi_camera_stream as pcs
    ann, raw = _annotated_jpeg()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), pcs.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        pcs._latest, pcs._latest_raw = None, None
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/raw", timeout=5)
        assert e.value.code == 503
        pcs._latest, pcs._latest_raw = ann, raw
        assert urllib.request.urlopen(base + "/raw", timeout=5).read() == raw
        assert urllib.request.urlopen(base + "/snapshot", timeout=5).read() == ann
        cam = RoverCamera(base)
        assert cam.read()[0] and cam.mode == "raw"
    finally:
        srv.shutdown()
        pcs._latest = pcs._latest_raw = None


def test_control_server_raw_route_serves_the_clean_frame():
    pytest.importorskip("flask")        # installed on the Pi, not in the laptop venv
    import pi_rover_control as prc
    ann, raw = _annotated_jpeg()
    c = prc.app.test_client()
    prc._frame, prc._raw = None, None
    assert c.get("/raw").status_code == 503
    prc._frame, prc._raw = ann, raw
    assert c.get("/raw").data == raw and c.get("/snapshot").data == ann
    prc._frame = prc._raw = None


def test_live_sprayer_opens_a_webcam_index_or_the_rover(monkeypatch):
    import live_sprayer as ls

    class FakeCapture:
        def __init__(self, index):
            self.index = index
    monkeypatch.setattr(ls.cv2, "VideoCapture", FakeCapture)
    cap, source, misses = ls.open_camera("0")
    assert isinstance(cap, FakeCapture) and cap.index == 0
    assert source == "camera:0" and misses == 1               # unchanged webcam behaviour
    cap, source, misses = ls.open_camera("192.168.137.167:8080")
    assert isinstance(cap, RoverCamera) and source == "rover:http://192.168.137.167:8080"
    assert misses > 1                                         # a dropped packet is not the end
