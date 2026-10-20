"""Frames from the rover's camera, for the laptop-side tools.

The rover's camera servers (rover/pi_rover_control.py, rover/pi_camera_stream.py)
annotate every frame on /snapshot and /stream with a status banner across the top.
That banner must not reach the classifier or the leaf-tissue measurement: it is
black, which drags the brightness median down, and its text can be green, which
the tissue mask counts as plant. So frames are taken, in order of preference, from

  /raw       the un-annotated frame, served by the updated rover servers; or
  /snapshot  with the banner cropped off, for servers not yet updated.

One fresh frame is fetched per read() rather than decoding the MJPEG stream, so the
frame judged is the frame now, not one queued in a decoder buffer. The endpoint in
use is reported in `.mode`, and belongs in every log record.

    cam = RoverCamera("192.168.137.167:8080")
    ok, bgr = cam.read()          # same shape as cv2.VideoCapture.read()
"""
from __future__ import annotations

import urllib.error
import urllib.request

import cv2
import numpy as np

#: Rows the servers' banner covers: cv2.rectangle((0, 0), (w, 24)) fills rows 0-24.
BANNER_ROWS = 25
#: Rows cropped from an annotated frame. JPEG compression smears the banner's lower
#: edge through the rest of its 8-row block (measured: rows 25-31 deviate by up to
#: 35 grey levels, row 32 onward by at most 3), so the whole block row goes.
CROP_ROWS = 32


def base_url(url: str) -> str:
    """'192.168.137.167:8080', 'http://host:8080/stream' etc. -> 'http://host:8080'."""
    u = url.strip().rstrip("/")
    if "://" not in u:
        u = "http://" + u
    for suffix in ("/snapshot", "/stream", "/raw"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
    return u


class RoverCamera:
    """A cv2.VideoCapture-shaped reader (read / isOpened / release) for the rover."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.base = base_url(url)
        self.timeout = timeout
        #: "raw" or "snapshot-cropped"; decided on the first successful contact.
        self.mode: str | None = None

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return r.read()

    def _probe(self) -> None:
        """Prefer /raw. A 404 means an older server: fall back to /snapshot. A 503
        means /raw exists but has no frame yet, which is still /raw."""
        try:
            self._get("/raw")
            self.mode = "raw"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                self.mode = "snapshot-cropped"
            elif e.code == 503:
                self.mode = "raw"
            else:
                raise

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            if self.mode is None:
                self._probe()
            data = self._get("/raw" if self.mode == "raw" else "/snapshot")
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None and self.mode == "snapshot-cropped":
                frame = frame[CROP_ROWS:].copy()
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            # HTTPError is a URLError: a 503 (no frame yet) or a dropped hotspot
            # packet is a missed frame, for the caller to retry, never a crash.
            return False, None
        return frame is not None, frame

    def isOpened(self) -> bool:  # noqa: N802 - cv2.VideoCapture's name
        return self.read()[0]

    def release(self) -> None:
        pass
