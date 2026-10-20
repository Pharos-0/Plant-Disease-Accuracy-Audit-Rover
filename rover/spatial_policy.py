"""The spatial-only spray decision, for the rover's on-board AUTO mode.

This is the policy the project recommends - spray where plant tissue is, and never
let a disease classifier withhold treatment - implemented exactly as the analysed
demo implements it (demo/live_sprayer.py; docs/camera_experiment_protocol.md,
sections 4 and 9, version 3):

  * leaf tissue of ANY plant colour (a green-only test is blind to late blight),
    judged on a brightness-normalised value channel, the gain capped at MAX_GAIN;
  * a frame whose median brightness is below 128 / MAX_GAIN, or a blank frame from
    a dead camera, is DEFERRED - too dark to judge: no spray, and flagged, never
    silently counted as a decision;
  * otherwise spray when any of the N_ZONES vertical nozzle zones holds at least
    ZONE_MIN tissue.

It depends only on OpenCV and NumPy, so it runs on the Pi and is testable anywhere;
tests/test_spatial_policy.py checks frame-for-frame agreement with the demo.
"""
from __future__ import annotations

import cv2
import numpy as np

MAX_GAIN = 4.0
MIN_EXPOSURE = 128.0 / MAX_GAIN
N_ZONES = 4
ZONE_MIN = 0.15


def foliage_mask(hsv: np.ndarray) -> np.ndarray:
    """Plant tissue of any colour on an exposure-normalised V channel."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2].astype(np.float32)
    gain = min(MAX_GAIN, 128.0 / max(1.0, float(np.median(v))))
    v = np.minimum(255.0, v * gain)
    return ((h <= 95) | (h >= 170)) & (s >= 40) & (v >= 25)


def spatial_decision(bgr: np.ndarray) -> dict:
    """{'zones': per-zone tissue fractions, 'tissue_frac': whole-frame fraction,
    'deferred': too dark or blank to judge, 'fire': spray this frame}."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blank = float(bgr.std()) < 0.5
    deferred = blank or float(np.median(hsv[..., 2])) < MIN_EXPOSURE
    veg = foliage_mask(hsv)
    zones = [float(b.mean()) for b in np.array_split(veg, N_ZONES, axis=1)]
    return {"zones": zones, "tissue_frac": float(veg.mean()), "deferred": bool(deferred),
            "fire": bool((not deferred) and any(z >= ZONE_MIN for z in zones))}
