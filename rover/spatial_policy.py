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
  * so is a stuck sensor's frame - static stripes, varying between rows but not
    within them - a CAMERA FAULT. Observed on the rover 2026-10-21: high contrast,
    so not "blank", and the tissue test read the stripes as 86% leaf;
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
#: A stuck sensor (seen on the rover 2026-10-21) delivers static stripes: frames
#: that vary strongly BETWEEN rows and not at all WITHIN them. Measured: stuck
#: frames within 0.09-0.14 and between 71-83; real rover frames within ~38; of the
#: 1,800 dry-run frames none has both (a blown-out frame has neither, and remains an
#: exposure matter, not a camera fault).
STRIPE_WITHIN_MAX = 0.5
STRIPE_BETWEEN_MIN = 10.0


def row_structure(bgr: np.ndarray) -> tuple[float, float]:
    """(grey variation along rows, pooled over all rows; variation between row means)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = g.mean(axis=1, keepdims=True)
    return float((g - m).std()), float(m.std())


def is_stuck_stripes(bgr: np.ndarray) -> bool:
    within, between = row_structure(bgr)
    return within < STRIPE_WITHIN_MAX and between > STRIPE_BETWEEN_MIN


def foliage_mask(hsv: np.ndarray) -> np.ndarray:
    """Plant tissue of any colour on an exposure-normalised V channel."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2].astype(np.float32)
    gain = min(MAX_GAIN, 128.0 / max(1.0, float(np.median(v))))
    v = np.minimum(255.0, v * gain)
    return ((h <= 95) | (h >= 170)) & (s >= 40) & (v >= 25)


def spatial_decision(bgr: np.ndarray) -> dict:
    """{'zones': per-zone tissue fractions, 'tissue_frac': whole-frame fraction,
    'camera_fault': blank, or a stuck sensor's stripes, 'deferred': fault or too dark,
    'fire': spray this frame}."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    fault = float(bgr.std()) < 0.5 or is_stuck_stripes(bgr)
    deferred = fault or float(np.median(hsv[..., 2])) < MIN_EXPOSURE
    veg = foliage_mask(hsv)
    zones = [float(b.mean()) for b in np.array_split(veg, N_ZONES, axis=1)]
    return {"zones": zones, "tissue_frac": float(veg.mean()), "camera_fault": bool(fault),
            "deferred": bool(deferred),
            "fire": bool((not deferred) and any(z >= ZONE_MIN for z in zones))}
