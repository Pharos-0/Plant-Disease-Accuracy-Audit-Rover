"""Tests for the camera demo's decision logic.

The leaf-tissue test was revised twice during the dry run, each time because it
missed diseased leaves: green-only was blind to late-blight browning, and a fixed
brightness floor lost leaves in dim light. These tests pin both fixes, and the
cases that must still be excluded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

from live_sprayer import MAX_GAIN, MIN_EXPOSURE, SprayPipeline, foliage_mask  # noqa: E402


def patch(bgr, n=64):
    return cv2.cvtColor(np.full((n, n, 3), bgr, np.uint8), cv2.COLOR_BGR2HSV)


def frac(bgr):
    return float(foliage_mask(patch(bgr)).mean())


def test_green_leaf_is_tissue():
    assert frac((40, 140, 50)) == 1.0


def test_brown_late_blight_lesion_is_tissue():
    """The defect green-only had: a brown lesion is diseased leaf, not background."""
    assert frac((30, 60, 110)) == 1.0


def test_yellow_chlorosis_is_tissue():
    assert frac((30, 170, 190)) == 1.0


def test_blue_backdrop_is_not_tissue():
    assert frac((200, 120, 40)) == 0.0


def test_grey_white_and_black_backdrops_are_not_tissue():
    for c in ((128, 128, 128), (240, 240, 240), (5, 5, 5)):
        assert frac(c) == 0.0, c


def test_dim_leaf_is_still_tissue():
    """The second defect: a leaf at a third of normal brightness vanished."""
    leaf = np.full((64, 64, 3), (40, 140, 50), np.float32) * 0.3
    hsv = cv2.cvtColor(leaf.astype(np.uint8), cv2.COLOR_BGR2HSV)
    assert foliage_mask(hsv).mean() == 1.0


def test_deferral_threshold_matches_the_gain_cap():
    """The two thresholds must agree on what 'too dark' means."""
    assert MIN_EXPOSURE == 128.0 / MAX_GAIN


def test_abstention_threshold_meets_its_accuracy_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(0, 1, (2000, 3))
    logits[np.arange(2000), y] += rng.uniform(0, 4, 2000)   # confident ones mostly right
    tau = SprayPipeline._abstention_threshold(logits, y)
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    keep = p.max(1) >= tau
    assert keep.any() and (p.argmax(1)[keep] == y[keep]).mean() >= 0.99
