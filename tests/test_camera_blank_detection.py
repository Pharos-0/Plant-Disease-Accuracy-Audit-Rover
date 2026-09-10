"""The camera probe must fail on a blind camera, not pass it.

Regression test for a measured defect: on 2026-09-10 the development machine's
camera streamed 60 frames at 640x480 in which every pixel was the constant value
2 (per-frame std 0.0000), and the probe reported HEALTHY because its rule was
``frames > 0``. For a rover whose safety rule is "sensor failure -> stop, spray
off, require attention", a blind sensor reporting health is the wrong failure
direction.

These tests run without hardware.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rover" / "hardware_tests"))

from test_components import UNIFORM_FRAME_STD, frame_is_blank  # noqa: E402


def test_the_exact_observed_blind_frame_is_rejected():
    """640x480, every pixel = 2 - the frame actually produced by the hardware."""
    observed = np.full((480, 640, 3), 2, dtype=np.uint8)
    assert frame_is_blank(observed) is True


def test_pure_black_and_pure_white_are_rejected():
    for value in (0, 255):
        assert frame_is_blank(np.full((64, 64, 3), value, dtype=np.uint8)) is True


def test_a_dark_but_real_frame_is_accepted():
    """A capped lens in a dark room still produces sensor noise, and that is a
    working camera. Only a *uniform* frame indicates no sensor data."""
    rng = np.random.default_rng(0)
    dark_with_noise = rng.integers(0, 12, size=(480, 640, 3)).astype(np.uint8)
    assert frame_is_blank(dark_with_noise) is False


def test_a_normal_scene_is_accepted():
    rng = np.random.default_rng(1)
    scene = rng.integers(0, 256, size=(240, 320, 3)).astype(np.uint8)
    assert frame_is_blank(scene) is False


def test_threshold_is_tight_enough_to_catch_near_uniform_frames():
    """A frame with a single stuck pixel is still effectively blind."""
    almost = np.full((480, 640, 3), 7, dtype=np.uint8)
    almost[0, 0] = 255
    assert frame_is_blank(almost) is True


def test_threshold_constant_is_small_and_positive():
    assert 0 < UNIFORM_FRAME_STD < 5


@pytest.mark.parametrize("std_target", [0.0, 0.1, 0.4])
def test_low_variation_frames_are_blank(std_target):
    rng = np.random.default_rng(2)
    base = np.full((128, 128, 3), 30.0)
    frame = base + rng.normal(0, std_target, base.shape)
    assert frame_is_blank(frame) is True


def test_frame_saving_is_opt_in_by_signature():
    """The probe must not write a JPEG unless explicitly asked.

    Checked on the signature so the test needs no camera: writing a photo of
    whatever the camera sees on every test run is a privacy problem once the
    camera works.
    """
    import inspect

    from test_components import test_camera

    sig = inspect.signature(test_camera)
    assert "save_frame" in sig.parameters
    assert sig.parameters["save_frame"].default is False
