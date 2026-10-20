"""The rover's on-board AUTO must run exactly the spatial-only policy that is analysed.

demo/live_sprayer.py is the implementation the dry run and the camera protocol
analyse; rover/spatial_policy.py is what the Pi runs in AUTO. If they disagreed on
any frame, a result measured with one would not describe the robot. So they are
compared frame for frame: on real tomato-leaf photographs (healthy, early blight,
late blight), dimmed past the deferral threshold, a blank frame, noise, and the case
the old whole-frame AUTO test got wrong - a plant filling only one nozzle zone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "rover"))

import live_sprayer as ls  # noqa: E402
import spatial_policy as sp  # noqa: E402

PV = Path(r"C:/ds/plantvillage/raw/color")


def test_constants_match_the_analysed_demo():
    assert sp.MAX_GAIN == ls.MAX_GAIN and sp.MIN_EXPOSURE == ls.MIN_EXPOSURE
    assert sp.N_ZONES == ls.N_NOZZLES


def test_foliage_masks_are_identical_on_random_colours():
    rng = np.random.default_rng(0)
    for _ in range(50):
        hsv = cv2.cvtColor(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8),
                           cv2.COLOR_BGR2HSV)
        assert np.array_equal(sp.foliage_mask(hsv), ls.foliage_mask(hsv))


def test_one_zone_plant_fires_although_the_whole_frame_is_under_threshold():
    """The defect of the first AUTO: whole-frame tissue >= 15% missed a plant that
    fills one nozzle zone (12.5% of the frame). The policy sprays that zone."""
    f = np.full((480, 640, 3), (128, 128, 128), np.uint8)       # grey bench
    f[:, :80] = (40, 140, 50)                                   # leaf: half of zone 0
    d = sp.spatial_decision(f)
    assert d["tissue_frac"] == pytest.approx(0.125)
    assert d["zones"][0] == pytest.approx(0.5) and d["fire"] is True


def test_too_dark_and_blank_frames_are_deferred_not_sprayed():
    leaf = np.full((480, 640, 3), (40, 140, 50), np.float32)
    dark = (leaf * 0.1).astype(np.uint8)                         # median V ~14 < 32
    assert sp.spatial_decision(dark) == {**sp.spatial_decision(dark),
                                         "deferred": True, "fire": False}
    blank = np.full((480, 640, 3), 90, np.uint8)
    assert sp.spatial_decision(blank)["deferred"] is True


def _frames():
    out = [("blank", np.full((480, 640, 3), 90, np.uint8)),
           ("noise", np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8))]
    for folder in ("Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight"):
        for p in sorted((PV / folder).glob("*.JPG"))[:4]:
            bgr = cv2.resize(cv2.imread(str(p)), (640, 480))
            for g in (1.0, 0.35, 0.12):                          # normal, dim, too dark
                out.append((f"{folder}/{p.name}@{g}",
                            (bgr.astype(np.float32) * g).clip(0, 255).astype(np.uint8)))
    return out


@pytest.fixture(scope="module")
def pipe():
    return ls.SprayPipeline(device="cpu")


@pytest.mark.skipif(not PV.exists(), reason="PlantVillage photographs not on this machine")
def test_rover_decision_equals_the_demo_decision_frame_for_frame(pipe):
    frames = _frames()
    deferred_seen = fired_seen = False
    for name, bgr in frames:
        demo = pipe.decide(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        rover = sp.spatial_decision(bgr)
        assert rover["zones"] == pytest.approx(demo.canopy, abs=1e-12), name
        assert rover["deferred"] == demo.deferred, name
        assert rover["fire"] == demo.spray_spatial, name
        deferred_seen |= rover["deferred"]
        fired_seen |= rover["fire"]
    assert deferred_seen and fired_seen        # both branches actually exercised
