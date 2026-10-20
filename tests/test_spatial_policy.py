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
#: A real frame from the rover's camera on 2026-10-21, with its sensor stuck: static,
#: perfectly uniform-row stripes - high contrast, so not "blank".
STUCK = ROOT / "tests" / "fixtures" / "rover_stuck_sensor_20261021.jpg"
#: Part of a real rover frame from 2026-10-21 with the camera's blue channel lost after a
#: reopen: a white panel reads yellow-green, and the tissue test calls most of it leaf.
DEAD_BLUE = ROOT / "tests" / "fixtures" / "rover_dead_blue_channel_20261021.jpg"


def test_constants_match_the_analysed_demo():
    assert sp.MAX_GAIN == ls.MAX_GAIN and sp.MIN_EXPOSURE == ls.MIN_EXPOSURE
    assert sp.N_ZONES == ls.N_NOZZLES
    assert (sp.STRIPE_WITHIN_MAX, sp.STRIPE_BETWEEN_MIN) == (ls.STRIPE_WITHIN_MAX,
                                                          ls.STRIPE_BETWEEN_MIN)
    assert (sp.DEAD_CHANNEL_BRIGHT, sp.DEAD_CHANNEL_RATIO) == (ls.DEAD_CHANNEL_BRIGHT,
                                                            ls.DEAD_CHANNEL_RATIO)


def test_the_real_stuck_sensor_frame_is_a_camera_fault_not_a_leaf():
    """The failure seen on the robot: the tissue test alone reads these stripes as
    mostly leaf, so without the fault check AUTO would spray on a stuck camera."""
    f = cv2.imread(str(STUCK))
    assert f.std() > 50                                  # passes the old blank check
    d = sp.spatial_decision(f)
    assert d["tissue_frac"] > 0.5                        # what fooled the tissue test
    assert d["camera_fault"] is True and d["deferred"] is True and d["fire"] is False


def test_a_leaf_on_a_blown_out_background_is_not_a_camera_fault():
    """The first version of the check used the MEDIAN row and called 90 harsh-light
    dry-run frames - real leaves on saturated white - camera faults. Stripes need
    variation between rows AND none within them; this frame has rows of leaf."""
    f = np.full((256, 256, 3), 255, np.uint8)
    f[90:170, 40:215] = (40, 160, 60)                    # leaf across 31% of the rows
    assert sp.spatial_decision(f)["camera_fault"] is False


DRY = Path(r"C:/ds/camera_dryrun/manifest.csv")


@pytest.mark.skipif(not DRY.exists(), reason="dry-run images not on this machine")
def test_no_recorded_dry_run_frame_is_a_camera_fault():
    """The fault check was added after the dry run was analysed, so it must not
    change a single recorded decision: none of the 1,800 frames may trip it."""
    import csv
    rows = list(csv.DictReader(DRY.open(encoding="utf-8")))
    flagged = [r["path"] for r in rows
               if sp.is_stuck_stripes(im := cv2.imread(r["path"])) or sp.is_dead_channel(im)]
    assert len(rows) == 1800 and flagged == []


PILOT = sorted(ROOT.glob("camera_sessions/20261021_pilot_P01*/frames/*.jpg"))


@pytest.mark.skipif(not PILOT, reason="pilot frames not on this machine (they are not in git)")
def test_no_pilot_frame_is_a_camera_fault():
    """The dead-channel check was added after the pilot: it must change none of its 32."""
    assert len(PILOT) == 32
    assert [p.name for p in PILOT if sp.spatial_decision(cv2.imread(str(p)))["camera_fault"]] == []


def test_a_camera_that_lost_its_blue_channel_is_a_camera_fault_not_a_leaf():
    """The second failure seen on the robot: everything yellow-green, read as leaf."""
    f = cv2.imread(str(DEAD_BLUE))
    d = sp.spatial_decision(f)
    assert d["tissue_frac"] > 0.5                        # what the tissue test alone sees
    assert d["camera_fault"] is True and d["deferred"] is True and d["fire"] is False


def test_a_dark_frame_is_too_dark_not_a_dead_channel():
    dark = (np.full((480, 640, 3), (60, 140, 90), np.float32) * 0.2).astype(np.uint8)
    assert sp.is_dead_channel(dark) is False             # every channel is low: darkness


def test_a_real_but_dim_scene_is_not_a_camera_fault():
    rng = np.random.default_rng(3)
    leaf = np.full((480, 640, 3), (40, 140, 50), np.float32)
    leaf[:, ::7] *= 0.6                                  # some texture across the row
    dim = ((leaf + rng.normal(0, 3, leaf.shape)) * 0.4).clip(0, 255).astype(np.uint8)
    d = sp.spatial_decision(dim)
    assert d["camera_fault"] is False and d["fire"] is True


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
           ("noise", np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)),
           ("stuck sensor (real, 2026-10-21)", cv2.imread(str(STUCK))),
           ("dead blue channel (real, 2026-10-21)", cv2.imread(str(DEAD_BLUE)))]
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
        assert rover["camera_fault"] == demo.camera_fault, name
        deferred_seen |= rover["deferred"]
        fired_seen |= rover["fire"]
    assert deferred_seen and fired_seen        # both branches actually exercised
