"""Leaf-crop proposal, tested offline against real field frames.

The study protocol requires the rover modules to be "offline-tested against
saved frames" before they go near hardware. Tomato-Village is real Rajasthan field
photography, so it stands in for rover frames until arm E imagery exists.

Tests that need those images skip cleanly when the dataset is absent, so the suite
still runs on a machine without `C:\\ds`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data import paths
from rover.leaf_proposal import (
    ProposalConfig,
    adaptive_vision,
    adjust_gamma,
    detect_green,
    green_coverage,
    propose_leaf_crops,
)

cv2 = pytest.importorskip("cv2")


def field_frames(limit: int = 12) -> list[Path]:
    root = paths.TOMATOVILLAGE_ROOT / "train" / "Late_blight"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() == ".jpg")[:limit]


# --------------------------------------------------------------------------
# gamma: the D1 defect and its fix
# --------------------------------------------------------------------------


def test_gamma_convention_matches_the_audit() -> None:
    """The audit measured the LUT numerically; reproduce those numbers.

    `docs/rover_code_audit.md` D1 states that with this convention the 0..255
    mean maps to 169.3 at gamma=2.0 and to 104.6 at gamma=0.7.
    """
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    assert adjust_gamma(ramp, 2.0).mean() == pytest.approx(169.3, abs=0.6)
    assert adjust_gamma(ramp, 0.7).mean() == pytest.approx(104.6, abs=0.6)


def test_gamma_direction() -> None:
    """gamma > 1 brightens, gamma < 1 darkens."""
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    assert adjust_gamma(img, 2.0).mean() > img.mean()
    assert adjust_gamma(img, 0.7).mean() < img.mean()


def test_d1_fix_moves_exposure_the_right_way() -> None:
    """The delivered code amplified the exposure problem in both branches.

    An over-exposed frame must come out darker and an under-exposed one
    brighter. Before the fix both got pushed further from the middle.
    """
    cfg = ProposalConfig()
    bright = np.full((64, 64, 3), 230, dtype=np.uint8)
    dark = np.full((64, 64, 3), 30, dtype=np.uint8)

    assert bright.mean() > cfg.bright_threshold
    assert dark.mean() < cfg.dark_threshold

    assert adaptive_vision(bright, cfg).mean() < bright.mean(), "over-exposed frame not darkened"
    assert adaptive_vision(dark, cfg).mean() > dark.mean(), "under-exposed frame not brightened"


# --------------------------------------------------------------------------
# mask and proposal behaviour on synthetic input
# --------------------------------------------------------------------------


def test_green_mask_selects_green_and_rejects_grey() -> None:
    green = np.zeros((128, 128, 3), dtype=np.uint8)
    green[:, :] = (40, 180, 60)  # BGR, a saturated leaf green
    grey = np.full((128, 128, 3), 128, dtype=np.uint8)

    assert green_coverage(detect_green(green)) > 0.9
    assert green_coverage(detect_green(grey)) < 0.05, (
        "D10: a desaturated grey surface must not read as vegetation"
    )


def test_proposal_finds_a_planted_blob() -> None:
    frame = np.full((240, 320, 3), 40, dtype=np.uint8)
    cv2.circle(frame, (160, 120), 55, (40, 190, 60), -1)
    crops = propose_leaf_crops(frame)
    assert len(crops) >= 1
    c = crops[0]
    assert c.image.shape == (224, 224, 3)
    assert c.area >= ProposalConfig().min_component_area
    x, y, w, h = c.bbox
    assert 0 <= x < 320 and 0 <= y < 240 and w > 0 and h > 0


def test_proposal_returns_nothing_on_a_bare_frame() -> None:
    frame = np.full((240, 320, 3), 90, dtype=np.uint8)
    assert propose_leaf_crops(frame) == []


def test_crop_cap_is_respected() -> None:
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    for i in range(24):
        cv2.circle(frame, (30 + (i % 8) * 78, 60 + (i // 8) * 150), 34, (40, 190, 60), -1)
    cfg = ProposalConfig(max_crops=5)
    crops = propose_leaf_crops(frame, cfg)
    assert len(crops) <= 5
    areas = [c.area for c in crops]
    assert areas == sorted(areas, reverse=True), "crops must be returned largest-first"


# --------------------------------------------------------------------------
# real field frames
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", field_frames(6), ids=lambda p: p.name)
def test_runs_on_real_field_frames(path: Path) -> None:
    """It must not crash, and must return well-formed crops, on real photography."""
    frame = cv2.imread(str(path))
    assert frame is not None, f"could not read {path}"

    crops = propose_leaf_crops(frame)
    for c in crops:
        assert c.image.shape == (224, 224, 3)
        assert c.image.dtype == np.uint8
        assert c.area > 0
        x, y, w, h = c.bbox
        assert 0 <= x and 0 <= y
        assert x + w <= frame.shape[1] and y + h <= frame.shape[0]


def test_finds_vegetation_in_most_real_field_frames() -> None:
    """A leaf-region proposer that finds nothing in real crop photos is useless.

    This is a behavioural floor, not a quality metric: Tomato-Village frames are
    close-ups of diseased tomato foliage, so most should yield at least one crop.
    """
    frames = field_frames(12)
    if not frames:
        pytest.skip("Tomato-Village not present")

    found = 0
    coverage = []
    for p in frames:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        coverage.append(green_coverage(detect_green(adaptive_vision(frame))))
        if propose_leaf_crops(frame):
            found += 1

    assert found >= len(frames) // 2, (
        f"only {found}/{len(frames)} real field frames yielded a leaf crop; "
        f"mean vegetation coverage {np.mean(coverage):.3f}"
    )
