"""Tests for spray-deposition scoring on synthetic cards.

The sign convention is the thing most worth pinning: if "late" came out
positive, the headline figure of the whole experiment would be inverted and
every conclusion with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rover.deposition import (
    CardGeometry,
    DepositionResult,
    score_card,
    summarise_runs,
)

CARD = CardGeometry(width_m=0.10, height_m=0.05, target_frac=0.5, travel_axis=1)


def make_card(stain_centre_frac: float | None, *, w: int = 200, h: int = 100,
              stain_width_px: int = 20, clean: float = 230.0,
              stained: float = 30.0) -> np.ndarray:
    """A synthetic card: bright background, a dark vertical stain band."""
    img = np.full((h, w), clean, dtype=float)
    if stain_centre_frac is not None:
        c = int(round(stain_centre_frac * (w - 1)))
        lo = max(0, c - stain_width_px // 2)
        hi = min(w, c + stain_width_px // 2)
        img[:, lo:hi] = stained
    return img


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_geometry_rejects_bad_values():
    with pytest.raises(ValueError):
        CardGeometry(width_m=0, height_m=0.05)
    with pytest.raises(ValueError):
        CardGeometry(width_m=0.1, height_m=0.05, target_frac=1.5)
    with pytest.raises(ValueError):
        CardGeometry(width_m=0.1, height_m=0.05, travel_axis=2)


def test_length_along_travel_follows_the_axis():
    assert CardGeometry(0.10, 0.05, travel_axis=1).length_along_travel_m() == 0.10
    assert CardGeometry(0.10, 0.05, travel_axis=0).length_along_travel_m() == 0.05


# ---------------------------------------------------------------------------
# the sign convention -- the most important test in this file
# ---------------------------------------------------------------------------


def test_sign_convention_late_is_negative():
    """Spray landing BEHIND the target must give a NEGATIVE offset.

    The rover travels left to right (travel_axis=1), so "behind" is to the
    left, at a lower column index than the target at the card's centre. An
    uncompensated rover fires late and stains the left of the card.
    """
    late = score_card(make_card(0.25), CARD)      # stain left of centre
    assert late.hit
    assert late.offset_m < 0
    assert late.offset_m == pytest.approx(-0.025, abs=0.003)   # ~2.5 cm behind


def test_sign_convention_early_is_positive():
    early = score_card(make_card(0.75), CARD)
    assert early.hit
    assert early.offset_m > 0


def test_on_target_is_near_zero():
    on = score_card(make_card(0.50), CARD)
    assert on.hit
    assert abs(on.offset_m) < 0.002


# ---------------------------------------------------------------------------
# hits and misses
# ---------------------------------------------------------------------------


def test_clean_card_is_a_miss_with_no_offset():
    r = score_card(make_card(None), CARD)
    assert not r.hit
    assert r.offset_m is None
    assert "below min_stain_fraction" in r.note


def test_a_few_noisy_pixels_do_not_count_as_a_hit():
    img = make_card(None)
    img[0, 0] = 10.0                        # one dark pixel
    r = score_card(img, CARD, min_stain_fraction=0.001)
    assert not r.hit
    assert r.coverage > 0, "marginal coverage must stay visible in the data"


def test_coverage_scales_with_stain_width():
    narrow = score_card(make_card(0.5, stain_width_px=10), CARD)
    wide = score_card(make_card(0.5, stain_width_px=40), CARD)
    assert wide.coverage > narrow.coverage
    assert wide.coverage == pytest.approx(4 * narrow.coverage, rel=0.15)


def test_stained_and_total_pixels_are_consistent():
    r = score_card(make_card(0.5, w=200, h=100), CARD)
    assert r.total_pixels == 200 * 100
    assert 0 < r.stained_pixels < r.total_pixels
    assert r.coverage == pytest.approx(r.stained_pixels / r.total_pixels)


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------


def test_threshold_adapts_to_exposure():
    """The same card photographed darker must score the same."""
    bright = make_card(0.30, clean=240, stained=40)
    dim = make_card(0.30, clean=120, stained=20)
    assert score_card(bright, CARD).offset_m == pytest.approx(
        score_card(dim, CARD).offset_m, abs=0.002)


def test_colour_images_are_accepted():
    grey = make_card(0.30)
    rgb = np.stack([grey, grey, grey], axis=-1)
    assert score_card(rgb, CARD).offset_m == pytest.approx(
        score_card(grey, CARD).offset_m, abs=1e-9)


def test_rgba_is_accepted_and_alpha_ignored():
    grey = make_card(0.30)
    rgba = np.stack([grey, grey, grey, np.full_like(grey, 255)], axis=-1)
    assert score_card(rgba, CARD).hit


def test_rejects_an_unusable_shape():
    with pytest.raises(ValueError, match="2-D or 3-channel"):
        score_card(np.zeros((4, 4, 7)), CARD)


def test_rejects_an_empty_image():
    with pytest.raises(ValueError, match="empty"):
        score_card(np.zeros((0, 0)), CARD)


def test_uniform_card_does_not_crash_otsu():
    r = score_card(np.full((50, 50), 200.0), CARD)
    assert not r.hit


def test_inverted_polarity_for_dye_on_dark_paper():
    """A light stain on dark paper: same result with stain_is_darker=False."""
    img = make_card(0.25, clean=30.0, stained=230.0)
    r = score_card(img, CARD, stain_is_darker=False)
    assert r.hit
    assert r.offset_m < 0


def test_vertical_travel_axis():
    """Rover travelling top-to-bottom: the offset must read along rows."""
    geom = CardGeometry(width_m=0.10, height_m=0.05, travel_axis=0)
    img = np.full((100, 200), 230.0)
    img[20:30, :] = 30.0                    # stain near the top = behind
    r = score_card(img, geom)
    assert r.hit
    assert r.offset_m < 0


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_summarise_reports_hit_rate_and_offsets():
    results = [score_card(make_card(f), CARD) for f in (0.30, 0.35, 0.40)]
    results.append(score_card(make_card(None), CARD))      # one miss
    s = summarise_runs(results)
    assert s["n"] == 4
    assert s["n_hits"] == 3
    assert s["hit_rate"] == pytest.approx(0.75)
    assert s["mean_offset_cm"] < 0


def test_summarise_excludes_misses_from_the_offset():
    """A miss has no centroid; substituting one would invent data."""
    hits = [score_card(make_card(0.30), CARD) for _ in range(3)]
    with_miss = hits + [score_card(make_card(None), CARD)]
    assert summarise_runs(hits)["mean_offset_cm"] == pytest.approx(
        summarise_runs(with_miss)["mean_offset_cm"])


def test_summarise_with_no_hits_omits_offset_keys():
    s = summarise_runs([score_card(make_card(None), CARD) for _ in range(3)])
    assert s["n_hits"] == 0
    assert s["hit_rate"] == 0.0
    assert "mean_offset_cm" not in s


def test_summarise_rejects_empty():
    with pytest.raises(ValueError, match="no results"):
        summarise_runs([])


def test_single_hit_has_zero_sd_not_a_crash():
    s = summarise_runs([score_card(make_card(0.4), CARD)])
    assert s["sd_offset_m"] == 0.0


def test_the_headline_contrast_is_expressible():
    """Uncompensated lands late and scattered; compensated centres on target."""
    naive = [score_card(make_card(f), CARD) for f in (0.20, 0.24, 0.28, 0.22)]
    comp = [score_card(make_card(f), CARD) for f in (0.48, 0.52, 0.50, 0.49)]

    n, c = summarise_runs(naive), summarise_runs(comp)
    assert n["mean_offset_cm"] < -1.0            # consistently behind
    assert abs(c["mean_offset_cm"]) < 0.5        # on target
    assert c["mean_abs_offset_cm"] < n["mean_abs_offset_cm"]
