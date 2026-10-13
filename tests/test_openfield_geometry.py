"""Geometry of the open-field experiment: rotation, cell labelling, aggregation.

An off-by-one in any of these silently changes every number in the results, and
none of them would raise. Each is pinned against a case worked out by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openfield_sprayer import GX, GY, aggregate, grid_truth, to_landscape  # noqa: E402


def test_landscape_is_untouched():
    w, h, b = to_landscape(1600, 1200, [("LYPES", 10, 20, 30, 40)])
    assert (w, h) == (1600, 1200) and b == [("LYPES", 10, 20, 30, 40)]


def test_portrait_box_matches_np_rot90():
    """Paint a box into a portrait array, rotate the array with np.rot90, and
    check the rotated box covers exactly the painted pixels."""
    w, h = 40, 60
    img = np.zeros((h, w), bool)
    x0, y0, x1, y1 = 5, 10, 15, 30
    img[y0:y1, x0:x1] = True
    rot = np.rot90(img, 1)
    nw, nh, [(_, a0, b0, a1, b1)] = to_landscape(w, h, [("LYPES", x0, y0, x1, y1)])
    assert (nw, nh) == (h, w) == (rot.shape[1], rot.shape[0])
    mask = np.zeros_like(rot)
    mask[int(b0):int(b1), int(a0):int(a1)] = True
    assert (mask == rot).all()


def test_box_centre_marks_its_cell_even_when_small():
    w, h = GX * 100, GY * 100
    crop, other, tom, n = grid_truth(w, h, [("LYPES", 250, 350, 260, 360)])
    assert n == 1
    assert crop[3, 2] and crop.sum() == 1
    assert tom[3][2] == [0] and not other.any()


def test_weeds_and_not_recognised_are_never_crop():
    w, h = GX * 100, GY * 100
    crop, other, _, n = grid_truth(w, h, [("SOLNI", 0, 0, 100, 100),
                                          ("NR", 500, 500, 600, 600)])
    assert n == 0 and not crop.any()
    assert other[0, 0] and other[5, 5]


def test_out_of_bounds_box_is_clipped_not_wrapped():
    w, h = GX * 100, GY * 100
    crop, _, _, _ = grid_truth(w, h, [("LYPES", -500, -500, 50, 50)])
    assert crop[0, 0] and crop.sum() == 1


def test_large_plant_spans_every_cell_it_covers():
    w, h = GX * 100, GY * 100
    crop, _, tom, _ = grid_truth(w, h, [("LYPES", 0, 0, 300, 200)])
    assert crop[:2, :3].all() and crop.sum() == 6
    assert all(tom[r][c] == [0] for r in range(2) for c in range(3))


def test_aggregation_fires_a_nozzle_if_any_subcell_does():
    a = np.zeros((GY, GX), bool)
    a[0, 0] = True
    z = aggregate(a, 4, 3, "max")
    assert z.shape == (3, 4) and z[0, 0] and z.sum() == 1


def test_mean_aggregation_of_vegetation():
    a = np.zeros((GY, GX))
    a[:4, :4] = 1.0                       # exactly one 4x3-layout zone
    z = aggregate(a, 4, 3, "mean")
    assert z[0, 0] == 1.0 and z.sum() == 1.0
