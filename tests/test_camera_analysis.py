"""Tests for the camera-session analysis: the resampling unit and the policy comparison.

The real camera session will be small, so the two things that most decide what its
intervals mean are pinned here: frames of one plant must be resampled together, and
the gated vs spatial-only comparison must be computed on the same frames.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_camera_session import load, policy_comparison  # noqa: E402


def _frame(plant, leaf, truth, cond, spray, spatial, pred="healthy", msp=0.9):
    return {"plant_id": plant, "leaf_id": leaf, "truth": truth, "condition": cond,
            "pred": pred, "msp": msp, "abstain": False, "spray": spray,
            "spray_spatial": spatial, "deferred": False}


def _session(tmp_path, frames, name="s1"):
    d = tmp_path / name
    d.mkdir()
    (d / "log.jsonl").write_text("\n".join(json.dumps(f) for f in frames), encoding="utf-8")
    return d


def test_resamples_plants_when_every_frame_names_one(tmp_path):
    frames = [_frame("P01", "P01L01", "late_blight", "low_light", True, True),
              _frame("P01", "P01L02", "late_blight", "low_light", False, True),
              _frame("P02", "P02L01", "healthy", "low_light", True, True)]
    df = load([_session(tmp_path, frames)])
    assert df.attrs["unit_kind"] == "plant"
    assert df.unit.nunique() == 2          # two plants, three leaves


def test_falls_back_to_leaves_when_a_plant_is_missing(tmp_path):
    frames = [_frame("P01", "L01", "late_blight", "normal", True, True),
              _frame("", "L02", "late_blight", "normal", True, True)]
    df = load([_session(tmp_path, frames)])
    assert df.attrs["unit_kind"] == "leaf"
    assert df.unit.nunique() == 2


def test_policy_comparison_is_computed_on_the_same_frames(tmp_path):
    # 4 diseased frames in low light: gated misses 3, spatial-only misses 1.
    frames = [_frame("P01", "P01L01", "early_blight", "low_light", False, True),
              _frame("P01", "P01L01", "early_blight", "low_light", False, False),
              _frame("P02", "P02L01", "late_blight", "low_light", False, True),
              _frame("P02", "P02L01", "late_blight", "low_light", True, True),
              _frame("P03", "P03L01", "healthy", "low_light", True, True)]
    df = load([_session(tmp_path, frames)])
    pc = policy_comparison(df, np.random.default_rng(0)).set_index("condition")
    r = pc.loc["low_light"]
    assert r.diseased_frames == 4 and r.units == 2
    assert r.missed_gated == 0.75 and r.missed_spatial == 0.25
    assert abs(r.difference - 0.5) < 1e-12
    assert r.difference_lo <= r.difference <= r.difference_hi


def test_deferred_frames_are_not_counted_as_missed(tmp_path):
    f = _frame("P01", "P01L01", "late_blight", "low_light", False, False)
    f["deferred"] = True
    df = load([_session(tmp_path, [f, _frame("P01", "P01L01", "late_blight",
                                             "low_light", True, True)])])
    pc = policy_comparison(df, np.random.default_rng(0)).set_index("condition")
    assert pc.loc["low_light"].missed_gated == 0.0
