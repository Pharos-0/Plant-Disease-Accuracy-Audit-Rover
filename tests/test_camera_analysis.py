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

import analyse_camera_session as acs  # noqa: E402
from analyse_camera_session import (cluster_ci, load, policy_comparison,  # noqa: E402
                                    session_status)


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


# --- what a small session cannot measure must not be reported as a number ------------

def _one_healthy_plant(tmp_path):
    """The likely first real session: one shop-bought, healthy plant."""
    frames = [_frame("P01", f"P01L0{leaf}", "healthy", cond, True, True)
              for leaf in (1, 2) for cond in ("normal", "low_light") for _ in range(3)]
    return _session(tmp_path, frames)


def test_one_plant_gives_no_interval_rather_than_a_zero_width_one(tmp_path):
    df = load([_one_healthy_plant(tmp_path)])
    point, lo, hi = cluster_ci(df, lambda d: float(d.spray.mean()),
                               np.random.default_rng(0))
    assert point == 1.0
    assert np.isnan(lo) and np.isnan(hi)


def test_policy_comparison_with_one_plant_keeps_the_point_but_no_interval(tmp_path):
    frames = [_frame("P01", "P01L01", "late_blight", "low_light", False, True),
              _frame("P01", "P01L01", "late_blight", "low_light", True, True)]
    df = load([_session(tmp_path, frames)])
    r = policy_comparison(df, np.random.default_rng(0)).set_index("condition").loc["low_light"]
    assert r.missed_gated == 0.5 and r.missed_spatial == 0.0
    assert np.isnan(r.difference_lo) and np.isnan(r.difference_hi)


def test_status_of_a_healthy_single_plant_session(tmp_path):
    st = session_status(load([_one_healthy_plant(tmp_path)]))
    assert st["unit_kind"] == "plant" and st["units"] == 1
    assert st["diseased_plants"] == 0 and st["healthy_plants"] == 1
    assert st["pilot"] is True
    assert st["intervals_estimable"] is False
    assert st["primary_outcome_estimable"] is False
    assert st["wasted_spray_estimable"] is True


def test_protocol_minimum_decides_pilot(tmp_path):
    def plants(n_sick, n_well):
        f = [_frame(f"D{i}", f"D{i}L1", "late_blight", "normal", True, True)
             for i in range(n_sick)]
        return f + [_frame(f"H{i}", f"H{i}L1", "healthy", "normal", True, True)
                    for i in range(n_well)]
    assert session_status(load([_session(tmp_path, plants(6, 4), "full")]))["pilot"] is False
    assert session_status(load([_session(tmp_path, plants(5, 4), "short")]))["pilot"] is True


def test_zero_event_upper_bound_is_the_exact_clopper_pearson_value():
    assert abs(acs.zero_event_upper(6) - (1 - 0.05 ** (1 / 6))) < 1e-12   # ~0.393
    assert 0.39 < acs.zero_event_upper(6) < 0.40
    assert acs.zero_event_upper(30) < acs.zero_event_upper(6)              # more plants, tighter
    assert np.isnan(acs.zero_event_upper(0))


def test_zero_misses_get_an_exact_bound_beside_the_degenerate_bootstrap(tmp_path, capsys,
                                                                        monkeypatch):
    frames = [_frame(f"D{i}", f"D{i}L1", "late_blight", "normal", True, True)
              for i in range(6)]
    frames += [_frame(f"H{i}", f"H{i}L1", "healthy", "normal", True, True) for i in range(4)]
    d = _session(tmp_path, frames)
    monkeypatch.setattr(sys, "argv", ["analyse", str(d)])
    assert acs.main() == 0
    out = capsys.readouterr().out
    assert "zero misses are not proof of zero risk" in out
    assert "0 missed across 6 diseased plants" in out and "39%" in out
    st = json.loads((d / "analysis_status.json").read_text())
    assert st["pilot"] is False and len(st["zero_miss_notes"]) == 2      # both policies


def test_healthy_only_session_reports_missed_as_unmeasured_not_zero(tmp_path, capsys,
                                                                    monkeypatch):
    d = _one_healthy_plant(tmp_path)
    monkeypatch.setattr(sys, "argv", ["analyse", str(d)])
    assert acs.main() == 0
    out = capsys.readouterr().out
    assert "PILOT" in out and "NOT MEASURED" in out and "NOT ESTIMABLE" in out
    import pandas as pd
    t = pd.read_csv(d / "analysis_by_condition.csv", index_col=0)
    assert t.missed_treatment.isna().all()          # not 0.0: there was nothing to miss
    assert (t.wasted_spray == 1.0).all()            # measurable: healthy leaves were sprayed
    assert not (d / "policy_comparison.csv").exists()
    assert json.loads((d / "analysis_status.json").read_text())["pilot"] is True


def test_pilot_report_gives_the_pre_registered_quantities(tmp_path, capsys, monkeypatch):
    """Protocol section 11.2: P2 false alarms, P3 tissue detection/deferral, P4 confidence."""
    frames = []
    for i in range(4):          # normal light: 1 of 4 healthy frames called diseased
        frames.append(_frame("P01", "P01L01", "healthy", "normal", True, True,
                             pred="late_blight" if i == 0 else "healthy", msp=0.95))
    for i in range(4):          # low light: abstains, spatial misses one, one deferred
        f = _frame("P01", "P01L01", "healthy", "low_light", True, i != 3, msp=0.5)
        f["abstain"] = True
        f["deferred"] = i == 3
        frames.append(f)
    d = _session(tmp_path, frames)
    monkeypatch.setattr(sys, "argv", ["analyse", str(d)])
    assert acs.main() == 0
    assert "PILOT QUANTITIES" in capsys.readouterr().out
    import pandas as pd
    pr = pd.read_csv(d / "pilot_report.csv", index_col=0)
    assert pr.loc["normal", "p2_pred_diseased_on_healthy"] == 0.25
    assert pr.loc["normal", "p3_spatial_fires"] == 1.0
    assert pr.loc["low_light", "p2_abstain_on_healthy"] == 1.0
    assert pr.loc["low_light", "p3_spatial_fires"] == 0.75
    assert pr.loc["low_light", "p3_deferred"] == 0.25
    assert pr.loc["low_light", "p4_confidence"] == 0.5


def test_full_study_gets_no_pilot_report(tmp_path, monkeypatch, capsys):
    frames = [_frame(f"D{i}", f"D{i}L1", "late_blight", "normal", True, True) for i in range(6)]
    frames += [_frame(f"H{i}", f"H{i}L1", "healthy", "normal", True, True) for i in range(4)]
    d = _session(tmp_path, frames)
    monkeypatch.setattr(sys, "argv", ["analyse", str(d)])
    assert acs.main() == 0
    assert "PILOT" not in capsys.readouterr().out
    assert not (d / "pilot_report.csv").exists()
