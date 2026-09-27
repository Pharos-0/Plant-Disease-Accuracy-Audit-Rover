"""Tests for the latency budget, the lead compensator and the speed model.

All of this runs without a rover. The point is that when the hardware session
happens, the only unknowns are the measured numbers -- not whether the code that
consumes them is right.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rover.speed import SpeedRun, SpeedModel, fit_speed_model
from rover.timing import (
    LatencyBudget,
    LeadCompensator,
    StageTiming,
    predicted_miss_m,
)


# ---------------------------------------------------------------------------
# StageTiming
# ---------------------------------------------------------------------------


def test_stage_rejects_empty_samples():
    with pytest.raises(ValueError, match="no samples"):
        StageTiming("capture", ())


def test_stage_rejects_negative_time():
    with pytest.raises(ValueError, match="negative"):
        StageTiming("capture", (0.01, -0.002))


def test_stage_statistics():
    s = StageTiming("capture", (0.10, 0.20, 0.30))
    assert s.mean_s == pytest.approx(0.20)
    assert s.n == 3
    assert s.sd_s > 0


def test_single_sample_has_zero_spread_not_a_crash():
    s = StageTiming("relay", (0.004,))
    assert s.sd_s == 0.0
    assert s.quantile_s(0.95) == 0.004


def test_quantile_is_nearest_rank_and_in_range():
    s = StageTiming("infer", tuple(i / 100 for i in range(1, 101)))
    assert s.quantile_s(0.95) == pytest.approx(0.95)
    assert s.quantile_s(0.0) == pytest.approx(0.01)
    assert s.quantile_s(1.0) == pytest.approx(1.00)


def test_quantile_rejects_out_of_range_q():
    s = StageTiming("infer", (0.01, 0.02))
    with pytest.raises(ValueError):
        s.quantile_s(1.5)


# ---------------------------------------------------------------------------
# LatencyBudget
# ---------------------------------------------------------------------------


def _budget(capture=0.08, process=0.05, actuate=0.04) -> LatencyBudget:
    return LatencyBudget(stages=(
        StageTiming("capture", (capture, capture, capture)),
        StageTiming("process", (process, process, process)),
        StageTiming("actuate", (actuate, actuate, actuate)),
    ))


def test_budget_rejects_duplicate_stage_names():
    with pytest.raises(ValueError, match="duplicate"):
        LatencyBudget(stages=(StageTiming("a", (0.1,)), StageTiming("a", (0.2,))))


def test_budget_rejects_empty():
    with pytest.raises(ValueError, match="at least one stage"):
        LatencyBudget(stages=())


def test_total_mean_is_the_sum_of_stage_means():
    b = _budget(0.08, 0.05, 0.04)
    assert b.total_mean_s == pytest.approx(0.17)


def test_measured_total_is_preferred_over_the_normal_approximation():
    """A measured end-to-end total captures correlation the RSS sum cannot."""
    stages = (StageTiming("a", (0.10, 0.12)), StageTiming("b", (0.05, 0.07)))
    measured = StageTiming("total", (0.30, 0.32, 0.50))  # a long real tail
    b = LatencyBudget(stages=stages, measured_total=measured)
    assert b.total_p95_s() == pytest.approx(0.50)
    # the approximation would have been far smaller
    approx = LatencyBudget(stages=stages)
    assert approx.total_p95_s() < 0.30


def test_dominant_stage_is_the_slowest():
    b = _budget(0.08, 0.05, 0.04)
    assert b.dominant_stage().name == "capture"


def test_table_includes_a_total_row():
    rows = _budget().table()
    assert any("TOTAL" in str(r["stage"]) for r in rows)


# ---------------------------------------------------------------------------
# the miss, which is the quantity the whole experiment is about
# ---------------------------------------------------------------------------


def test_miss_grows_linearly_with_speed():
    b = _budget(0.10, 0.10, 0.10)          # 300 ms
    assert predicted_miss_m(b, 0.0) == pytest.approx(0.0)
    assert predicted_miss_m(b, 0.30) == pytest.approx(0.09)   # 9 cm
    assert predicted_miss_m(b, 0.60) == pytest.approx(0.18)   # doubles


def test_miss_rejects_negative_speed():
    with pytest.raises(ValueError):
        predicted_miss_m(_budget(), -0.1)


# ---------------------------------------------------------------------------
# LeadCompensator
# ---------------------------------------------------------------------------


def test_lead_distance_equals_the_miss_it_cancels():
    b = _budget(0.10, 0.10, 0.10)
    c = LeadCompensator(budget=b, min_lead_m=0.0)
    assert c.lead_distance_m(0.30) == pytest.approx(predicted_miss_m(b, 0.30))


def test_min_lead_floor_applies_at_low_speed():
    c = LeadCompensator(budget=_budget(), min_lead_m=0.02)
    assert c.lead_distance_m(0.0) == pytest.approx(0.02)


def test_fires_once_inside_the_lead_distance():
    c = LeadCompensator(budget=_budget(0.10, 0.10, 0.10))   # 0.09 m at 0.3 m/s
    far = c.decide(target_detected=True, range_m=0.50, speed_mps=0.30)
    near = c.decide(target_detected=True, range_m=0.05, speed_mps=0.30)
    assert not far.fire and "beyond lead" in far.reason
    assert near.fire


def test_does_not_fire_without_a_target():
    c = LeadCompensator(budget=_budget())
    assert not c.decide(target_detected=False, range_m=0.05, speed_mps=0.3).fire


def test_sensor_timeout_refuses_rather_than_defaulting_to_close():
    """A missing range must never be read as 'the target is here'."""
    c = LeadCompensator(budget=_budget())
    d = c.decide(target_detected=True, range_m=None, speed_mps=0.30)
    assert not d.fire
    assert "timeout" in d.reason


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_implausible_range_refuses(bad):
    c = LeadCompensator(budget=_budget())
    d = c.decide(target_detected=True, range_m=bad, speed_mps=0.30)
    assert not d.fire


def test_range_beyond_sensor_limit_refuses():
    c = LeadCompensator(budget=_budget(), max_range_m=2.0)
    d = c.decide(target_detected=True, range_m=5.0, speed_mps=0.30)
    assert not d.fire
    assert "beyond sensor limit" in d.reason


def test_faster_travel_fires_earlier():
    """The whole point: at higher speed the trigger must move further out."""
    c = LeadCompensator(budget=_budget(0.10, 0.10, 0.10))
    slow = c.lead_distance_m(0.20)
    fast = c.lead_distance_m(0.60)
    assert fast > slow
    # at 0.6 m/s a target 0.15 m away should fire; at 0.2 m/s it should not
    assert c.decide(target_detected=True, range_m=0.15, speed_mps=0.60).fire
    assert not c.decide(target_detected=True, range_m=0.15, speed_mps=0.20).fire


def test_history_records_every_decision_for_the_manifest():
    c = LeadCompensator(budget=_budget())
    c.decide(target_detected=True, range_m=0.5, speed_mps=0.3)
    c.decide(target_detected=False, range_m=0.5, speed_mps=0.3)
    assert len(c.history) == 2
    assert [d.fire for d in c.history] == [False, False]


def test_record_false_leaves_history_untouched():
    c = LeadCompensator(budget=_budget())
    c.decide(target_detected=True, range_m=0.5, speed_mps=0.3, record=False)
    assert c.history == []


def test_p95_budget_gives_a_longer_lead_than_the_mean():
    stages = (StageTiming("a", (0.05, 0.05, 0.25)),)
    b = LatencyBudget(stages=stages)
    mean_c = LeadCompensator(budget=b, use_p95=False, min_lead_m=0.0)
    p95_c = LeadCompensator(budget=b, use_p95=True, min_lead_m=0.0)
    assert p95_c.lead_distance_m(0.3) > mean_c.lead_distance_m(0.3)


# ---------------------------------------------------------------------------
# the V0 baseline, which must stay a fair contrast
# ---------------------------------------------------------------------------


def test_v0_baseline_ignores_range_entirely_when_it_is_missing():
    """V0 fires on green pixels alone. The baseline must reproduce that."""
    c = LeadCompensator(budget=_budget())
    d = c.naive_decide(target_detected=True, range_m=None)
    assert d.fire
    assert "range ignored" in d.reason


def test_v0_baseline_has_zero_lead():
    c = LeadCompensator(budget=_budget())
    assert c.naive_decide(target_detected=True, range_m=0.1).lead_distance_m == 0.0


def test_v0_and_compensated_differ_exactly_where_it_matters():
    """At speed, the compensator fires on a target V0 is still approaching."""
    c = LeadCompensator(budget=_budget(0.10, 0.10, 0.10))
    r = 0.20                                  # 20 cm away
    assert c.decide(target_detected=True, range_m=r, speed_mps=0.70).fire
    assert not c.naive_decide(target_detected=True, range_m=r,
                              trigger_range_m=0.10).fire


# ---------------------------------------------------------------------------
# SpeedModel
# ---------------------------------------------------------------------------


def test_speed_run_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        SpeedRun(duty=120, distance_m=1.0, duration_s=1.0)
    with pytest.raises(ValueError):
        SpeedRun(duty=40, distance_m=0.0, duration_s=1.0)
    with pytest.raises(ValueError):
        SpeedRun(duty=40, distance_m=1.0, duration_s=0.0)


def test_fit_recovers_a_known_linear_relationship():
    """speed = 0.01 * (duty - 20), sampled cleanly and never stalling.

    The deadband has to come from the fit's x-intercept, because no run in this
    set falls below the motion threshold -- which is the normal case when
    calibration starts above the stall duty.
    """
    runs = [SpeedRun(duty=d, distance_m=2.0, duration_s=2.0 / (0.01 * (d - 20)))
            for d in (30, 40, 50, 60, 70)]
    m = fit_speed_model(runs)
    assert m.deadband_duty == pytest.approx(20.0, abs=0.5)
    assert m.slope_mps_per_duty == pytest.approx(0.01, rel=0.02)
    assert m.speed_for(70) == pytest.approx(0.50, rel=0.02)
    assert m.speed_for(20) == 0.0
    assert m.r_squared > 0.99


def test_deadband_is_recovered_without_any_stalled_run():
    """The bug this guards: assuming deadband = 0 whenever nothing stalled.

    A rover with a real 25% stall duty, calibrated only from 40% upward, never
    stalls during calibration. Forcing the fit through the origin then predicts
    motion at duties that produce none, and the compensator fires early into
    empty ground.
    """
    runs = [SpeedRun(duty=d, distance_m=2.0, duration_s=2.0 / (0.012 * (d - 25)))
            for d in (40, 55, 70, 85)]
    m = fit_speed_model(runs)
    assert m.deadband_duty == pytest.approx(25.0, abs=1.0)
    assert m.speed_for(25) == 0.0
    assert m.speed_for(20) == 0.0


def test_deadband_is_the_highest_stalled_duty():
    runs = [SpeedRun(duty=10, distance_m=1.0, duration_s=1000.0),   # stalled
            SpeedRun(duty=20, distance_m=1.0, duration_s=1000.0),   # stalled
            SpeedRun(duty=40, distance_m=1.0, duration_s=5.0),
            SpeedRun(duty=60, distance_m=1.0, duration_s=2.5),
            SpeedRun(duty=80, distance_m=1.0, duration_s=1.67)]
    m = fit_speed_model(runs)
    assert m.deadband_duty == pytest.approx(20)
    assert m.speed_for(20) == 0.0
    assert m.speed_for(15) == 0.0
    assert m.speed_for(60) > 0


def test_fit_refuses_when_there_are_too_few_runs():
    with pytest.raises(ValueError, match="at least 3"):
        fit_speed_model([SpeedRun(40, 1.0, 5.0), SpeedRun(60, 1.0, 2.5)])


def test_fit_refuses_when_everything_stalled():
    runs = [SpeedRun(duty=d, distance_m=1.0, duration_s=1000.0) for d in (10, 20, 30)]
    with pytest.raises(ValueError, match="cannot fit a slope"):
        fit_speed_model(runs)


def test_fit_refuses_a_single_duty():
    runs = [SpeedRun(duty=40, distance_m=1.0, duration_s=d) for d in (5.0, 5.1, 4.9)]
    with pytest.raises(ValueError, match="unidentifiable"):
        fit_speed_model(runs)


def test_fit_refuses_when_speed_falls_with_duty():
    """Reversed wiring or a mistimed run must fail loudly, not fit backwards."""
    runs = [SpeedRun(duty=40, distance_m=1.0, duration_s=1.0),
            SpeedRun(duty=60, distance_m=1.0, duration_s=5.0),
            SpeedRun(duty=80, distance_m=1.0, duration_s=20.0)]
    with pytest.raises(ValueError, match="non-positive|do not increase"):
        fit_speed_model(runs)


def test_duty_for_inverts_speed_for():
    runs = [SpeedRun(duty=d, distance_m=2.0, duration_s=2.0 / (0.01 * d))
            for d in (30, 45, 60, 75)]
    m = fit_speed_model(runs)
    target = 0.4
    assert m.speed_for(m.duty_for(target)) == pytest.approx(target, rel=1e-6)


def test_duty_for_zero_speed_is_zero():
    runs = [SpeedRun(duty=d, distance_m=2.0, duration_s=2.0 / (0.01 * d))
            for d in (30, 45, 60)]
    assert fit_speed_model(runs).duty_for(0.0) == 0.0


def test_extrapolation_is_detectable():
    runs = [SpeedRun(duty=d, distance_m=2.0, duration_s=2.0 / (0.01 * d))
            for d in (30, 45, 60)]
    m = fit_speed_model(runs)
    assert not m.extrapolating(45)
    assert m.extrapolating(95)
    assert m.extrapolating(10)


def test_battery_droop_is_flagged():
    runs = [SpeedRun(duty=d, distance_m=2.0, duration_s=2.0 / (0.01 * d),
                     battery_v=8.0) for d in (30, 45, 60)]
    m = fit_speed_model(runs)
    assert m.mean_battery_v == pytest.approx(8.0)
    assert not m.stale_for(7.9)
    assert m.stale_for(7.0)
    assert not m.stale_for(None)          # unknown voltage is not a failure


def test_model_and_compensator_compose():
    """End to end: timed runs -> speed -> lead distance, no hardware."""
    runs = [SpeedRun(duty=d, distance_m=2.0, duration_s=2.0 / (0.008 * d))
            for d in (30, 45, 60, 75)]
    model = fit_speed_model(runs)
    comp = LeadCompensator(budget=_budget(0.09, 0.06, 0.05), min_lead_m=0.0)

    v = model.speed_for(60)
    lead = comp.lead_distance_m(v)
    assert lead == pytest.approx(v * 0.20, rel=1e-6)
    assert math.isfinite(lead) and lead > 0
