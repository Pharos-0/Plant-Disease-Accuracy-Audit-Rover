"""Perception-to-actuation timing: the latency budget and the lead compensator.

The problem this module exists to solve. A spot sprayer decides from a camera
frame, but the frame is already old when the decision is made, and the nozzle
takes further time to produce a droplet. By then the rover has moved. Firing at
the moment of decision therefore sprays *behind* the target, and the miss grows
linearly with travel speed:

    miss = v * t_total

At 0.30 m/s with a 300 ms budget that is 9 cm, which is larger than a tomato
leaf. The frozen V0 rover (`rover/original/web_rover.py`) fires the instant its
green-pixel count crosses 1500, with no compensation of any kind.

The fix is to fire *early*, when the target is still `v * t_total` ahead of the
spray point, rather than when it is level with it. That needs two measured
quantities and neither can be assumed:

* the **latency budget** -- how long each stage of the pipeline actually takes
  on this hardware, measured with `LatencyProbe` (see `measure.py`);
* the **speed model** -- what ground speed a given PWM duty produces, since the
  rover has no encoders (see `speed.py`).

Nothing here talks to hardware. Everything in this module is a pure function of
measured numbers, so the compensator can be tested without a rover and the same
code runs in the analysis and on the Pi.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, pstdev

__all__ = [
    "StageTiming",
    "LatencyBudget",
    "LeadCompensator",
    "TriggerDecision",
    "predicted_miss_m",
]


# ---------------------------------------------------------------------------
# a measured stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageTiming:
    """Repeated measurements of one pipeline stage, in seconds.

    Kept as the raw samples rather than a mean, because the spread matters as
    much as the centre: a stage with a 20 ms mean and a 40 ms tail will miss
    targets that a steady 30 ms stage would hit, and averaging hides that.
    """

    name: str
    samples: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError(f"stage {self.name!r} has no samples")
        if any(s < 0 for s in self.samples):
            raise ValueError(f"stage {self.name!r} has a negative sample")

    @property
    def mean_s(self) -> float:
        return mean(self.samples)

    @property
    def sd_s(self) -> float:
        return pstdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def n(self) -> int:
        return len(self.samples)

    def quantile_s(self, q: float) -> float:
        """Nearest-rank quantile. Used for the pessimistic budget."""
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must lie in [0, 1]")
        ordered = sorted(self.samples)
        idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return ordered[idx]

    def summary(self) -> dict[str, float | str | int]:
        return {"stage": self.name, "n": self.n, "mean_ms": self.mean_s * 1e3,
                "sd_ms": self.sd_s * 1e3, "p95_ms": self.quantile_s(0.95) * 1e3}


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyBudget:
    """The end-to-end perception-to-droplet delay, stage by stage.

    Stages are summed in pipeline order. They are treated as independent when
    combining spreads, which is an approximation and is stated as one: capture
    and processing share a CPU and will correlate under thermal throttling. The
    approximation is conservative for the mean (exact) and optimistic for the
    tail, so the p95 budget is also reported directly from measured totals where
    those are available.
    """

    stages: tuple[StageTiming, ...]
    measured_total: StageTiming | None = None

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a budget needs at least one stage")
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stage names: {names}")

    @property
    def total_mean_s(self) -> float:
        """Sum of stage means. Exact regardless of correlation."""
        return sum(s.mean_s for s in self.stages)

    @property
    def total_sd_s(self) -> float:
        """Independence approximation: root-sum-square of stage spreads."""
        return math.sqrt(sum(s.sd_s ** 2 for s in self.stages))

    def total_p95_s(self) -> float:
        """Prefer a directly measured total; fall back to a normal approximation.

        A measured end-to-end total captures whatever correlation exists between
        stages, so it is used when present. The fallback is mean + 1.645 sd,
        which assumes normality and independence -- both stated as assumptions
        rather than believed.
        """
        if self.measured_total is not None:
            return self.measured_total.quantile_s(0.95)
        return self.total_mean_s + 1.645 * self.total_sd_s

    def table(self) -> list[dict[str, float | str | int]]:
        rows = [s.summary() for s in self.stages]
        rows.append({"stage": "TOTAL (sum of means)", "n": min(s.n for s in self.stages),
                     "mean_ms": self.total_mean_s * 1e3,
                     "sd_ms": self.total_sd_s * 1e3,
                     "p95_ms": self.total_p95_s() * 1e3})
        if self.measured_total is not None:
            rows.append(self.measured_total.summary() | {"stage": "TOTAL (measured)"})
        return rows

    def dominant_stage(self) -> StageTiming:
        """The stage worth optimising first."""
        return max(self.stages, key=lambda s: s.mean_s)


def predicted_miss_m(budget: LatencyBudget, speed_mps: float,
                     use_p95: bool = False) -> float:
    """Distance the rover travels during the budget: the uncompensated miss.

    This is the quantity the whole experiment is about. With no compensation the
    spray lands this far *behind* the target.
    """
    if speed_mps < 0:
        raise ValueError("speed must be non-negative")
    t = budget.total_p95_s() if use_p95 else budget.total_mean_s
    return speed_mps * t


# ---------------------------------------------------------------------------
# the compensator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerDecision:
    """Why the compensator did or did not fire, so a run can be audited."""

    fire: bool
    reason: str
    lead_distance_m: float
    range_to_target_m: float | None
    predicted_error_m: float | None = None


@dataclass
class LeadCompensator:
    """Fire when the target is `v * t_total` ahead, not when it is level.

    The rover measures range to the target with the HC-SR04 and its speed from
    the calibrated PWM model. The compensator converts the latency budget into a
    *lead distance* and fires once the measured range falls inside it.

    Two guards, both of which exist because the naive version is unsafe:

    * **`max_range_m`** -- the HC-SR04 returns nonsense beyond roughly 2 m and
      `None` on timeout. A missing or implausible range must never be read as
      "target is close", so it refuses to fire rather than defaulting.
    * **`min_lead_m`** -- at very low speed the lead distance tends to zero and
      the compensator degenerates into the uncompensated trigger. A floor keeps
      the nozzle open long enough to wet the target at all.

    The compensator is deliberately stateless. A rover that has to remember
    whether it already fired is a rover that fires twice when the state is lost;
    dwell and re-trigger suppression belong to the caller, which owns the clock.
    """

    budget: LatencyBudget
    max_range_m: float = 2.0
    min_lead_m: float = 0.02
    use_p95: bool = False

    #: recorded for the manifest, so a run can be tied to the budget that drove it
    history: list[TriggerDecision] = field(default_factory=list, repr=False)

    def lead_distance_m(self, speed_mps: float) -> float:
        """How far ahead of the spray point to fire, at this speed."""
        if speed_mps < 0:
            raise ValueError("speed must be non-negative")
        raw = predicted_miss_m(self.budget, speed_mps, use_p95=self.use_p95)
        return max(self.min_lead_m, raw)

    def decide(self, *, target_detected: bool, range_m: float | None,
               speed_mps: float, record: bool = True) -> TriggerDecision:
        """Should the nozzle open now?"""
        lead = self.lead_distance_m(speed_mps)

        if not target_detected:
            d = TriggerDecision(False, "no target detected", lead, range_m)
        elif range_m is None:
            # An ultrasonic timeout. V0 would have sprayed anyway, because its
            # trigger never consulted range at all.
            d = TriggerDecision(False, "range unavailable (sensor timeout)", lead, None)
        elif not math.isfinite(range_m) or range_m < 0:
            d = TriggerDecision(False, f"range implausible ({range_m})", lead, range_m)
        elif range_m > self.max_range_m:
            d = TriggerDecision(False, f"range {range_m:.3f} m beyond sensor limit",
                                lead, range_m)
        elif range_m > lead:
            d = TriggerDecision(False, "target still beyond lead distance", lead,
                                range_m, predicted_error_m=range_m - lead)
        else:
            d = TriggerDecision(True, "target inside lead distance", lead, range_m,
                                predicted_error_m=lead - range_m)

        if record:
            self.history.append(d)
        return d

    def naive_decide(self, *, target_detected: bool, range_m: float | None,
                     trigger_range_m: float = 0.25) -> TriggerDecision:
        """The V0 behaviour, for the paired comparison.

        V0 fires on the green-pixel count alone and ignores range entirely. We
        give the baseline a fixed trigger range instead of no range at all,
        because a baseline that fires continuously would not be a fair contrast
        -- it would lose on spray volume rather than on timing. This is the
        stronger baseline, deliberately.
        """
        if not target_detected:
            return TriggerDecision(False, "no target detected", 0.0, range_m)
        if range_m is None or not math.isfinite(range_m) or range_m < 0:
            # V0 has no range guard: detection alone is sufficient.
            return TriggerDecision(True, "V0: detection only, range ignored", 0.0, range_m)
        fire = range_m <= trigger_range_m
        return TriggerDecision(fire, "V0: fixed trigger range", 0.0, range_m)
