"""Ground-speed calibration: PWM duty cycle to metres per second.

The rover has no wheel encoders, so it cannot measure its own speed. The
compensator in `timing.py` needs a speed to convert a latency into a lead
distance, which means speed has to come from an open-loop model fitted to timed
runs over a measured distance.

Two properties of small DC gearmotors make a straight-line fit through the
origin wrong, and both are handled here:

* **Deadband.** Below some duty the motors do not turn at all -- friction and
  the H-bridge drop exceed the available torque. Extrapolating a line through
  the origin predicts motion where there is none.
* **Battery droop.** Speed at a fixed duty falls as the pack discharges. The
  model therefore records the pack voltage alongside each run, and
  `SpeedModel.stale_for` reports when a calibration was taken under conditions
  too different to trust.

The fit is a least-squares line above the deadband, which is the simplest model
the data can support. A four-motor skid-steer rover on a cardboard chassis does
not warrant a richer one, and a richer one fitted to a handful of runs would be
fitting noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean

__all__ = ["SpeedRun", "SpeedModel", "fit_speed_model"]


@dataclass(frozen=True)
class SpeedRun:
    """One timed traverse of a measured distance at a fixed PWM duty."""

    duty: float
    distance_m: float
    duration_s: float
    battery_v: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.duty <= 100:
            raise ValueError(f"duty {self.duty} outside 0-100")
        if self.distance_m <= 0:
            raise ValueError("distance must be positive")
        if self.duration_s <= 0:
            raise ValueError("duration must be positive")

    @property
    def speed_mps(self) -> float:
        return self.distance_m / self.duration_s


@dataclass(frozen=True)
class SpeedModel:
    """speed = slope * (duty - deadband_duty), clamped at zero below deadband."""

    slope_mps_per_duty: float
    deadband_duty: float
    n_runs: int
    r_squared: float
    residual_sd_mps: float
    duty_range: tuple[float, float]
    mean_battery_v: float | None = None

    def speed_for(self, duty: float) -> float:
        """Predicted ground speed. Zero at or below the deadband."""
        if duty <= self.deadband_duty:
            return 0.0
        return self.slope_mps_per_duty * (duty - self.deadband_duty)

    def duty_for(self, speed_mps: float) -> float:
        """Inverse: the duty that should produce this speed."""
        if speed_mps <= 0:
            return 0.0
        if self.slope_mps_per_duty <= 0:
            raise ValueError("model has non-positive slope; refit before inverting")
        return self.deadband_duty + speed_mps / self.slope_mps_per_duty

    def extrapolating(self, duty: float) -> bool:
        """True when `duty` lies outside the range the model was fitted on.

        Worth checking before every use. A compensator that silently
        extrapolates is a compensator that misses at the speeds nobody tested.
        """
        lo, hi = self.duty_range
        return not (lo <= duty <= hi)

    def stale_for(self, battery_v: float | None, tolerance_v: float = 0.4) -> bool:
        """True when the pack voltage has drifted far from calibration."""
        if battery_v is None or self.mean_battery_v is None:
            return False
        return abs(battery_v - self.mean_battery_v) > tolerance_v

    def summary(self) -> dict[str, float | int | None]:
        return {
            "slope_mps_per_duty": self.slope_mps_per_duty,
            "deadband_duty": self.deadband_duty,
            "n_runs": self.n_runs,
            "r_squared": self.r_squared,
            "residual_sd_mps": self.residual_sd_mps,
            "duty_min": self.duty_range[0],
            "duty_max": self.duty_range[1],
            "mean_battery_v": self.mean_battery_v,
        }


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Correlation coefficient. Carries the *sign* of the relationship.

    The sign matters more than the magnitude here. An earlier version of this
    module guarded reversed wiring by checking the sign of the through-origin
    slope, `sum(xy) / sum(x^2)` -- which is always positive when duties and
    speeds are both positive, whatever the trend. Reversed motors would have
    produced a confident, positive, entirely wrong model.
    """
    mx, my = mean(xs), mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def fit_speed_model(runs: list[SpeedRun], moving_eps_mps: float = 0.01) -> SpeedModel:
    """Fit `speed = slope * (duty - deadband)` to timed runs.

    Two stages, because the deadband cannot simply be read off the data:

    1. **Free-intercept least squares** on the moving runs. Its x-intercept
       estimates the duty at which motion stops, which is the deadband -- and it
       works even when calibration never went low enough to stall the motors,
       which is the normal case. Observed stalls raise that estimate but never
       lower it.
    2. **Constrained refit** through the chosen deadband, so the returned model
       predicts exactly zero speed there rather than a small positive one.

    Raises rather than returning a degenerate model when the data cannot support
    a fit. A silently flat or reversed model would make the compensator fire at
    the wrong lead at every speed, which is the failure it exists to prevent.
    """
    if len(runs) < 3:
        raise ValueError(f"need at least 3 runs to fit, got {len(runs)}")

    observed_stall = max((r.duty for r in runs if r.speed_mps < moving_eps_mps),
                         default=0.0)
    moving = [r for r in runs if r.speed_mps >= moving_eps_mps]
    if len(moving) < 2:
        raise ValueError(
            f"only {len(moving)} run(s) exceeded {moving_eps_mps} m/s; cannot fit "
            "a slope. Collect runs at higher duty.")

    duties = [r.duty for r in moving]
    speeds = [r.speed_mps for r in moving]
    if len(set(duties)) < 2:
        raise ValueError("all moving runs share one duty; the slope is unidentifiable")

    # --- stage 1: free intercept, and the sign guard -----------------------
    if _pearson(duties, speeds) <= 0:
        raise ValueError(
            "speed does not increase with duty (correlation is non-positive). "
            "Check motor wiring, the timing procedure, or whether the runs are "
            "mislabelled.")

    mx, my = mean(duties), mean(speeds)
    sxx = sum((d - mx) ** 2 for d in duties)
    b = sum((d - mx) * (s - my) for d, s in zip(duties, speeds)) / sxx
    a = my - b * mx
    if b <= 0:                                   # unreachable given the guard above
        raise ValueError(f"fitted slope is non-positive ({b:.5f})")

    fitted_deadband = -a / b                     # x-intercept of the free fit
    deadband = max(0.0, observed_stall, fitted_deadband)
    if deadband >= max(duties):
        raise ValueError(
            f"estimated deadband ({deadband:.0f}% duty) is at or above every "
            "moving run. The runs cannot identify a usable speed range.")

    # --- stage 2: refit through the deadband -------------------------------
    xs = [d - deadband for d in duties]
    slope = sum(x * s for x, s in zip(xs, speeds)) / sum(x * x for x in xs)
    if slope <= 0:
        raise ValueError(f"constrained slope is non-positive ({slope:.5f})")

    resid = [s - slope * x for x, s in zip(xs, speeds)]
    ss_res = sum(r * r for r in resid)
    ss_tot = sum((s - my) ** 2 for s in speeds)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    resid_sd = math.sqrt(ss_res / len(resid))

    volts = [r.battery_v for r in runs if r.battery_v is not None]

    return SpeedModel(
        slope_mps_per_duty=slope,
        deadband_duty=deadband,
        n_runs=len(runs),
        r_squared=r2,
        residual_sd_mps=resid_sd,
        duty_range=(min(duties), max(duties)),
        mean_battery_v=mean(volts) if volts else None,
    )
