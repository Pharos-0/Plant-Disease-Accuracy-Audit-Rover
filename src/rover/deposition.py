"""Scoring spray deposition from photographed water-sensitive paper.

Water-sensitive paper is the standard outcome measure in spray research: yellow
card that stains blue where a droplet lands. It is cheap, it needs no
instrumentation, and it turns "did the rover hit the target?" into a number
that can carry an error bar. Ordinary paper with a dye tracer works the same way
if WSP is unavailable; only `stain_is_darker` changes.

What this module computes, per target card:

* **hit** -- did any spray land on the card at all
* **coverage** -- stained fraction of the card, the usual WSP statistic
* **centroid offset** -- signed distance along the direction of travel between
  where the spray landed and where the target was. **This is the measurement
  the whole experiment exists for.** An uncompensated rover should show a
  consistently *negative* offset (spray lands behind the target) growing with
  speed; a compensated one should centre near zero.

The sign convention is fixed once, here, and tested: **negative means the spray
landed behind the target**, i.e. the rover fired too late. Getting this backwards
would invert the headline figure, so `test_rover_deposition.py` pins it.

No OpenCV dependency: the analysis is thresholding and moments over a numpy
array, and keeping it to numpy means it runs in the same environment as the
rest of the analysis rather than only on the Pi.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["CardGeometry", "DepositionResult", "score_card", "summarise_runs"]


@dataclass(frozen=True)
class CardGeometry:
    """Physical size of the card and where the target sits on it.

    `travel_axis` is the image axis along which the rover moves: 1 when the
    rover travels left-to-right across the photograph, 0 when it travels top to
    bottom. Everything downstream depends on this being right, so it is
    explicit rather than assumed.
    """

    width_m: float
    height_m: float
    target_frac: float = 0.5
    travel_axis: int = 1

    def __post_init__(self) -> None:
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("card dimensions must be positive")
        if not 0.0 <= self.target_frac <= 1.0:
            raise ValueError("target_frac must lie in [0, 1]")
        if self.travel_axis not in (0, 1):
            raise ValueError("travel_axis must be 0 (rows) or 1 (columns)")

    def length_along_travel_m(self) -> float:
        return self.width_m if self.travel_axis == 1 else self.height_m


@dataclass(frozen=True)
class DepositionResult:
    """One scored card."""

    hit: bool
    coverage: float
    offset_m: float | None
    stained_pixels: int
    total_pixels: int
    note: str = ""

    def summary(self) -> dict[str, float | bool | str | None]:
        return {"hit": self.hit, "coverage": self.coverage,
                "offset_m": self.offset_m, "offset_cm":
                    None if self.offset_m is None else self.offset_m * 100,
                "stained_pixels": self.stained_pixels,
                "total_pixels": self.total_pixels, "note": self.note}


def _to_grey(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=float)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] in (3, 4):
        # Rec. 601 luma over the first three channels; alpha ignored.
        return a[..., :3] @ np.array([0.299, 0.587, 0.114])
    raise ValueError(f"expected a 2-D or 3-channel image, got shape {a.shape}")


def score_card(image: np.ndarray, geometry: CardGeometry, *,
               threshold: float | None = None,
               min_stain_fraction: float = 0.001,
               stain_is_darker: bool = True) -> DepositionResult:
    """Score one photographed card.

    `threshold` defaults to Otsu's method, computed here rather than imported,
    so a card photographed under different lighting still separates correctly.
    A fixed threshold would make the result depend on how bright the room was
    when the photograph was taken.

    `min_stain_fraction` guards against calling sensor noise a hit. Below it the
    card is reported as a miss with the coverage still recorded, so a marginal
    result stays visible in the data rather than being silently rounded away.
    """
    grey = _to_grey(image)
    if grey.size == 0:
        raise ValueError("empty image")

    thr = _otsu(grey) if threshold is None else threshold
    mask = (grey < thr) if stain_is_darker else (grey > thr)

    stained = int(mask.sum())
    total = int(mask.size)
    coverage = stained / total

    if coverage < min_stain_fraction:
        return DepositionResult(False, coverage, None, stained, total,
                                note=f"coverage {coverage:.5f} below "
                                     f"min_stain_fraction {min_stain_fraction}")

    # Centroid of the stain along the travel axis, in pixels.
    idx = np.nonzero(mask)
    centroid_px = float(idx[geometry.travel_axis].mean())
    n_px = grey.shape[geometry.travel_axis]
    target_px = geometry.target_frac * (n_px - 1)

    metres_per_px = geometry.length_along_travel_m() / n_px
    # Negative => stain centroid sits BEFORE the target along travel, i.e. the
    # rover sprayed behind it. Pinned by test_sign_convention_late_is_negative.
    offset_m = (centroid_px - target_px) * metres_per_px

    return DepositionResult(True, coverage, offset_m, stained, total)


def _otsu(grey: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold. Falls back to the midpoint on a uniform image."""
    lo, hi = float(grey.min()), float(grey.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return (lo + hi) / 2.0

    hist, edges = np.histogram(grey, bins=bins, range=(lo, hi))
    hist = hist.astype(float)
    centres = (edges[:-1] + edges[1:]) / 2.0

    w0 = np.cumsum(hist)
    w1 = hist.sum() - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.cumsum(hist * centres) / w0
        m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between = np.nan_to_num(between, nan=-np.inf)
    return float(centres[int(np.argmax(between))])


def summarise_runs(results: list[DepositionResult]) -> dict[str, float | int]:
    """Aggregate one experimental condition: hit rate and offset statistics.

    Offsets are summarised over hits only -- a miss has no centroid, and
    substituting one would invent data. The miss count is reported alongside so
    the hit rate and the offset are read together: an estimator that hits rarely
    but centres well is not better than one that always hits slightly late.
    """
    if not results:
        raise ValueError("no results to summarise")

    hits = [r for r in results if r.hit and r.offset_m is not None]
    offsets = np.array([r.offset_m for r in hits], dtype=float)

    out: dict[str, float | int] = {
        "n": len(results),
        "n_hits": len(hits),
        "hit_rate": len(hits) / len(results),
        "mean_coverage": float(np.mean([r.coverage for r in results])),
    }
    if len(hits):
        out |= {
            "mean_offset_m": float(offsets.mean()),
            "mean_offset_cm": float(offsets.mean() * 100),
            "sd_offset_m": float(offsets.std(ddof=1)) if len(hits) > 1 else 0.0,
            "mean_abs_offset_cm": float(np.abs(offsets).mean() * 100),
            "median_offset_cm": float(np.median(offsets) * 100),
        }
    return out
