"""Labelled-sample estimation of target accuracy (RQ5).

Deliberately the simplest transparent estimator available: label a handful of
field photographs and take the sample accuracy. No learned calibration, no
weighting, no correction. The research question is whether *this* beats the
label-free estimators, and it must be answered before anything more elaborate is
considered.

**Photographs are the independent unit.** Arm D holds 1,616 crops from 391
photographs, and the crop counts are bimodal - every healthy photograph yields
one crop and every blight photograph yields eight. A uniform sample of
photographs is therefore a *cluster sample with unequal cluster sizes*, and the
estimator below is a ratio estimator rather than a simple mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabelledEstimate:
    """One labelled-sample estimate, plus what an annotator would actually see."""

    accuracy: float
    n_photographs: int
    n_crops: int
    n_correct: int
    ci_low: float
    ci_high: float


def photograph_index(group_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Map each photograph id to the crop indices belonging to it."""
    return {g: np.flatnonzero(group_ids == g) for g in np.unique(group_ids)}


def sample_photographs(
    group_ids: np.ndarray, n_photographs: int, seed: int
) -> np.ndarray:
    """Draw `n_photographs` distinct photographs and return **all** their crop indices.

    A photograph is labelled once and all of its crops inherit that label, which
    is what a human annotator would actually do. Sampling is without
    replacement, so a photograph can never appear twice.
    """
    by_photo = photograph_index(group_ids)
    photos = np.array(sorted(by_photo))
    if n_photographs > len(photos):
        raise ValueError(
            f"asked for {n_photographs} photographs but only {len(photos)} exist"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(photos, size=n_photographs, replace=False)
    return np.sort(np.concatenate([by_photo[p] for p in chosen]))


def wilson_interval(n_correct: int, n_total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval, chosen in advance (see docs/rq5_plan.md section J).

    Computed on the number of labelled **crops**. It treats those crops as
    independent, which they are not - crops from one photograph are correlated -
    so it is expected to be **anti-conservative**. Its empirical coverage is
    measured rather than assumed; that measurement is the honest way to present
    it, and is reported in the results.
    """
    if n_total == 0:
        return float("nan"), float("nan")
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def labelled_estimate(
    correct: np.ndarray, group_ids: np.ndarray, n_photographs: int, seed: int
) -> LabelledEstimate:
    """Estimate target accuracy from `n_photographs` labelled photographs.

    `correct` is the per-crop correctness of the frozen classifier. Only the
    sampled entries are read; everything else is evaluation truth and is not
    touched here.
    """
    idx = sample_photographs(group_ids, n_photographs, seed)
    sampled = correct[idx]
    n_crops = int(len(sampled))
    n_correct = int(sampled.sum())
    lo, hi = wilson_interval(n_correct, n_crops)
    return LabelledEstimate(
        accuracy=n_correct / n_crops if n_crops else float("nan"),
        n_photographs=n_photographs,
        n_crops=n_crops,
        n_correct=n_correct,
        ci_low=lo,
        ci_high=hi,
    )


def seed_for(repetition: int, base_seed: int = 20260909) -> int:
    """The pre-registered seed convention: base + repetition index."""
    return base_seed + repetition
