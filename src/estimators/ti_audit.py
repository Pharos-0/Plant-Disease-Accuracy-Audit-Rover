"""Target-Informativeness audit for label-free accuracy estimators.

Measures how much of an estimator's output is driven by the unlabelled target
sample and how much by the source-side calibration it was fitted on.

Definition (``docs/NEW_ESTIMATOR_NOVELTY_AUDIT.md`` section 3). For an estimator
``A(S, T)`` with source information ``S`` and unlabelled target ``T``:

    V_T = Var_T( A(S0, T) )      source held fixed, target resampled
    V_S = Var_S( A(S,  T0) )     target held fixed, source resampled
    TI  = V_T / (V_T + V_S)

``TI = 0`` means the estimator ignores the target; ``TI = 1`` means the source
anchor is irrelevant. **This is a first-order Sobol' index over two input
groups** - established mathematics applied to a new object, not a new metric.
That equivalence is documented, not hidden.

Two design points that are not free choices:

* **Resampling is cluster-level, never row-level.** The target pool is 1,616
  crops from 391 photographs, and the source pool is grouped by leaf. Bootstrapping
  crops independently would overstate the available information and understate the
  variance - the same error the project bars everywhere else. Both sides resample
  *groups*.
* **Estimators are re-fitted from scratch** for each source resample, because the
  anchor is created at fit time.

Nothing here touches target labels. The audit consumes only what the estimator
itself is allowed to see.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import Estimator, SourceStats

#: an estimator factory - called with no arguments, returns an unfitted estimator
Factory = Callable[[], Estimator]


@dataclass
class TIResult:
    estimator: str
    target_sd: float          # sd of the estimate under target resampling
    anchor_sd: float          # sd of the estimate under source resampling
    target_var: float
    anchor_var: float
    ti: float                 # V_T / (V_T + V_S)
    point_estimate: float     # the estimate at the full source and full target
    n_target_boot: int
    n_source_boot: int


def _cluster_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap whole groups, returning row indices of the resampled sample.

    Groups are drawn with replacement; every row of a drawn group is included, so
    the resample respects the correlation structure inside a photograph or a leaf.
    """
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    picked = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([rows_by_group[g] for g in picked])


def target_sensitivity(
    factory: Factory,
    source: SourceStats,
    target_logits: np.ndarray,
    target_features: np.ndarray | None,
    target_groups: np.ndarray,
    n_boot: int = 50,
    seed: int = 0,
) -> tuple[float, list[float]]:
    """Spread of the estimate when the SOURCE is fixed and the TARGET is resampled."""
    rng = np.random.default_rng(seed)
    est = factory().fit(source)
    out: list[float] = []
    for _ in range(n_boot):
        idx = _cluster_indices(target_groups, rng)
        feats = None if target_features is None else target_features[idx]
        try:
            out.append(est.estimate(target_logits[idx], feats))
        except (ValueError, RuntimeError):
            continue
    return (float(np.std(out)) if len(out) > 1 else 0.0), out


def anchor_sensitivity(
    factory: Factory,
    source: SourceStats,
    source_groups: np.ndarray,
    target_logits: np.ndarray,
    target_features: np.ndarray | None,
    n_boot: int = 50,
    seed: int = 0,
) -> tuple[float, list[float]]:
    """Spread of the estimate when the TARGET is fixed and the SOURCE is resampled.

    The estimator is refitted on every source resample, because the anchor -
    threshold, marginal, intercept, source accuracy - is created during ``fit``.
    """
    rng = np.random.default_rng(seed + 9973)
    out: list[float] = []
    for _ in range(n_boot):
        idx = _cluster_indices(source_groups, rng)
        s = SourceStats(
            logits=source.logits[idx],
            labels=source.labels[idx],
            features=None if source.features is None else source.features[idx],
        )
        try:
            out.append(factory().fit(s).estimate(target_logits, target_features))
        except (ValueError, RuntimeError):
            continue
    return (float(np.std(out)) if len(out) > 1 else 0.0), out


def target_informativeness(
    name: str,
    factory: Factory,
    source: SourceStats,
    source_groups: np.ndarray,
    target_logits: np.ndarray,
    target_features: np.ndarray | None,
    target_groups: np.ndarray,
    n_boot: int = 50,
    seed: int = 0,
) -> TIResult:
    t_sd, _ = target_sensitivity(factory, source, target_logits, target_features,
                                 target_groups, n_boot, seed)
    a_sd, _ = anchor_sensitivity(factory, source, source_groups, target_logits,
                                 target_features, n_boot, seed)
    v_t, v_a = t_sd ** 2, a_sd ** 2
    denom = v_t + v_a
    ti = float(v_t / denom) if denom > 0 else float("nan")
    point = factory().fit(source).estimate(target_logits, target_features)
    return TIResult(name, t_sd, a_sd, v_t, v_a, ti, float(point), n_boot, n_boot)


# ---------------------------------------------------------------------------
# Target-information removal (Phase 8)
# ---------------------------------------------------------------------------

def removal_surrogates(
    logits: np.ndarray,
    features: np.ndarray | None,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Targets with their information progressively destroyed.

    An estimator that still moves between these, or that returns the same number
    for a real target and a noise target, is telling us something about how much
    of its output the target was ever responsible for.
    """
    n = len(logits)
    out: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    # 1. row order shuffled - every estimator here should be permutation
    #    invariant, EXCEPT GdScore, whose score is computed per fixed-size batch.
    perm = rng.permutation(n)
    out["shuffled"] = (logits[perm], None if features is None else features[perm])

    # 2. one sample duplicated n times - all target variety removed
    k = int(rng.integers(0, n))
    out["duplicated"] = (np.repeat(logits[k:k + 1], n, axis=0),
                         None if features is None else
                         np.repeat(features[k:k + 1], n, axis=0))

    # 3. Gaussian noise matched to the target's own first two moments
    out["moment_matched_noise"] = (
        rng.normal(logits.mean(0), logits.std(0) + 1e-9, size=logits.shape).astype(np.float32),
        None if features is None else
        rng.normal(features.mean(0), features.std(0) + 1e-9,
                   size=features.shape).astype(np.float32),
    )

    # 4. uniform-random logits - no relationship to the model or the data
    out["random_logits"] = (
        rng.normal(0.0, 1.0, size=logits.shape).astype(np.float32),
        None if features is None else
        np.abs(rng.normal(0.0, 1.0, size=features.shape)).astype(np.float32),
    )
    return out
