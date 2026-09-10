"""Tests for the Target-Informativeness audit (Phase 11).

Covers the properties the measure is supposed to have, the cluster-bootstrap
machinery, the leakage firewall, and the numerical edge cases that would
otherwise produce a quietly wrong TI.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from estimators.base import Estimator, SourceStats, max_confidence
from estimators.methods import REGISTRY
from estimators.ti_audit import (
    _cluster_indices,
    anchor_sensitivity,
    removal_surrogates,
    target_informativeness,
    target_sensitivity,
)


# ---------------------------------------------------------------- helpers
class _Mixture(Estimator):
    """alpha * mean confidence(T) + (1 - alpha) * source accuracy."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.name = f"mix{alpha}"
        self.anchor_ = 0.0

    def _fit(self, source: SourceStats) -> None:
        self.anchor_ = source.accuracy

    def _estimate(self, logits, features) -> float:
        return self.alpha * float(max_confidence(logits).mean()) + \
            (1 - self.alpha) * self.anchor_


def _world(seed: int = 0, n_src: int = 400, n_tgt: int = 600):
    """A deliberately NON-separable world.

    Separation matters: with well-separated classes the source accuracy is 1.0
    with zero bootstrap variance, so TI collapses to 1 for every estimator. That
    is a measurement artefact and it is what these tests guard against.
    """
    rng = np.random.default_rng(seed)
    k, d = 3, 8
    means = rng.normal(0, 0.35, size=(k, d))
    ys = rng.integers(0, k, n_src)
    xs = (means[ys] + rng.normal(0, 1, size=(n_src, d))).astype(np.float32)
    cent = np.stack([xs[ys == c].mean(0) for c in range(k)])
    lg = lambda x: (-((x[:, None, :] - cent[None]) ** 2).sum(-1)).astype(np.float32)
    yt = rng.integers(0, k, n_tgt)
    xt = (means[yt] + 0.6 + rng.normal(0, 1, size=(n_tgt, d))).astype(np.float32)
    src = SourceStats(logits=lg(xs), labels=ys, features=xs)
    gs = np.repeat(np.arange(n_src // 4), 4).astype(str)
    gt = np.repeat(np.arange(n_tgt // 4), 4).astype(str)
    return src, lg(xt), xt, gs[:n_src], gt[:n_tgt]


# ---------------------------------------------------------------- cluster bootstrap
def test_cluster_indices_resamples_whole_groups():
    groups = np.repeat(np.arange(10), 5).astype(str)
    rng = np.random.default_rng(0)
    idx = _cluster_indices(groups, rng)
    # every group present appears with all five of its rows
    counts = np.unique(groups[idx], return_counts=True)[1]
    assert set(counts % 5) == {0}


def test_cluster_indices_length_matches_when_groups_are_equal_sized():
    groups = np.repeat(np.arange(20), 3).astype(str)
    idx = _cluster_indices(groups, np.random.default_rng(1))
    assert len(idx) == 60


def test_cluster_bootstrap_is_not_row_bootstrap():
    """Rows from one group must move together, or the variance is understated."""
    groups = np.repeat(np.arange(8), 6).astype(str)
    idx = _cluster_indices(groups, np.random.default_rng(3))
    picked = groups[idx]
    for g in np.unique(picked):
        assert (picked == g).sum() % 6 == 0


# ---------------------------------------------------------------- TI properties
def test_ti_is_zero_for_an_estimator_that_ignores_the_target():
    src, lgt, ft, gs, gt = _world()
    res = target_informativeness("const", lambda: _Mixture(0.0), src, gs,
                                 lgt, ft, gt, n_boot=25, seed=0)
    assert res.ti == pytest.approx(0.0, abs=1e-9)
    assert res.target_sd == pytest.approx(0.0, abs=1e-12)


def test_ti_is_one_for_an_estimator_with_no_anchor():
    src, lgt, ft, gs, gt = _world()
    res = target_informativeness("pure", lambda: _Mixture(1.0), src, gs,
                                 lgt, ft, gt, n_boot=25, seed=0)
    assert res.ti == pytest.approx(1.0, abs=1e-9)
    assert res.anchor_sd == pytest.approx(0.0, abs=1e-12)


def test_ti_increases_with_target_dependence():
    src, lgt, ft, gs, gt = _world()
    tis = [target_informativeness(f"m{a}", (lambda a=a: _Mixture(a)), src, gs,
                                  lgt, ft, gt, n_boot=40, seed=0).ti
           for a in (0.1, 0.5, 0.9)]
    assert tis[0] < tis[1] < tis[2]


def test_ti_is_bounded():
    src, lgt, ft, gs, gt = _world()
    for a in (0.0, 0.3, 0.7, 1.0):
        ti = target_informativeness("m", (lambda a=a: _Mixture(a)), src, gs,
                                    lgt, ft, gt, n_boot=20, seed=0).ti
        assert 0.0 <= ti <= 1.0


# ---------------------------------------------------------------- real estimators
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registry_estimator_yields_finite_sensitivities(name):
    src, lgt, ft, gs, gt = _world()
    fac = REGISTRY[name]["factory"]
    t_sd, _ = target_sensitivity(fac, src, lgt, ft, gt, n_boot=8, seed=0)
    a_sd, _ = anchor_sensitivity(fac, src, gs, lgt, ft, n_boot=8, seed=0)
    assert np.isfinite(t_sd) and t_sd >= 0
    assert np.isfinite(a_sd) and a_sd >= 0


def test_shuffling_the_target_does_not_move_permutation_invariant_estimators():
    """A sanity check on the audit itself: if a shuffle moved these, the
    machinery would be wrong rather than the estimator being interesting."""
    src, lgt, ft, gs, gt = _world()
    rng = np.random.default_rng(0)
    sur = removal_surrogates(lgt, ft, gt, rng)
    lg_s, ft_s = sur["shuffled"]
    # Tolerance is 1e-6, not 0: these are float32 arrays and summation is not
    # associative, so a reordered mean differs in the last few digits (~6e-8
    # observed). That is arithmetic, not sensitivity to the shuffle.
    for name in ("confidence", "entropy", "atc_mc", "doc"):
        est = REGISTRY[name]["factory"]().fit(src)
        assert est.estimate(lgt, ft) == pytest.approx(est.estimate(lg_s, ft_s), abs=1e-6)


def test_removal_surrogates_preserve_shape_and_destroy_variety():
    src, lgt, ft, gs, gt = _world()
    sur = removal_surrogates(lgt, ft, gt, np.random.default_rng(0))
    assert set(sur) == {"shuffled", "duplicated", "moment_matched_noise", "random_logits"}
    for lg, f in sur.values():
        assert lg.shape == lgt.shape
        assert f is None or f.shape == ft.shape
    # every row identical - checked directly rather than via std(), which
    # accumulates float32 error over 600 rows and returns ~3e-05 for a constant
    dup_lg, _ = sur["duplicated"]
    assert np.array_equal(dup_lg, np.repeat(dup_lg[:1], len(dup_lg), axis=0))


# ---------------------------------------------------------------- leakage firewall
def test_audit_functions_never_take_target_labels():
    """Negative control: there is no argument through which a target label could
    reach the audit, so it cannot leak one even by mistake."""
    for fn in (target_sensitivity, anchor_sensitivity, target_informativeness):
        params = set(inspect.signature(fn).parameters)
        assert not {"target_labels", "y_target", "truth", "true_accuracy"} & params


def test_estimator_interface_exposes_no_target_label_channel():
    sig = set(inspect.signature(Estimator.estimate).parameters)
    assert sig == {"self", "target_logits", "target_features"}


# ---------------------------------------------------------------- edge cases
def test_single_group_target_gives_zero_target_variance():
    src, lgt, ft, gs, _ = _world()
    one = np.array(["g0"] * len(lgt))
    sd, _ = target_sensitivity(REGISTRY["confidence"]["factory"], src, lgt, ft,
                               one, n_boot=10, seed=0)
    assert sd == pytest.approx(0.0, abs=1e-12)


def test_ti_is_nan_when_neither_input_moves_the_estimate():
    """A constant estimator has no variance from either side; TI is undefined and
    must be reported as NaN rather than silently defaulting to 0 or 1."""
    class _Const(Estimator):
        name = "const"
        needs_source = False

        def _estimate(self, logits, features) -> float:
            return 0.5

    src, lgt, ft, gs, gt = _world()
    res = target_informativeness("c", _Const, src, gs, lgt, ft, gt, n_boot=6, seed=0)
    assert np.isnan(res.ti)
