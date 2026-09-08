"""RQ5 sampling and estimator tests (brief section 36).

Covers photograph-level sampling, duplicate prevention, seed determinism,
labelled-only access, large-N convergence, and the structural facts the
pre-registration depends on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sessions.labelled_sample import (
    labelled_estimate,
    photograph_index,
    sample_photographs,
    seed_for,
    wilson_interval,
)


# --------------------------------------------------------------------------
# synthetic fixture mirroring Arm D's bimodal structure
# --------------------------------------------------------------------------


@pytest.fixture
def toy():
    """40 photographs: 20 singles (like healthy) and 20 with 8 crops (like blight)."""
    groups, correct = [], []
    rng = np.random.default_rng(0)
    for i in range(20):
        groups.append(f"ts:S{i:03d}")
        correct.append(rng.integers(0, 2))
    for i in range(20):
        for _ in range(8):
            groups.append(f"ts:B{i:03d}")
            correct.append(rng.integers(0, 2))
    return np.array(groups), np.array(correct)


def test_photograph_index_partitions_every_crop(toy):
    g, c = toy
    idx = photograph_index(g)
    assert len(idx) == 40
    allidx = np.sort(np.concatenate(list(idx.values())))
    assert np.array_equal(allidx, np.arange(len(g))), "index must cover every crop exactly once"


def test_sample_returns_whole_photographs(toy):
    g, _ = toy
    idx = sample_photographs(g, 10, seed=1)
    chosen = set(g[idx])
    assert len(chosen) == 10
    # every crop of each chosen photograph must be present
    for p in chosen:
        assert set(np.flatnonzero(g == p)) <= set(idx), f"photograph {p} was split"


def test_no_duplicate_photograph_in_a_sample(toy):
    g, _ = toy
    for s in range(25):
        idx = sample_photographs(g, 12, seed=s)
        assert len(idx) == len(set(idx)), "a crop index appeared twice"
        assert len(set(g[idx])) == 12, "a photograph appeared twice"


def test_seeds_are_deterministic_and_distinct(toy):
    g, _ = toy
    a = sample_photographs(g, 8, seed=42)
    b = sample_photographs(g, 8, seed=42)
    c = sample_photographs(g, 8, seed=43)
    assert np.array_equal(a, b), "same seed must give the same sample"
    assert not np.array_equal(a, c), "different seed must give a different sample"


def test_seed_convention_matches_preregistration():
    assert seed_for(0) == 20260909
    assert seed_for(7) == 20260916
    assert len({seed_for(i) for i in range(100)}) == 100


def test_estimator_reads_only_sampled_labels(toy):
    """Corrupting the unsampled entries must not change the estimate."""
    g, c = toy
    idx = sample_photographs(g, 10, seed=5)
    est_before = labelled_estimate(c, g, 10, seed=5)

    poisoned = c.copy()
    mask = np.ones(len(c), bool)
    mask[idx] = False
    poisoned[mask] = 1 - poisoned[mask]  # flip everything NOT sampled
    est_after = labelled_estimate(poisoned, g, 10, seed=5)

    assert est_before.accuracy == est_after.accuracy, (
        "the estimate changed when unsampled labels changed - it is reading held-out truth"
    )


def test_large_n_converges_to_full_pool_accuracy(toy):
    g, c = toy
    est = labelled_estimate(c, g, n_photographs=40, seed=0)
    assert est.accuracy == pytest.approx(c.mean())
    assert est.n_crops == len(c)


def test_asking_for_too_many_photographs_raises(toy):
    g, c = toy
    with pytest.raises(ValueError, match="only 40"):
        sample_photographs(g, 41, seed=0)


def test_crop_count_is_random_because_clusters_are_unequal(toy):
    """A fixed photograph budget does not fix the number of labelled crops."""
    g, c = toy
    counts = {labelled_estimate(c, g, 10, seed=s).n_crops for s in range(30)}
    assert len(counts) > 1, "cluster sizes are unequal, so crop count must vary"


def test_wilson_interval_basic_properties():
    lo, hi = wilson_interval(5, 10)
    assert 0 <= lo < 0.5 < hi <= 1
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and lo < 1.0
    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0.0) and hi > 0.0
    # wider for smaller samples
    w_small = np.subtract(*reversed(wilson_interval(5, 10)))
    w_large = np.subtract(*reversed(wilson_interval(50, 100)))
    assert w_small > w_large
    assert wilson_interval(0, 0) == pytest.approx((float("nan"), float("nan")), nan_ok=True)


# --------------------------------------------------------------------------
# the real Arm-D structure the pre-registration relies on
# --------------------------------------------------------------------------


def arm_d():
    from inference.cache import CacheKey, load_features, load_labels

    key = CacheKey("V2", "resnet50", 0, "tomatovillage", "SHARED_3")
    if not key.features_path.exists():
        pytest.skip("Arm-D cache not present")
    f = load_features(key)
    return f["group_ids"], f["logits"], load_labels(key)


def test_arm_d_independent_units_are_as_preregistered():
    g, logits, y = arm_d()
    assert len(g) == 1616, "crop count changed"
    assert len(np.unique(g)) == 391, "photograph count changed"
    assert all(str(x).startswith("ts:") for x in np.unique(g)), "an unparsed group appeared"


def test_arm_d_photographs_have_a_single_true_class():
    """Photograph-level labelling is only well defined if this holds."""
    g, _, y = arm_d()
    for p in np.unique(g):
        assert len(np.unique(y[g == p])) == 1, f"photograph {p} spans multiple classes"


def test_arm_d_cluster_sizes_are_bimodal():
    """The structural fact recorded in the pre-registration, section B.1."""
    g, _, _ = arm_d()
    sizes = np.array([int((g == p).sum()) for p in np.unique(g)])
    assert set(np.unique(sizes)) == {1, 8}
    assert int((sizes == 1).sum()) == 216
    assert int((sizes == 8).sum()) == 175


def test_arm_d_large_n_convergence_on_real_data():
    """Section 28: at the full budget the estimate must equal the pool accuracy."""
    g, logits, y = arm_d()
    correct = (logits.argmax(1) == y).astype(int)
    est = labelled_estimate(correct, g, n_photographs=391, seed=0)
    assert est.n_crops == 1616
    assert est.accuracy == pytest.approx(float(correct.mean()), abs=1e-12)
