"""GdScore: verify our implementation against the official computation.

The official release (github.com/Renchunzi-Xie/GdScore, `algs/gdscore.py`) does:

    loss = CrossEntropyLoss(outputs, pseudo_labels)
    loss.backward()
    score = torch.norm(base_model.fc.weight.grad, p=norm_type)

averaged over batches. We compute the same gradient in closed form,
``(1/B) R.T @ H``, which avoids autograd entirely. These tests prove the two
agree, which is what justifies the L3 claim in
``docs/estimator_reproducibility.md``.

Torch is used *here*, in the test, never inside ``src/estimators/`` - the
firewall permits numpy only there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from estimators.methods import GdScore

torch = pytest.importorskip("torch")


def reference_gdscore(logits_np, feats_np, norm_type=0.3, batch_size=128, drop_last=True):
    """The official path, using autograd on a real linear head.

    Threshold is fixed at 0 so no random pseudo-label substitution occurs and
    the comparison is deterministic.
    """
    import torch.nn as nn

    scores = []
    for start in range(0, len(logits_np), batch_size):
        lg = torch.tensor(logits_np[start:start + batch_size], dtype=torch.float64)
        h = torch.tensor(feats_np[start:start + batch_size], dtype=torch.float64)
        if drop_last and len(lg) < batch_size:
            continue
        k, d = lg.shape[1], h.shape[1]

        # a linear head that reproduces these logits exactly is not needed - the
        # gradient wrt W depends only on (softmax(logits) - onehot) and h - but
        # we build one anyway so autograd does the real work.
        head = nn.Linear(d, k, bias=True).double()
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.zeros(k, dtype=torch.float64))
        out = head(h) + lg  # logits are what the official forward produced
        with torch.no_grad():
            pseudo = torch.softmax(out, 1).argmax(1)
        head.zero_grad()
        nn.CrossEntropyLoss()(out, pseudo).backward()
        scores.append(torch.norm(head.weight.grad, p=norm_type).item())
    return float(np.mean(scores))


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n, d, k = 400, 64, 3
    feats = rng.normal(0, 1, (n, d))
    logits = rng.normal(0, 2, (n, k))
    return logits, feats


def test_matches_autograd_reference(data):
    """The closed-form gradient must reproduce the official autograd path."""
    logits, feats = data
    g = GdScore(threshold=0.0, norm_type=0.3, batch_size=128, drop_last=True)
    ours = g.raw_score(logits, feats)
    theirs = reference_gdscore(logits, feats, norm_type=0.3, batch_size=128)
    assert ours == pytest.approx(theirs, rel=1e-6), f"ours={ours} reference={theirs}"


@pytest.mark.parametrize("norm_type", [0.3, 0.5, 1.0, 2.0])
def test_matches_reference_across_norm_types(data, norm_type):
    logits, feats = data
    ours = GdScore(threshold=0.0, norm_type=norm_type, batch_size=128).raw_score(logits, feats)
    theirs = reference_gdscore(logits, feats, norm_type=norm_type, batch_size=128)
    assert ours == pytest.approx(theirs, rel=1e-6)


def test_pnorm_matches_torch_for_fractional_p():
    """p=0.3 is a fractional entrywise quantity, not the Frobenius norm."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (3, 64))
    for p in (0.3, 0.5, 1.0, 2.0):
        assert GdScore._pnorm(a, p) == pytest.approx(
            torch.norm(torch.tensor(a), p=p).item(), rel=1e-9
        )


def test_default_is_the_official_fractional_norm_not_frobenius():
    g = GdScore()
    assert g.norm_type == 0.3, "official scripts use --norm_type 0.3"
    assert g.threshold == 0.5, "official scripts use --threshold 0.5"
    assert g.batch_size == 128, "official scripts use --batch_size 128"
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, (3, 2048))
    assert GdScore._pnorm(a, 0.3) > 1e6 * GdScore._pnorm(a, 2.0), (
        "p=0.3 and Frobenius differ by orders of magnitude; they are not interchangeable"
    )


def test_random_pseudo_labels_are_seeded(data):
    """Below the threshold the official code substitutes RANDOM labels."""
    logits, feats = data
    a = GdScore(threshold=1.0, gd_seed=1).raw_score(logits, feats)
    b = GdScore(threshold=1.0, gd_seed=1).raw_score(logits, feats)
    c = GdScore(threshold=1.0, gd_seed=2).raw_score(logits, feats)
    assert a == b, "same seed must give the same score"
    assert a != c, "different seed must give a different score"


def test_threshold_zero_disables_substitution(data):
    logits, feats = data
    a = GdScore(threshold=0.0, gd_seed=1).raw_score(logits, feats)
    b = GdScore(threshold=0.0, gd_seed=999).raw_score(logits, feats)
    assert a == b, "with threshold 0 no label is randomised, so the seed is irrelevant"


def test_score_is_batch_size_dependent(data):
    """A documented property, not a bug - it is why GdScore stays out of the N-sweep."""
    logits, feats = data
    scores = {bs: GdScore(threshold=0.0, batch_size=bs, drop_last=False).raw_score(logits, feats)
              for bs in (25, 64, 128, 400)}
    assert max(scores.values()) > 2 * min(scores.values()), (
        f"expected strong batch-size dependence, got {scores}"
    )


def test_requires_features(data):
    logits, _ = data
    with pytest.raises(ValueError, match="features"):
        GdScore(threshold=0.0).raw_score(logits, None)


def test_raises_when_no_full_batch_available(data):
    logits, feats = data
    g = GdScore(threshold=0.0, batch_size=1000, drop_last=True)
    with pytest.raises(ValueError, match="no full batch"):
        g.raw_score(logits[:100], feats[:100])


def test_length_mismatch_is_rejected(data):
    logits, feats = data
    with pytest.raises(ValueError, match="same length"):
        GdScore(threshold=0.0).raw_score(logits, feats[:10])
