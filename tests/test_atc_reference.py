"""Check our ATC against the official release.

MASTER_PROMPT section 8 requires the official implementation where one exists.
``find_atc_threshold`` is a vectorised transcription of ``find_ATC_threshold``
from github.com/saurabhgarg1996/ATC_code; this test loads that file directly
and asserts the two agree, which is what lets the reproducibility table claim
level L3 for ATC rather than "written from the paper".

Skipped when the reference checkout is absent, so CI stays green on a machine
that has not cloned it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

REFERENCE = Path(r"C:\ds\_refrepos\ATC_code\ATC_helper.py")


def load_reference():
    if not REFERENCE.exists():
        pytest.skip(f"official ATC reference not present at {REFERENCE}")
    spec = importlib.util.spec_from_file_location("atc_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_threshold_matches_official(seed: int) -> None:
    from estimators.methods import find_atc_threshold

    ref = load_reference()
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 800))
    scores = rng.normal(0, 1, n)
    # vary the error rate so the sweep is exercised across regimes
    correct = (rng.random(n) < rng.uniform(0.5, 0.99)).astype(int)

    ref_min, ref_thres = ref.find_ATC_threshold(scores.copy(), correct.copy())
    our_min, our_thres = find_atc_threshold(scores, correct)

    assert our_min == pytest.approx(ref_min), "|fp - fn| disagrees with the reference"
    assert our_thres == pytest.approx(ref_thres), "threshold disagrees with the reference"


def test_atc_accuracy_matches_official() -> None:
    """End-to-end: fit on source, estimate on target, compare with the reference."""
    from estimators.base import SourceStats, max_confidence
    from estimators.methods import ATC

    ref = load_reference()
    rng = np.random.default_rng(42)
    k = 3

    def make(n: int, sep: float):
        y = rng.integers(0, k, n)
        logits = rng.normal(0, 1, (n, k))
        logits[np.arange(n), y] += sep
        return logits, y

    src_logits, src_y = make(2000, 3.0)
    tgt_logits, _ = make(1000, 1.5)

    source = SourceStats(logits=src_logits, labels=src_y)
    est = ATC("mc").fit(source)
    ours = est.estimate(tgt_logits)

    src_conf = max_confidence(src_logits)
    _, thres = ref.find_ATC_threshold(src_conf, (src_logits.argmax(1) == src_y).astype(int))
    theirs = ref.get_ATC_acc(thres, max_confidence(tgt_logits)) / 100.0

    assert ours == pytest.approx(theirs, abs=1e-12)


def test_entropy_score_matches_official() -> None:
    """Our negative-entropy score is the reference's ``get_entropy``."""
    from estimators.base import negative_entropy, softmax

    ref = load_reference()
    rng = np.random.default_rng(7)
    logits = rng.normal(0, 2, (500, 5))
    probs = softmax(logits)
    assert np.allclose(negative_entropy(logits), ref.get_entropy(probs), atol=1e-9)


def test_doc_matches_official_formula() -> None:
    """DoC estimate = source acc + (mean target conf - mean source conf)."""
    from estimators.base import SourceStats, max_confidence
    from estimators.methods import DoC

    rng = np.random.default_rng(3)
    k = 4
    y = rng.integers(0, k, 1500)
    src_logits = rng.normal(0, 1, (1500, k))
    src_logits[np.arange(1500), y] += 2.5
    tgt_logits = rng.normal(0, 1, (900, k))

    source = SourceStats(logits=src_logits, labels=y)
    ours = DoC().fit(source).estimate(tgt_logits)

    # official: v1acc + get_doc(conf_source, conf_target)
    v1acc = float((src_logits.argmax(1) == y).mean())
    doc = float(max_confidence(tgt_logits).mean() - max_confidence(src_logits).mean())
    assert ours == pytest.approx(np.clip(v1acc + doc, 0, 1), abs=1e-12)


def test_cot_matches_official_optimal_transport() -> None:
    """Our semi-discrete COT equals the official n x n EMD formulation.

    The official implementation (github.com/luyuzhe111/COT, ``run_estimation.py``)
    builds one-hot vectors sampled from the source label distribution and solves
    a balanced EMD with cost ``||e_k - p||_1 / 2``. That cost equals ``1 - p_k``
    exactly, which is the cost we use; we solve the equivalent semi-discrete
    problem against the exact marginal instead of a sampled one, which is the
    same estimator with the sampling noise removed.
    """
    ot = pytest.importorskip("ot")

    from estimators.base import SourceStats, softmax
    from estimators.methods import COT

    rng = np.random.default_rng(0)
    k = 3
    for _ in range(3):
        n_src, n_tgt = 1200, 300
        ys = rng.integers(0, k, n_src)
        src_logits = rng.normal(0, 1, (n_src, k))
        src_logits[np.arange(n_src), ys] += 2.5
        tgt_logits = rng.normal(0, 1, (n_tgt, k))
        tgt_logits[np.arange(n_tgt), rng.integers(0, k, n_tgt)] += 1.2

        source = SourceStats(logits=src_logits, labels=ys)
        ours = COT().fit(source).estimate(tgt_logits)

        # one-hot set realising the source marginal exactly (no sampling noise)
        q = source.class_marginal
        counts = np.floor(q * n_tgt).astype(int)
        while counts.sum() < n_tgt:
            counts[np.argmax(q * n_tgt - counts)] += 1
        iid = np.repeat(np.eye(k), counts, axis=0)

        probs = softmax(tgt_logits)
        cost = np.abs(iid[:, None, :] - probs[None, :, :]).sum(-1) / 2.0
        w = np.ones(n_tgt) / n_tgt
        official_error = ot.emd2(w, w, cost, numItermax=int(1e8))

        assert ours == pytest.approx(1.0 - official_error, abs=1e-3)
