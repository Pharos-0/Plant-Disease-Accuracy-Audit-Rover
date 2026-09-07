"""The session gate, and the per-image prohibition it has to enforce."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from estimators.base import SourceStats
from estimators.methods import build
from rover.session_gate import (
    GateConfig,
    GateDecision,
    PerImageQueryError,
    SessionGate,
)


def fitted_estimator(name: str = "atc_mc", sep: float = 3.0):
    rng = np.random.default_rng(0)
    n, k = 2000, 3
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += sep
    return build(name).fit(SourceStats(logits=logits, labels=y))


def make_logits(n: int, sep: float, seed: int = 1, k: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += sep
    return logits


def test_per_image_query_is_refused() -> None:
    """Section 2: a dataset-level estimator must never answer per-image."""
    gate = SessionGate(fitted_estimator())
    for method in ("is_image_correct", "estimate_for_image", "should_spray_this_leaf"):
        with pytest.raises(PerImageQueryError):
            getattr(gate, method)(np.zeros(3))


def test_gate_withholds_until_enough_observations() -> None:
    gate = SessionGate(fitted_estimator(), GateConfig(min_session_size=100))
    gate.observe_batch(make_logits(50, sep=3.0))
    v = gate.verdict()
    assert v.decision is GateDecision.INSUFFICIENT
    assert v.estimated_accuracy is None
    assert v.spray_enabled is False, "an under-filled session must never arm the sprayer"


def test_gate_operates_on_a_clean_session() -> None:
    gate = SessionGate(fitted_estimator(), GateConfig(min_session_size=100, tau_operate=0.80))
    gate.observe_batch(make_logits(200, sep=3.0, seed=5))
    v = gate.verdict()
    assert v.decision is GateDecision.OPERATE
    assert v.spray_enabled is True
    assert 0.0 <= v.estimated_accuracy <= 1.0


def test_gate_withholds_on_a_degraded_session() -> None:
    gate = SessionGate(
        fitted_estimator(), GateConfig(min_session_size=100, tau_operate=0.80, tau_review=0.65)
    )
    gate.observe_batch(make_logits(200, sep=0.05, seed=7))
    v = gate.verdict()
    assert v.decision in (GateDecision.WITHHOLD, GateDecision.REVIEW)
    assert v.spray_enabled is False


def test_review_band_is_reachable() -> None:
    """Between the two thresholds the gate asks for a human, not for a spray."""
    gate = SessionGate(
        fitted_estimator(), GateConfig(min_session_size=50, tau_operate=0.99, tau_review=0.10)
    )
    gate.observe_batch(make_logits(150, sep=1.5, seed=11))
    v = gate.verdict()
    assert v.decision is GateDecision.REVIEW
    assert v.spray_enabled is False


def test_reset_clears_the_session() -> None:
    gate = SessionGate(fitted_estimator(), GateConfig(min_session_size=10))
    gate.observe_batch(make_logits(20, sep=3.0))
    assert len(gate) == 20
    gate.reset()
    assert len(gate) == 0
    assert gate.verdict().decision is GateDecision.INSUFFICIENT


def test_verdict_record_is_serialisable_and_complete() -> None:
    gate = SessionGate(fitted_estimator(), GateConfig(min_session_size=10))
    gate.observe_batch(make_logits(120, sep=3.0))
    rec = gate.verdict().as_record()
    for field in (
        "timestamp", "decision", "estimated_accuracy", "n_observations",
        "estimator", "tau_operate", "spray_enabled",
    ):
        assert field in rec
    import json

    json.dumps(rec)  # must not raise


def test_gate_never_receives_labels() -> None:
    """The gate's public surface has no route for a label."""
    import inspect

    for name in ("observe", "observe_batch", "verdict"):
        params = set(inspect.signature(getattr(SessionGate, name)).parameters) - {"self"}
        assert not (params & {"y", "labels", "target", "truth"}), (
            f"SessionGate.{name} exposes a label-shaped parameter"
        )
