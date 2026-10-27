"""Estimator interface and the label-free contract.

LEAKAGE FIREWALL (study protocol). Everything in ``src/estimators/``
may see source validation labels and source/target logits and features. It may
**never** see target labels. The contract is structural, not a convention:
``Estimator.estimate`` is handed target logits and features and nothing else,
so there is no argument through which a target label could arrive.

Scoring an estimate against the truth happens in ``src/evaluation/``, strictly
after estimates are written to disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def max_confidence(logits: np.ndarray) -> np.ndarray:
    return softmax(logits).max(axis=1)


def negative_entropy(logits: np.ndarray) -> np.ndarray:
    """Higher means more confident, matching ATC-NE's score convention."""
    p = softmax(logits)
    return np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)


def normalised_entropy(logits: np.ndarray) -> np.ndarray:
    p = softmax(logits)
    h = -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)
    return h / np.log(p.shape[1])


@dataclass
class SourceStats:
    """Everything an estimator may learn from the source side.

    Deliberately holds no target quantity. ``labels`` here are *source*
    validation labels, which the protocol permits.
    """

    logits: np.ndarray
    labels: np.ndarray
    features: np.ndarray | None = None

    @property
    def probs(self) -> np.ndarray:
        return softmax(self.logits)

    @property
    def preds(self) -> np.ndarray:
        return self.logits.argmax(axis=1)

    @property
    def correct(self) -> np.ndarray:
        return (self.preds == self.labels).astype(np.float64)

    @property
    def accuracy(self) -> float:
        return float(self.correct.mean())

    @property
    def error(self) -> float:
        return 1.0 - self.accuracy

    @property
    def class_marginal(self) -> np.ndarray:
        k = self.logits.shape[1]
        counts = np.bincount(self.labels, minlength=k).astype(np.float64)
        return counts / counts.sum()


class Estimator:
    """Base class. Subclasses override `_fit` and `_estimate`."""

    name: str = "base"
    #: set False for estimators that need no source calibration
    needs_source: bool = True

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, source: SourceStats) -> "Estimator":
        self._fit(source)
        self._fitted = True
        return self

    def estimate(
        self, target_logits: np.ndarray, target_features: np.ndarray | None = None
    ) -> float:
        """Estimated accuracy on an unlabeled target set, clipped to [0, 1].

        There is no target-label parameter, and there never may be.
        """
        if self.needs_source and not self._fitted:
            raise RuntimeError(f"{self.name} must be fitted on source stats before use")
        if target_logits.ndim != 2 or len(target_logits) == 0:
            raise ValueError(f"{self.name}: target_logits must be a non-empty 2-D array")
        value = self._estimate(target_logits, target_features)
        if not np.isfinite(value):
            raise ValueError(f"{self.name} produced a non-finite estimate: {value}")
        return float(np.clip(value, 0.0, 1.0))

    # -- subclass hooks ----------------------------------------------------
    def _fit(self, source: SourceStats) -> None:  # noqa: B027 - optional hook
        return None

    def _estimate(self, logits: np.ndarray, features: np.ndarray | None) -> float:
        raise NotImplementedError


@dataclass
class EstimatorSpec:
    """Registry entry, carrying the honesty metadata section 8 requires."""

    factory: type[Estimator]
    kind: str  # "baseline" or "method"
    reference: str
    notes: str = ""
    kwargs: dict = field(default_factory=dict)
