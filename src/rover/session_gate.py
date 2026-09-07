"""Session-level deployment gate.

THE CONCEPTUAL CONSTRAINT (MASTER_PROMPT section 2). ATC, DoC, MaNo and COT
estimate accuracy over a **dataset / session**. They do not produce a per-image
correctness probability. This module therefore refuses to answer questions about
a single image:

    session of N unlabeled images -> classifier -> logits
        -> dataset-level estimator -> estimated session accuracy
        -> gate -> operate / withhold / request human review

The green-region rule inherited from the delivered rover decides *where* to
spray. This gate decides *whether the rover should be operating at all*. The two
levels are separate and must stay separate.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimators.base import Estimator


class GateDecision(str, Enum):
    OPERATE = "operate"
    WITHHOLD = "withhold"
    REVIEW = "request_human_review"
    INSUFFICIENT = "insufficient_observations"


class PerImageQueryError(RuntimeError):
    """Raised on any attempt to ask this gate about one image.

    Per-image correctness is selective prediction - a different literature and
    a different guarantee. Confusing the two is the error section 2 forbids.
    """


@dataclass
class GateConfig:
    #: minimum observations before the estimate is trusted at all. Defaults to
    #: 100 but the defensible value is N* measured in the benchmark, not a guess.
    min_session_size: int = 100
    #: operate only if the estimated session accuracy clears this
    tau_operate: float = 0.80
    #: below this, withhold outright; between the two, ask for a human
    tau_review: float = 0.65
    #: rolling buffer capacity
    capacity: int = 2000


@dataclass
class GateVerdict:
    decision: GateDecision
    estimated_accuracy: float | None
    n_observations: int
    estimator: str
    config: GateConfig
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def spray_enabled(self) -> bool:
        """The single flag the rover reads. Only OPERATE arms the sprayer."""
        return self.decision is GateDecision.OPERATE

    def as_record(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "decision": self.decision.value,
            "estimated_accuracy": self.estimated_accuracy,
            "n_observations": self.n_observations,
            "estimator": self.estimator,
            "tau_operate": self.config.tau_operate,
            "tau_review": self.config.tau_review,
            "min_session_size": self.config.min_session_size,
            "spray_enabled": self.spray_enabled,
        }


class SessionGate:
    """Accumulates logits over a session and gates operation on the estimate.

    The estimator must already be fitted on source validation statistics; this
    class never sees a label of any kind.
    """

    def __init__(self, estimator: Estimator, config: GateConfig | None = None) -> None:
        self.estimator = estimator
        self.config = config or GateConfig()
        self._buffer: deque[np.ndarray] = deque(maxlen=self.config.capacity)
        self._features: deque[np.ndarray] = deque(maxlen=self.config.capacity)

    # -- accumulation ------------------------------------------------------
    def observe(self, logits: np.ndarray, features: np.ndarray | None = None) -> None:
        """Add one crop's logits to the session buffer.

        Adding an observation is *not* a decision about that observation. The
        gate only ever speaks about the session as a whole.
        """
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        self._buffer.append(logits)
        if features is not None:
            self._features.append(np.asarray(features, dtype=np.float64).reshape(-1))

    def observe_batch(self, logits: np.ndarray, features: np.ndarray | None = None) -> None:
        logits = np.atleast_2d(np.asarray(logits, dtype=np.float64))
        for i, row in enumerate(logits):
            self.observe(row, None if features is None else np.asarray(features)[i])

    def reset(self) -> None:
        self._buffer.clear()
        self._features.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    # -- the prohibition, enforced ----------------------------------------
    def is_image_correct(self, *args, **kwargs):  # noqa: D401, ANN201
        raise PerImageQueryError(
            "A dataset-level accuracy estimator cannot say whether one image is "
            "correctly classified. That is selective prediction, a different "
            "method family with different guarantees. Use the session verdict."
        )

    estimate_for_image = is_image_correct
    should_spray_this_leaf = is_image_correct

    # -- the decision ------------------------------------------------------
    def verdict(self) -> GateVerdict:
        n = len(self._buffer)
        if n < self.config.min_session_size:
            return GateVerdict(
                decision=GateDecision.INSUFFICIENT,
                estimated_accuracy=None,
                n_observations=n,
                estimator=self.estimator.name,
                config=self.config,
            )

        logits = np.vstack(self._buffer)
        feats = np.vstack(self._features) if len(self._features) == n else None
        est = self.estimator.estimate(logits, feats)

        if est >= self.config.tau_operate:
            decision = GateDecision.OPERATE
        elif est >= self.config.tau_review:
            decision = GateDecision.REVIEW
        else:
            decision = GateDecision.WITHHOLD

        return GateVerdict(
            decision=decision,
            estimated_accuracy=est,
            n_observations=n,
            estimator=self.estimator.name,
            config=self.config,
        )
