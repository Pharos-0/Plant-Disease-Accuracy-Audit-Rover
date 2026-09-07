"""The estimator suite (V4-V8).

Every method here is *imported*, not invented. Reproduction level and any
modification are recorded in ``docs/estimator_reproducibility.md``.

None of these functions accepts a target label. See ``base.py`` for the firewall.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .base import (
    Estimator,
    SourceStats,
    max_confidence,
    negative_entropy,
    normalised_entropy,
    softmax,
)

# ---------------------------------------------------------------------------
# V4 / V5 - trivial baselines. These are BASELINES, never presented as novel.
# ---------------------------------------------------------------------------


class MeanConfidence(Estimator):
    """Average maximum softmax probability. Needs no source calibration."""

    name = "confidence"
    needs_source = False

    def _estimate(self, logits: np.ndarray, features: np.ndarray | None) -> float:
        return float(max_confidence(logits).mean())


class MeanEntropy(Estimator):
    """One minus mean entropy, normalised by log K so it lands in [0, 1]."""

    name = "entropy"
    needs_source = False

    def _estimate(self, logits: np.ndarray, features: np.ndarray | None) -> float:
        return float(1.0 - normalised_entropy(logits).mean())


# ---------------------------------------------------------------------------
# V6 - Average Thresholded Confidence (Garg et al., ICLR 2022)
# ---------------------------------------------------------------------------


class ATC(Estimator):
    """Average Thresholded Confidence.

    Pick a threshold t on a confidence score so that the fraction of *source
    validation* points scoring below t equals the source validation error.
    The estimate on the target is then the fraction scoring at or above t.

    ``score="mc"`` uses maximum confidence, ``score="ne"`` negative entropy -
    the two variants reported in the paper.
    """

    def __init__(self, score: str = "mc") -> None:
        super().__init__()
        if score not in ("mc", "ne"):
            raise ValueError("score must be 'mc' or 'ne'")
        self.score = score
        self.name = f"atc_{score}"
        self.threshold_: float | None = None
        self.source_error_: float | None = None

    def _score(self, logits: np.ndarray) -> np.ndarray:
        return max_confidence(logits) if self.score == "mc" else negative_entropy(logits)

    def _fit(self, source: SourceStats) -> None:
        s = self._score(source.logits)
        self.source_error_ = source.error
        # t is the source-error quantile of the score distribution, so that
        # #{s < t} matches the number of source validation mistakes.
        self.threshold_ = float(np.quantile(s, source.error))

    def _estimate(self, logits: np.ndarray, features: np.ndarray | None) -> float:
        return float((self._score(logits) >= self.threshold_).mean())


# ---------------------------------------------------------------------------
# V7 - Difference of Confidences
# ---------------------------------------------------------------------------


class DoC(Estimator):
    """Difference of Confidences.

    Assumes the drop in mean confidence from source to target tracks the drop
    in accuracy:  acc_target ~= acc_source - (conf_source - conf_target).
    """

    name = "doc"

    def __init__(self) -> None:
        super().__init__()
        self.source_acc_: float | None = None
        self.source_conf_: float | None = None

    def _fit(self, source: SourceStats) -> None:
        self.source_acc_ = source.accuracy
        self.source_conf_ = float(max_confidence(source.logits).mean())

    def _estimate(self, logits: np.ndarray, features: np.ndarray | None) -> float:
        target_conf = float(max_confidence(logits).mean())
        return self.source_acc_ - (self.source_conf_ - target_conf)


# ---------------------------------------------------------------------------
# V8 - non-confidence estimators
# ---------------------------------------------------------------------------


class _SourceCalibratedScore(Estimator):
    """Shared machinery for scores that correlate with accuracy but are not one.

    Nuclear norm and MaNo both yield a *score*, not a probability. To turn a
    score into an accuracy we fit a linear map on bootstrap subsets of the
    source validation set - which uses source labels only, so the target stays
    unlabeled. The limitation is real and worth stating: a map fitted on source
    cannot know how the score behaves once the distribution moves, and that is
    part of what this benchmark measures.
    """

    n_calibration_subsets = 200
    subset_size = 200

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed
        self.slope_: float = 1.0
        self.intercept_: float = 0.0
        self.calibration_r_: float | None = None

    def _score_matrix(self, logits: np.ndarray) -> float:
        raise NotImplementedError

    def _fit(self, source: SourceStats) -> None:
        rng = np.random.default_rng(self.seed)
        n = len(source.logits)
        size = min(self.subset_size, n)
        scores, accs = [], []
        for _ in range(self.n_calibration_subsets):
            idx = rng.choice(n, size=size, replace=True)
            scores.append(self._score_matrix(source.logits[idx]))
            accs.append(float(source.correct[idx].mean()))
        scores_arr, accs_arr = np.asarray(scores), np.asarray(accs)
        if np.std(scores_arr) < 1e-12:
            self.slope_, self.intercept_ = 0.0, float(accs_arr.mean())
            self.calibration_r_ = 0.0
            return
        self.slope_, self.intercept_ = np.polyfit(scores_arr, accs_arr, 1)
        self.calibration_r_ = float(np.corrcoef(scores_arr, accs_arr)[0, 1])

    def _estimate(self, logits: np.ndarray, features: np.ndarray | None) -> float:
        return float(self.slope_ * self._score_matrix(logits) + self.intercept_)


class NuclearNorm(_SourceCalibratedScore):
    """Confidence & Dispersity (Deng et al., ICML 2023).

    The nuclear norm of the m x K softmax prediction matrix rewards being both
    confident (peaked rows) and dispersed (classes actually used). Normalised
    by sqrt(m * K) so sessions of different sizes stay comparable.
    """

    name = "nuclear_norm"

    def _score_matrix(self, logits: np.ndarray) -> float:
        p = softmax(logits)
        m, k = p.shape
        return float(np.linalg.norm(p, ord="nuc") / np.sqrt(m * k))


class MaNo(_SourceCalibratedScore):
    """MaNo (Xie et al., NeurIPS 2024) - matrix-norm score on normalised logits.

    Softmax-free: logits are recentred and passed through a bounded map before
    the L_p norm, which the paper motivates as avoiding softmax saturation
    under shift.
    """

    name = "mano"

    def __init__(self, seed: int = 0, p: int = 4) -> None:
        super().__init__(seed=seed)
        self.p = p

    def _score_matrix(self, logits: np.ndarray) -> float:
        z = logits - logits.mean(axis=1, keepdims=True)
        # bounded, softmax-free normalisation
        q = 1.0 / (1.0 + np.exp(-z))
        q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)
        m, k = q.shape
        return float((np.abs(q) ** self.p).sum() / (m * k)) ** (1.0 / self.p)


class COT(Estimator):
    """Confidence Optimal Transport (Lu et al., NeurIPS 2023).

    Transports the empirical distribution of softmax vectors onto the class
    marginal taken from the source, with cost ``1 - p_ik``. The optimal cost is
    the expected 0-1 error under the best coupling, so the accuracy estimate is
    one minus that cost.

    The target label marginal is not observable, so the source marginal stands
    in for it. Under prevalence shift that substitution is wrong by
    construction - which is exactly the failure mode H2 predicts.
    """

    name = "cot"

    def __init__(self) -> None:
        super().__init__()
        self.marginal_: np.ndarray | None = None

    def _fit(self, source: SourceStats) -> None:
        self.marginal_ = source.class_marginal

    def _estimate(self, logits: np.ndarray, features: np.ndarray | None) -> float:
        p = softmax(logits)
        m, k = p.shape
        q = self.marginal_
        if q is None or len(q) != k:
            q = np.full(k, 1.0 / k)

        cost = (1.0 - p).ravel()  # variable order: (i, k) row-major

        # row marginals: each sample carries mass 1/m
        rows = sparse.kron(sparse.eye(m, format="csr"), np.ones((1, k)))
        # column marginals: class k receives q_k
        cols = sparse.kron(np.ones((1, m)), sparse.eye(k, format="csr"))
        a_eq = sparse.vstack([rows, cols], format="csr")
        b_eq = np.concatenate([np.full(m, 1.0 / m), q])

        res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        if not res.success:
            # fall back to the independent coupling rather than inventing a number
            return float((p * q).sum() / m)
        return float(1.0 - res.fun)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, dict] = {
    "confidence": {
        "factory": MeanConfidence,
        "kind": "baseline",
        "version": "V4",
        "reference": "mean max-softmax; standard baseline",
    },
    "entropy": {
        "factory": MeanEntropy,
        "kind": "baseline",
        "version": "V5",
        "reference": "mean normalised entropy; standard baseline",
    },
    "atc_mc": {
        "factory": lambda: ATC("mc"),
        "kind": "method",
        "version": "V6",
        "reference": "Garg et al., ICLR 2022 (ATC-MC)",
    },
    "atc_ne": {
        "factory": lambda: ATC("ne"),
        "kind": "method",
        "version": "V6",
        "reference": "Garg et al., ICLR 2022 (ATC-NE)",
    },
    "doc": {
        "factory": DoC,
        "kind": "method",
        "version": "V7",
        "reference": "Guillory et al., ICCV 2021 (Difference of Confidences)",
    },
    "nuclear_norm": {
        "factory": NuclearNorm,
        "kind": "method",
        "version": "V8",
        "reference": "Deng et al., ICML 2023 (Confidence & Dispersity)",
    },
    "mano": {
        "factory": MaNo,
        "kind": "method",
        "version": "V8",
        "reference": "Xie et al., NeurIPS 2024 (MaNo)",
    },
    "cot": {
        "factory": COT,
        "kind": "method",
        "version": "V8",
        "reference": "Lu et al., NeurIPS 2023 (Confidence Optimal Transport)",
    },
}

ALL_ESTIMATORS = list(REGISTRY)


def build(name: str) -> Estimator:
    return REGISTRY[name]["factory"]()


def build_all() -> dict[str, Estimator]:
    return {n: build(n) for n in ALL_ESTIMATORS}
