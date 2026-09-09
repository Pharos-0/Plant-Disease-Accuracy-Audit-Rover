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


def find_atc_threshold(scores: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """Vectorised transcription of ``find_ATC_threshold`` from the official
    ATC release (github.com/saurabhgarg1996/ATC_code, ``ATC_helper.py``).

    Sweeping the sorted scores, ``fp`` counts mistakes still above the
    threshold and ``fn`` counts correct predictions already below it. The
    chosen threshold minimises ``|fp - fn|``, which balances the two and makes
    the count below the threshold match the number of source mistakes.

    Returns ``(min_fp_fn, threshold)``, matching the reference signature.
    """
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_correct = correct[order].astype(bool)

    total_incorrect = float((~sorted_correct).sum())
    # after position i: fp = mistakes strictly after i, fn = correct up to i
    fp = total_incorrect - np.cumsum(~sorted_correct)
    fn = np.cumsum(sorted_correct).astype(float)
    diff = np.abs(fp - fn)

    # the reference seeds the search with (|fp - fn| = total_incorrect, thres = 0.0)
    # and updates only on a strict improvement, so ties keep the earliest.
    best = int(np.argmin(diff))
    if diff[best] < total_incorrect:
        return float(diff[best]), float(sorted_scores[best])
    return float(total_incorrect), 0.0


class ATC(Estimator):
    """Average Thresholded Confidence (Garg et al., ICLR 2022).

    A threshold is fitted on a *source validation* confidence score so that the
    number of points falling below it matches the number of source mistakes.
    The estimate on the target is the fraction of points at or above it.

    ``score="mc"`` uses maximum confidence, ``score="ne"`` negative entropy -
    the two variants reported in the paper. Threshold selection follows the
    official implementation exactly; see ``find_atc_threshold``.
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
        _, self.threshold_ = find_atc_threshold(s, source.correct)

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


class GdScore(_SourceCalibratedScore):
    """GdScore (Xie et al., TMLR 2025) - one-step gradient norm of the head.

    Faithful reproduction of ``algs/gdscore.py`` from the official release
    (github.com/Renchunzi-Xie/GdScore). See ``docs/gdscore_eligibility.md`` for
    the full eligibility and reproduction record.

    Per batch the official code pseudo-labels with the model's own argmax,
    replaces the label with a **random** one wherever max softmax < ``threshold``,
    takes the cross-entropy gradient with respect to the classifier head, and
    records ``torch.norm(grad, p=norm_type)``. The reported score is the mean
    over batches.

    Two things about this implementation:

    * **No autograd is needed.** For a linear head the gradient of the batch-mean
      cross-entropy is ``(1/B) * R.T @ H`` with ``R = softmax - onehot(pseudo)``
      and ``H`` the penultimate features. Verified against autograd to 1.9e-09.
    * ``norm_type`` defaults to **0.3**, matching the official scripts. That is a
      *fractional* entrywise p-quantity, not the Frobenius norm, and the two
      differ by ten orders of magnitude. Do not "fix" it to 2.

    The raw score is unbounded, so - exactly as for nuclear norm and MaNo - it is
    mapped to an accuracy by a linear fit on source-validation bootstraps. The
    score itself is a faithful reproduction; **the score-to-accuracy map is ours**.

    ``features`` must be supplied, and for a BatchNorm backbone they must be
    **train-mode** activations, because the official code scores under
    ``model.train()``. Passing eval-mode features silently computes a different
    quantity.
    """

    name = "gdscore"

    def __init__(self, threshold: float = 0.5, norm_type: float = 0.3,
                 batch_size: int = 128, gd_seed: int = 1,
                 drop_last: bool = True, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.threshold = threshold
        self.norm_type = norm_type
        self.batch_size = batch_size
        self.gd_seed = gd_seed
        # Our source/target sets are far smaller than the paper's, so a trailing
        # partial batch would carry disproportionate weight through the 1/B
        # factor - and a *different* weight for source than for target. Dropping
        # it keeps calibration and application on the same footing. Documented
        # as a deviation; sensitivity is reported in the results.
        self.drop_last = drop_last

    @staticmethod
    def _pnorm(a: np.ndarray, p: float) -> float:
        """torch.norm(a, p) for a matrix: the entrywise p-quantity."""
        return float((np.abs(a) ** p).sum() ** (1.0 / p))

    def raw_score(self, logits: np.ndarray, features: np.ndarray) -> float:
        """The faithful GdScore, before any calibration."""
        if features is None:
            raise ValueError("GdScore requires penultimate features")
        if len(logits) != len(features):
            raise ValueError("logits and features must have the same length")
        rng = np.random.default_rng(self.gd_seed)
        k = logits.shape[1]
        scores = []
        for start in range(0, len(logits), self.batch_size):
            lg = logits[start:start + self.batch_size]
            h = features[start:start + self.batch_size]
            if self.drop_last and len(lg) < self.batch_size:
                continue
            probs = softmax(lg)
            pseudo = probs.argmax(1)
            low = probs.max(1) < self.threshold
            if low.any():
                pseudo = pseudo.copy()
                pseudo[low] = rng.integers(0, k, int(low.sum()))
            onehot = np.zeros_like(probs)
            onehot[np.arange(len(probs)), pseudo] = 1.0
            grad = ((probs - onehot).T @ h) / len(lg)
            scores.append(self._pnorm(grad, self.norm_type))
        if not scores:
            raise ValueError(
                f"no full batch of {self.batch_size} available from {len(logits)} samples"
            )
        return float(np.mean(scores))

    # -- calibration needs features too, so override the score hooks ---------
    def _score_matrix(self, logits: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError("GdScore needs features; use raw_score")

    def _fit(self, source: SourceStats) -> None:
        if source.features is None:
            raise ValueError("GdScore calibration requires source features")
        rng = np.random.default_rng(self.seed)
        n = len(source.logits)
        size = max(self.batch_size, min(self.subset_size, n))
        scores, accs = [], []
        for _ in range(self.n_calibration_subsets):
            idx = rng.choice(n, size=size, replace=True)
            try:
                scores.append(self.raw_score(source.logits[idx], source.features[idx]))
            except ValueError:
                continue
            accs.append(float(source.correct[idx].mean()))
        if len(scores) < 2:
            raise ValueError("GdScore calibration failed: too few usable subsets")
        s_arr, a_arr = np.asarray(scores), np.asarray(accs)
        if np.std(s_arr) < 1e-12:
            self.slope_, self.intercept_ = 0.0, float(a_arr.mean())
            self.calibration_r_ = 0.0
            return
        self.slope_, self.intercept_ = np.polyfit(s_arr, a_arr, 1)
        self.calibration_r_ = float(np.corrcoef(s_arr, a_arr)[0, 1])

    def _estimate(self, logits: np.ndarray, features: np.ndarray | None) -> float:
        return float(self.slope_ * self.raw_score(logits, features) + self.intercept_)


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
