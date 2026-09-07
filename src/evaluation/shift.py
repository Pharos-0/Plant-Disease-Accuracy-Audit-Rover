"""Measure how far each target domain actually sits from the source.

MASTER_PROMPT section 6 is explicit: the arms represent different *types* of
shift and must not be assumed to form an increasing severity ordering.
Whether an ordering exists is measured here, on penultimate features, and
reported as a result.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg


def _sqrtm(a: np.ndarray) -> np.ndarray:
    """``scipy.linalg.sqrtm`` across versions.

    SciPy dropped the ``disp`` keyword (and the ``(root, errest)`` return) in
    1.18, so probe for it rather than pinning a version.
    """
    try:
        out = linalg.sqrtm(a, disp=False)
    except TypeError:
        out = linalg.sqrtm(a)
    return out[0] if isinstance(out, tuple) else out


def reduce_dims(
    source: np.ndarray, target: np.ndarray, n_components: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Project both sides onto source-fitted principal components.

    Penultimate features are 2048-d while a source split here holds only a few
    hundred images, so a full-rank covariance is not estimable and a raw 2048-d
    Frechet distance would be dominated by noise. Projecting onto the leading
    source components first makes the covariance well posed. The projection is
    fitted on the source only, so it does not peek at the target's geometry.
    """
    k = int(min(n_components, source.shape[1], max(source.shape[0] - 1, 1)))
    mu = source.mean(0, keepdims=True)
    # economy SVD of the centred source gives the principal directions
    _u, _s, vt = np.linalg.svd(source - mu, full_matrices=False)
    basis = vt[:k].T
    return (source - mu) @ basis, (target - mu) @ basis


def frechet_distance(
    x: np.ndarray, y: np.ndarray, eps: float = 1e-6, n_components: int | None = 64
) -> float:
    """Frechet (Wasserstein-2 between Gaussians) distance, as used by FID.

    When ``n_components`` is set, both sides are first projected onto
    source-fitted principal components; see ``reduce_dims``.
    """
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if n_components:
        x, y = reduce_dims(x, y, n_components)

    mu1, mu2 = x.mean(0), y.mean(0)
    s1, s2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)

    diff = mu1 - mu2
    covmean = _sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = _sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def _median_bandwidth(z: np.ndarray, rng: np.random.Generator, cap: int = 1000) -> float:
    if len(z) > cap:
        z = z[rng.choice(len(z), cap, replace=False)]
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    return float(np.sqrt(med / 2.0)) or 1.0


def mmd_rbf(
    x: np.ndarray, y: np.ndarray, *, cap: int = 1000, seed: int = 0
) -> float:
    """Unbiased MMD^2 with an RBF kernel at the median-heuristic bandwidth."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) > cap:
        x = x[rng.choice(len(x), cap, replace=False)]
    if len(y) > cap:
        y = y[rng.choice(len(y), cap, replace=False)]

    sigma = _median_bandwidth(np.vstack([x, y]), rng, cap)
    g = 1.0 / (2 * sigma**2)

    def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None, :] - 2 * a @ b.T
        return np.exp(-g * np.maximum(d2, 0))

    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    n, m = len(x), len(y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return float(
        kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()
    )


def label_shift(source_labels: np.ndarray, target_labels: np.ndarray, k: int) -> dict[str, float]:
    """Total-variation and chi-square distance between class priors.

    Uses labels from both sides, so it belongs to evaluation and is reported as
    a property of the benchmark - never consumed by an estimator.
    """
    p = np.bincount(source_labels, minlength=k).astype(float)
    q = np.bincount(target_labels, minlength=k).astype(float)
    p, q = p / p.sum(), q / q.sum()
    tv = float(0.5 * np.abs(p - q).sum())
    chi2 = float((((q - p) ** 2) / np.clip(p, 1e-12, None)).sum())
    return {"tv_distance": tv, "chi2_distance": chi2,
            "source_prior": p.tolist(), "target_prior": q.tolist()}


def shift_report(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray | None = None,
    target_labels: np.ndarray | None = None,
    n_classes: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    out: dict[str, float] = {
        "frechet_distance": frechet_distance(source_features, target_features),
        "mmd_rbf": mmd_rbf(source_features, target_features, seed=seed),
        "n_source": int(len(source_features)),
        "n_target": int(len(target_features)),
    }
    if source_labels is not None and target_labels is not None and n_classes:
        out.update(label_shift(source_labels, target_labels, n_classes))
    return out
