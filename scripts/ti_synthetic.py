"""Synthetic validation of the Target-Informativeness measure (Phase 5).

Two questions the agricultural data cannot answer, because there the true target
dependence of an estimator is unknown:

    V1  **Does TI recover a known answer?** Construct estimators whose target
        dependence is fixed by design - a pure constant (TI must be 0), a pure
        target statistic (TI must be 1), and a mixture
        ``alpha * target + (1 - alpha) * anchor`` for a grid of alpha. If TI does
        not track alpha, the measure is broken and nothing downstream is worth
        reading.

    V2  **Does TI behave sensibly as the shift grows and as the target sample
        shrinks?** With synthetic data the true target accuracy is known exactly,
        so estimator error and TI can be compared without any label-free proxy.

Data are generated, not borrowed: K Gaussian classes in d dimensions, a linear
classifier fitted on the source, and a target drawn with controllable covariate
shift (a mean displacement) and label shift (altered class priors). True target
accuracy is computed exactly from the labels, which exist only in this synthetic
setting and never touch an estimator.

NEW ESTIMATOR EXPERIMENT. No repository data are read.

    python scripts/ti_synthetic.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from estimators.base import Estimator, SourceStats, max_confidence
from estimators.methods import REGISTRY
from estimators.ti_audit import anchor_sensitivity, target_sensitivity
from utils.manifest import write_experiment_manifest

EXPERIMENT_ID = "ti_synthetic"
OUT = paths.RESULTS_DIR / EXPERIMENT_ID
K, D = 3, 16
#: Class-mean spread. Calibrated so source accuracy lands near 0.72 rather than
#: 1.000. At the original 1.5 the classifier was perfect, the source accuracy had
#: bootstrap sd 0.0000, and TI therefore collapsed to 1.0 for EVERY probe - a
#: measurement artefact, not a property of the estimators. Recorded because it is
#: also a genuine limitation of TI: the measure depends on how precisely the
#: anchor is determined, which is a function of the source sample, not only of
#: the estimator.
SEPARATION = 0.35
SHIFT_SCALE = 1.2


# ---------------------------------------------------------------- probes
class MixtureProbe(Estimator):
    """An estimator with a *known* target dependence, for validating TI.

    ``estimate = alpha * mean_confidence(T) + (1 - alpha) * source_accuracy``

    The anchor is the source accuracy, fixed at fit time; the target term is a
    statistic of T. TI measured on this estimator must increase monotonically
    with ``alpha``, hit 0 at ``alpha = 0`` and 1 at ``alpha = 1``.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.name = f"mixture_a{alpha:.2f}"
        self.anchor_ = 0.0

    def _fit(self, source: SourceStats) -> None:
        self.anchor_ = source.accuracy

    def _estimate(self, logits: np.ndarray, features: np.ndarray | None) -> float:
        return self.alpha * float(max_confidence(logits).mean()) + \
            (1.0 - self.alpha) * self.anchor_


# ---------------------------------------------------------------- data
def make_world(shift: float, n_src: int, n_tgt: int, seed: int,
               label_shift: bool = True) -> tuple[SourceStats, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, float]:
    """Source stats, target logits/features/groups, and the EXACT target accuracy."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0, SEPARATION, size=(K, D))
    direction = rng.normal(0, 1, size=D)
    direction /= np.linalg.norm(direction)

    def draw(n, priors, mu_offset):
        y = rng.choice(K, size=n, p=priors)
        x = means[y] + mu_offset + rng.normal(0, 1.0, size=(n, D))
        return x.astype(np.float32), y

    src_priors = np.full(K, 1.0 / K)
    x_s, y_s = draw(n_src, src_priors, 0.0)

    # a linear classifier fitted on the source, in closed form (class centroids)
    centroids = np.stack([x_s[y_s == k].mean(0) if (y_s == k).any() else means[k]
                          for k in range(K)])

    def logits_of(x):
        # negative squared distance to each centroid, a valid linear-in-x score
        return (-((x[:, None, :] - centroids[None]) ** 2).sum(-1)).astype(np.float32)

    tgt_priors = (np.array([0.6, 0.3, 0.1]) if label_shift else src_priors)
    x_t, y_t = draw(n_tgt, tgt_priors, shift * direction * SHIFT_SCALE)

    lg_s, lg_t = logits_of(x_s), logits_of(x_t)
    true_acc = float((lg_t.argmax(1) == y_t).mean())          # EXACT, synthetic only
    src = SourceStats(logits=lg_s, labels=y_s, features=x_s)
    # groups: 4 rows per synthetic "photograph", mirroring the real clustering
    g_s = np.repeat(np.arange(int(np.ceil(n_src / 4))), 4)[:n_src].astype(str)
    g_t = np.repeat(np.arange(int(np.ceil(n_tgt / 4))), 4)[:n_tgt].astype(str)
    return src, lg_t, x_t, g_s, g_t, true_acc


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    NB = 60

    # ---- V1: does TI recover a known alpha? ---------------------------
    print("=== V1: TI on probes with KNOWN target dependence ===", flush=True)
    v1 = []
    for rep in range(5):
        src, lg_t, ft_t, g_s, g_t, _ = make_world(0.5, 600, 800, seed=100 + rep)
        for a in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            fac = (lambda a=a: MixtureProbe(a))
            t_sd, _ = target_sensitivity(fac, src, lg_t, ft_t, g_t, NB, seed=rep)
            a_sd, _ = anchor_sensitivity(fac, src, g_s, lg_t, ft_t, NB, seed=rep)
            vt, va = t_sd ** 2, a_sd ** 2
            v1.append({"rep": rep, "alpha": a,
                       "ti": float(vt / (vt + va)) if (vt + va) > 0 else np.nan,
                       "target_sd": t_sd, "anchor_sd": a_sd})
    d1 = pd.DataFrame(v1)
    d1.to_csv(OUT / "v1_known_alpha.csv", index=False)
    g = d1.groupby("alpha").ti.agg(["mean", "std"])
    print(g.round(4).to_string())
    ok = d1.groupby("alpha").ti.mean()
    rho = sps.spearmanr(ok.index, ok.values)
    print(f"  monotonicity of TI in alpha: Spearman = {rho.statistic:.3f} "
          f"(p={rho.pvalue:.4f})")
    print(f"  TI at alpha=0: {ok.loc[0.0]:.4f}   (should be ~0)")
    print(f"  TI at alpha=1: {ok.loc[1.0]:.4f}   (should be ~1)")

    # ---- V2: shift severity and target sample size --------------------
    print("\n=== V2: real estimators under controlled shift, exact true accuracy ===",
          flush=True)
    v2 = []
    facs = {n: s["factory"] for n, s in REGISTRY.items()}
    for shift in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for n_tgt in [100, 400]:
            for rep in range(3):
                src, lg_t, ft_t, g_s, g_t, truth = make_world(
                    shift, 600, n_tgt, seed=7000 + rep + int(shift * 100) + n_tgt)
                for name, fac in facs.items():
                    try:
                        est = float(fac().fit(src).estimate(lg_t, ft_t))
                    except (ValueError, RuntimeError):
                        continue
                    t_sd, _ = target_sensitivity(fac, src, lg_t, ft_t, g_t, NB, rep)
                    a_sd, _ = anchor_sensitivity(fac, src, g_s, lg_t, ft_t, NB, rep)
                    vt, va = t_sd ** 2, a_sd ** 2
                    v2.append({"shift": shift, "n_target": n_tgt, "rep": rep,
                               "estimator": name, "estimate": est,
                               "true_accuracy": truth, "abs_error": abs(est - truth),
                               "ti": float(vt / (vt + va)) if (vt + va) > 0 else np.nan})
    d2 = pd.DataFrame(v2)
    d2.to_csv(OUT / "v2_shift_and_size.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n  MAE by shift severity:")
    print(d2.pivot_table(index="estimator", columns="shift",
                         values="abs_error").round(3).to_string())
    print("\n  TI by shift severity:")
    print(d2.pivot_table(index="estimator", columns="shift",
                         values="ti").round(3).to_string())
    print("\n  MAE by target sample size:")
    print(d2.pivot_table(index="estimator", columns="n_target",
                         values="abs_error").round(3).to_string())

    r = sps.spearmanr(d2.ti, d2.abs_error, nan_policy="omit")
    print(f"\n  Q4 (synthetic, pooled n={len(d2)}): Spearman TI vs |error| = "
          f"{r.statistic:.3f} (p={r.pvalue:.2e})")

    write_experiment_manifest(
        EXPERIMENT_ID,
        params={"purpose": "validate TI against known target dependence, and observe it "
                           "under controlled shift with exact true accuracy",
                "K": K, "D": D, "n_boot": NB,
                "alphas": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                "shifts": [0.0, 0.25, 0.5, 0.75, 1.0], "n_target": [100, 400],
                "label_shift": True,
                "status": "NEW ESTIMATOR EXPERIMENT - synthetic, no repository data"},
        inputs={"data": "generated in-process from a fixed seed"},
        outputs=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))],
        metrics={"ti_alpha_monotonicity_spearman": float(rho.statistic),
                 "ti_at_alpha0": float(ok.loc[0.0]), "ti_at_alpha1": float(ok.loc[1.0]),
                 "synthetic_spearman_ti_vs_error": float(r.statistic)},
        runtime_seconds=time.time() - t0,
        notes="Synthetic ground truth. Labels exist here only to compute exact target "
              "accuracy for scoring; no estimator sees them.",
        out_dir=OUT,
    )
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
