"""RQ5 crossover analysis — the pre-registered primary comparison.

Separated from `rq5_labelled_sample.py` so the experiment and its analysis are
distinct steps. Reads only `results/rq5/` tables and writes
`rq5_crossover_analysis.csv` plus the summary tables quoted in
`docs/rq5_results.md`.

Primary unit is the **trained model** (n = 20), paired: per model the labelled
error is the mean over 100 repetitions and the label-free error is a single
value on the full unlabelled pool.

Pre-registered crossover (docs/rq5_plan.md section K): the smallest budget where
the mean paired difference is negative **and** the 95% bootstrap CI excludes
zero. The Holm-corrected Wilcoxon p is reported alongside because the two
criteria can disagree, and both are stated rather than the more convenient one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths

RQ5 = paths.RESULTS_DIR / "rq5"
N_BOOT = 2000
BOOT_SEED = 7


def holm(pvals: list[float]) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_ci(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(N_BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    a = pd.read_csv(RQ5 / "rq5_settingA_labelled.csv")
    a = a[a.n_label_photographs != 391]
    lf = pd.read_csv(RQ5 / "rq5_labelfree_full_pool.csv")

    lab = (a.groupby(["backbone", "seed", "n_label_photographs"])
             .abs_error.mean().reset_index())
    rng = np.random.default_rng(BOOT_SEED)

    rows = []
    for budget, gl in lab.groupby("n_label_photographs"):
        pvals, block = [], []
        for est, gf in lf.groupby("estimator"):
            m = gl.merge(gf[["backbone", "seed", "abs_error"]],
                         on=["backbone", "seed"], suffixes=("_lab", "_lf"))
            if len(m) != 20:
                raise RuntimeError(f"expected 20 paired models, got {len(m)}")
            diff = (m.abs_error_lab - m.abs_error_lf).to_numpy()  # <0 = labelling wins
            lo, hi = bootstrap_ci(diff, rng)
            p = float(sps.wilcoxon(m.abs_error_lab, m.abs_error_lf).pvalue)
            pvals.append(p)
            block.append({
                "N": int(budget), "baseline": est,
                "lab_mae": float(m.abs_error_lab.mean()),
                "lf_mae": float(m.abs_error_lf.mean()),
                "diff": float(diff.mean()), "ci_lo": lo, "ci_hi": hi,
                "p": p, "n_models": len(m),
            })
        for entry, padj in zip(block, holm(pvals)):
            entry["p_holm"] = float(padj)
            entry["CROSSOVER"] = bool(entry["diff"] < 0 and entry["ci_hi"] < 0)
            entry["CROSSOVER_with_holm"] = bool(entry["CROSSOVER"] and padj < 0.05)
            rows.append(entry)

    cross = pd.DataFrame(rows)
    cross.to_csv(RQ5 / "rq5_crossover_analysis.csv", index=False)

    print("=== Table 3: crossover budget by baseline ===")
    for est, s in cross.groupby("baseline"):
        s = s.sort_values("N")
        ci_first = s[s.CROSSOVER].N.min() if s.CROSSOVER.any() else None
        holm_first = (s[s.CROSSOVER_with_holm].N.min()
                      if s.CROSSOVER_with_holm.any() else None)
        print(f"  {est:14s} baseline MAE={s.lf_mae.iloc[0]:.4f}  "
              f"crossover(CI)={ci_first}  crossover(CI+Holm)={holm_first}")

    print("\n=== Table 4: tail risk, 95th percentile absolute error ===")
    print(a.groupby("n_label_photographs").abs_error
           .quantile(0.95).round(4).to_string())

    print("\n=== Wilson interval coverage (nominal 95%) ===")
    print((a.groupby("n_label_photographs").ci_covers_truth.mean() * 100)
          .round(1).to_string())

    print(f"\nwrote {RQ5 / 'rq5_crossover_analysis.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
