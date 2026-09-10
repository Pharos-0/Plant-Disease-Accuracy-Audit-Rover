"""Figures for the target sample-size sweep.

Every panel is generated from `results/ti_sample_size/*.csv`; no value is typed
in. Run after `scripts/ti_sample_size.py`.

    python scripts/make_ti_sample_size_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths

IN = paths.RESULTS_DIR / "ti_sample_size"
OUT = paths.RESULTS_DIR / "figures"
ARM_LABEL = {"source_test": "source_test\n(zero shift)",
             "plantdoc": "PlantDoc\n(severe shift)",
             "tomatovillage": "Tomato-Village\n(severe shift)"}
ARMS = ["source_test", "plantdoc", "tomatovillage"]


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


def fig1_rho(t3: pd.DataFrame) -> None:
    """Spearman(TI, |error|) against N, one line per arm, with bootstrap CIs."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm, c in zip(ARMS, ("#1b7837", "#762a83", "#b35806")):
        g = t3[t3.arm == arm].sort_values("N")
        if g.empty:
            continue
        ax.errorbar(g.N, g["mean"],
                    yerr=[g["mean"] - g.ci_lo, g.ci_hi - g["mean"]],
                    marker="o", capsize=3, color=c, label=ARM_LABEL[arm].replace("\n", " "))
    ax.axhline(0, color="k", lw=1, ls="--", alpha=.6)
    ax.set_xscale("log")
    ax.set_xticks(sorted(t3.N.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("target sample size N (photographs)")
    ax.set_ylabel(r"mean per-model Spearman(TI, |error|)")
    ax.set_title("Figure 1  The TI–error relationship reverses with shift severity,\n"
                 "and the reversal is present at every feasible sample size")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "ti_ss_fig1_rho_vs_N.png")


def fig2_mae(err: pd.DataFrame) -> None:
    """MAE against N, one panel per arm, one line per estimator."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    ests = sorted(err.estimator.unique())
    cmap = plt.get_cmap("tab10")
    for ax, arm in zip(axes, ARMS):
        g = err[err.arm == arm]
        for i, e in enumerate(ests):
            s = g[g.estimator == e].groupby("N").abs_error.mean().sort_index()
            if s.empty:
                continue
            ax.plot(s.index, s.values, marker="o", ms=3.5, color=cmap(i % 10), label=e)
        ax.set_xscale("log")
        ax.set_xticks(sorted(g.N.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(ARM_LABEL[arm], fontsize=10)
        ax.set_xlabel("N (photographs)")
    axes[0].set_ylabel("MAE")
    axes[-1].legend(frameon=False, fontsize=7, ncol=1, loc="upper right")
    fig.suptitle("Figure 2  Estimator MAE against target sample size", y=1.02)
    _save(fig, "ti_ss_fig2_mae_vs_N.png")


def fig3_sensitivities(ti: pd.DataFrame) -> None:
    """Target vs anchor sensitivity as N grows."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for ax, arm in zip(axes, ARMS):
        g = ti[ti.arm == arm]
        if g.empty:
            continue
        t = g.groupby("N").target_sd.mean().sort_index()
        a = g.groupby("N").anchor_sd.mean().sort_index()
        ax.plot(t.index, t.values, marker="o", color="#1b7837", label="target sensitivity")
        ax.plot(a.index, a.values, marker="s", color="#762a83", label="anchor sensitivity")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sorted(g.N.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(ARM_LABEL[arm], fontsize=10)
        ax.set_xlabel("N (photographs)")
    axes[0].set_ylabel("sd of the estimate (log scale)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Figure 3  Target sensitivity rises with N; anchor sensitivity does not",
                 y=1.02)
    _save(fig, "ti_ss_fig3_sensitivity_vs_N.png")


def fig4_interaction(t4: pd.DataFrame) -> None:
    """The zero-shift minus severe-shift contrast at each N."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for comp, c, m in zip(sorted(t4.comparison.unique()), ("#762a83", "#b35806"), "os"):
        g = t4[t4.comparison == comp].sort_values("N")
        ax.errorbar(g.N, g.mean_diff,
                    yerr=[g.mean_diff - g.ci_lo, g.ci_hi - g.mean_diff],
                    marker=m, capsize=3, color=c, label=comp)
    ax.axhline(0, color="k", lw=1, ls="--", alpha=.6)
    ax.set_xscale("log")
    ax.set_xticks(sorted(t4.N.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("target sample size N (photographs)")
    ax.set_ylabel("difference in mean per-model Spearman")
    ax.set_title("Figure 4  Shift severity × sample size:\n"
                 "the contrast stays above zero at every N (95% bootstrap CI)")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "ti_ss_fig4_interaction.png")


def main() -> int:
    if not (IN / "table3_rho_by_arm_N.csv").exists():
        print("run scripts/ti_sample_size.py first")
        return 2
    err = pd.read_csv(IN / "sample_size_errors.csv")
    ti = pd.read_csv(IN / "sample_size_ti.csv")
    t3 = pd.read_csv(IN / "table3_rho_by_arm_N.csv")
    t4 = pd.read_csv(IN / "table4_reversal_by_N.csv")
    fig1_rho(t3)
    fig2_mae(err)
    fig3_sensitivities(ti)
    fig4_interaction(t4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
