"""RQ5 figures, generated from `results/rq5/` tables only.

Label-free estimators consume no labels, so they are drawn as **horizontal
reference lines**, never as points on the labelled-budget axis. The central
visual question is how far right the labelled curve must travel before it drops
below each line.
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

RQ5 = paths.RESULTS_DIR / "rq5"
FIGS = paths.FIGURES_DIR
BUDGETS = [5, 10, 20, 30, 50, 75, 100]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out


def _lab_curve(a, group=None):
    d = a[a.n_label_photographs != 391]
    if group:
        d = d[d.backbone == group]
    g = d.groupby("n_label_photographs").abs_error
    return g.mean(), g.quantile(0.95)


# --------------------------------------------------------------- Fig 1
def fig1(a, lf):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    mae, p95 = _lab_curve(a)
    ax.plot(mae.index, mae.values, "o-", color="#c44e52", lw=2.4, ms=7,
            label="labelled sample (MAE)", zorder=5)
    ax.plot(p95.index, p95.values, "o--", color="#c44e52", lw=1.3, ms=4, alpha=0.65,
            label="labelled sample (95th pct error)", zorder=5)

    base = lf.groupby("estimator").abs_error.mean().sort_values()
    cmap = plt.get_cmap("tab10")
    # several baselines sit within ~0.002 of each other, so nudge labels apart
    # rather than letting them overprint
    min_gap, last = 0.016, -np.inf
    for i, (est, v) in enumerate(base.items()):
        ax.axhline(v, color=cmap(i % 10), lw=1.4, ls=":", alpha=0.9)
        y = max(v, last + min_gap)
        last = y
        ax.annotate(f"{est}  {v:.3f}", xy=(103, y), fontsize=7.5,
                    color=cmap(i % 10), va="center")
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(4.4, 160)
    ax.set_xlabel("labelled field photographs (independent units)")
    ax.set_ylabel("absolute error of the estimated field accuracy")
    ax.set_title("Fig 1  Arm D — how many labelled photographs beat a label-free estimate?\n"
                 "Dotted lines are label-free estimators using all 1,616 unlabelled crops "
                 "and zero labels.", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save(fig, "rq5_fig01_budget_vs_error.png")


# --------------------------------------------------------------- Fig 2
def fig2(cross):
    ests = ["entropy", "cot", "nuclear_norm", "atc_ne",
            "confidence", "atc_mc", "doc", "mano"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True, sharey=True)
    for ax, est in zip(axes.ravel(), ests):
        s = cross[cross.baseline == est].sort_values("N")
        ax.axhline(0, color="k", lw=1.2)
        ax.errorbar(s.N, s["diff"],
                    yerr=[s["diff"] - s.ci_lo, s.ci_hi - s["diff"]],
                    fmt="o-", color="#4c72b0", capsize=3, ms=5, lw=1.6)
        first = s[s.CROSSOVER].N.min() if s.CROSSOVER.any() else None
        if first is not None:
            ax.axvline(first, color="#c44e52", ls="--", lw=1.4)
            ax.annotate(f"crossover\nN={int(first)}", xy=(first, s["diff"].max()),
                        fontsize=7.5, color="#c44e52", ha="left", va="top")
        ax.set_xscale("log"); ax.set_xticks(BUDGETS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"vs {est}  (MAE {s.lf_mae.iloc[0]:.3f})", fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("labelled photographs")
    for ax in axes[:, 0]:
        ax.set_ylabel("paired difference\n(labelled − label-free)")
    fig.suptitle("Fig 2  Paired difference in absolute error, unit = trained model (n=20), "
                 "95% bootstrap CI.\nBelow zero means labelling wins.", y=1.03)
    fig.tight_layout()
    return _save(fig, "rq5_fig02_paired_difference.png")


# --------------------------------------------------------------- Fig 3
def fig3(a, lf):
    bbs = sorted(a.backbone.unique())
    fig, axes = plt.subplots(1, len(bbs), figsize=(3.6 * len(bbs), 3.8), sharey=True)
    for ax, bb in zip(axes, bbs):
        mae, _ = _lab_curve(a, bb)
        ax.plot(mae.index, mae.values, "o-", color="#c44e52", lw=2.2, ms=6,
                label="labelled", zorder=5)
        sub = lf[lf.backbone == bb].groupby("estimator").abs_error.mean().sort_values()
        best, bv = sub.index[0], sub.iloc[0]
        ax.axhline(bv, color="#4c72b0", ls=":", lw=1.8, label=f"best: {best} ({bv:.3f})")
        ax.axhline(sub.iloc[-1], color="#8172b3", ls=":", lw=1.2,
                   label=f"worst: {sub.index[-1]} ({sub.iloc[-1]:.3f})")
        ax.set_xscale("log"); ax.set_xticks(BUDGETS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(bb, fontsize=8.5); ax.set_xlabel("labelled photographs")
        ax.legend(fontsize=6.5)
    axes[0].set_ylabel("absolute error")
    fig.suptitle("Fig 3  Per-backbone. On ResNet-50 and ViT-S/16 the best label-free "
                 "estimator (COT) is already strong, so labelling helps far less.", y=1.04)
    fig.tight_layout()
    return _save(fig, "rq5_fig03_per_backbone.png")


# --------------------------------------------------------------- Fig 4
def fig4(b):
    lab = b[b.method == "labelled"]
    lfb = b[b.method != "labelled"].copy()
    lfb["estimator"] = lfb.method.str.replace("labelfree_", "", regex=False)
    days = sorted(lab.capture_date.unique())
    fig, axes = plt.subplots(1, len(days), figsize=(4.2 * len(days), 3.9), sharey=True)
    for ax, day in zip(axes, days):
        m = lab[lab.capture_date == day].groupby("n_label_photographs").abs_error.mean()
        ax.plot(m.index, m.values, "o-", color="#c44e52", lw=2.2, ms=6, label="labelled",
                zorder=5)
        sub = lfb[lfb.capture_date == day].groupby("estimator").abs_error.mean().sort_values()
        ax.axhline(sub.iloc[0], color="#4c72b0", ls=":", lw=1.8,
                   label=f"best: {sub.index[0]} ({sub.iloc[0]:.3f})")
        nph = int(b[b.capture_date == day].day_photographs.iloc[0])
        acc = float(lab[lab.capture_date == day].day_accuracy.mean())
        ax.set_title(f"{day}\n{nph} photographs, true acc {acc:.3f}", fontsize=8.5)
        ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 30, 50])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("labelled photographs"); ax.legend(fontsize=7)
    axes[0].set_ylabel("absolute error")
    fig.suptitle("Fig 4  Capture-day holdout — labels from one day only. The label-free "
                 "reference degrades as the day gets harder; the labelled sample does not.",
                 y=1.04)
    fig.tight_layout()
    return _save(fig, "rq5_fig04_capture_day.png")


# --------------------------------------------------------------- Fig 5
def fig5(a):
    """Wilson coverage — the interval a practitioner would actually quote."""
    d = a[a.n_label_photographs != 391]
    cov = d.groupby("n_label_photographs").ci_covers_truth.mean() * 100
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(cov.index, cov.values, "o-", color="#c44e52", lw=2.2, ms=7)
    ax.axhline(95, color="k", ls="--", lw=1.4)
    ax.annotate("nominal 95%", xy=(cov.index[0], 95.6), fontsize=8)
    ax.set_xscale("log"); ax.set_xticks(BUDGETS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(50, 100)
    ax.set_xlabel("labelled field photographs")
    ax.set_ylabel("actual coverage of the true accuracy (%)")
    ax.set_title("Fig 5  The naive Wilson interval is badly anti-conservative.\n"
                 "It treats clustered crops as independent; real coverage is 63–75%, "
                 "not 95%.", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, "rq5_fig05_wilson_coverage.png")


def main() -> int:
    a = pd.read_csv(RQ5 / "rq5_settingA_labelled.csv")
    lf = pd.read_csv(RQ5 / "rq5_labelfree_full_pool.csv")
    b = pd.read_csv(RQ5 / "rq5_settingB_capture_day.csv")
    cross = pd.read_csv(RQ5 / "rq5_crossover_analysis.csv")
    print("generating RQ5 figures ...")
    fig1(a, lf); fig2(cross); fig3(a, lf); fig4(b); fig5(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
