"""Figures, generated from logged results only (MASTER_PROMPT section 15).

Every function here reads a table written by ``run_evaluation.py`` or a run
manifest. Nothing is hand-drawn and no number is typed in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import paths

ARM_LABEL = {
    "A": "A: PlantVillage held-out\n(no shift)",
    "B": "B: PlantDoc\n(web, SHARED_8)",
    "C": "C: PlantDoc\n(web, SHARED_3)",
    "D": "D: Tomato-Village\n(real field)",
}
ARM_ORDER = ["A", "C", "B", "D"]

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out


def fig_estimated_vs_actual(per_session: pd.DataFrame, n: int = 100) -> Path:
    """Estimated against true session accuracy, with the y = x line."""
    d = per_session[per_session.n_requested == n]
    ests = sorted(d.estimator.unique())
    ncol = 4
    nrow = int(np.ceil(len(ests) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)

    for ax, est in zip(axes.ravel(), ests):
        sub = d[d.estimator == est]
        for arm in ARM_ORDER:
            s = sub[sub.arm == arm]
            if len(s):
                ax.scatter(s.actual, s.estimate, s=3, alpha=0.25, label=arm)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(est, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("true session accuracy")
        ax.set_ylabel("estimated")
    for ax in axes.ravel()[len(ests):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", markerscale=4, title="arm")
    fig.suptitle(f"Estimated vs true session accuracy (N={n})", y=1.01)
    fig.tight_layout()
    return _save(fig, f"fig05_estimated_vs_actual_N{n}.png")


def fig_mae_by_estimator(agg: pd.DataFrame, n: int = 100) -> Path:
    d = agg[agg.n_requested == n]
    piv = d.pivot_table(index="estimator", columns="arm", values="mae", aggfunc="mean")
    piv = piv.reindex(columns=[a for a in ARM_ORDER if a in piv.columns])
    piv = piv.sort_values(piv.columns[-1])

    fig, ax = plt.subplots(figsize=(7.5, 4))
    piv.plot(kind="barh", ax=ax, width=0.8)
    ax.set_xlabel("mean absolute error of the accuracy estimate")
    ax.set_ylabel("")
    ax.set_title(f"Estimator error by target arm (N={n})")
    ax.legend(title="arm", fontsize=8)
    return _save(fig, f"fig06_mae_by_estimator_N{n}.png")


def fig_n_sweep(agg: pd.DataFrame) -> tuple[Path, Path]:
    """The headline sample-efficiency result: MAE and spread against N."""
    arms = [a for a in ARM_ORDER if a in set(agg.arm)]
    ests = sorted(agg.estimator.unique())
    cmap = plt.get_cmap("tab10")
    colors = {e: cmap(i % 10) for i, e in enumerate(ests)}

    outs = []
    for metric, fname, ylab in (
        ("mae", "fig07_N_vs_mae.png", "mean absolute error"),
        ("std_error", "fig08_N_vs_variance.png", "std. dev. of estimation error"),
    ):
        fig, axes = plt.subplots(1, len(arms), figsize=(3.4 * len(arms), 3.4), squeeze=False)
        for ax, arm in zip(axes[0], arms):
            sub = agg[agg.arm == arm]
            for est in ests:
                s = sub[sub.estimator == est].groupby("n_requested")[metric].mean()
                if len(s):
                    ax.plot(s.index, s.values, marker="o", ms=3.5, lw=1.4,
                            label=est, color=colors[est])
            ax.set_xscale("log")
            ax.set_xticks([25, 50, 100, 200, 400])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("session size N")
            ax.set_title(ARM_LABEL.get(arm, arm), fontsize=8)
            ax.set_ylim(bottom=0)
        axes[0, 0].set_ylabel(ylab)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(ests),
                   fontsize=7.5, bbox_to_anchor=(0.5, 1.10))
        fig.suptitle(f"Session size vs {ylab}", y=1.16)
        fig.tight_layout()
        outs.append(_save(fig, fname))
    return outs[0], outs[1]


def fig_bias_by_shift(agg: pd.DataFrame, shift: pd.DataFrame | None, n: int = 100) -> Path:
    """Signed bias per arm. Positive means the estimator over-promises."""
    d = agg[agg.n_requested == n]
    piv = d.pivot_table(index="estimator", columns="arm", values="bias", aggfunc="mean")
    piv = piv.reindex(columns=[a for a in ARM_ORDER if a in piv.columns])

    fig, ax = plt.subplots(figsize=(7.5, 4))
    piv.plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("bias  (estimate - truth)")
    ax.set_title(f"Signed bias by arm (N={n}); above zero = over-estimating accuracy")
    ax.legend(title="arm", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, f"fig09_bias_by_shift_N{n}.png")


def fig_gate_oc(gate: pd.DataFrame, arm: str = "D", n: int = 100) -> Path:
    """Deployment gate operating characteristic - the framing contribution."""
    d = gate[(gate.arm == arm) & (gate.n_requested == n)]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for est, g in d.groupby("estimator"):
        g = g.groupby("tau")[["unsafe_operate_rate", "wasted_withhold_rate"]].mean().reset_index()
        ax.plot(g.wasted_withhold_rate, g.unsafe_operate_rate, marker=".", ms=3,
                lw=1.2, label=est)
    ax.set_xlabel("wasted withhold rate")
    ax.set_ylabel("unsafe operate rate  (the costly error)")
    ax.set_title(f"Deployment gate operating characteristic\narm {arm}, N={n}, tau swept 0.50-0.95")
    ax.legend(fontsize=7.5)
    return _save(fig, f"fig10_gate_oc_arm{arm}_N{n}.png")


def fig_v1_v2_leakage(agg: pd.DataFrame, n: int = 100) -> Path | None:
    """V1 vs V2: does source-side leakage propagate into target estimates?"""
    d = agg[(agg.n_requested == n) & (agg.version.isin(["V1", "V2"]))]
    if d.version.nunique() < 2:
        return None
    piv = d.pivot_table(index="estimator", columns="version", values="bias", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    piv.plot(kind="bar", ax=ax, width=0.8, color=["#c44e52", "#4c72b0"])
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("bias  (estimate - truth)")
    ax.set_title(
        f"Leakage propagation into estimator bias (N={n})\n"
        "V1 = random image split (leaky) vs V2 = leaf-grouped split"
    )
    ax.legend(title="protocol")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, f"fig11_v1_v2_leakage_N{n}.png")


def fig_shift_magnitude(shift: pd.DataFrame) -> Path | None:
    """Is there actually a severity ordering across arms? Measured, not assumed."""
    if shift is None or not len(shift):
        return None
    d = shift.groupby("arm").agg(
        frechet=("frechet_distance", "mean"),
        mmd=("mmd_rbf", "mean"),
        tv=("tv_distance", "mean"),
        target_acc=("target_accuracy", "mean"),
    )
    d = d.reindex([a for a in ARM_ORDER if a in d.index])

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, (col, title) in zip(
        axes,
        [
            ("frechet", "Frechet distance\n(penultimate features)"),
            ("mmd", "MMD (RBF)"),
            ("tv", "label-prior TV distance"),
            ("target_acc", "true target accuracy"),
        ],
    ):
        ax.bar(d.index, d[col], color="#4c72b0")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("arm")
    fig.suptitle("Measured shift magnitude per arm - ordering is a result, not an assumption", y=1.04)
    fig.tight_layout()
    return _save(fig, "fig12_shift_magnitude.png")


def generate_all() -> list[Path]:
    per_session = pd.read_parquet(paths.TABLES_DIR / "per_session.parquet")
    agg = pd.read_parquet(paths.TABLES_DIR / "aggregate_metrics.parquet")
    gate_path = paths.TABLES_DIR / "gate_operating_characteristic.parquet"
    shift_path = paths.TABLES_DIR / "shift_magnitude.csv"
    gate = pd.read_parquet(gate_path) if gate_path.exists() else None
    shift = pd.read_csv(shift_path) if shift_path.exists() else None

    made: list[Path] = []
    print("generating figures ...")
    made.append(fig_estimated_vs_actual(per_session))
    made.append(fig_mae_by_estimator(agg))
    made.extend(fig_n_sweep(agg))
    made.append(fig_bias_by_shift(agg, shift))
    if gate is not None and len(gate):
        made.append(fig_gate_oc(gate))
    v = fig_v1_v2_leakage(agg)
    if v:
        made.append(v)
    s = fig_shift_magnitude(shift)
    if s:
        made.append(s)
    return [m for m in made if m]


if __name__ == "__main__":
    generate_all()
