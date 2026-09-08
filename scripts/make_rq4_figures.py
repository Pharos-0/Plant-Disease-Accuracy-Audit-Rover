"""Figures for the RQ4 diagnostics, generated from `results/rq4/` tables only.

Every panel is titled with its deployability class, because the central
distinction in this phase is between what a deployed machine could compute and
what needs labels. Figure 1 reloads cached logits to draw distributions; the
rest read the diagnostic CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from estimators.base import max_confidence
from inference.cache import CacheKey, load_features, load_labels

RQ4 = paths.RESULTS_DIR / "rq4"
FIGS = paths.FIGURES_DIR
BACKBONES = ["resnet50", "efficientnet_b0", "mobilenetv3_large", "vit_small_patch16"]
ARM_CACHE = {"A": "source_test", "C": "plantdoc", "D": "tomatovillage"}

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


def _version_of(bb):
    return "V2" if bb == "resnet50" else "V9"


# --------------------------------------------------------------- Fig 1
def fig1_confidence_distributions(seed=0):
    fig, axes = plt.subplots(2, len(BACKBONES), figsize=(3.3 * len(BACKBONES), 5.6),
                             sharex=True, sharey="row")
    for j, bb in enumerate(BACKBONES):
        for i, arm in enumerate(["A", "D"]):
            ax = axes[i, j]
            key = CacheKey(_version_of(bb), bb, seed, ARM_CACHE[arm], "SHARED_3")
            if not key.features_path.exists():
                ax.axis("off"); continue
            f = load_features(key); y = load_labels(key)
            conf = max_confidence(f["logits"]); ok = f["logits"].argmax(1) == y
            bins = np.linspace(0.3, 1.0, 40)
            ax.hist(conf[ok], bins=bins, density=True, alpha=0.6, label="correct", color="#4c72b0")
            ax.hist(conf[~ok], bins=bins, density=True, alpha=0.6, label="incorrect", color="#c44e52")
            if i == 0:
                ax.set_title(bb, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"arm {arm}\ndensity")
            if i == 1:
                ax.set_xlabel("max softmax confidence")
            if (~ok).any() and ok.any():
                ax.text(0.03, 0.92, f"sep={conf[ok].mean()-conf[~ok].mean():+.3f}",
                        transform=ax.transAxes, fontsize=7.5)
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Fig 1  POST-HOC — confidence for correct vs incorrect predictions.\n"
                 "Top: no shift (arm A). Bottom: real field (arm D). The distributions merge.",
                 y=1.03)
    fig.tight_layout()
    return _save(fig, "rq4_fig01_confidence_distributions.png")


# --------------------------------------------------------------- Fig 2
def fig2_calibration(ab):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (col, title) in zip(axes, [
        ("conf_minus_acc", "confidence − accuracy\n(>0 = overconfident)"),
        ("ece", "expected calibration error"),
        ("auroc_conf_correct", "AUROC: confidence ranks correctness\n(0.5 = uninformative)"),
    ]):
        piv = ab.pivot_table(index="arm", columns="backbone", values=col)
        piv = piv.reindex(["A", "C", "D"])
        piv.plot(kind="bar", ax=ax, width=0.8, legend=False)
        ax.set_title(title, fontsize=8.5); ax.set_xlabel("arm")
        ax.tick_params(axis="x", rotation=0)
        if col == "auroc_conf_correct":
            ax.axhline(0.5, color="k", ls="--", lw=1.2)
            ax.annotate("chance", xy=(2.2, 0.51), fontsize=7)
        if col == "conf_minus_acc":
            ax.axhline(0, color="k", lw=1)
    axes[2].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Fig 2  POST-HOC — calibration collapse under shift (mean over 5 seeds)", y=1.04)
    fig.tight_layout()
    return _save(fig, "rq4_fig02_calibration.png")


# --------------------------------------------------------------- Fig 3
def fig3_estimated_vs_actual(n=100):
    ps = pd.read_parquet(paths.TABLES_DIR / "per_session.parquet")
    d = ps[(ps.n_requested == n) & (ps.version.isin(["V2", "V9"]))]
    ests = sorted(d.estimator.unique())
    fig, axes = plt.subplots(2, len(ests), figsize=(2.3 * len(ests), 5.0),
                             sharex=True, sharey=True)
    for j, est in enumerate(ests):
        for i, arm in enumerate(["A", "D"]):
            ax = axes[i, j]
            s = d[(d.estimator == est) & (d.arm == arm)]
            ax.scatter(s.actual, s.estimate, s=2, alpha=0.15,
                       color="#4c72b0" if arm == "A" else "#c44e52")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            if len(s) > 2 and s.actual.std() > 1e-9:
                sl = np.polyfit(s.actual, s.estimate, 1)[0]
                ax.text(0.04, 0.88, f"slope={sl:.2f}", transform=ax.transAxes, fontsize=7)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if i == 0: ax.set_title(est, fontsize=8)
            if j == 0: ax.set_ylabel(f"arm {arm}\nestimated")
            if i == 1: ax.set_xlabel("true")
    fig.suptitle(f"Fig 3  POST-HOC — estimated vs true session accuracy, N={n}. "
                 "Dashed = y=x. A flat cloud means the estimate ignores the session.", y=1.03)
    fig.tight_layout()
    return _save(fig, "rq4_fig03_estimated_vs_actual.png")


# --------------------------------------------------------------- Fig 4
def fig4_sensitivity(ci, n=100):
    c = ci[ci.n_requested == n]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (col, title, ref) in zip(axes, [
        ("slope_est_on_true", "slope of estimate on true accuracy\n(1.0 = tracks perfectly)", 1.0),
        ("var_ratio", "var(estimate) / var(true accuracy)\n(1.0 = matched spread)", 1.0),
    ]):
        piv = c.pivot_table(index="estimator", columns="arm", values=col).reindex(columns=["A", "C", "D"])
        piv.plot(kind="barh", ax=ax, width=0.8, legend=(col == "var_ratio"))
        ax.axvline(ref, color="k", ls="--", lw=1.2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=9); ax.set_ylabel("")
        if col == "var_ratio":
            ax.set_xscale("symlog", linthresh=0.1)
    fig.suptitle(f"Fig 4  POST-HOC — loss of between-session sensitivity (N={n})", y=1.03)
    fig.tight_layout()
    return _save(fig, "rq4_fig04_session_sensitivity.png")


# --------------------------------------------------------------- Fig 5
def fig5_shift_vs_truth(de):
    d = de[de.arm == "D"]
    sigs = ["mmd_features_vs_source", "dominant_class_share", "mean_conf", "marginal_tv_vs_source"]
    fig, axes = plt.subplots(1, len(sigs), figsize=(3.3 * len(sigs), 3.4))
    for ax, s in zip(axes, sigs):
        for bb, g in d.groupby("backbone"):
            r = sps.spearmanr(g[s], g.true_accuracy_POSTHOC).statistic
            ax.scatter(g[s], g.true_accuracy_POSTHOC, s=3, alpha=0.25, label=f"{bb} ρ={r:+.2f}")
        pooled = sps.spearmanr(d[s], d.true_accuracy_POSTHOC).statistic
        ax.set_title(f"{s}\npooled ρ={pooled:+.2f}", fontsize=8)
        ax.set_xlabel("unlabelled signal (DEPLOYABLE)")
    axes[0].set_ylabel("true session accuracy (POST-HOC)")
    axes[0].legend(fontsize=5.5, loc="lower left")
    fig.suptitle("Fig 5  Arm D — deployable signals vs truth. Pooling across backbones REVERSES "
                 "the sign of MMD (Simpson's paradox): pooled ρ>0, every within-backbone ρ<0.",
                 y=1.05)
    fig.tight_layout()
    return _save(fig, "rq4_fig05_shift_vs_truth.png")


# --------------------------------------------------------------- Fig 6
def fig6_marginal_vs_cot(de, ci):
    c = ci[(ci.n_requested == 100) & (ci.arm == "D")]
    m = de[de.arm == "D"].groupby("backbone").agg(
        marg_tv=("marginal_tv_vs_source", "mean"),
        marg_ent=("marginal_entropy", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, xcol in zip(axes, ["marg_tv", "marg_ent"]):
        for est, colour in [("cot", "#c44e52"), ("mano", "#dd8452"),
                            ("doc", "#4c72b0"), ("atc_mc", "#55a868")]:
            b = c[c.estimator == est].set_index("backbone").bias
            y = [b[bb] for bb in m.backbone]
            r = np.corrcoef(m[xcol], y)[0, 1]
            ax.plot(m[xcol], y, "o-", color=colour, ms=6, label=f"{est}  r={r:+.2f}")
        for _, row in m.iterrows():
            ax.annotate(row.backbone.replace("_", "\n"), (row[xcol], 0.0),
                        fontsize=5.5, ha="center", va="bottom", alpha=0.6)
        ax.set_xlabel(f"{xcol} (DEPLOYABLE)"); ax.set_ylabel("estimator bias on arm D (POST-HOC)")
        ax.legend(fontsize=7)
    fig.suptitle("Fig 6  n=4 architectures, DESCRIPTIVE ONLY. COT's bias tracks predicted-marginal "
                 "shift — but so does MaNo's and DoC's, so the pattern is NOT COT-specific.", y=1.04)
    fig.tight_layout()
    return _save(fig, "rq4_fig06_marginal_vs_cot.png")


# --------------------------------------------------------------- Fig 7
def fig7_class_conditional(f):
    names = ["healthy", "early_blight", "late_blight"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, arm, title in zip(axes[:2], ["A", "D"], ["arm A (no shift)", "arm D (real field)"]):
        g = f[f.arm == arm].groupby("class_idx")[["precision", "mean_conf_when_predicted"]].mean()
        x = np.arange(len(names)); w = 0.38
        ax.bar(x - w / 2, g.precision, w, label="precision (POST-HOC)", color="#4c72b0")
        ax.bar(x + w / 2, g.mean_conf_when_predicted, w, label="mean confidence (DEPLOYABLE)",
               color="#c44e52")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7); axes[0].set_ylabel("value")

    g = f[f.arm == "D"].groupby("class_idx")[["true_share", "pred_share", "recall"]].mean()
    x = np.arange(len(names)); w = 0.27
    axes[2].bar(x - w, g.true_share, w, label="true share", color="#4c72b0")
    axes[2].bar(x, g.pred_share, w, label="predicted share", color="#dd8452")
    axes[2].bar(x + w, g.recall, w, label="recall", color="#55a868")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, fontsize=8)
    axes[2].set_title("arm D: prediction mass moves to early blight", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.suptitle("Fig 7  Class-conditional failure. On arm D the model predicts early blight far "
                 "more often than it occurs, confidently, at 0.39 precision.", y=1.04)
    fig.tight_layout()
    return _save(fig, "rq4_fig07_class_conditional.png")


# --------------------------------------------------------------- Fig 8
def fig8_capture_day(g):
    t = g.groupby("capture_date").mean(numeric_only=True)
    days = [str(d) for d in t.index]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    axes[0].plot(days, t.true_accuracy_POSTHOC, "o-", color="#c44e52", lw=2)
    axes[0].set_title("true accuracy (POST-HOC)\nfalls 21 points", fontsize=8.5)
    axes[0].set_ylim(0, 1)
    for ax, col, lab in zip(axes[1:], ["mean_conf", "mean_entropy", "mmd_features_vs_source"],
                            ["mean confidence", "mean entropy", "MMD vs source"]):
        ax.plot(days, t[col], "o-", color="#4c72b0", lw=2)
        rng = t[col].max() - t[col].min()
        ax.set_title(f"{lab} (DEPLOYABLE)\nrange = {rng:.3f}", fontsize=8.5)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle("Fig 8  Three real acquisition days. True accuracy moves 21 points; the "
                 "deployable signals move 1–3 points, and MMD moves the wrong way.", y=1.05)
    fig.tight_layout()
    return _save(fig, "rq4_fig08_capture_day.png")


# --------------------------------------------------------------- Fig 9
def fig9_bias_variance(ci):
    d = ci[ci.arm == "D"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for est, g in d.groupby("estimator"):
        s = g.groupby("n_requested")[["var_estimate", "abs_bias", "bias2_share_of_mse"]].mean()
        axes[0].plot(s.index, s.var_estimate, "o-", ms=4, label=est)
        axes[1].plot(s.index, s.abs_bias, "o-", ms=4, label=est)
        axes[2].plot(s.index, s.bias2_share_of_mse, "o-", ms=4, label=est)
    for ax, ylab, log in zip(axes,
                             ["variance of the estimate", "|bias|", "bias² share of MSE"],
                             [True, False, False]):
        ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 200, 400])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("session size N"); ax.set_ylabel(ylab)
        if log: ax.set_yscale("log")
    axes[1].set_ylim(bottom=0); axes[2].set_ylim(0, 1.02)
    axes[2].axhline(0.5, color="k", ls="--", lw=1)
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Fig 9  POST-HOC, arm D — variance falls with N, |bias| does not. "
                 "At N=400, bias² is ~85% of MSE.", y=1.04)
    fig.tight_layout()
    return _save(fig, "rq4_fig09_bias_variance_vs_N.png")


# --------------------------------------------------------------- Fig 10
def fig10_architecture(ab, ci):
    c = ci[(ci.n_requested == 100) & (ci.arm == "D")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    piv = c.pivot_table(index="estimator", columns="backbone", values="bias")
    piv.plot(kind="barh", ax=axes[0], width=0.8)
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_title("estimator bias on arm D by architecture", fontsize=9)
    axes[0].set_xlabel("bias"); axes[0].set_ylabel("")
    axes[0].legend(fontsize=6)

    a = ab.pivot_table(index="backbone", columns="arm", values="auroc_conf_correct")
    a = a.reindex(columns=["A", "C", "D"])
    a.plot(kind="bar", ax=axes[1], width=0.8)
    axes[1].axhline(0.5, color="k", ls="--", lw=1.2)
    axes[1].set_title("confidence→correctness AUROC\n(0.5 = uninformative)", fontsize=9)
    axes[1].tick_params(axis="x", rotation=20, labelsize=7); axes[1].legend(fontsize=7)

    src = ab[ab.arm == "D"].groupby("backbone").accuracy.mean()
    bias = c.groupby("backbone").bias.mean()
    axes[2].scatter(src, bias, s=70, color="#c44e52")
    for bb in src.index:
        axes[2].annotate(bb.replace("_", "\n"), (src[bb], bias[bb]), fontsize=6.5,
                         ha="center", va="bottom")
    axes[2].set_xlabel("target accuracy on arm D")
    axes[2].set_ylabel("mean estimator bias")
    axes[2].set_title("no monotone relation between\nmodel quality and estimator bias", fontsize=9)
    fig.suptitle("Fig 10  Architecture summary — the failure is architecture-independent "
                 "in direction, and every backbone loses confidence-correctness ranking.", y=1.05)
    fig.tight_layout()
    return _save(fig, "rq4_fig10_architecture.png")


def main() -> int:
    ab = pd.read_csv(RQ4 / "diagA_B_confidence_calibration.csv")
    ci = pd.read_csv(RQ4 / "diagC_session_sensitivity.csv")
    de = pd.read_csv(RQ4 / "diagD_E_session_signals.csv")
    f = pd.read_csv(RQ4 / "diagF_class_conditional.csv")
    g = pd.read_csv(RQ4 / "diagG_capture_day.csv")

    print("generating RQ4 figures ...")
    fig1_confidence_distributions()
    fig2_calibration(ab)
    fig3_estimated_vs_actual()
    fig4_sensitivity(ci)
    fig5_shift_vs_truth(de)
    fig6_marginal_vs_cot(de, ci)
    fig7_class_conditional(f)
    fig8_capture_day(g)
    fig9_bias_variance(ci)
    fig10_architecture(ab, ci)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
