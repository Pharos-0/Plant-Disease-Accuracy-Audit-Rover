"""Generate the thesis figures that did not already exist.

27 figures already exist from earlier phases and are NOT regenerated. This adds
only the ones the thesis figure list still lacks, each built from a committed
results table:

    F11  target-information removal          results/ti_audit/ti_removal.csv
    F12  GdScore train/eval BatchNorm        results/rq7b_batchnorm/
    F13  within-model session ranking        results/gate_evaluation/
    F16  gate operating characteristic       results/gate_evaluation/

Every panel carries its sample size, and the ranking figure deliberately shows
the per-model paired result rather than the pooled one, because the pooled
version is a Simpson artefact.

    python scripts/make_thesis_figures.py
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

R = paths.RESULTS_DIR
OUT = R / "figures"


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/figures/{name}")


def f11_removal() -> None:
    f = R / "ti_audit" / "ti_removal.csv"
    if not f.exists():
        print("  SKIP F11 - ti_removal.csv not found")
        return
    d = pd.read_csv(f)
    d = d[d.arm == "tomatovillage"] if "arm" in d.columns else d
    piv = d.pivot_table(index="estimator", columns="surrogate",
                        values="shift_from_real").sort_values("random_logits")
    order = ["shuffled", "moment_matched_noise", "duplicated", "random_logits"]
    piv = piv[[c for c in order if c in piv.columns]]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    piv.plot(kind="barh", ax=ax, width=.8,
             color=["#c7c7c7", "#9ecae1", "#4292c6", "#08519c"])
    ax.set_xlabel("mean |change in the estimate| when target information is destroyed")
    ax.set_ylabel("")
    ax.set_title("Figure 11  Target-information removal (Tomato-Village, 20 models)\n"
                 "estimators near zero barely notice that the target was replaced")
    ax.legend(title="surrogate target", frameon=False, fontsize=8)
    _save(fig, "thesis_f11_target_removal.png")


def f12_gdscore_bn() -> None:
    f = R / "rq7b_batchnorm" / "rq7b_train_vs_eval.csv"
    if not f.exists():
        print("  SKIP F12 - rq7b_train_vs_eval.csv not found")
        return
    d = pd.read_csv(f)
    piv = d.pivot_table(index="backbone", columns="mode", values="abs_error")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(len(piv))
    ax.bar(x - .19, piv["train"], .38, label="train mode (official)", color="#2166ac")
    ax.bar(x + .19, piv["eval"], .38, label="eval mode (deployed model)", color="#b2182b")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=12, fontsize=9)
    ax.set_ylabel("GdScore MAE")
    ax.set_title("Figure 12  GdScore under train- vs eval-mode activations\n"
                 "ViT has no BatchNorm and is the control (n = 5 seeds per backbone)")
    ax.legend(frameon=False, fontsize=9)
    for i, b in enumerate(piv.index):
        if "vit" in b:
            ax.annotate("no BatchNorm\n(control)", (i, max(piv.loc[b]) + .02),
                        ha="center", fontsize=8)
    _save(fig, "thesis_f12_gdscore_batchnorm.png")


def f13_session_ranking() -> None:
    f = R / "gate_evaluation" / "gate_sessions.csv"
    if not f.exists():
        print("  SKIP F13 - gate_sessions.csv not found")
        return
    g = pd.read_csv(f)
    field = g[g.arm.isin(["plantdoc", "tomatovillage"])]
    rows = []
    for (arm, bb, sd), d in field.groupby(["arm", "backbone", "seed"]):
        for sig, h in d.groupby("signal"):
            if h.true_accuracy.std() > 0:
                rows.append({"signal": sig,
                             "rho": sps.spearmanr(h.score, h.true_accuracy).statistic})
    r = pd.DataFrame(rows)
    st = r.groupby("signal").rho.agg(["mean", "std", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.barh(st.index, st["mean"],
            xerr=st["std"] / np.sqrt(st["count"]), color="#4a7c59", capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel(r"per-model Spearman($\mathrm{signal}$, true session accuracy)")
    ax.set_title("Figure 13  Can a signal rank field sessions within one model?\n"
                 f"paired per model, n = {int(st['count'].max())} model-arm pairs; "
                 "best is 0.22 - none ranks usefully")
    ax.annotate("pooling across models inflates this\n(Simpson artefact) - not shown",
                xy=(0.55, 0.15), xycoords="axes fraction", fontsize=8, style="italic",
                color="#7a2020")
    _save(fig, "thesis_f13_session_ranking.png")


def f16_gate_oc() -> None:
    f = R / "gate_evaluation" / "gate_operating_characteristic.csv"
    if not f.exists():
        print("  SKIP F16 - operating characteristic not found")
        return
    d = pd.read_csv(f)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for ax, arm in zip(axes, ("source_test", "plantdoc", "tomatovillage")):
        g = d[d.arm == arm]
        cmap = plt.get_cmap("tab10")
        for i, (sig, h) in enumerate(g.groupby("signal")):
            h = h.sort_values("unsafe_operate_rate")
            ax.plot(h.unsafe_operate_rate, h.coverage, marker="", lw=1.4,
                    color=cmap(i % 10), label=sig)
        ax.set_xlabel("unsafe-operate rate")
        ax.set_title(arm, fontsize=10)
    axes[0].set_ylabel("coverage (fraction of sessions operated on)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Figure 16  Deployment-gate operating characteristic, no threshold tuned\n"
                 "under field shift essentially no session is safe at any threshold", y=1.04)
    _save(fig, "thesis_f16_gate_operating_characteristic.png")


def main() -> int:
    print("generating thesis figures that do not already exist:")
    f11_removal()
    f12_gdscore_bn()
    f13_session_ranking()
    f16_gate_oc()
    n = len(list(OUT.glob("*.png")))
    print(f"\nresults/figures now holds {n} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
