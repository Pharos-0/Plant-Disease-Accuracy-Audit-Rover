"""Pesticide-use and disease-coverage figure for the selective sprayer.

Drawn from results/sprayer/sprayer_summary.csv, never from typed-in numbers, so
the chart cannot drift from the experiment.

Design decisions, per the project's data-visualisation method:
  * two panels, one axis each - use and coverage are different measures, and a
    dual-axis chart would let one be read against the other's scale;
  * one hue per panel - the bars are already identified by their axis labels,
    so colour carries no extra information and is not spent on decoration;
  * the oracle ceiling is muted and hatched, because it is a different kind of
    thing: not a deployable sprayer, but the most any sprayer could achieve;
  * bars start at zero. Coverage values sit at 98.9-100%, and showing them on a
    zero baseline is the honest picture: they are nearly equal.

    python scripts/make_sprayer_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import paths  # noqa: E402

SERIES = "#2a78d6"        # reference palette, categorical slot 1
CEILING = "#a8a7a0"       # muted neutral for the non-deployable ceiling
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"

ORDER = ["blanket", "vision_only", "selective", "oracle"]
LABELS = {"blanket": "Blanket\n(manual)", "vision_only": "Vision-only\nautonomous",
          "selective": "Proposed\nselective", "oracle": "Ceiling\n(perfect\nclassifier)"}


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def panel(ax, s: pd.DataFrame, col: str, lo: str, hi: str, title: str,
          annotate_reduction: bool) -> None:
    x = range(len(ORDER))
    for i, pol in enumerate(ORDER):
        r = s.loc[pol]
        ceiling = pol == "oracle"
        ax.bar(i, r[col], width=0.58, color=CEILING if ceiling else SERIES,
               hatch="////" if ceiling else None, edgecolor="white", linewidth=2,
               zorder=3)
        ax.errorbar(i, r[col], yerr=[[r[col] - r[lo]], [r[hi] - r[col]]],
                    fmt="none", ecolor=INK, capsize=2.5, lw=0.9, zorder=4)
        # Reductions sit above the bar in ink, never inside it: white text on
        # the hatched grey ceiling bar measured 2.41:1 and was unreadable.
        ax.text(i, r[col] + 2.2, f"{r[col]:.1f}%", ha="center", va="bottom",
                fontsize=7.8, color=INK, fontweight="bold")
        if annotate_reduction and pol != "blanket":
            ax.text(i, r[col] + 8.4, f"−{100 - r[col]:.1f}%", ha="center",
                    va="bottom", fontsize=7.0, color=INK_2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[p] for p in ORDER], fontsize=7.2, color=INK)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80, 100)])
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=8)
    style(ax)


def main() -> int:
    s = pd.read_csv(paths.RESULTS_DIR / "sprayer" / "sprayer_summary.csv",
                    index_col="policy")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    panel(ax[0], s, "pesticide_use_pct", "use_ci_lo", "use_ci_hi",
          "(a) Pesticide use, % of blanket spraying", annotate_reduction=True)
    ax[0].set_ylabel("pesticide consumption", fontsize=8, color=INK)
    panel(ax[1], s, "disease_coverage_pct", "cov_ci_lo", "cov_ci_hi",
          "(b) Diseased zones that received spray", annotate_reduction=False)
    ax[1].set_ylabel("disease coverage", fontsize=8, color=INK)
    fig.text(0.5, -0.04,
             "Simulation over real imagery: traverse geometry from a rover-captured "
             "greenhouse dataset (AgRobTomato); disease state and classifier\n"
             "outputs from Tomato-Village (Rajasthan, India). 200 traversals per "
             "policy; 95% bootstrap intervals. Not a field trial.",
             ha="center", fontsize=6.4, color=INK_2)
    fig.tight_layout()
    out = ROOT / "paper" / "figures" / "fig11_sprayer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
