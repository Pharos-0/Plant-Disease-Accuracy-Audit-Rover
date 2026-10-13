"""Open-field sprayer figure: use, precision and coverage, primary layout.

Drawn from results/openfield/spray_summary.csv. Three measures, three panels,
one axis each: pesticide use alone would reward a sprayer that misses plants,
and precision alone would reward one that sprays almost nothing.

    python scripts/make_openfield_figure.py
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

SERIES, CEILING, INK, INK_2, GRID = "#2a78d6", "#a8a7a0", "#0b0b0b", "#52514e", "#e4e3de"
ORDER = ["manual", "vision_only", "proposed", "ceiling"]
LABELS = {"manual": "Manual\n(blanket)", "vision_only": "Vision-only\n(green)",
          "proposed": "Proposed\n(detector)", "ceiling": "Ceiling\n(perfect)"}
LAYOUT = "4x3"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=7, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def panel(ax, s, m, title, note_fn=None):
    for i, pol in enumerate(ORDER):
        r = s.loc[pol]
        ceil = pol == "ceiling"
        ax.bar(i, r[m], width=0.6, color=CEILING if ceil else SERIES,
               hatch="////" if ceil else None, edgecolor="white", linewidth=2, zorder=3)
        ax.errorbar(i, r[m], yerr=[[r[m] - r[m + "_lo"]], [r[m + "_hi"] - r[m]]],
                    fmt="none", ecolor=INK, capsize=2.5, lw=0.9, zorder=4)
        top = max(r[m + "_hi"], r[m])
        ax.text(i, top + 2.5, f"{r[m]:.1f}%", ha="center", va="bottom",
                fontsize=7.4, color=INK, fontweight="bold")
        if note_fn:
            note = note_fn(pol, r)
            if note:
                ax.text(i, top + 9.5, note, ha="center", va="bottom",
                        fontsize=6.6, color=INK_2)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABELS[p] for p in ORDER], fontsize=6.8, color=INK)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
    ax.set_title(title, fontsize=8.2, color=INK, loc="left", pad=8)
    style(ax)


def main() -> int:
    s = pd.read_csv(paths.RESULTS_DIR / "openfield" / "spray_summary.csv")
    s = s[s.layout == LAYOUT].set_index("policy")
    man_prec = s.loc["manual", "precision"]
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.3))
    panel(ax[0], s, "use", "(a) Pesticide use, % of manual",
          lambda p, r: None if p == "manual" else f"−{100 - r.use:.1f}%")
    panel(ax[1], s, "precision", "(b) Spray precision: sprayed zones that needed it",
          lambda p, r: None if p == "manual" else f"{r.precision / man_prec:.1f}× manual")
    panel(ax[2], s, "coverage", "(c) Diseased-plant zones that received spray")
    fig.text(0.5, -0.06,
             "Open-field tomato, zenithal imagery (GROUNDBASED_WEED, Badajoz, Spain); disease "
             "state and classifier outputs from Tomato-Village (Rajasthan, India).\n"
             "Pre-registered 4-nozzle layout; 60 held-out images; 200 runs per policy; 95% "
             "bootstrap intervals over images. Simulation over real imagery, not a field trial.",
             ha="center", fontsize=6.2, color=INK_2)
    fig.tight_layout()
    out = ROOT / "paper" / "figures" / "fig12_openfield.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
