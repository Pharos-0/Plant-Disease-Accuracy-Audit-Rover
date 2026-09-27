"""Step 7: measure the pilot, deduplicate it, and show representative images.

Three questions, in order:

1. **Does the pilot imagery actually look rover-relevant?** Measured with the
   rubric calibrated in `inventory_local.py`, not asserted from the dataset
   description.
2. **Does the dedup pipeline work?** This has a known answer. The authors of
   SRC015 state that part of their dataset was taken from SRC014, so genuine
   cross-source duplicates should exist and must be found. A pipeline that
   reports zero here is broken, not clean.
3. **How does the pilot compare with what we already hold?**

Perceptual hashing is implemented directly rather than pulled from a dependency:
the DCT-based pHash is eight lines and vendoring it keeps the result
reproducible without an extra package.

    python dataset/scripts/analyse_pilot.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rover_relevance import extract, classify, image_type_hint  # noqa: E402

ROOT = HERE.parents[1]
CFG = yaml.safe_load((ROOT / "dataset" / "config.yaml").read_text(encoding="utf-8"))
META = ROOT / "dataset" / "metadata"
SAMPLES = ROOT / CFG["paths"]["samples"]

PHASH_SIDE = 32          # DCT input side; the top-left 8x8 block becomes the hash
PHASH_BLOCK = CFG["dedup"]["phash_size"]
HAMMING_MAX = CFG["dedup"]["phash_hamming_max"]


def phash(bgr: np.ndarray) -> int:
    """64-bit DCT perceptual hash. Robust to rescaling and mild compression."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (PHASH_SIDE, PHASH_SIDE), interpolation=cv2.INTER_AREA)
    d = cv2.dct(np.float32(g))[:PHASH_BLOCK, :PHASH_BLOCK]
    flat = d.flatten()[1:]                      # drop DC, it only carries brightness
    bits = flat > np.median(flat)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def imread(p: str | Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> int:
    pilot = pd.read_csv(META / "pilot_downloads.csv")
    print(f"pilot rows: {len(pilot)} from {pilot.source_id.nunique()} sources\n")

    recs = []
    for _, r in pilot.iterrows():
        img = imread(r.stored_path)
        if img is None:
            print(f"  unreadable: {r.stored_path}")
            continue
        f = extract(img)
        score, reason = classify(f)
        recs.append({"source_id": r.source_id, "path": r.stored_path,
                     "sha256": r.sha256, "phash": phash(img),
                     "rover_relevance": score, "reason": reason,
                     "image_type_hint": image_type_hint(f, score), **f.as_dict()})
    d = pd.DataFrame(recs)
    d.to_csv(META / "pilot_features.csv", index=False)

    print("=" * 70)
    print("ROVER-RELEVANCE OF THE PILOT")
    tab = d.pivot_table(index="source_id", columns="rover_relevance",
                        values="path", aggfunc="count").fillna(0).astype(int)
    print(tab.to_string())
    print()
    for sid, g in d.groupby("source_id"):
        hi = float((g.rover_relevance >= 2).mean())
        print(f"  {sid}: {hi:.0%} scored 2-3, mean {g.rover_relevance.mean():.2f}, "
              f"border_scene_frac {g.border_scene_frac.mean():.3f}")

    # ---- exact duplicates ------------------------------------------------
    exact = d[d.duplicated("sha256", keep=False)]
    print(f"\nEXACT DUPLICATES (sha256): {len(exact)} rows in "
          f"{exact.sha256.nunique()} groups")

    # ---- near duplicates, the known-answer test --------------------------
    print(f"\nNEAR DUPLICATES (pHash hamming <= {HAMMING_MAX})")
    pairs = []
    rows = d.reset_index(drop=True)
    for i, j in combinations(range(len(rows)), 2):
        h = hamming(int(rows.at[i, "phash"]), int(rows.at[j, "phash"]))
        if h <= HAMMING_MAX:
            pairs.append({"a": rows.at[i, "path"], "b": rows.at[j, "path"],
                          "source_a": rows.at[i, "source_id"],
                          "source_b": rows.at[j, "source_id"], "hamming": h})
    dup = pd.DataFrame(pairs)
    if dup.empty:
        print("  none found")
    else:
        dup.to_csv(META / "duplicate_report.csv", index=False)
        within = dup[dup.source_a == dup.source_b]
        across = dup[dup.source_a != dup.source_b]
        print(f"  {len(dup)} pairs: {len(within)} within a source, "
              f"{len(across)} across sources")
        if not across.empty:
            print("  cross-source pairs by source:")
            print(across.groupby(["source_a", "source_b"]).size().to_string())

    print("\n  KNOWN-ANSWER CHECK: SRC015's authors state part of their dataset")
    print("  was taken from SRC014, so cross-source duplicates SHOULD exist.")
    n_cross = 0 if dup.empty else int((dup.source_a != dup.source_b).sum())
    if n_cross > 0:
        print(f"  -> found {n_cross}. Pipeline detects genuine cross-source reuse.")
    else:
        print("  -> found 0 in this 60+60 sample. That is NOT evidence of no overlap:")
        print("     each sample is the first 60 archive members, so the overlapping")
        print("     portion may simply not be in the sampled range. Re-test on a")
        print("     random or full sample before concluding anything.")

    # ---- sample grid -----------------------------------------------------
    SAMPLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260928)
    srcs = sorted(d.source_id.unique())
    n = 6
    fig, axes = plt.subplots(len(srcs), n, figsize=(16, 3.1 * len(srcs)))
    axes = np.atleast_2d(axes)
    for r, sid in enumerate(srcs):
        g = d[d.source_id == sid]
        take = g.iloc[rng.choice(len(g), min(n, len(g)), replace=False)]
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(take):
                continue
            row = take.iloc[c]
            img = imread(row.path)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{sid}  score {int(row.rover_relevance)}\n"
                         f"scene {row.border_scene_frac:.2f}  soil {row.soil_fraction:.2f}",
                         fontsize=7)
    fig.suptitle("Pilot sample. Score is a rover-view PROXY rating, not a claim "
                 "that a rover captured these.", y=1.0, fontsize=10)
    fig.tight_layout()
    out = SAMPLES / "pilot_grid.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")
    print(f"wrote {(META / 'pilot_features.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
