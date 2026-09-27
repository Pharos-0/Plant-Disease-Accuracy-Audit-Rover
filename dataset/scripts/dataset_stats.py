"""Dataset statistics, plots and a final sample grid.

Reports source imbalance explicitly rather than letting the largest source speak
for the dataset, per the build plan's rule that a source must not dominate
merely because it is big.

    python dataset/scripts/dataset_stats.py
"""
from __future__ import annotations

import sys
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

ROOT = HERE.parents[1]
CFG = yaml.safe_load((ROOT / "dataset" / "config.yaml").read_text(encoding="utf-8"))
META = ROOT / "dataset" / "metadata"
DOCS = ROOT / "dataset" / "documentation"
SAMPLES = ROOT / CFG["paths"]["samples"]

NAMES = {"SRC002": "PlantDoc (local, not redistributable)",
         "SRC014": "AgRobTomato (rover-captured)",
         "SRC015": "Greenhouse tomatoes (handheld)"}


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> int:
    d = pd.read_csv(META / "metadata_split.csv")
    rej = pd.read_csv(META / "rejected_images.csv")
    lic = pd.read_csv(META / "license_registry.csv")

    rows = []

    def stat(k, v):
        rows.append({"statistic": k, "value": v})

    stat("total images retained", len(d))
    stat("total rejected as duplicates", len(rej))
    stat("sources contributing", d.source_id.nunique())
    stat("synthetic images", 0)
    stat("leakage groups", d.leakage_group.nunique())
    stat("guard-band frames excluded from splits", int(d.guard_drop.sum()))

    print("=" * 74)
    print("SOURCE DISTRIBUTION  (imbalance reported, not smoothed away)")
    src = (d.groupby("source_id")
             .agg(images=("image_id", "size"),
                  groups=("leakage_group", "nunique"),
                  mean_relevance=("rover_relevance", "mean"),
                  wide=("camera_distance_estimate", lambda s: (s == "WIDE").sum()))
             .assign(share=lambda x: (x.images / len(d) * 100).round(1)))
    src.insert(0, "name", [NAMES.get(i, i) for i in src.index])
    print(src.to_string())
    for sid, r in src.iterrows():
        stat(f"images from {sid}", int(r.images))
        stat(f"share of dataset from {sid} (%)", float(r.share))

    print("\nThe largest source is "
          f"{src.images.idxmax()} at {src.share.max():.1f}% of the dataset. "
          f"The genuinely rover-captured\nsource is SRC014 at "
          f"{src.loc['SRC014', 'share']:.1f}%, contributing "
          f"{int(src.loc['SRC014', 'wide'])} of the "
          f"{int((d.camera_distance_estimate == 'WIDE').sum())} WIDE images.")

    print("\nROVER RELEVANCE")
    rel = d.rover_relevance.value_counts().sort_index()
    print(rel.to_string())
    for k, v in rel.items():
        stat(f"rover_relevance == {k}", int(v))

    print("\nCAMERA DISTANCE")
    dist = d.camera_distance_estimate.value_counts()
    print(dist.to_string())
    for k, v in dist.items():
        stat(f"camera_distance == {k}", int(v))

    print("\nSPLITS")
    sp = d.train_val_test_split.value_counts()
    print(sp.to_string())
    for k, v in sp.items():
        stat(f"split {k}", int(v))

    print("\nLICENCES")
    merged = d.merge(lic[["source_id", "license_name", "license_status"]],
                     on="source_id", how="left")
    licc = merged.license_name.fillna("UNVERIFIED").value_counts()
    print(licc.to_string())
    for k, v in licc.items():
        stat(f"licence {k}", int(v))
    redistributable = int((merged.license_status == "VERIFIED").sum())
    stat("redistributable images (licence VERIFIED)", redistributable)
    print(f"\n  {redistributable} of {len(d)} images are redistributable; the rest "
          "are usable locally only.")

    print("\nRESOLUTION")
    mp = (d.width * d.height / 1e6)
    stat("median megapixels", round(float(mp.median()), 3))
    stat("min megapixels", round(float(mp.min()), 3))
    stat("max megapixels", round(float(mp.max()), 3))
    print(f"  median {mp.median():.2f} MP, range {mp.min():.2f}-{mp.max():.2f} MP")
    ar = (d.width / d.height)
    print(f"  aspect ratio median {ar.median():.2f}, range {ar.min():.2f}-{ar.max():.2f}")

    pd.DataFrame(rows).to_csv(META / "dataset_statistics.csv", index=False)

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    o = ["WIDE", "MEDIUM", "CLOSE", "MACRO"]
    ct = (d.pivot_table(index="source_id", columns="camera_distance_estimate",
                        values="image_id", aggfunc="count")
            .reindex(columns=o).fillna(0))
    ct.plot(kind="bar", stacked=True, ax=ax[0, 0], colormap="viridis")
    ax[0, 0].set_title("Camera distance by source\n(WIDE is the rover viewpoint)")
    ax[0, 0].set_ylabel("images")
    ax[0, 0].tick_params(axis="x", rotation=15)

    d.pivot_table(index="source_id", columns="rover_relevance",
                  values="image_id", aggfunc="count").fillna(0).plot(
        kind="bar", stacked=True, ax=ax[0, 1], colormap="cividis")
    ax[0, 1].set_title("Rover-relevance score by source")
    ax[0, 1].tick_params(axis="x", rotation=15)

    for sid, g in d.groupby("source_id"):
        ax[1, 0].hist(g.half_detail_scale, bins=24, alpha=0.55,
                      label=NAMES.get(sid, sid))
    ax[1, 0].set_title("Structure scale\n(small = wide/distant view)")
    ax[1, 0].set_xlabel("half_detail_scale")
    ax[1, 0].legend(fontsize=7)

    for sid, g in d.groupby("source_id"):
        ax[1, 1].hist(g.border_scene_frac, bins=24, alpha=0.55,
                      label=NAMES.get(sid, sid))
    ax[1, 1].set_title("Scene continuation at the frame border")
    ax[1, 1].set_xlabel("border_scene_frac")
    ax[1, 1].legend(fontsize=7)

    fig.tight_layout()
    DOCS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS / "dataset_statistics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------- WIDE-only grid
    wide = d[(d.camera_distance_estimate == "WIDE") & (d.rover_relevance == 3)]
    rng = np.random.default_rng(20260928)
    take = wide.iloc[rng.choice(len(wide), min(8, len(wide)), replace=False)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax_, (_, r) in zip(axes.ravel(), take.iterrows()):
        img = imread(r.stored_path)
        ax_.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_.axis("off")
        ax_.set_title(f"{r.source_id} {r.image_id}\nscene {r.border_scene_frac:.2f} "
                      f"scale {r.half_detail_scale:.2f}", fontsize=7.5)
    fig.suptitle("The rover-view core: WIDE viewpoint and relevance 3. "
                 f"{len(wide)} images qualify.", y=1.0)
    fig.tight_layout()
    fig.savefig(SAMPLES / "rover_view_core.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWIDE + relevance 3 (the rover-view core): {len(wide)} images")
    print(wide.source_id.value_counts().to_string())
    print(f"\nwrote {META / 'dataset_statistics.csv'}")
    print(f"wrote {DOCS / 'dataset_statistics.png'}")
    print(f"wrote {SAMPLES / 'rover_view_core.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
