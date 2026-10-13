"""Photometric perturbations of real labelled field images, for a ground-truthed
dry run of the camera experiment.

These are EDITED REAL PHOTOGRAPHS, not camera captures: a dimmed image is not a
leaf photographed in dim light (real low light adds sensor noise, white-balance
shift and motion blur). They exist to (1) exercise the demo and analysis code
end to end with true labels, including the leaf-paired comparisons, and (2)
preview whether the deployed decision rule ever leaves a diseased leaf unsprayed
when an image is degraded. They do not replace the real camera session.

Perturbations, fixed before running:
  normal          the original
  low_light       intensity x 0.30
  harsh_light     intensity x 1.9, clipped (blown highlights)
  occlusion       the left 45% of the image covered
  distance_angle  downsampled to 35% then restored, and rotated 25 degrees

Source: Tomato-Village (Rajasthan, India), SHARED_3 crops, sampled by
photograph so that every crop of a chosen photograph is included and the
photograph is the clustering unit.

    python scripts/make_perturbation_manifest.py --photos 80
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("C:/ds/camera_dryrun")
SEED = 20261013


def perturb(img: np.ndarray, kind: str) -> np.ndarray:
    f = img.astype(np.float32)
    if kind == "normal":
        return img
    if kind == "low_light":
        return np.clip(f * 0.30, 0, 255).astype(np.uint8)
    if kind == "harsh_light":
        return np.clip(f * 1.9, 0, 255).astype(np.uint8)
    if kind == "occlusion":
        out = img.copy()
        out[:, : int(0.45 * img.shape[1])] = 20
        return out
    if kind == "distance_angle":
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(8, int(w * 0.35)), max(8, int(h * 0.35))),
                           interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1.0)
        return cv2.warpAffine(back, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=int, default=80)
    a = ap.parse_args()
    d = pd.read_parquet(ROOT / "data" / "index.parquet")
    tv = d[(d.domain == "tomatovillage") &
           d.label.isin(["healthy", "early_blight", "late_blight"])]
    rng = np.random.default_rng(SEED)
    groups = tv.group_id.unique()
    chosen = set(rng.choice(groups, min(a.photos, len(groups)), replace=False))
    sel = tv[tv.group_id.isin(chosen)]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ("normal", "low_light", "harsh_light", "occlusion", "distance_angle"):
        (OUT / kind).mkdir(exist_ok=True)
        for _, r in sel.iterrows():
            img = cv2.imdecode(np.fromfile(r.path, np.uint8), cv2.IMREAD_COLOR)
            out = OUT / kind / Path(r.path).name
            cv2.imencode(".jpg", perturb(img, kind))[1].tofile(str(out))
            rows.append({"path": str(out), "truth": r.label,
                         "leaf_id": str(r.group_id).replace(":", "_"), "condition": kind})
    man = OUT / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "truth", "leaf_id", "condition"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(sel)} crops from {len(chosen)} photographs x 5 conditions = "
          f"{len(rows)} frames -> {man}")
    print("truth:", sel.label.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
