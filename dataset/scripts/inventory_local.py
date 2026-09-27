"""Step 1-2: inventory the tomato imagery already on this machine, and use it to
calibrate the rover-relevance rubric against a known answer.

Three datasets are already here from the thesis work, and their acquisition
protocols were established by direct visual inspection:

  plantvillage   single leaf on a coarse speckled grey stone surface (studio)
  tomatovillage  single detached leaf on plain paper or cloth, outdoors
  plantdoc       web-scraped and mixed: leaves on the plant with straw and soil,
                 but also potted seedlings against black cloth or a white wall

The rubric must reproduce that structure without being told which dataset an
image came from. It is scored on separation and ordering rather than on one
label per source, because PlantDoc genuinely contains both kinds of image and a
per-source label would invite fitting thresholds until they passed. If the
calibration fails, the rubric must be understood before it filters anything.

    python dataset/scripts/inventory_local.py [--per-source 250]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rover_relevance import extract, classify, image_type_hint  # noqa: E402

ROOT = HERE.parents[1]
CFG = yaml.safe_load((ROOT / "dataset" / "config.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "dataset" / "metadata"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#: What direct inspection established about each acquisition PROTOCOL. These are
#: not expectations about individual images, and deliberately so: PlantDoc is
#: web-scraped and genuinely mixed - field shots with straw and soil alongside
#: potted seedlings against black cloth - so demanding one label for all of it
#: would test the wrong thing and invite threshold-fitting until it passed.
#:
#: The rubric is therefore required to do three things it cannot fake:
#:   1. keep the two specimen-on-backdrop protocols at 0-1,
#:   2. lift a substantial share of the in-the-wild source to 2-3,
#:   3. order the three sources correctly by mean score.
SPECIMEN_PROTOCOL = {
    "plantvillage": "single leaf on a coarse grey stone surface (studio)",
    "tomatovillage": "single detached leaf on plain paper or cloth (outdoors)",
}
WILD_PROTOCOL = {
    "plantdoc": "web-scraped, mixed: leaves on the plant and potted specimens",
}
#: Fraction of a specimen-protocol source that must score 0-1.
SPECIMEN_LOW_MIN = 0.85
#: Fraction of the in-the-wild source that must reach 2-3.
WILD_HIGH_MIN = 0.45


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(name: str, spec: dict) -> list[Path]:
    root = Path(spec["root"])
    if not root.is_dir():
        print(f"  [skip] {name}: {root} not present")
        return []
    files = [p for p in root.rglob("*") if p.suffix.lower() in EXTS]
    if name == "plantvillage":
        files = [p for p in files if "Tomato" in str(p)]
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=250,
                    help="images sampled per source for feature measurement")
    ap.add_argument("--seed", type=int, default=20260928)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    totals = {}

    for name, spec in CFG["local_sources"].items():
        files = discover(name, spec)
        totals[name] = len(files)
        if not files:
            continue
        take = files if len(files) <= args.per_source else [
            files[i] for i in rng.choice(len(files), args.per_source, replace=False)]
        print(f"  {name}: {len(files)} tomato images on disk, measuring {len(take)}")
        for p in take:
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                rows.append({"source_dataset": name, "path": str(p),
                             "unreadable": True})
                continue
            f = extract(img)
            score, reason = classify(f)
            rows.append({"source_dataset": name, "path": str(p),
                         "unreadable": False, "rover_relevance": score,
                         "reason": reason,
                         "image_type_hint": image_type_hint(f, score),
                         **f.as_dict()})

    if not rows:
        print("no local imagery found; nothing to calibrate against")
        return 1

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "local_inventory_features.csv", index=False)

    ok = df[~df.unreadable]
    print("\n" + "=" * 70)
    print("ON-DISK TOTALS (every tomato image, not a sample)")
    for k, v in totals.items():
        print(f"  {k:15s} {v:6d}")
    print(f"  {'TOTAL':15s} {sum(totals.values()):6d}")

    print("\nROVER-RELEVANCE DISTRIBUTION (measured sample)")
    tab = ok.pivot_table(index="source_dataset", columns="rover_relevance",
                         values="path", aggfunc="count").fillna(0).astype(int)
    print(tab.to_string())

    print("\nCALIBRATION AGAINST DIRECT INSPECTION")
    all_pass = True
    for name, note in SPECIMEN_PROTOCOL.items():
        g = ok[ok.source_dataset == name]
        if g.empty:
            continue
        frac = float((g.rover_relevance <= 1).mean())
        good = frac >= SPECIMEN_LOW_MIN
        all_pass &= good
        print(f"  {name:15s} specimen protocol: {note}")
        print(f"  {'':15s} {frac:6.1%} scored 0-1 (need >={SPECIMEN_LOW_MIN:.0%})  "
              f"[{'PASS' if good else 'FAIL'}]")
    for name, note in WILD_PROTOCOL.items():
        g = ok[ok.source_dataset == name]
        if g.empty:
            continue
        frac = float((g.rover_relevance >= 2).mean())
        good = frac >= WILD_HIGH_MIN
        all_pass &= good
        print(f"  {name:15s} in-the-wild protocol: {note}")
        print(f"  {'':15s} {frac:6.1%} scored 2-3 (need >={WILD_HIGH_MIN:.0%})  "
              f"[{'PASS' if good else 'FAIL'}]")

    means = ok.groupby("source_dataset").rover_relevance.mean()
    order = list(means.sort_values().index)
    expected_order = ["tomatovillage", "plantvillage", "plantdoc"]
    order_ok = order == expected_order
    all_pass &= order_ok
    print(f"  ordering by mean score: {' < '.join(order)}")
    print(f"  {'':15s} expected {' < '.join(expected_order)}  "
          f"[{'PASS' if order_ok else 'FAIL'}]")

    print("\nFEATURE MEANS BY SOURCE")
    feats = ["border_scene_frac", "border_achromatic_frac", "border_hue_entropy",
             "border_texture", "edge_density", "green_fraction", "soil_fraction",
             "largest_green_blob", "sharpness"]
    print(ok.groupby("source_dataset")[feats].mean().round(3).to_string())

    print("\n" + "=" * 70)
    if all_pass:
        print("RUBRIC CALIBRATED: it recovers the inspected ordering on all three "
              "sources without seeing dataset labels.")
    else:
        print("RUBRIC NOT CALIBRATED. Do not use it to filter downloads until the "
              "failing source is understood.")
    print(f"wrote {(OUT / 'local_inventory_features.csv').relative_to(ROOT)}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
