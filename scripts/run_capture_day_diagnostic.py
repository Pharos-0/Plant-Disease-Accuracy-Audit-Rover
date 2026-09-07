"""Capture-day diagnostic on Tomato-Village (arm D).

`docs/dataset_class_mapping.md` §6.2 notes that the three capture days
(2022-03-23/24/25) are a *natural, defensible* basis for an easy / medium / hard
session comparison - real acquisition conditions rather than manufactured
difficulty. A session in the field is one outing, so a capture day is much
closer to a real session than a random bootstrap sample is.

This script treats each day as a session and asks: does estimator error track
the day's true difficulty, or is it constant regardless?

Estimates are produced without target labels; the true per-day accuracy is
computed afterwards, in the same two-phase order the firewall requires.

    python scripts/run_capture_day_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from data.splits import apply_exclusions, load_index
from estimators.base import SourceStats
from estimators.methods import ALL_ESTIMATORS, build
from inference.cache import CacheKey, load_features, load_labels


def main() -> int:
    paths.ensure_dirs()

    # capture date per Tomato-Village image, keyed by path
    df, _ = apply_exclusions(load_index())
    tv = df[df.domain == "tomatovillage"]
    date_of = dict(zip(tv["path"], tv["capture_date"]))


    models = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        c = m["config"]
        if c["label_set"] != "SHARED_3":
            continue
        models.append((c["version"], c["backbone"], c["seed"], c["label_set"]))

    if not models:
        print("no SHARED_3 models found")
        return 1

    rows = []
    for version, backbone, seed, label_set in models:
        src_key = CacheKey(version, backbone, seed, "source_val", label_set)
        tgt_key = CacheKey(version, backbone, seed, "tomatovillage", label_set)
        if not (src_key.features_path.exists() and tgt_key.features_path.exists()):
            continue

        src = load_features(src_key)
        source = SourceStats(logits=src["logits"], labels=load_labels(src_key))
        fitted = {n: build(n).fit(source) for n in ALL_ESTIMATORS}

        tgt = load_features(tgt_key)
        days = np.array([date_of.get(p) for p in tgt["paths"]])

        # --- phase 1: estimates, no labels ---
        estimates = {}
        for day in sorted({d for d in days if d}):
            mask = days == day
            for name, est in fitted.items():
                estimates[(day, name)] = est.estimate(
                    tgt["logits"][mask], tgt["features"][mask]
                )

        # --- phase 2: truth ---
        labels = load_labels(tgt_key)
        preds = tgt["logits"].argmax(1)
        for day in sorted({d for d in days if d}):
            mask = days == day
            actual = float((preds[mask] == labels[mask]).mean())
            for name in ALL_ESTIMATORS:
                rows.append(
                    {
                        "version": version,
                        "backbone": backbone,
                        "seed": seed,
                        "capture_date": day,
                        "n_images": int(mask.sum()),
                        "estimator": name,
                        "estimate": estimates[(day, name)],
                        "actual": actual,
                        "error": estimates[(day, name)] - actual,
                    }
                )

    out = pd.DataFrame(rows)
    if not len(out):
        print("no cached tomatovillage arms found")
        return 1

    out.to_csv(paths.TABLES_DIR / "capture_day_diagnostic.csv", index=False)

    print("=== true accuracy per capture day (is there an easy/medium/hard ordering?) ===")
    truth = out.groupby("capture_date").agg(
        n_images=("n_images", "first"),
        actual=("actual", "mean"),
    )
    print(truth.round(4).to_string())
    print()
    print("=== estimator error per capture day (signed) ===")
    piv = out.pivot_table(index="estimator", columns="capture_date", values="error", aggfunc="mean")
    print(piv.round(4).to_string())
    print()
    spread = truth.actual.max() - truth.actual.min()
    print(f"true accuracy spread across days: {spread:.4f}")
    if spread < 0.05:
        print("-> the three days are NOT an easy/medium/hard ladder; they are comparable.")
        print("   Report that rather than presenting them as graded difficulty.")
    else:
        print("-> the days do differ in difficulty; usable as a graded diagnostic.")
    print(f"\nwrote {paths.TABLES_DIR / 'capture_day_diagnostic.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
