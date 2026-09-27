"""Split by group, then VERIFY the split rather than trusting the grouping.

Two things happen here, and the second is the one that matters:

  1. Groups from `leakage_groups.py` are assigned whole to train / val / test,
     stratified by source so no split is starved of a source.
  2. Every train-test and val-test pair is then compared by perceptual hash. If
     any pair is closer than the dedup threshold, the split has leaked and the
     script FAILS. A grouping that looks right is not evidence; a verified split
     is.

    python dataset/scripts/create_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakage_groups  # noqa: E402
from analyse_pilot import hamming  # noqa: E402

ROOT = HERE.parents[1]
CFG = yaml.safe_load((ROOT / "dataset" / "config.yaml").read_text(encoding="utf-8"))
META = ROOT / "dataset" / "metadata"
SPLITS = ROOT / "dataset" / "splits"
HAMMING_MAX = CFG["dedup"]["phash_hamming_max"]

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 20260928


def main() -> int:
    d = pd.read_csv(META / "metadata.csv")
    d = leakage_groups.assign(d)

    print("LEAKAGE GROUPING")
    print(leakage_groups.summarise(d).to_string())
    print(f"\n  {d.leakage_group.nunique()} groups over {len(d)} images")
    print(f"  {int(d.guard_drop.sum())} frames dropped as guard band at block "
          "boundaries (retained in the dataset, excluded from every split)")

    pool = d[~d.guard_drop].copy()
    rng = np.random.default_rng(SEED)
    assign: dict[str, str] = {}

    # Stratify by source: allocate each source's groups independently, so a
    # source with few groups cannot vanish from a split by chance.
    for sid, g in pool.groupby("source_id"):
        gs = g.leakage_group.unique()
        rng.shuffle(gs)
        n = len(gs)
        n_tr = max(1, int(round(n * FRACTIONS["train"])))
        n_va = max(0, int(round(n * FRACTIONS["val"])))
        if n_tr + n_va >= n:                       # guarantee a non-empty test
            n_va = max(0, n - n_tr - 1)
        for i, grp in enumerate(gs):
            assign[grp] = ("train" if i < n_tr
                           else "val" if i < n_tr + n_va else "test")
        print(f"  {sid}: {n} groups -> train {n_tr}, val {n_va}, test {n - n_tr - n_va}")

    pool["train_val_test_split"] = pool.leakage_group.map(assign)

    print("\nSPLIT SIZES")
    print(pool.pivot_table(index="source_id", columns="train_val_test_split",
                           values="image_id", aggfunc="count")
          .fillna(0).astype(int).to_string())
    print("\nBY ROVER RELEVANCE")
    print(pool.pivot_table(index="train_val_test_split", columns="rover_relevance",
                           values="image_id", aggfunc="count")
          .fillna(0).astype(int).to_string())

    # ------------------------------------------------------------- verify
    print(f"\nLEAKAGE VERIFICATION (pHash hamming <= {HAMMING_MAX} across splits)")
    leaks = []
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        ha = pool[pool.train_val_test_split == a]
        hb = pool[pool.train_val_test_split == b]
        pa, pb = ha.phash.to_numpy(), hb.phash.to_numpy()
        ia, ib = ha.image_id.to_numpy(), hb.image_id.to_numpy()
        found = 0
        for i, x in enumerate(pa):
            for j, y in enumerate(pb):
                if hamming(int(x), int(y)) <= HAMMING_MAX:
                    leaks.append({"split_a": a, "split_b": b,
                                  "image_a": ia[i], "image_b": ib[j],
                                  "hamming": hamming(int(x), int(y))})
                    found += 1
        print(f"  {a:5s} x {b:5s}: {found} near-duplicate pairs")

    SPLITS.mkdir(parents=True, exist_ok=True)
    for name in FRACTIONS:
        s = pool[pool.train_val_test_split == name]
        s[["image_id", "source_id", "stored_path", "leakage_group",
           "rover_relevance", "camera_distance_estimate"]].to_csv(
            SPLITS / f"{name}.csv", index=False)
    pool.to_csv(META / "metadata_split.csv", index=False)

    if leaks:
        pd.DataFrame(leaks).to_csv(META / "leakage_violations.csv", index=False)
        print(f"\n  FAIL: {len(leaks)} cross-split near-duplicate pairs. The split "
              "is NOT clean; see metadata/leakage_violations.csv")
        return 2

    print("\n  PASS: no cross-split near-duplicate pairs.")
    print("\n  WHAT THIS DOES NOT CLAIM: SRC014's frames come from only TWO capture")
    print("  sessions, both on 2020-08-06 in one greenhouse. Temporal blocks are")
    print("  segments of two traverses, not independent samples. The split prevents")
    print("  frame-level leakage; it cannot manufacture independence.")
    print(f"\nwrote splits to {SPLITS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
