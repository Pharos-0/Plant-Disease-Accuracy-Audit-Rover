"""Merge the manually-registered sources into the corpus, with dedup.

Provenance ranking decides which copy of a duplicated image survives, and it is
deliberately NOT "whichever licence is most permissive".

SRC004 redistributes 311 images that are byte-identical to PlantDoc originals,
under its own CC BY 4.0 claim. Re-publishing someone else's images under a new
licence does not make them so licensed. Keeping the SRC004 copy would launder an
unverified licence into a verified one; keeping the PlantDoc copy preserves the
conservative status (local use only, not redistributable). SRC004 therefore ranks
LAST, so that for any duplicate it loses and the cautious record survives.

    python dataset/scripts/merge_manual.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse_pilot import hamming  # noqa: E402

ROOT = HERE.parents[1]
CFG = yaml.safe_load((ROOT / "dataset" / "config.yaml").read_text(encoding="utf-8"))
META = ROOT / "dataset" / "metadata"
HAMMING_MAX = CFG["dedup"]["phash_hamming_max"]

#: Lower rank wins a duplicate contest. See the module docstring for why SRC004
#: is last despite carrying the more permissive licence claim.
PROVENANCE_RANK = {
    "SRC014": 0,   # rover-captured, CC BY 4.0, original
    "SRC015": 1,   # handheld, CC BY 4.0, original
    "SRC005": 2,   # CC BY 4.0, no reuse detected
    "SRC002": 3,   # PlantDoc original; licence unverified, local only
    "SRC004": 4,   # redistributes PlantDoc under its own CC BY 4.0 claim
}


def main() -> int:
    existing = pd.read_csv(META / "metadata.csv")
    manual = pd.read_csv(META / "manual_downloads.csv")
    print(f"existing corpus: {len(existing)}   newly registered: {len(manual)}")

    for col in set(existing.columns) - set(manual.columns):
        manual[col] = ""
    for col in set(manual.columns) - set(existing.columns):
        existing[col] = ""
    both = pd.concat([existing, manual], ignore_index=True)
    both["prov_rank"] = both.source_id.map(PROVENANCE_RANK).fillna(99).astype(int)
    both = both.sort_values(["prov_rank", "source_id", "original_filename"]).reset_index(drop=True)

    print("\nDEDUP across all sources")
    kept: list[tuple[int, int]] = []
    keep = np.ones(len(both), dtype=bool)
    dup_of = [""] * len(both)
    dup_src = [""] * len(both)
    hashes = both.phash.to_numpy()
    for i, h in enumerate(hashes):
        hit = None
        for hh, j in kept:
            if hamming(int(h), int(hh)) <= HAMMING_MAX:
                hit = j
                break
        if hit is None:
            kept.append((int(h), i))
        else:
            keep[i] = False
            dup_of[i] = str(both.at[hit, "sha256"])
            dup_src[i] = str(both.at[hit, "source_id"])
    both["retained"] = keep
    both["duplicate_of_sha256"] = dup_of
    both["duplicate_of_source"] = dup_src

    dropped = both[~both.retained]
    print(f"  {len(dropped)} duplicates dropped, {int(keep.sum())} retained")
    if not dropped.empty:
        print("\n  dropped-from -> kept-in:")
        print(dropped.groupby(["source_id", "duplicate_of_source"]).size().to_string())

    retained = both[both.retained].copy()
    retained["image_id"] = [f"TOM_{i:06d}" for i in range(1, len(retained) + 1)]
    retained.to_csv(META / "metadata.csv", index=False)
    pd.concat([pd.read_csv(META / "rejected_images.csv"), dropped],
              ignore_index=True).to_csv(META / "rejected_images.csv", index=False)

    print("\nRETAINED BY SOURCE")
    print(retained.groupby("source_id")
          .agg(images=("image_id", "size"),
               mean_relevance=("rover_relevance", "mean"),
               wide=("camera_distance_estimate", lambda s: (s == "WIDE").sum()))
          .round(2).to_string())
    print(f"\ntotal retained: {len(retained)}")
    print(f"wrote {META / 'metadata.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
