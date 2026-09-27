"""Known-answer validation of the duplicate pipeline, on the full archives.

SRC015's Zenodo description states that part of that dataset "was obtained from
https://zenodo.org/record/5596799" - which is SRC014. So genuine cross-source
duplicates exist by the authors' own account, and a dedup pipeline that reports
none is broken rather than clean.

The pilot sample could not settle this: it took the first 60 archive members of
each, which is a biased slice. This script hashes EVERY image member of both
archives, in memory, so the answer is definitive rather than suggestive.

It also answers the question that decides the real dataset size: how many
distinct images do the two sources contribute together?

    python dataset/scripts/overlap_test.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse_pilot import PHASH_BLOCK, PHASH_SIDE, hamming  # noqa: E402

ROOT = HERE.parents[1]
CFG = yaml.safe_load((ROOT / "dataset" / "config.yaml").read_text(encoding="utf-8"))
META = ROOT / "dataset" / "metadata"
CACHE = Path(CFG["paths"]["store"]) / "_archives"

UA = {"User-Agent": "rover-tomato-dataset/0.1 (academic research)"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HAMMING_MAX = CFG["dedup"]["phash_hamming_max"]

ARCHIVES = {
    "SRC014": "10.5281/zenodo.5596799",
    "SRC015": "10.5281/zenodo.8429227",
}


def phash_bytes(body: bytes) -> int | None:
    arr = np.frombuffer(body, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    g = cv2.resize(img, (PHASH_SIDE, PHASH_SIDE), interpolation=cv2.INTER_AREA)
    d = cv2.dct(np.float32(g))[:PHASH_BLOCK, :PHASH_BLOCK]
    flat = d.flatten()[1:]
    out = 0
    for b in flat > np.median(flat):
        out = (out << 1) | int(b)
    return out


def fetch_archive(sid: str, doi: str) -> bytes:
    """Download once, cache on the dataset volume, reuse thereafter."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{sid}.zip"
    if cached.exists():
        print(f"  {sid}: using cached archive ({cached.stat().st_size/1e6:,.1f} MB)")
        return cached.read_bytes()
    rec_id = doi.rsplit("zenodo.", 1)[1]
    rec = requests.get(f"https://zenodo.org/api/records/{rec_id}",
                       headers=UA, timeout=60).json()
    zips = [f for f in rec.get("files", []) if f["key"].lower().endswith(".zip")]
    if not zips:
        raise RuntimeError(f"{sid}: no .zip in record {rec_id}")
    f = zips[0]
    print(f"  {sid}: downloading {f['key']} ({f['size']/1e6:,.1f} MB)")
    body = requests.get(f["links"]["self"], headers=UA, timeout=900).content
    cached.write_bytes(body)
    return body


def main() -> int:
    per_source = {}
    for sid, doi in ARCHIVES.items():
        body = fetch_archive(sid, doi)
        rows = []
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            members = [m for m in zf.namelist()
                       if Path(m).suffix.lower() in IMAGE_EXTS and not m.endswith("/")]
            for m in members:
                h = phash_bytes(zf.read(m))
                if h is not None:
                    rows.append({"source_id": sid, "member": m, "phash": h})
        per_source[sid] = pd.DataFrame(rows)
        print(f"  {sid}: hashed {len(rows)} of {len(members)} image members")

    a, b = per_source["SRC014"], per_source["SRC015"]
    print(f"\ntotal image members: {len(a)} + {len(b)} = {len(a) + len(b)}")

    ha = a.phash.to_numpy()
    hb = b.phash.to_numpy()

    print(f"\nCROSS-SOURCE NEAR DUPLICATES (hamming <= {HAMMING_MAX})")
    pairs = []
    for i, x in enumerate(ha):
        for j, y in enumerate(hb):
            d = hamming(int(x), int(y))
            if d <= HAMMING_MAX:
                pairs.append({"src014_member": a.member.iloc[i],
                              "src015_member": b.member.iloc[j], "hamming": d})
    cross = pd.DataFrame(pairs)
    print(f"  {len(cross)} pairs")
    if not cross.empty:
        cross.to_csv(META / "overlap_src014_src015.csv", index=False)
        n_b = cross.src015_member.nunique()
        print(f"  {cross.src014_member.nunique()} distinct SRC014 images matched")
        print(f"  {n_b} distinct SRC015 images matched "
              f"({n_b / len(b):.1%} of SRC015)")
        print(f"  exact pHash matches (hamming 0): {(cross.hamming == 0).sum()}")

    print("\nKNOWN-ANSWER VERDICT")
    if len(cross) > 0:
        print("  PASS - the pipeline recovers the reuse the SRC015 authors declared.")
    else:
        print("  FAIL - the authors state SRC015 reuses SRC014 imagery, and the")
        print("  pipeline found none. Either the threshold is too tight or the")
        print("  hash is wrong. Do NOT treat the corpus as duplicate-free.")

    uniq_b = len(b) - (cross.src015_member.nunique() if not cross.empty else 0)
    print(f"\nDISTINCT CONTRIBUTION: {len(a)} + {uniq_b} = {len(a) + uniq_b} "
          f"unique images from these two sources")
    return 0 if len(cross) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
