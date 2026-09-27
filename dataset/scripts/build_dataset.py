"""Scale up: extract every image from the licence-verified archives, score it,
deduplicate across sources, and write the full provenance table.

Order matters and is deliberate:

  1. extract every image member, storing by SHA-256 so an exact duplicate
     cannot be written twice even across runs;
  2. measure the rover-relevance features on all of them;
  3. deduplicate by perceptual hash ACROSS sources, keeping the copy from the
     source with the stronger provenance;
  4. assign a leakage group - for video-derived sources consecutive frames are
     near-identical, so the group is the capture sequence, not the image;
  5. write metadata.csv with one row per retained image and a rejection reason
     for every image not retained.

Nothing is deleted. A dropped duplicate stays on disk and is recorded in
rejected_images.csv with the id of the copy that was kept.

    python dataset/scripts/build_dataset.py
"""
from __future__ import annotations

import io
import re
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
from rover_relevance import extract, classify, camera_distance_estimate  # noqa: E402
from analyse_pilot import PHASH_BLOCK, PHASH_SIDE, hamming  # noqa: E402

ROOT = HERE.parents[1]
CFG = yaml.safe_load((ROOT / "dataset" / "config.yaml").read_text(encoding="utf-8"))
META = ROOT / "dataset" / "metadata"
RAW = Path(CFG["paths"]["raw"])
CACHE = Path(CFG["paths"]["store"]) / "_archives"
UA = {"User-Agent": "rover-tomato-dataset/0.1 (academic research)"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HAMMING_MAX = CFG["dedup"]["phash_hamming_max"]

#: Archives to build from, in provenance order. When two sources hold the same
#: image the EARLIER one wins, because its provenance is the stronger claim:
#: SRC014 is rover-captured and SRC015 states it reused SRC014 imagery.
ARCHIVES = [
    ("SRC014", "10.5281/zenodo.5596799"),
    ("SRC015", "10.5281/zenodo.8429227"),
]

#: Locally held sources, usable but NOT redistributable (licence unverified).
LOCAL = {"SRC002": Path(CFG["local_sources"]["plantdoc"]["root"])}


def phash_arr(gray: np.ndarray) -> int:
    g = cv2.resize(gray, (PHASH_SIDE, PHASH_SIDE), interpolation=cv2.INTER_AREA)
    d = cv2.dct(np.float32(g))[:PHASH_BLOCK, :PHASH_BLOCK]
    flat = d.flatten()[1:]
    out = 0
    for b in flat > np.median(flat):
        out = (out << 1) | int(b)
    return out


def sequence_group(sid: str, member: str) -> str:
    """Leakage group. Video-derived frames must never straddle a split.

    SRC014 is a ZED camera recording driven along inter-rows, so its frames come
    in sequences and consecutive ones are near-identical. The archive encodes
    that in the directory path and the frame index; the group is therefore the
    directory plus the non-numeric stem, never the individual frame.
    """
    p = Path(member)
    parent = "/".join(p.parts[:-1]) or "root"
    stem = re.sub(r"\d+", "#", p.stem)
    return f"{sid}:{parent}:{stem}"


def fetch(sid: str, doi: str) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{sid}.zip"
    if cached.exists():
        print(f"  {sid}: cached archive ({cached.stat().st_size/1e6:,.1f} MB)")
        return cached.read_bytes()
    rec = requests.get(f"https://zenodo.org/api/records/{doi.rsplit('zenodo.', 1)[1]}",
                       headers=UA, timeout=60).json()
    f = [x for x in rec["files"] if x["key"].lower().endswith(".zip")][0]
    print(f"  {sid}: downloading {f['key']} ({f['size']/1e6:,.1f} MB)")
    body = requests.get(f["links"]["self"], headers=UA, timeout=1800).content
    cached.write_bytes(body)
    return body


def ingest_archive(sid: str, doi: str) -> list[dict]:
    body = fetch(sid, doi)
    dest = RAW / sid
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        members = [m for m in zf.namelist()
                   if Path(m).suffix.lower() in IMAGE_EXTS and not m.endswith("/")]
        print(f"  {sid}: {len(members)} image members")
        for i, m in enumerate(members):
            raw = zf.read(m)
            import hashlib
            digest = hashlib.sha256(raw).hexdigest()
            out = dest / f"{digest[:16]}{Path(m).suffix.lower()}"
            if not out.exists():
                out.write_bytes(raw)
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                rows.append({"source_id": sid, "original_filename": m,
                             "stored_path": str(out), "sha256": digest,
                             "unreadable": True})
                continue
            f = extract(arr)
            score, reason = classify(f)
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            rows.append({
                "source_id": sid, "original_filename": m, "stored_path": str(out),
                "sha256": digest, "unreadable": False, "phash": phash_arr(gray),
                "rover_relevance": score, "relevance_reason": reason,
                "camera_distance_estimate": camera_distance_estimate(f),
                "leakage_group": sequence_group(sid, m),
                "source_url": f"https://doi.org/{doi}",
                "bytes": len(raw), **f.as_dict(),
            })
            if (i + 1) % 400 == 0:
                print(f"      {i+1}/{len(members)}")
    return rows


def ingest_local(sid: str, root: Path) -> list[dict]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    print(f"  {sid}: {len(files)} local files (NOT redistributable)")
    rows = []
    import hashlib
    for i, p in enumerate(files):
        raw = p.read_bytes()
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            continue
        f = extract(arr)
        score, reason = classify(f)
        rows.append({
            "source_id": sid, "original_filename": str(p.relative_to(root)),
            "stored_path": str(p), "sha256": hashlib.sha256(raw).hexdigest(),
            "unreadable": False, "phash": phash_arr(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)),
            "rover_relevance": score, "relevance_reason": reason,
            "camera_distance_estimate": camera_distance_estimate(f),
            # PlantDoc is web-scraped stills: the class directory is the only
            # grouping available, and it is weaker than a capture sequence.
            "leakage_group": f"{sid}:{p.parent.name}",
            "source_url": "local, licence unverified",
            "bytes": len(raw), **f.as_dict(),
        })
        if (i + 1) % 300 == 0:
            print(f"      {i+1}/{len(files)}")
    return rows


def main() -> int:
    rows: list[dict] = []
    print("INGEST")
    for sid, doi in ARCHIVES:
        rows += ingest_archive(sid, doi)
    for sid, root in LOCAL.items():
        if root.is_dir():
            rows += ingest_local(sid, root)

    d = pd.DataFrame(rows)
    ok = d[~d.unreadable].reset_index(drop=True).copy()
    print(f"\ningested {len(d)} files, {len(ok)} readable")

    # ---- cross-source dedup ------------------------------------------------
    print("\nDEDUP")
    order = {sid: i for i, (sid, _) in enumerate(ARCHIVES)}
    order.update({sid: 100 + i for i, sid in enumerate(LOCAL)})
    ok["prov_rank"] = ok.source_id.map(order).fillna(999).astype(int)
    ok = ok.sort_values(["prov_rank", "source_id", "original_filename"]).reset_index(drop=True)

    kept_hashes: list[tuple[int, int]] = []       # (phash, row index of keeper)
    keep_flag = np.ones(len(ok), dtype=bool)
    dup_of = [""] * len(ok)
    for i, h in enumerate(ok.phash.to_numpy()):
        hit = None
        for hh, j in kept_hashes:
            if hamming(int(h), int(hh)) <= HAMMING_MAX:
                hit = j
                break
        if hit is None:
            kept_hashes.append((int(h), i))
        else:
            keep_flag[i] = False
            dup_of[i] = str(ok.at[hit, "sha256"])
    ok["retained"] = keep_flag
    ok["duplicate_of_sha256"] = dup_of
    print(f"  {int((~keep_flag).sum())} duplicates dropped, "
          f"{int(keep_flag.sum())} retained")
    cross = ok[(~ok.retained)]
    if not cross.empty:
        print("  dropped, by source:")
        print(cross.groupby("source_id").size().to_string())

    # ---- write -------------------------------------------------------------
    ok["image_id"] = [f"TOM_{i:06d}" for i in range(1, len(ok) + 1)]
    ok["image_type"] = np.where(ok.rover_relevance >= 2,
                                "REAL_SCENE_ROVER_RELEVANT",
                                np.where(ok.rover_relevance == 0,
                                         "REAL_ISOLATED_LEAF_OR_STUDIO", "UNKNOWN"))
    ok["is_synthetic"] = False
    ok["processing_applied"] = ""
    ok["train_val_test_split"] = "UNASSIGNED"

    retained = ok[ok.retained]
    retained.to_csv(META / "metadata.csv", index=False)
    ok[~ok.retained].to_csv(META / "rejected_images.csv", index=False)
    print(f"\nwrote metadata.csv ({len(retained)} rows) and "
          f"rejected_images.csv ({len(ok) - len(retained)} rows)")

    print("\nRETAINED BY SOURCE AND RELEVANCE")
    print(retained.pivot_table(index="source_id", columns="rover_relevance",
                               values="image_id", aggfunc="count")
          .fillna(0).astype(int).to_string())
    print("\nRETAINED BY CAMERA DISTANCE")
    print(retained.pivot_table(index="source_id", columns="camera_distance_estimate",
                               values="image_id", aggfunc="count")
          .fillna(0).astype(int).to_string())
    print(f"\nleakage groups: {retained.leakage_group.nunique()} over "
          f"{len(retained)} images "
          f"({len(retained)/max(1, retained.leakage_group.nunique()):.1f} per group)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
