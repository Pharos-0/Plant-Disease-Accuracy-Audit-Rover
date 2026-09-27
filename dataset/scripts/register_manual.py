"""Register archives fetched by hand, with the same provenance discipline as the
automated path.

Mendeley Data answers its documented API with HTTP 403 from this environment - a
bot wall, not a licence restriction. Rather than forge headers to get past it,
these two CC BY 4.0 archives were downloaded through a real browser, which is
the access route the publisher intends. That means the bytes arrive without the
API metadata the automated downloader records, so this script reconstructs the
same provenance fields from the registry and from the files themselves.

Every registered image gets: source DOI, licence and licence URL, the archive's
own SHA-256, the member path inside it, the member's SHA-256, and the download
date. Nothing is asserted that was not read from the registry or computed here.

    python dataset/scripts/register_manual.py --verify-only
    python dataset/scripts/register_manual.py
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rover_relevance import extract, classify, camera_distance_estimate  # noqa: E402
from analyse_pilot import PHASH_BLOCK, PHASH_SIDE  # noqa: E402

ROOT = HERE.parents[1]
CFG = yaml.safe_load((ROOT / "dataset" / "config.yaml").read_text(encoding="utf-8"))
META = ROOT / "dataset" / "metadata"
RAW = Path(CFG["paths"]["raw"])
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#: Archives downloaded through the browser, matched by glob because the browser
#: truncates long filenames. `exclude` is applied to the member path INSIDE the
#: archive and is not optional: it is how synthetic content is kept out.
MANUAL = {
    "SRC005": {
        "glob": "Tomato Leaf Image Dataset for Disease Analysis*.zip",
        "doi": "10.17632/rnbsw72zb5.1",
        "expect_images": 1028,
        "exclude": (),
        "note": "publisher states 1,028; no augmentation markers found",
    },
    "SRC004": {
        "glob": "Tomato Leaf Disease Classification Dataset in Paki*.zip",
        "doi": "10.17632/3mbnb82mxd.2",
        "expect_images": 7200,
        # Every one of the 7,200 members under this prefix ends `aug<N>`.
        # They are synthetic augmentations of the 830 raw images, and the build
        # plan bars synthetic content from the dataset.
        "exclude": ("Augmented Dataset/",),
        "note": ("publisher's advertised 7,200 IS the augmented folder; only the "
                 "830-image raw folder is admitted, then deduplicated"),
    },
}

#: A member is synthetic if its name carries an augmentation marker. Checked in
#: addition to the folder exclusion, so a stray augmented file outside the
#: Augmented folder is still caught.
AUG_MARKER = re.compile(r"aug\d+\.(?:jpg|jpeg|png|bmp|tif|tiff)$", re.I)


def phash_gray(gray: np.ndarray) -> int:
    g = cv2.resize(gray, (PHASH_SIDE, PHASH_SIDE), interpolation=cv2.INTER_AREA)
    d = cv2.dct(np.float32(g))[:PHASH_BLOCK, :PHASH_BLOCK]
    out = 0
    for b in d.flatten()[1:] > np.median(d.flatten()[1:]):
        out = (out << 1) | int(b)
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def image_members(zf: zipfile.ZipFile, exclude: tuple = ()) -> list[str]:
    """Image members, minus anything excluded or carrying an augmentation marker."""
    out = []
    for m in zf.namelist():
        if m.endswith("/") or Path(m).suffix.lower() not in IMAGE_EXTS:
            continue
        if any(x in m for x in exclude) or AUG_MARKER.search(m):
            continue
        out.append(m)
    return out


def open_payload(path: Path):
    """Yield (label, ZipFile). Mendeley bundles wrap the authors' own zip."""
    outer = zipfile.ZipFile(path)
    inner_zips = [m for m in outer.namelist() if m.lower().endswith(".zip")]
    if image_members(outer, ()):
        yield path.name, outer
    for m in inner_zips:
        import io
        yield f"{path.name}!{m}", zipfile.ZipFile(io.BytesIO(outer.read(m)))


def verify(sid: str, spec: dict) -> tuple[bool, str]:
    p: Path = spec["archive"]
    if not p.exists():
        return False, "archive not present"
    try:
        total = admitted = 0
        for label, zf in open_payload(p):
            allm = len(image_members(zf, ()))
            keep = len(image_members(zf, spec["exclude"]))
            total += allm
            admitted += keep
            print(f"    {label}: {allm} image members, {keep} admitted "
                  f"({allm - keep} excluded as synthetic or filtered)")
        if admitted == 0:
            return False, "no admissible image members"
        exp = spec["expect_images"]
        flag = "" if total == exp else "  <-- differs from the publisher's figure"
        return True, (f"{total} in archive vs {exp} stated{flag}; "
                      f"{admitted} admitted. {spec['note']}")
    except zipfile.BadZipFile as e:
        return False, f"not a valid zip ({e}) - download likely incomplete"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    lic = pd.read_csv(META / "license_registry.csv").set_index("source_id")

    print("VERIFY")
    ok_sources = {}
    for sid, spec in MANUAL.items():
        found = sorted(ROOT.glob(spec["glob"]))
        if not found:
            print(f"  {sid}  no archive matching {spec['glob']} - skipping\n")
            continue
        spec["archive"] = found[0]
        print(f"  {sid}  {spec['archive'].name}")
        print(f"    {spec['archive'].stat().st_size:,} bytes")
        good, note = verify(sid, spec)
        print(f"    {'OK' if good else 'FAIL'}: {note}\n")
        if good:
            ok_sources[sid] = spec

    if args.verify_only or not ok_sources:
        return 0 if ok_sources else 1

    print("REGISTER")
    rows = []
    for sid, spec in ok_sources.items():
        row_lic = lic.loc[sid]
        if row_lic.license_status != "VERIFIED":
            print(f"  {sid}: licence is {row_lic.license_status}, refusing")
            continue
        archive_sha = sha256_file(spec["archive"])
        print(f"  {sid}: archive sha256 {archive_sha[:16]}...")
        dest = RAW / sid
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for label, zf in open_payload(spec["archive"]):
            for m in image_members(zf, spec["exclude"]):
                body = zf.read(m)
                digest = hashlib.sha256(body).hexdigest()
                out = dest / f"{digest[:16]}{Path(m).suffix.lower()}"
                if not out.exists():
                    out.write_bytes(body)
                arr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    continue
                f = extract(arr)
                score, reason = classify(f)
                rows.append({
                    "source_id": sid,
                    "original_filename": f"{label}::{m}",
                    "stored_path": str(out),
                    "sha256": digest,
                    "archive_sha256": archive_sha,
                    "unreadable": False,
                    "phash": phash_gray(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)),
                    "rover_relevance": score,
                    "relevance_reason": reason,
                    "camera_distance_estimate": camera_distance_estimate(f),
                    "source_url": f"https://doi.org/{spec['doi']}",
                    "license": row_lic.license_name,
                    "license_url": row_lic.license_url,
                    "download_date": time.strftime("%Y-%m-%d"),
                    "acquisition_route": "manual browser download (API returns 403)",
                    "is_synthetic": False,
                    "admission_note": spec["note"],
                    "bytes": len(body), **f.as_dict(),
                })
                n += 1
                if n % 500 == 0:
                    print(f"      {n} registered")
        print(f"    {n} images registered\n")

    if not rows:
        print("nothing registered")
        return 1

    d = pd.DataFrame(rows)
    d.to_csv(META / "manual_downloads.csv", index=False)
    print(f"wrote {META / 'manual_downloads.csv'} ({len(d)} rows)")
    print("\nROVER RELEVANCE")
    print(d.pivot_table(index="source_id", columns="rover_relevance",
                        values="sha256", aggfunc="count").fillna(0).astype(int).to_string())
    print("\nCAMERA DISTANCE")
    print(d.pivot_table(index="source_id", columns="camera_distance_estimate",
                        values="sha256", aggfunc="count").fillna(0).astype(int).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
