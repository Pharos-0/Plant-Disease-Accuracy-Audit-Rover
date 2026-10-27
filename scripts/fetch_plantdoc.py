"""Fetch PlantDoc tomato classes on Windows.

PlantDoc filenames are scraped web URLs and 20 of the 747 tomato files contain
characters ( ? : * < > | " ) that NTFS refuses, which makes `git checkout` abort
the whole worktree. We therefore pull the zipball and extract only the tomato
entries, sanitising illegal characters and recording the original -> written
name map so provenance stays traceable (study protocol).
"""
from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import requests

ZIP_URL = "https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master"
DEST = Path(r"C:\ds\plantdoc_files")
ILLEGAL = re.compile(r'[?:*<>|"]')


def sanitise(name: str) -> str:
    """Replace NTFS-illegal characters; keep the name otherwise byte-faithful."""
    return ILLEGAL.sub("_", name).rstrip(" .")


def main() -> int:
    print(f"downloading {ZIP_URL}")
    resp = requests.get(ZIP_URL, timeout=600, stream=True)
    resp.raise_for_status()
    blob = io.BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        blob.write(chunk)
        total += len(chunk)
        print(f"\r  {total / 1024**2:8.1f} MB", end="", flush=True)
    print(f"\n  downloaded {total / 1024**2:.1f} MB")

    DEST.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    written = renamed = 0

    with zipfile.ZipFile(blob) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # strip the "PlantDoc-Dataset-master/" prefix the zipball adds
            rel = info.filename.split("/", 1)[1] if "/" in info.filename else info.filename
            parts = rel.split("/")
            if len(parts) != 3:
                continue
            split, cls, fname = parts
            if split not in ("train", "test") or not cls.lower().startswith("tomato"):
                continue

            safe = sanitise(fname)
            out_dir = DEST / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / safe

            # disambiguate the rare case where sanitising collides
            if out_path.exists():
                stem, suffix = out_path.stem, out_path.suffix
                n = 1
                while out_path.exists():
                    out_path = out_dir / f"{stem}__dup{n}{suffix}"
                    n += 1

            out_path.write_bytes(zf.read(info))
            written += 1
            if safe != fname:
                renamed += 1
            rows.append(
                {
                    "split": split,
                    "class": cls,
                    "original_name": fname,
                    "written_name": out_path.name,
                    "sanitised": str(safe != fname),
                }
            )

    map_path = DEST / "filename_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["split", "class", "original_name", "written_name", "sanitised"]
        )
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {written} tomato files ({renamed} sanitised) -> {DEST}")
    print(f"provenance map -> {map_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
