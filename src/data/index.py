"""Build a unified image index for the three domains.

Each row carries a ``group_id``: the unit that must never be split across a
train/test boundary or across sessions.

* PlantVillage - physical leaf, from ``leaf_grouping/filtered_leafmaps/*.csv``.
  The dataset ships 4 photographs per leaf, so an image-level split leaks the
  same leaf into train and test (docs/dataset_class_mapping.md section 6.1).
* Tomato-Village - source photograph, from the ``IMG<date><time>[_n]`` filename
  stem; ``_n`` marks several crops of one photograph (section 6.2).
* PlantDoc - web-scraped singletons; no known duplicate structure.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import paths
from .mapping import canonical_label

# Tomato-Village: IMG<YYYYMMDD><HHMMSS> optionally followed by _<crop index>
TV_STEM = re.compile(r"^(IMG(\d{8})(\d{6}))(?:_(\d+))?$")


def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in paths.IMAGE_SUFFIXES)


def _load_leafmaps() -> dict[tuple[str, str], str]:
    """(class folder, original file name) -> leaf id."""
    out: dict[tuple[str, str], str] = {}
    for csv_path in sorted(paths.PLANTVILLAGE_LEAFMAPS.glob("Tomato*.csv")):
        cls = csv_path.stem
        df = pd.read_csv(csv_path, dtype=str)
        for fname, leaf in zip(df["File Name"], df["Leaf #"]):
            if isinstance(fname, str) and isinstance(leaf, str):
                out[(cls, fname.strip())] = leaf.strip()
    return out


def build_plantvillage() -> pd.DataFrame:
    leafmaps = _load_leafmaps()
    rows = []
    for cls_dir in sorted(paths.PLANTVILLAGE_IMAGES.glob("Tomato___*")):
        folder = cls_dir.name
        label = canonical_label("plantvillage", folder)
        for p in _images_in(cls_dir):
            # PlantVillage prefixes a uuid: "<uuid>___<original name>"
            original = p.name.split("___", 1)[1] if "___" in p.name else p.name
            leaf = leafmaps.get((folder, original))
            rows.append(
                {
                    "path": str(p),
                    "domain": "plantvillage",
                    "folder": folder,
                    "label": label,
                    "orig_split": "raw",
                    "original_name": original,
                    "group_id": f"leaf:{folder}:{leaf}" if leaf else None,
                    "group_kind": "leaf" if leaf else "ungrouped",
                }
            )
    return pd.DataFrame(rows)


def build_plantdoc() -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        split_dir = paths.PLANTDOC_ROOT / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            folder = cls_dir.name
            label = canonical_label("plantdoc", folder)
            for p in _images_in(cls_dir):
                rows.append(
                    {
                        "path": str(p),
                        "domain": "plantdoc",
                        "folder": folder,
                        "label": label,
                        "orig_split": split,
                        "original_name": p.name,
                        "group_id": f"img:{split}:{folder}:{p.name}",
                        "group_kind": "image",
                    }
                )
    return pd.DataFrame(rows)


def build_tomatovillage() -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        split_dir = paths.TOMATOVILLAGE_ROOT / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            folder = cls_dir.name
            label = canonical_label("tomatovillage", folder)
            for p in _images_in(cls_dir):
                m = TV_STEM.match(p.stem)
                if m:
                    base, date, _time, _crop = m.group(1), m.group(2), m.group(3), m.group(4)
                    group_id, group_kind, capture_date = f"ts:{base}", "photograph", date
                else:
                    group_id, group_kind, capture_date = f"file:{p.name}", "unparsed", None
                rows.append(
                    {
                        "path": str(p),
                        "domain": "tomatovillage",
                        "folder": folder,
                        "label": label,
                        "orig_split": split,
                        "original_name": p.name,
                        "group_id": group_id,
                        "group_kind": group_kind,
                        "capture_date": capture_date,
                    }
                )
    return pd.DataFrame(rows)


def build_pakistan() -> pd.DataFrame:
    """Fourth target arm: the raw folder of Mendeley 10.17632/3mbnb82mxd.2.

    Flat class directories, one image per group: these are individual stills
    from mixed origins (phone captures and web imagery), with no multi-crop or
    sequence structure to group by. Files are named by SHA-256 prefix, which is
    how the dataset build guarantees an image cannot be admitted twice.
    """
    rows = []
    root = paths.PAKISTAN_ROOT
    if not root.is_dir():
        return pd.DataFrame(rows)
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        folder = cls_dir.name
        label = canonical_label("pakistan", folder)
        for p in _images_in(cls_dir):
            rows.append(
                {
                    "path": str(p),
                    "domain": "pakistan",
                    "folder": folder,
                    "label": label,
                    "orig_split": "all",
                    "original_name": p.name,
                    "group_id": f"img:pakistan:{folder}:{p.name}",
                    "group_kind": "image",
                }
            )
    return pd.DataFrame(rows)


BUILDERS = {
    "plantvillage": build_plantvillage,
    "plantdoc": build_plantdoc,
    "tomatovillage": build_tomatovillage,
    "pakistan": build_pakistan,
}


def build_all() -> pd.DataFrame:
    # NOTE: "pakistan" is deliberately NOT in this tuple. `index.parquet` is hashed
    # into all 47 experiment manifests by utils/manifest.py, so regenerating it
    # with a fourth domain would invalidate the provenance chain of every frozen
    # result. The fourth arm is built on demand via BUILDERS["pakistan"] and kept
    # in its own index file. See docs/arm4_pakistan.md.
    frames = [BUILDERS[d]() for d in ("plantvillage", "plantdoc", "tomatovillage")]
    df = pd.concat(frames, ignore_index=True)
    if "capture_date" not in df.columns:
        df["capture_date"] = None
    return df


def labelled(df: pd.DataFrame, label_set: list[str]) -> pd.DataFrame:
    """Rows whose canonical label is inside `label_set`."""
    return df[df["label"].isin(label_set)].copy()
