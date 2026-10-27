"""Filesystem roots. Datasets live at a short root (C:/ds) to stay clear of MAX_PATH."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DS_ROOT = Path(r"C:\ds")
PLANTVILLAGE_ROOT = DS_ROOT / "plantvillage"
PLANTVILLAGE_IMAGES = PLANTVILLAGE_ROOT / "raw" / "color"
PLANTVILLAGE_LEAFMAPS = PLANTVILLAGE_ROOT / "leaf_grouping" / "filtered_leafmaps"
PLANTDOC_ROOT = DS_ROOT / "plantdoc_files"
#: Fourth target arm, added 2026-09-28. See docs/arm4_pakistan.md for what it
#: is and what it is NOT independent of.
PAKISTAN_ROOT = DS_ROOT / "pakistan_tomato"
TOMATOVILLAGE_ROOT = DS_ROOT / "tomatovillage" / "Variant-a(Multiclass Classification)"

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RESULTS = RESULTS_DIR / "raw"
ESTIMATES_DIR = RAW_RESULTS / "estimates"
CACHE_DIR = RAW_RESULTS / "cache"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = PROJECT_ROOT / "reports"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_RESULTS, ESTIMATES_DIR, CACHE_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
