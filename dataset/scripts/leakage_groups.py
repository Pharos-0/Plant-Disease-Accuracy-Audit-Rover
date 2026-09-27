"""Leakage grouping, derived from what each source's filenames actually encode.

A random image-level split would be dishonest here, and the reason is visible in
the filenames rather than assumed:

  SRC014  `tomate_barroselas_20200806_0000.jpg`,
          `tomates_2020-08-06-11-35-15_side_0248.jpg`
          A ZED camera on a robot driven along greenhouse inter-rows. Frames are
          consecutive video. **Only 2 capture sessions exist across 446 frames,
          both on 2020-08-06.** Neighbouring frames overlap heavily, so the group
          is a contiguous temporal block, and a guard band of dropped frames sits
          at every block boundary so that no two frames straddling a split can
          share content.

  SRC015  `image/1.jpg` ... `image/1474.jpg`
          Flat sequential numbering, no session metadata of any kind. The index
          is assumed to be acquisition order - which is a HEURISTIC, stated as
          such - so consecutive runs of indices are chunked into groups.

  SRC002  `train/Tomato leaf late blight/dscn3175.jpg`
          Web-scraped stills from unrelated sources; each image is treated as its
          own group. Exact and near-exact reuse was already removed by the
          perceptual-hash dedup, so what remains is genuinely distinct imagery.

What this CANNOT fix: SRC014's two sessions are one day, one greenhouse. Blocks
within a session are not independent samples, they are segments of two traverses.
The split prevents frame-level leakage; it does not manufacture independence, and
`documentation/PILOT_REPORT.md` says so.
"""
from __future__ import annotations

import re

import pandas as pd

#: Frames per temporal block for video-derived sources.
BLOCK_FRAMES = 40
#: Frames discarded either side of a block boundary. A ZED camera on a slow rover
#: overlaps heavily between neighbours; this buys separation at the cost of data.
GUARD_FRAMES = 5
#: Images per chunk for a source whose only ordering is a filename index.
INDEX_CHUNK = 25

_TRAIL_NUM = re.compile(r"(\d+)\.(?:jpg|jpeg|png|bmp|tif|tiff)$", re.I)
_SESSION = re.compile(r"_?\d+\.(?:jpg|jpeg|png|bmp|tif|tiff)$", re.I)


def _basename(path: str) -> str:
    return re.split(r"[\\/]", str(path))[-1]


def _frame_index(path: str) -> int | None:
    m = _TRAIL_NUM.search(_basename(path))
    return int(m.group(1)) if m else None


def _session(path: str) -> str:
    return _SESSION.sub("", _basename(path))


def assign(df: pd.DataFrame) -> pd.DataFrame:
    """Add `leakage_group` and `guard_drop` columns.

    `guard_drop` marks frames sitting on a block boundary. They are retained in
    the dataset but excluded from every split, so that no split boundary cuts
    through overlapping video.
    """
    group_by_idx: dict = {}
    guard_by_idx: dict = {}

    for sid, g in df.groupby("source_id", sort=False):
        idx = g.original_filename.map(_frame_index)
        has_index = bool(idx.notna().all())

        if sid == "SRC014" and has_index:
            sess = g.original_filename.map(_session)
            for row_idx, s, i in zip(g.index, sess, idx.astype(int)):
                block, pos = int(i) // BLOCK_FRAMES, int(i) % BLOCK_FRAMES
                group_by_idx[row_idx] = f"{sid}:{s}:block{block:03d}"
                guard_by_idx[row_idx] = bool(
                    pos < GUARD_FRAMES or pos >= BLOCK_FRAMES - GUARD_FRAMES)
        elif has_index:
            for row_idx, i in zip(g.index, idx.astype(int)):
                group_by_idx[row_idx] = f"{sid}:chunk{int(i) // INDEX_CHUNK:04d}"
                guard_by_idx[row_idx] = False
        else:
            for row_idx, fn in zip(g.index, g.original_filename):
                group_by_idx[row_idx] = f"{sid}:{_basename(fn)}"
                guard_by_idx[row_idx] = False

    out = df.copy()
    out["leakage_group"] = pd.Series(group_by_idx, dtype="object").reindex(out.index)
    out["guard_drop"] = pd.Series(guard_by_idx, dtype="bool").reindex(out.index).fillna(False)
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("source_id")
              .agg(images=("leakage_group", "size"),
                   groups=("leakage_group", "nunique"),
                   guard_dropped=("guard_drop", "sum"))
              .assign(per_group=lambda d: (d.images / d.groups).round(1)))
