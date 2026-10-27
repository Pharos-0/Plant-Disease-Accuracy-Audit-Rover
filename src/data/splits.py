"""Exclusions and the V1 / V2 source splits.

Two split protocols over the *same* image pool, so that V1 vs V2 isolates
exactly one factor - the grouping - and nothing else:

* **V1** random split by image. Sibling photographs of one physical leaf land
  on both sides of the boundary. This is the protocol the literature uses and
  the one that produces the inflated PlantVillage number.
* **V2** split by physical leaf. No leaf appears in more than one partition.

Both draw from the leafmap-covered pool. Restricting V1 to that pool costs some
images but keeps the comparison single-factor; splitting V1 over the full 4,500
images would confound grouping with pool size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import paths
from data.mapping import SHARED_3, SHARED_8

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_index() -> pd.DataFrame:
    return pd.read_parquet(paths.DATA_DIR / "index.parquet")


def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop unusable rows. Returns (kept, dropped-with-reason).

    Two rules, in order:

    1. Byte-identical images carrying *different* labels are unlearnable and
       their true label is unknown, so **every** copy goes. PlantDoc's
       ``tomato_V8.jpg`` is filed as both Septoria leaf spot and bacterial spot.
    2. Byte-identical images carrying the same label are redundant; keep the
       first occurrence by path order and drop the rest.
    """
    df = df.copy()
    dropped = []

    counts = df.groupby("md5").agg(n=("path", "size"), labels=("label", "nunique"))
    contradictory = set(counts[(counts.n > 1) & (counts.labels > 1)].index)

    mask_contra = df.md5.isin(contradictory)
    if mask_contra.any():
        d = df[mask_contra].copy()
        d["exclusion_reason"] = "byte-identical duplicate with contradictory labels"
        dropped.append(d)
        df = df[~mask_contra]

    df = df.sort_values("path")
    is_dup = df.md5.notna() & df.duplicated(subset="md5", keep="first")
    if is_dup.any():
        d = df[is_dup].copy()
        d["exclusion_reason"] = "byte-identical duplicate of an earlier image (same label)"
        dropped.append(d)
        df = df[~is_dup]

    dropped_df = (
        pd.concat(dropped, ignore_index=True)
        if dropped
        else pd.DataFrame(columns=[*df.columns, "exclusion_reason"])
    )
    return df.reset_index(drop=True), dropped_df


def source_pool(df: pd.DataFrame, label_set: list[str]) -> pd.DataFrame:
    """PlantVillage rows in `label_set` that carry a physical-leaf group id."""
    pool = df[
        (df.domain == "plantvillage")
        & (df.label.isin(label_set))
        & (df.group_id.notna())
    ]
    return pool.reset_index(drop=True)


def _assign(keys: np.ndarray, rng: np.random.Generator) -> dict[str, str]:
    """Shuffle `keys` and cut them into train/val/test by SPLIT_FRACTIONS."""
    keys = np.array(sorted(set(keys)))
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(round(n * SPLIT_FRACTIONS["train"]))
    n_val = int(round(n * SPLIT_FRACTIONS["val"]))
    out: dict[str, str] = {}
    for i, k in enumerate(keys):
        if i < n_train:
            out[k] = "train"
        elif i < n_train + n_val:
            out[k] = "val"
        else:
            out[k] = "test"
    return out


def make_split(pool: pd.DataFrame, protocol: str, seed: int) -> pd.DataFrame:
    """Return `pool` with a `split` column.

    `protocol` is ``"v1_image"`` (split by image) or ``"v2_leaf"`` (split by
    physical leaf). Stratified by label so class balance is preserved.
    """
    if protocol not in ("v1_image", "v2_leaf"):
        raise ValueError(f"unknown protocol {protocol!r}")

    out = pool.copy()
    key_col = "path" if protocol == "v1_image" else "group_id"
    rng = np.random.default_rng(seed)

    assignment: dict[str, str] = {}
    for label in sorted(out.label.unique()):
        keys = out.loc[out.label == label, key_col].to_numpy()
        assignment.update(_assign(keys, rng))

    out["split"] = out[key_col].map(assignment)
    out["split_protocol"] = protocol
    out["split_seed"] = seed
    return out


def check_group_leakage(split_df: pd.DataFrame) -> dict[str, int]:
    """Count groups appearing in more than one partition.

    Zero is required for ``v2_leaf``. For ``v1_image`` this is the leakage the
    ablation is designed to measure, so a non-zero count is the expected result.
    """
    per_group = split_df.groupby("group_id").split.nunique()
    straddling = per_group[per_group > 1]
    n_imgs = int(split_df.group_id.isin(straddling.index).sum())
    return {"straddling_groups": int(len(straddling)), "affected_images": n_imgs}


def target_set(df: pd.DataFrame, domain: str, label_set: list[str]) -> pd.DataFrame:
    return df[(df.domain == domain) & (df.label.isin(label_set))].reset_index(drop=True)


ARMS = {
    "A": ("plantvillage", SHARED_3, "held-out source, leaf-grouped - no shift (control)"),
    "B": ("plantdoc", SHARED_8, "web / in-the-wild"),
    "C": ("plantdoc", SHARED_3, "web, label-space control against arm B"),
    "D": ("tomatovillage", SHARED_3, "real Indian field"),
    "E": ("rover", SHARED_3, "robot deployment - frames to be collected"),
}
