"""Stage 2 - cache logits and penultimate features for every model x domain pair.

This is the load-bearing engineering decision (study protocol). Training
happens once; estimator evaluation happens hundreds of thousands of times. Once
logits and features are on disk, every estimator, session, N and seed downstream
is array arithmetic.

Labels are written to a **separate** file from logits and features. Estimator
code loads only the feature file, so the firewall is a property of the storage
layout and not only of the call signatures.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import paths
from data.mapping import LABEL_SETS, label_index
from data.splits import apply_exclusions, load_index, target_set
from models.backbones import create_model, forward_features_and_logits
from training.dataset import make_loader


@dataclass(frozen=True)
class CacheKey:
    version: str
    backbone: str
    seed: int
    arm: str  # source_val | source_test | plantdoc | tomatovillage
    label_set: str

    @property
    def stem(self) -> str:
        return f"{self.version}_{self.backbone}_seed{self.seed}_{self.label_set}_{self.arm}"

    @property
    def features_path(self) -> Path:
        return paths.CACHE_DIR / f"{self.stem}.npz"

    @property
    def labels_path(self) -> Path:
        return paths.CACHE_DIR / f"{self.stem}__labels.npz"


def load_features(key: CacheKey) -> dict[str, np.ndarray]:
    """Logits, features and identifiers. Contains no labels, by construction."""
    with np.load(key.features_path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def load_labels(key: CacheKey) -> np.ndarray:
    """Target labels. For src/evaluation/ only, never for an estimator."""
    with np.load(key.labels_path, allow_pickle=True) as z:
        return z["labels"]


@torch.no_grad()
def _infer(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    feats, logits, idxs = [], [], []
    for x, _y, idx in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            f, lg = forward_features_and_logits(model, x)
        feats.append(f.float().cpu().numpy())
        logits.append(lg.float().cpu().numpy())
        idxs.append(idx.numpy())
    return (
        np.concatenate(feats).astype(np.float32),
        np.concatenate(logits).astype(np.float32),
        np.concatenate(idxs),
    )


def arm_frames(label_set: str, split_file: Path) -> dict[str, pd.DataFrame]:
    """The evaluation frames for one trained model.

    ``source_val`` calibrates the estimators; ``source_test`` is arm A; the rest
    are the shifted target domains.
    """
    labels = LABEL_SETS[label_set]
    split_df = pd.read_parquet(split_file)
    df, _ = apply_exclusions(load_index())

    frames = {
        "source_val": split_df[split_df.split == "val"],
        "source_test": split_df[split_df.split == "test"],
        "plantdoc": target_set(df, "plantdoc", labels),
        "tomatovillage": target_set(df, "tomatovillage", labels),
    }
    return {k: v.reset_index(drop=True) for k, v in frames.items() if len(v)}


def cache_model(
    version: str,
    backbone: str,
    seed: int,
    label_set: str,
    checkpoint: Path,
    split_file: Path,
    *,
    batch_size: int = 128,
    num_workers: int = 8,
    overwrite: bool = False,
) -> list[CacheKey]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = LABEL_SETS[label_set]
    l2i = label_index(label_set)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bundle = create_model(backbone, n_classes=len(labels), pretrained=False)
    bundle.model.load_state_dict(ckpt["state_dict"])
    model = bundle.model.to(device).to(memory_format=torch.channels_last)

    written: list[CacheKey] = []
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for arm, frame in arm_frames(label_set, split_file).items():
        key = CacheKey(version, backbone, seed, arm, label_set)
        if key.features_path.exists() and not overwrite:
            written.append(key)
            continue

        loader = make_loader(
            frame, l2i, train=False, batch_size=batch_size, num_workers=num_workers
        )
        feats, logits, order = _infer(model, loader, device)

        # restore dataset order, which the loader preserves but make explicit
        inv = np.argsort(order)
        feats, logits = feats[inv], logits[inv]

        np.savez_compressed(
            key.features_path,
            logits=logits,
            features=feats,
            paths=frame["path"].to_numpy(),
            group_ids=frame["group_id"].astype(str).to_numpy(),
        )
        np.savez_compressed(
            key.labels_path,
            labels=np.array([l2i[v] for v in frame["label"]], dtype=np.int64),
        )
        written.append(key)
        print(f"    cached {key.stem}: {len(logits)} images, {feats.shape[1]}-d features",
              flush=True)

    return written
