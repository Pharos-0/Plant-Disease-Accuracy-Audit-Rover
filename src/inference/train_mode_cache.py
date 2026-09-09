"""Train-mode activation cache, for GdScore only.

The official GdScore implementation scores under ``model.train()``, where
BatchNorm uses **batch statistics** rather than running statistics. Our Stage-2
cache stores ``eval()``-mode activations, and the two disagree on more than half
of all predictions for every BatchNorm backbone we train (see
``docs/gdscore_eligibility.md`` section 4). Scoring GdScore from the eval cache
would silently compute a different quantity.

This module produces a **separate** train-mode cache. It never touches the
existing eval-mode cache, and it guarantees the frozen classifier is returned
unchanged:

* BatchNorm running buffers are updated by a train-mode forward pass, so the
  full state dict is snapshotted before and restored afterwards;
* the restoration is verified tensor-by-tensor, and a mismatch raises.

Batching matters and is fixed at the official ``batch_size = 128`` in dataset
order: under BatchNorm the activations depend on which samples share a batch, so
the cache is only valid for scoring at the same batch size and order.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import paths
from data.mapping import LABEL_SETS, label_index
from inference.cache import arm_frames
from models.backbones import create_model, forward_features_and_logits
from training.dataset import make_loader

#: the official GdScore batch size; the cache is only valid at this value
GD_BATCH_SIZE = 128
TRAIN_CACHE_DIR = paths.RAW_RESULTS / "cache_trainmode"


@dataclass(frozen=True)
class TrainModeKey:
    version: str
    backbone: str
    seed: int
    arm: str
    label_set: str

    @property
    def path(self) -> Path:
        return TRAIN_CACHE_DIR / (
            f"{self.version}_{self.backbone}_seed{self.seed}_"
            f"{self.label_set}_{self.arm}_trainmode.npz"
        )


def load_train_mode(key: TrainModeKey) -> dict[str, np.ndarray]:
    with np.load(key.path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _assert_restored(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> None:
    after = model.state_dict()
    for k, v in before.items():
        if not torch.equal(after[k].detach().cpu(), v.cpu()):
            raise RuntimeError(
                f"frozen model was mutated by the train-mode pass: '{k}' differs. "
                "Refusing to continue - the classifier must be returned unchanged."
            )


@torch.no_grad()
def cache_train_mode(
    version: str,
    backbone: str,
    seed: int,
    label_set: str,
    checkpoint: Path,
    split_file: Path,
    arms: tuple[str, ...] = ("source_val", "tomatovillage"),
    num_workers: int = 4,
    overwrite: bool = False,
) -> list[TrainModeKey]:
    """Cache train-mode logits and penultimate features for the named arms.

    Only the arms GdScore needs: ``source_val`` for its score-to-accuracy
    calibration and ``tomatovillage`` (arm D) for the estimate itself.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = LABEL_SETS[label_set]
    l2i = label_index(label_set)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bundle = create_model(backbone, n_classes=len(labels), pretrained=False)
    bundle.model.load_state_dict(ckpt["state_dict"])
    model = bundle.model.to(device).to(memory_format=torch.channels_last)

    TRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = arm_frames(label_set, split_file)
    written: list[TrainModeKey] = []

    for arm in arms:
        if arm not in frames:
            continue
        key = TrainModeKey(version, backbone, seed, arm, label_set)
        if key.path.exists() and not overwrite:
            written.append(key)
            continue

        frame = frames[arm]
        # shuffle=False, drop_last=False: dataset order, matching how the score
        # will later be batched.
        loader = make_loader(frame, l2i, train=False,
                             batch_size=GD_BATCH_SIZE, num_workers=num_workers)

        before = _snapshot(model)
        model.train()                      # <- the whole point: batch statistics
        feats, logits, order = [], [], []
        for x, _y, idx in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            f, lg = forward_features_and_logits(model, x)
            feats.append(f.float().cpu().numpy())
            logits.append(lg.float().cpu().numpy())
            order.append(idx.numpy())
        model.eval()
        model.load_state_dict(before)      # undo BatchNorm buffer drift
        _assert_restored(model, before)

        feats = np.concatenate(feats).astype(np.float32)
        logits = np.concatenate(logits).astype(np.float32)
        inv = np.argsort(np.concatenate(order))
        # NOTE: activations are batch-dependent under BatchNorm, so we store them
        # in the order they were produced and record the batch size alongside.
        np.savez_compressed(
            key.path,
            logits=logits[inv], features=feats[inv],
            paths=frame["path"].to_numpy(),
            group_ids=frame["group_id"].astype(str).to_numpy(),
            batch_size=np.array([GD_BATCH_SIZE]),
        )
        np.savez_compressed(
            str(key.path).replace(".npz", "__labels.npz"),
            labels=np.array([l2i[v] for v in frame["label"]], dtype=np.int64),
        )
        written.append(key)
        print(f"    train-mode cached {key.path.name}: {len(logits)} images", flush=True)

    return written


def load_train_mode_labels(key: TrainModeKey) -> np.ndarray:
    """Labels for the train-mode cache. Evaluation only, never for an estimator."""
    with np.load(str(key.path).replace(".npz", "__labels.npz")) as z:
        return z["labels"]
