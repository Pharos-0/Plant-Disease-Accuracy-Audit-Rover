"""Image dataset and transforms driven by the verified index."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.backbones import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(INPUT_SIZE * 256 / 224)),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class LeafDataset(Dataset):
    """Rows of the verified index. `label_to_idx` fixes the class order."""

    def __init__(
        self,
        frame: pd.DataFrame,
        label_to_idx: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.paths = frame["path"].tolist()
        self.labels = [label_to_idx[v] for v in frame["label"]]
        self.group_ids = frame["group_id"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.labels[i], i


def make_loader(
    frame: pd.DataFrame,
    label_to_idx: dict[str, int],
    *,
    train: bool,
    batch_size: int = 64,
    num_workers: int = 8,
) -> DataLoader:
    """DataLoader tuned for this box.

    Windows spawns worker processes, so any entry point that calls this must sit
    behind ``if __name__ == "__main__":``.
    """
    ds = LeafDataset(frame, label_to_idx, train_transform() if train else eval_transform())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
