"""Train one backbone on a PlantVillage source split.

Usage (must be run as a script - DataLoader workers spawn on Windows):

    python src/training/train.py --version V2 --protocol v2_leaf --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import paths
from data.mapping import LABEL_SETS, label_index
from data.splits import apply_exclusions, check_group_leakage, load_index, make_split, source_pool
from models.backbones import create_model
from training.dataset import make_loader
from utils.manifest import write_manifest


@dataclass
class TrainConfig:
    version: str
    protocol: str
    backbone: str = "resnet50"
    label_set: str = "SHARED_3"
    seed: int = 0
    epochs: int = 15
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    num_workers: int = 8
    warmup_epochs: int = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    preds_all, labels_all = [], []
    crit = nn.CrossEntropyLoss()
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = crit(logits, y)
        loss_sum += float(loss) * y.size(0)
        pred = logits.argmax(1)
        correct += int((pred == y).sum())
        total += y.size(0)
        preds_all.append(pred.cpu().numpy())
        labels_all.append(y.cpu().numpy())
    return (
        correct / max(total, 1),
        loss_sum / max(total, 1),
        np.concatenate(preds_all),
        np.concatenate(labels_all),
    )


def run(cfg: TrainConfig) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required; refusing to train on CPU")
    set_seed(cfg.seed)

    labels = LABEL_SETS[cfg.label_set]
    l2i = label_index(cfg.label_set)

    df = load_index()
    kept, dropped = apply_exclusions(df)
    pool = source_pool(kept, labels)
    split_df = make_split(pool, cfg.protocol, seed=cfg.seed)
    leakage = check_group_leakage(split_df)

    parts = {s: split_df[split_df.split == s] for s in ("train", "val", "test")}
    print(
        f"[{cfg.version}/{cfg.protocol}/seed{cfg.seed}] "
        f"train {len(parts['train'])} | val {len(parts['val'])} | test {len(parts['test'])}"
    )
    print(f"  leaves straddling splits: {leakage['straddling_groups']} "
          f"({leakage['affected_images']} images)")

    loaders = {
        s: make_loader(
            parts[s],
            l2i,
            train=(s == "train"),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        for s in ("train", "val", "test")
    }

    bundle = create_model(cfg.backbone, n_classes=len(labels), pretrained=True)
    model = bundle.model.to(device).to(memory_format=torch.channels_last)

    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(loaders["train"]), 1)
    # Keep a real annealing phase even when epochs is small (e.g. a smoke run).
    pct_start = min(0.3, max(0.05, cfg.warmup_epochs / max(cfg.epochs, 1)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.lr,
        total_steps=cfg.epochs * steps_per_epoch,
        pct_start=pct_start,
    )
    scaler = torch.amp.GradScaler("cuda")

    history = []
    best_val, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for x, y, _ in loaders["train"]:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += float(loss.detach()) * y.size(0)
            seen += y.size(0)

        val_acc, val_loss, _, _ = evaluate(model, loaders["val"], device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": run_loss / max(seen, 1),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "lr": sched.get_last_lr()[0],
            }
        )
        print(f"  epoch {epoch:2d}  train_loss {run_loss / max(seen, 1):.4f}  "
              f"val_acc {val_acc:.4f}", flush=True)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_loss, preds, gts = evaluate(model, loaders["test"], device)
    train_time = time.time() - t0

    ckpt_dir = paths.RAW_RESULTS / cfg.version
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg.backbone}_seed{cfg.seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "labels": labels,
            "timm_id": bundle.timm_id,
        },
        ckpt_path,
    )

    split_path = ckpt_dir / f"split_{cfg.protocol}_seed{cfg.seed}.parquet"
    split_df[["path", "label", "group_id", "split"]].to_parquet(split_path, index=False)

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for g, p in zip(gts, preds):
        cm[g, p] += 1

    metrics = {
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "source_test_acc": test_acc,
        "source_test_loss": test_loss,
        "train_seconds": train_time,
        "n_train": len(parts["train"]),
        "n_val": len(parts["val"]),
        "n_test": len(parts["test"]),
        "leakage": leakage,
        "confusion_matrix": cm.tolist(),
        "history": history,
    }
    write_manifest(
        cfg.version,
        asdict(cfg),
        metrics,
        seed=cfg.seed,
        extra={
            "labels": labels,
            "checkpoint": str(ckpt_path),
            "split_file": str(split_path),
            "n_excluded_images": int(len(dropped)),
        },
        filename=f"manifest_{cfg.backbone}_seed{cfg.seed}.json",
    )
    print(f"  source test acc {test_acc:.4f}  ({train_time:.0f}s)", flush=True)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--protocol", required=True, choices=["v1_image", "v2_leaf"])
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--label-set", default="SHARED_3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    cfg = TrainConfig(
        version=args.version,
        protocol=args.protocol,
        backbone=args.backbone,
        label_set=args.label_set,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    metrics = run(cfg)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
