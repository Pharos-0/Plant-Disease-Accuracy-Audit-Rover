"""Run the training grid.

V1 (random image split) and V2 (leaf-grouped split) over 5 seeds each, then the
V9 architecture sweep on the V2 protocol. Sequential: one GPU.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.train import TrainConfig, run

SEEDS = [0, 1, 2, 3, 4]
V9_BACKBONES = ["mobilenetv3_large", "efficientnet_b0", "vit_small_patch16"]


def build_jobs(stage: str) -> list[TrainConfig]:
    jobs: list[TrainConfig] = []
    if stage in ("v1v2", "all"):
        for seed in SEEDS:
            jobs.append(TrainConfig(version="V1", protocol="v1_image", seed=seed, epochs=12))
            jobs.append(TrainConfig(version="V2", protocol="v2_leaf", seed=seed, epochs=12))
    if stage in ("v9", "all"):
        for backbone in V9_BACKBONES:
            for seed in SEEDS:
                jobs.append(
                    TrainConfig(
                        version="V9",
                        protocol="v2_leaf",
                        backbone=backbone,
                        seed=seed,
                        epochs=12,
                    )
                )
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="v1v2", choices=["v1v2", "v9", "all"])
    args = ap.parse_args()

    jobs = build_jobs(args.stage)
    print(f"{len(jobs)} training runs queued for stage {args.stage!r}", flush=True)
    t0 = time.time()
    failures = []

    for i, cfg in enumerate(jobs, 1):
        tag = f"{cfg.version}/{cfg.backbone}/seed{cfg.seed}"
        print(f"\n===== [{i}/{len(jobs)}] {tag} =====", flush=True)
        try:
            m = run(cfg)
            print(f"  -> source_test_acc={m['source_test_acc']:.4f}", flush=True)
        except Exception:  # noqa: BLE001 - keep the grid going, record the failure
            failures.append(tag)
            traceback.print_exc()

    mins = (time.time() - t0) / 60
    print(f"\nfinished {len(jobs) - len(failures)}/{len(jobs)} runs in {mins:.1f} min", flush=True)
    if failures:
        print("FAILED:", ", ".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
