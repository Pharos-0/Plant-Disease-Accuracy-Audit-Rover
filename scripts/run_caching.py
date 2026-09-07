"""Stage 2 driver - cache logits and penultimate features for every trained model.

    python scripts/run_caching.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from inference.cache import cache_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    manifests = sorted(paths.RAW_RESULTS.glob("*/manifest_*.json"))
    if not manifests:
        print("no manifests found - run scripts/run_training.py first")
        return 1

    print(f"{len(manifests)} trained models to cache", flush=True)
    t0 = time.time()
    total_keys = 0

    for i, mpath in enumerate(manifests, 1):
        m = json.loads(mpath.read_text(encoding="utf-8"))
        cfg = m["config"]
        tag = f"{cfg['version']}/{cfg['backbone']}/seed{cfg['seed']}/{cfg['label_set']}"
        print(f"\n[{i}/{len(manifests)}] {tag}", flush=True)
        keys = cache_model(
            version=cfg["version"],
            backbone=cfg["backbone"],
            seed=cfg["seed"],
            label_set=cfg["label_set"],
            checkpoint=Path(m["checkpoint"]),
            split_file=Path(m["split_file"]),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )
        total_keys += len(keys)

    print(f"\ncached {total_keys} model x arm pairs in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
