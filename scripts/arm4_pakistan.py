"""Fourth target arm: evaluate the frozen classifiers and estimators on new data.

WHY THIS EXISTS
---------------
The manuscript's largest stated limitation is dataset-level sample size:

    "One crop, one primary source dataset, three target conditions. Dataset-level
     evidence is effectively n=3 ... We cannot answer this with current data."

A fourth independent target arm is now available, so part of that can be
answered. The question this script asks is narrow and pre-stated:

    Does the anchor-versus-target reversal, measured on three arms, also appear
    on a fourth dataset that played no part in establishing it?

WHY IT DOES NOT TOUCH THE FROZEN PIPELINE
-----------------------------------------
`data/index.parquet` is hashed into all 47 experiment manifests by
`utils/manifest.py`. Regenerating it with a fourth domain would change that hash
and invalidate the provenance chain of every frozen result. So this script:

  * builds the arm-4 frame on demand from `BUILDERS["pakistan"]`,
  * writes its own index at `data/index_arm4.parquet`,
  * caches under the arm name `pakistan`, a new CacheKey namespace,

and never writes to `index.parquet`, to an existing cache key, or to any
existing results file.

WHAT THIS ARM IS, AND WHAT IT IS NOT
------------------------------------
It is the raw folder of Mendeley deposit 10.17632/3mbnb82mxd.2, restricted to
the three SHARED_3 classes, after two removals documented in
`dataset/documentation/PILOT_REPORT.md` section 16: the deposit's 7,200-image
"Augmented Dataset" folder, every filename of which ends `aug<N>`, and the 311
images byte-identical to PlantDoc.

Its independence is **partial and not established**. 1,528 filenames in the
original archive carry a Roboflow export signature, and some are URL-encoded web
filenames, so further overlap with public datasets we do not hold is possible.
Any finding here is therefore corroboration, not replication on clean data, and
must be reported that way.

    python scripts/arm4_pakistan.py --cache      # stage 2 for the new arm
    python scripts/arm4_pakistan.py --evaluate   # accuracy, estimators, TI
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths                                    # noqa: E402
from data.index import BUILDERS                           # noqa: E402
from data.mapping import LABEL_SETS, label_index          # noqa: E402

ARM = "pakistan"
LABEL_SET = "SHARED_3"
ARM_INDEX = paths.DATA_DIR / "index_arm4.parquet"
OUT = paths.RESULTS_DIR / "arm4_pakistan"


def build_frame() -> pd.DataFrame:
    """The arm-4 frame, restricted to the label set and written to its own index."""
    df = BUILDERS[ARM]()
    if df.empty:
        raise SystemExit(f"no images under {paths.PAKISTAN_ROOT}")
    labels = LABEL_SETS[LABEL_SET]
    df = df[df.label.isin(labels)].reset_index(drop=True)
    if "capture_date" not in df.columns:
        df["capture_date"] = None
    ARM_INDEX.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARM_INDEX, index=False)
    return df


def do_cache(batch_size: int, num_workers: int, overwrite: bool) -> int:
    """Run the frozen models over arm 4 and cache logits + features."""
    import torch
    from inference.cache import CacheKey, _infer
    from models.backbones import create_model
    from training.dataset import make_loader

    frame = build_frame()
    print(f"arm 4: {len(frame)} images")
    print(frame.label.value_counts().to_string())

    labels = LABEL_SETS[LABEL_SET]
    l2i = label_index(LABEL_SET)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    manifests = sorted(paths.RAW_RESULTS.glob("*/manifest_*.json"))
    targets = []
    for mp in manifests:
        cfg = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if cfg["label_set"] == LABEL_SET and cfg["version"] in ("V2", "V9"):
            targets.append((mp, cfg))
    print(f"{len(targets)} frozen SHARED_3 classifiers to run\n")

    t0 = time.time()
    done = 0
    for i, (mp, cfg) in enumerate(targets, 1):
        key = CacheKey(cfg["version"], cfg["backbone"], cfg["seed"], ARM, LABEL_SET)
        if key.features_path.exists() and not overwrite:
            print(f"[{i}/{len(targets)}] {key.stem}: cached")
            done += 1
            continue
        ckpt_path = mp.parent / f"{cfg['backbone']}_seed{cfg['seed']}.pt"
        if not ckpt_path.exists():
            print(f"[{i}/{len(targets)}] MISSING checkpoint {ckpt_path.name}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        bundle = create_model(cfg["backbone"], n_classes=len(labels), pretrained=False)
        bundle.model.load_state_dict(ckpt["state_dict"])
        model = bundle.model.to(device).to(memory_format=torch.channels_last)

        loader = make_loader(frame, l2i, train=False,
                             batch_size=batch_size, num_workers=num_workers)
        feats, logits, order = _infer(model, loader, device)
        inv = np.argsort(order)
        feats, logits = feats[inv], logits[inv]

        np.savez_compressed(key.features_path, logits=logits, features=feats,
                            paths=frame["path"].to_numpy(),
                            group_ids=frame["group_id"].astype(str).to_numpy())
        np.savez_compressed(key.labels_path,
                            labels=np.array([l2i[v] for v in frame["label"]],
                                            dtype=np.int64))
        acc = float((logits.argmax(1) == np.array([l2i[v] for v in frame["label"]])).mean())
        print(f"[{i}/{len(targets)}] {key.stem}: {len(logits)} images, accuracy {acc:.4f}")
        done += 1
        del model
        torch.cuda.empty_cache()

    print(f"\ncached {done} model x arm pairs in {(time.time() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cache:
        raise SystemExit(do_cache(a.batch_size, a.num_workers, a.overwrite))
    if a.evaluate:
        from arm4_evaluate import main as ev   # split out for readability
        raise SystemExit(ev())
    ap.print_help()
