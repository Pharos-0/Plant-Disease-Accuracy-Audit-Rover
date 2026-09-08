"""Phase 1 of the benchmark: produce accuracy estimates from unlabeled sessions.

This script loads **source validation labels** (permitted - they calibrate ATC,
DoC and the score-based methods) and target *logits and features only*. It never
opens a target label file. Estimates are written to ``results/raw/estimates/``
and are immutable; ``run_evaluation.py`` scores them afterwards.

    python scripts/run_estimates.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from estimators.base import SourceStats
from estimators.methods import ALL_ESTIMATORS, build
from inference.cache import CacheKey, load_features, load_labels
from sessions.build import SESSION_SIZES, N_BOOTSTRAP, build_sessions, overlap_stats, validate_plan

# arm label -> (cache arm name, description)
ARM_OF_CACHE = {
    "source_test": "A",
    "plantdoc": "B_or_C",
    "tomatovillage": "D",
}


def discover_models() -> list[dict]:
    """Every trained model that has a manifest on disk."""
    out = []
    for mpath in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        m = json.loads(mpath.read_text(encoding="utf-8"))
        cfg = m["config"]
        out.append(
            {
                "version": cfg["version"],
                "backbone": cfg["backbone"],
                "seed": cfg["seed"],
                "label_set": cfg["label_set"],
                "checkpoint": Path(m["checkpoint"]),
                "split_file": Path(m["split_file"]),
                "manifest": mpath,
            }
        )
    return out


def arm_name(cache_arm: str, label_set: str) -> str:
    if cache_arm == "source_test":
        return "A"
    if cache_arm == "plantdoc":
        return "C" if label_set == "SHARED_3" else "B"
    if cache_arm == "tomatovillage":
        return "D"
    return cache_arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--sizes", type=int, nargs="*", default=SESSION_SIZES)
    args = ap.parse_args()

    paths.ensure_dirs()
    models = discover_models()
    if not models:
        print("no trained models found - run scripts/run_training.py first")
        return 1
    print(f"{len(models)} trained models found", flush=True)

    rows: list[dict] = []
    overlap_rows: list[dict] = []
    problems: list[str] = []
    session_store: dict[str, np.ndarray] = {}

    for mi, mdl in enumerate(models, 1):
        version, backbone, seed, label_set = (
            mdl["version"], mdl["backbone"], mdl["seed"], mdl["label_set"]
        )
        tag = f"{version}/{backbone}/seed{seed}/{label_set}"

        # ---- fit estimators on the SOURCE validation set (source labels allowed)
        src_key = CacheKey(version, backbone, seed, "source_val", label_set)
        if not src_key.features_path.exists():
            print(f"  [{mi}/{len(models)}] {tag}: no source_val cache, skipping", flush=True)
            continue
        src = load_features(src_key)
        src_labels = load_labels(src_key)  # SOURCE labels - permitted
        source = SourceStats(
            logits=src["logits"], labels=src_labels, features=src["features"]
        )

        fitted = {}
        for name in ALL_ESTIMATORS:
            fitted[name] = build(name).fit(source)

        print(f"  [{mi}/{len(models)}] {tag}: source_val acc={source.accuracy:.4f}", flush=True)

        # ---- estimate on each target arm (NO target labels are loaded here)
        for cache_arm in ("source_test", "plantdoc", "tomatovillage"):
            key = CacheKey(version, backbone, seed, cache_arm, label_set)
            if not key.features_path.exists():
                continue
            tgt = load_features(key)
            logits, feats, groups = tgt["logits"], tgt["features"], tgt["group_ids"]
            arm = arm_name(cache_arm, label_set)

            for n in args.sizes:
                plan = build_sessions(
                    groups, n, arm=arm, seed=seed, n_sessions=args.n_bootstrap
                )
                problems.extend(validate_plan(plan))
                stats = overlap_stats(plan, pool_size=len(logits), n_groups=len(set(groups.tolist())))
                overlap_rows.append(
                    {"version": version, "backbone": backbone, "seed": seed,
                     "label_set": label_set, "arm": arm, **stats}
                )

                store_key = f"{version}|{backbone}|{seed}|{label_set}|{arm}|{n}"
                session_store[store_key] = np.array(
                    [np.pad(i, (0, n - len(i)), constant_values=-1) for i in plan.indices],
                    dtype=np.int32,
                )

                for si, idx in enumerate(plan.indices):
                    if len(idx) == 0:
                        continue
                    s_logits = logits[idx]
                    s_feats = feats[idx]
                    for name, est in fitted.items():
                        rows.append(
                            {
                                "version": version,
                                "backbone": backbone,
                                "seed": seed,
                                "label_set": label_set,
                                "arm": arm,
                                "cache_arm": cache_arm,
                                "n_requested": n,
                                "n_actual": int(len(idx)),
                                "session": si,
                                "estimator": name,
                                "estimate": est.estimate(s_logits, s_feats),
                            }
                        )
            print(f"      {arm:8s} ({cache_arm}): {len(logits)} images -> "
                  f"{len(args.sizes)} session sizes", flush=True)

    est_df = pd.DataFrame(rows)
    out = paths.ESTIMATES_DIR / "estimates.parquet"
    est_df.to_parquet(out, index=False)
    pd.DataFrame(overlap_rows).to_parquet(paths.ESTIMATES_DIR / "session_overlap.parquet", index=False)
    np.savez_compressed(paths.ESTIMATES_DIR / "sessions.npz", **session_store)

    if problems:
        (paths.ESTIMATES_DIR / "session_warnings.json").write_text(
            json.dumps(problems, indent=2), encoding="utf-8"
        )

    print(f"\nwrote {len(est_df):,} estimates -> {out}")
    print(f"wrote session index store -> {paths.ESTIMATES_DIR / 'sessions.npz'}")
    if problems:
        print(f"\n{len(problems)} session warnings (pool smaller than requested N):")
        for p in problems[:10]:
            print("  -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
