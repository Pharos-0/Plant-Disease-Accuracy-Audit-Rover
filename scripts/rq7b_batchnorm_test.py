"""Phase 7b: does GdScore's architecture pattern depend on BatchNorm train mode?

One controlled change against Phase 7 (`scripts/rq7_gdscore.py`): activations
come from the existing **eval()**-mode Stage-2 cache instead of the train-mode
cache. Everything else - models, target crops, threshold, norm_type, batch size,
seed, calibration procedure, estimand - is held identical.

Predictions were pre-registered in `docs/rq7b_batchnorm_plan.md` before this ran.

The built-in control is ViT-S/16: it has no BatchNorm, so train and eval are the
same computation and its result must be essentially unchanged. If it moves, the
comparison is invalid and nothing is concluded.

No GPU, no retraining, no new caching.

    python scripts/rq7b_batchnorm_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from estimators.base import SourceStats
from estimators.methods import GdScore
from inference.cache import CacheKey, load_features, load_labels
from inference.train_mode_cache import GD_BATCH_SIZE
from utils.manifest import cache_fingerprint, write_experiment_manifest

EXPERIMENT_ID = "rq7b_batchnorm"
OUT = paths.RESULTS_DIR / EXPERIMENT_ID
GD_SEED, THRESHOLD, NORM_TYPE = 1, 0.5, 0.3      # identical to Phase 7
HAS_BATCHNORM = {"resnet50": True, "efficientnet_b0": True,
                 "mobilenetv3_large": True, "vit_small_patch16": False}


def clean_models() -> list[dict]:
    out = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if c["version"] in ("V2", "V9") and c["label_set"] == "SHARED_3":
            out.append({"version": c["version"], "backbone": c["backbone"],
                        "seed": c["seed"], "label_set": c["label_set"]})
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    models = clean_models()
    print(f"models: {len(models)} | eval-mode activations, everything else as Phase 7",
          flush=True)

    rows = []
    for m in models:
        sk = CacheKey(m["version"], m["backbone"], m["seed"], "source_val", m["label_set"])
        tk = CacheKey(m["version"], m["backbone"], m["seed"], "tomatovillage", m["label_set"])
        s, t = load_features(sk), load_features(tk)
        source = SourceStats(logits=s["logits"], labels=load_labels(sk),
                             features=s["features"])
        y = load_labels(tk)                                   # POST-HOC truth only
        pool_acc = float((t["logits"].argmax(1) == y).mean())

        est = GdScore(threshold=THRESHOLD, norm_type=NORM_TYPE,
                      batch_size=GD_BATCH_SIZE, gd_seed=GD_SEED).fit(source)
        raw = est.raw_score(t["logits"], t["features"])
        calibrated = est.estimate(t["logits"], t["features"])
        rows.append({
            "backbone": m["backbone"], "seed": m["seed"], "mode": "eval",
            "has_batchnorm": HAS_BATCHNORM[m["backbone"]],
            "raw_score": raw, "estimate": calibrated, "pool_accuracy": pool_acc,
            "abs_error": abs(calibrated - pool_acc),
            "calibration_r": est.calibration_r_,
        })
    ev = pd.DataFrame(rows)
    ev.to_csv(OUT / "rq7b_gdscore_evalmode.csv", index=False)

    # ---- compare against the Phase-7 train-mode run --------------------
    tr = pd.read_csv(paths.RESULTS_DIR / "rq7_gdscore" / "rq7_gdscore_full_pool.csv")
    tr = tr.assign(mode="train", has_batchnorm=tr.backbone.map(HAS_BATCHNORM))
    both = pd.concat([tr[ev.columns.intersection(tr.columns)], ev], ignore_index=True)
    both.to_csv(OUT / "rq7b_train_vs_eval.csv", index=False)

    print("\n=== CONTROL: ViT has no BatchNorm, so train and eval must agree ===")
    ctl = both[both.backbone == "vit_small_patch16"].pivot_table(
        index="seed", columns="mode", values=["abs_error", "raw_score"])
    d_mae = float(np.abs(ctl[("abs_error", "train")] - ctl[("abs_error", "eval")]).max())
    rel_raw = float(np.abs(ctl[("raw_score", "train")] - ctl[("raw_score", "eval")]).max()
                    / max(ctl[("raw_score", "train")].abs().max(), 1e-12))
    print(f"  max |MAE difference| across seeds : {d_mae:.5f}")
    print(f"  max relative raw-score difference : {rel_raw:.2e}")
    control_ok = d_mae < 0.02
    print(f"  CONTROL {'PASS' if control_ok else 'FAIL'}")
    if not control_ok:
        print("  STOP: the control failed, so the comparison is invalid.")
        return 2

    print("\n=== GdScore MAE per backbone: train mode (Phase 7) vs eval mode ===")
    piv = both.pivot_table(index="backbone", columns="mode", values="abs_error")
    piv["change"] = piv["eval"] - piv["train"]
    piv["has_batchnorm"] = piv.index.map(HAS_BATCHNORM)
    print(piv.round(4).to_string())

    print("\n=== calibration correlation (near zero = the map is fitting noise) ===")
    print(both.pivot_table(index="backbone", columns="mode",
                           values="calibration_r").round(3).to_string())

    print("\n=== within-backbone Spearman, raw score vs true accuracy (n=5 seeds) ===")
    sp = []
    for (bb, mode), g in both.groupby(["backbone", "mode"]):
        sp.append({"backbone": bb, "mode": mode,
                   "spearman": float(sps.spearmanr(g.raw_score, g.pool_accuracy).statistic)})
    print(pd.DataFrame(sp).pivot(index="backbone", columns="mode",
                                 values="spearman").round(3).to_string())

    # ---- ranking against the other eight, under eval mode --------------
    lf = pd.read_csv(paths.RESULTS_DIR / "rq5" / "rq5_labelfree_full_pool.csv")
    ranks = []
    for bb in sorted(both.backbone.unique()):
        base = lf[lf.backbone == bb].groupby("estimator").abs_error.mean()
        for mode in ("train", "eval"):
            # NOTE: `both.mode` resolves to DataFrame.mode(), the statistical
            # method - not the column. Bracket access is required here.
            sel = both[(both["backbone"] == bb) & (both["mode"] == mode)]
            v = float(sel["abs_error"].mean())
            rank = int((base < v).sum()) + 1
            ranks.append({"backbone": bb, "mode": mode, "gdscore_mae": v,
                          "rank_of_9": rank, "best_other": base.min(),
                          "best_other_name": base.idxmin()})
    rk = pd.DataFrame(ranks)
    rk.to_csv(OUT / "rq7b_ranks.csv", index=False)
    print("\n=== GdScore rank among the nine, train vs eval ===")
    print(rk.pivot(index="backbone", columns="mode", values="rank_of_9").to_string())

    outputs = [str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))]
    write_experiment_manifest(
        EXPERIMENT_ID,
        params={
            "hypothesis": "H-BN: GdScore's advantage on BatchNorm backbones depends on "
                          "train()-mode batch statistics",
            "controlled_change": "eval-mode activations instead of train-mode; all else identical",
            "control": "ViT-S/16 has no BatchNorm and must be unchanged",
            "threshold": THRESHOLD, "norm_type": NORM_TYPE,
            "batch_size": GD_BATCH_SIZE, "gd_seed": GD_SEED,
            "n_models": len(models), "target_arm": "D", "label_set": "SHARED_3",
            "preregistered": "docs/rq7b_batchnorm_plan.md",
        },
        inputs={"eval_cache": cache_fingerprint(sorted(paths.CACHE_DIR.glob("*.npz"))),
                "phase7_trainmode": "results/rq7_gdscore/rq7_gdscore_full_pool.csv"},
        outputs=outputs,
        metrics={"control_max_mae_diff": d_mae, "control_passed": bool(control_ok)},
        runtime_seconds=time.time() - t0,
        notes="Phase 7b. Single controlled change. Consistency with H-BN only - not a "
              "causal claim about BatchNorm.",
        out_dir=OUT,
    )
    print(f"\nwrote {len(outputs)} tables + manifest in {time.time()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
