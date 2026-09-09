"""Phase 7 / Option C: GdScore on the complete unlabelled Arm-D pool.

Does a stronger, later label-free estimator change the RQ5 conclusions?

GdScore is evaluated on exactly the comparison RQ5 already established - all
1,616 unlabelled Tomato-Village crops, against the same true field accuracy,
with the same MAE metric - and **not** inside the N-sweep, because its score is
batch-size dependent (3.2x across batch sizes) and that would confound session
size with the method's own batching. See ``docs/gdscore_eligibility.md``.

Two metrics are reported:

* **MAE** after our source-fitted score-to-accuracy map, for comparability with
  the existing eight;
* **correlation with true accuracy across models**, which is the metric the
  GdScore authors themselves use. Judging a score-based method only by MAE after
  our own calibration would understate it.

Nothing existing is modified. No retraining. No N-sweep.

    python scripts/rq7_gdscore.py
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
from inference.train_mode_cache import (
    GD_BATCH_SIZE,
    TrainModeKey,
    cache_train_mode,
    load_train_mode,
    load_train_mode_labels,
)
from utils.manifest import cache_fingerprint, write_experiment_manifest

EXPERIMENT_ID = "rq7_gdscore"
OUT = paths.RESULTS_DIR / EXPERIMENT_ID
GD_SEED = 1          # official default
THRESHOLD = 0.5      # official default
NORM_TYPE = 0.3      # official default - fractional, NOT Frobenius


def clean_models() -> list[dict]:
    out = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        c = m["config"]
        if c["version"] in ("V2", "V9") and c["label_set"] == "SHARED_3":
            out.append({"version": c["version"], "backbone": c["backbone"],
                        "seed": c["seed"], "label_set": c["label_set"],
                        "checkpoint": Path(m["checkpoint"]),
                        "split_file": Path(m["split_file"])})
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    models = clean_models()
    print(f"models: {len(models)} | backbones: {sorted({m['backbone'] for m in models})}",
          flush=True)

    # ---- 1. train-mode cache (GPU; run alone) --------------------------
    print("\n[1/3] caching train-mode activations (BatchNorm needs batch statistics) ...",
          flush=True)
    for i, m in enumerate(models, 1):
        print(f"  [{i}/{len(models)}] {m['version']}/{m['backbone']}/seed{m['seed']}",
              flush=True)
        cache_train_mode(m["version"], m["backbone"], m["seed"], m["label_set"],
                         m["checkpoint"], m["split_file"])

    # ---- 2. sanity check: ViT must match the eval cache ----------------
    print("\n[2/3] sanity check - ViT has no BatchNorm, so train == eval ...", flush=True)
    checks = []
    for m in models:
        tk = TrainModeKey(m["version"], m["backbone"], m["seed"], "tomatovillage",
                          m["label_set"])
        ek = CacheKey(m["version"], m["backbone"], m["seed"], "tomatovillage",
                      m["label_set"])
        if not tk.path.exists():
            continue
        t, e = load_train_mode(tk), load_features(ek)
        d = float(np.abs(t["logits"] - e["logits"]).max())
        scale = float(np.abs(e["logits"]).max())
        agree = float((t["logits"].argmax(1) == e["logits"].argmax(1)).mean())
        checks.append({"backbone": m["backbone"], "seed": m["seed"],
                       "max_abs_logit_diff": d,
                       "rel_logit_diff": d / max(scale, 1e-9),
                       "prediction_agreement": agree})
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / "rq7_trainmode_vs_eval.csv", index=False)
    print(chk.groupby("backbone")[["rel_logit_diff", "prediction_agreement"]]
          .mean().round(4).to_string(), flush=True)
    # Semantic check, not bitwise: cuDNN TF32 is enabled, so identical maths at a
    # different batch size still differs in the last few digits. ViT has no
    # BatchNorm, so train and eval must agree on every PREDICTION.
    vit = chk[chk.backbone == "vit_small_patch16"]
    if len(vit):
        if vit.prediction_agreement.min() < 1.0:
            print("  STOP: ViT predictions differ between train and eval; cache suspect")
            return 2
        if vit.rel_logit_diff.max() > 1e-2:
            print("  STOP: ViT logit difference exceeds numerical tolerance")
            return 2
        print(f"  ViT: 100% prediction agreement, max relative logit diff "
              f"{vit.rel_logit_diff.max():.2e} (TF32 noise) - PASS", flush=True)

    # ---- 3. GdScore on the full pool -----------------------------------
    print("\n[3/3] GdScore on the complete 1,616-crop unlabelled pool ...", flush=True)
    rows = []
    for m in models:
        src = TrainModeKey(m["version"], m["backbone"], m["seed"], "source_val",
                           m["label_set"])
        tgt = TrainModeKey(m["version"], m["backbone"], m["seed"], "tomatovillage",
                           m["label_set"])
        if not (src.path.exists() and tgt.path.exists()):
            continue
        s, t = load_train_mode(src), load_train_mode(tgt)
        source = SourceStats(logits=s["logits"], labels=load_train_mode_labels(src),
                             features=s["features"])
        y = load_train_mode_labels(tgt)                     # POST-HOC truth only

        # true accuracy is defined by the DEPLOYED (eval-mode) classifier
        ek = CacheKey(m["version"], m["backbone"], m["seed"], "tomatovillage",
                      m["label_set"])
        e = load_features(ek)
        pool_acc = float((e["logits"].argmax(1) == load_labels(ek)).mean())

        est = GdScore(threshold=THRESHOLD, norm_type=NORM_TYPE,
                      batch_size=GD_BATCH_SIZE, gd_seed=GD_SEED).fit(source)
        raw = est.raw_score(t["logits"], t["features"])
        calibrated = est.estimate(t["logits"], t["features"])
        rows.append({
            "backbone": m["backbone"], "seed": m["seed"], "estimator": "gdscore",
            "raw_score": raw, "estimate": calibrated, "pool_accuracy": pool_acc,
            "abs_error": abs(calibrated - pool_acc),
            "calibration_r": est.calibration_r_,
            "n_unlabelled_crops": int(len(y)), "n_labels_used": 0,
        })
    gd = pd.DataFrame(rows)
    gd.to_csv(OUT / "rq7_gdscore_full_pool.csv", index=False)

    # ---- comparison against the existing eight -------------------------
    lf = pd.read_csv(paths.RESULTS_DIR / "rq5" / "rq5_labelfree_full_pool.csv")
    comp = (pd.concat([lf[["backbone", "seed", "estimator", "estimate",
                           "pool_accuracy", "abs_error"]],
                       gd[["backbone", "seed", "estimator", "estimate",
                           "pool_accuracy", "abs_error"]]], ignore_index=True))
    comp.to_csv(OUT / "rq7_comparison_full_pool.csv", index=False)

    print("\n=== MAE on the full unlabelled pool (lower is better) ===")
    summary = comp.groupby("estimator").abs_error.agg(["mean", "median", "std"]).sort_values("mean")
    print(summary.round(4).to_string())

    print("\n=== correlation with true accuracy across the 20 models "
          "(the metric GdScore's authors use) ===")
    corr = []
    for est, g in comp.groupby("estimator"):
        use_raw = gd.raw_score if est == "gdscore" else None
        v = gd.raw_score.to_numpy() if est == "gdscore" else g.estimate.to_numpy()
        a = gd.pool_accuracy.to_numpy() if est == "gdscore" else g.pool_accuracy.to_numpy()
        corr.append({"estimator": est,
                     "pearson_r": float(np.corrcoef(v, a)[0, 1]),
                     "spearman": float(sps.spearmanr(v, a).statistic),
                     "r2": float(np.corrcoef(v, a)[0, 1] ** 2)})
    cdf = pd.DataFrame(corr).sort_values("spearman", key=abs, ascending=False)
    cdf.to_csv(OUT / "rq7_correlation.csv", index=False)
    print(cdf.round(4).to_string(index=False))

    print("\n=== per-backbone MAE ===")
    print(comp.pivot_table(index="estimator", columns="backbone",
                           values="abs_error", aggfunc="mean").round(4).to_string())

    outputs = [str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))]
    write_experiment_manifest(
        EXPERIMENT_ID,
        params={
            "question": "does a later strong label-free estimator change the RQ5 conclusion",
            "protocol": "Option C - full unlabelled pool only, NOT the N-sweep",
            "reason_not_n_sweep": "GdScore score is batch-size dependent (3.2x)",
            "estimator": "GdScore (Xie et al., TMLR 2025)",
            "official_repo": "github.com/Renchunzi-Xie/GdScore",
            "threshold": THRESHOLD, "norm_type": NORM_TYPE,
            "batch_size": GD_BATCH_SIZE, "gd_seed": GD_SEED,
            "drop_last": True,
            "forward_mode": "train() - BatchNorm batch statistics, per official code",
            "buffers_restored": True,
            "score_to_accuracy_map": "ours - linear fit on source-val bootstraps, as for nuclear_norm/mano",
            "target_arm": "D", "label_set": "SHARED_3",
            "versions_used": ["V2", "V9"], "n_models": len(models),
            "backbones": sorted({m["backbone"] for m in models}),
        },
        inputs={"eval_cache": cache_fingerprint(sorted(paths.CACHE_DIR.glob("*.npz"))),
                "train_mode_cache": cache_fingerprint(
                    sorted((paths.RAW_RESULTS / "cache_trainmode").glob("*.npz"))),
                "rq5_baseline": "results/rq5/rq5_labelfree_full_pool.csv"},
        outputs=outputs,
        metrics={"gdscore_mae": float(gd.abs_error.mean()),
                 "gdscore_spearman": float(cdf[cdf.estimator == "gdscore"].spearman.iloc[0])},
        runtime_seconds=time.time() - t0,
        notes="Phase 7 Option C. GdScore evaluated only on the full pool. "
              "No retraining, no N-sweep, no existing result modified.",
        out_dir=OUT,
    )
    print(f"\nwrote {len(outputs)} tables + manifest in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
