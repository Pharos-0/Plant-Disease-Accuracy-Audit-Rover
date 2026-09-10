"""Target-Informativeness audit across all nine estimators (Phases 5-8).

Answers, from the existing cached activations and frozen checkpoints:

    Phase 6  How much does each estimate move when the SOURCE anchor is
             resampled and the target is held fixed?
    Phase 7  How much does it move when the TARGET is resampled and the source
             is held fixed?
    Phase 8  What does each estimator return when the target's information is
             destroyed - shuffled, collapsed to one repeated sample, replaced by
             moment-matched noise, or replaced by random logits?

    and the decisive one:

    Q4       Does TI predict estimator error?

Resampling is **cluster-level on both sides** - photographs on the target
(391 clusters over 1,616 crops), leaves on the source - because bootstrapping
crops independently would overstate the information available and understate the
variance.

NEW ESTIMATOR EXPERIMENT. Nothing historical is read for writing or modified.
No retraining, no GPU: everything runs from the Stage-2 eval cache.

    python scripts/ti_audit.py [--n-boot 100]
"""
from __future__ import annotations

import argparse
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
from estimators.methods import REGISTRY, GdScore
from estimators.ti_audit import (
    anchor_sensitivity,
    removal_surrogates,
    target_sensitivity,
)
from inference.cache import CacheKey, load_features, load_labels
from utils.manifest import cache_fingerprint, write_experiment_manifest

EXPERIMENT_ID = "ti_audit"
OUT = paths.RESULTS_DIR / EXPERIMENT_ID


def factories() -> dict:
    """The nine estimators, each as a zero-argument factory.

    GdScore is included with **eval-mode** activations so that every estimator is
    audited under one consistent forward mode. Its official mode is train(); that
    variant is examined separately in Phase 7b and is not mixed in here, because
    a sensitivity comparison across estimators requires them to share inputs.
    """
    f = {name: spec["factory"] for name, spec in REGISTRY.items()}
    f["gdscore"] = lambda: GdScore()
    return f


def clean_models() -> list[dict]:
    out = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if c["version"] in ("V2", "V9") and c["label_set"] == "SHARED_3":
            out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    models = clean_models()
    facs = factories()
    print(f"models: {len(models)} | estimators: {len(facs)} | "
          f"bootstraps: {args.n_boot} (cluster-level)", flush=True)

    rows, removal_rows = [], []
    for i, c in enumerate(models, 1):
        sk = CacheKey(c["version"], c["backbone"], c["seed"], "source_val", c["label_set"])
        tk = CacheKey(c["version"], c["backbone"], c["seed"], "tomatovillage", c["label_set"])
        s, t = load_features(sk), load_features(tk)
        src = SourceStats(logits=s["logits"], labels=load_labels(sk), features=s["features"])
        s_groups, t_groups = s["group_ids"], t["group_ids"]
        # POST-HOC truth: used only to score an estimate that is already written.
        truth = float((t["logits"].argmax(1) == load_labels(tk)).mean())

        for name, fac in facs.items():
            seed = 1000 * i + abs(hash(name)) % 997
            t_sd, _ = target_sensitivity(fac, src, t["logits"], t["features"],
                                         t_groups, args.n_boot, seed)
            a_sd, _ = anchor_sensitivity(fac, src, s_groups, t["logits"],
                                         t["features"], args.n_boot, seed)
            v_t, v_a = t_sd ** 2, a_sd ** 2
            ti = float(v_t / (v_t + v_a)) if (v_t + v_a) > 0 else np.nan
            point = float(fac().fit(src).estimate(t["logits"], t["features"]))
            rows.append({
                "backbone": c["backbone"], "seed": c["seed"], "estimator": name,
                "target_sd": t_sd, "anchor_sd": a_sd,
                "target_var": v_t, "anchor_var": v_a, "ti": ti,
                "estimate": point, "true_accuracy": truth,
                "abs_error": abs(point - truth),
            })

            # ---- Phase 8: target-information removal -----------------------
            rng = np.random.default_rng(seed)
            fitted = fac().fit(src)
            for label, (lg, ft) in removal_surrogates(
                    t["logits"], t["features"], t_groups, rng).items():
                try:
                    v = float(fitted.estimate(lg, ft))
                except (ValueError, RuntimeError):
                    v = np.nan
                removal_rows.append({
                    "backbone": c["backbone"], "seed": c["seed"], "estimator": name,
                    "surrogate": label, "estimate": v, "real_estimate": point,
                    "shift_from_real": abs(v - point) if np.isfinite(v) else np.nan,
                })
        print(f"  [{i}/{len(models)}] {c['backbone']}/seed{c['seed']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ti_per_model.csv", index=False)
    rem = pd.DataFrame(removal_rows)
    rem.to_csv(OUT / "ti_removal.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== Table E: target vs anchor sensitivity, per estimator ===")
    summ = (df.groupby("estimator")
              .agg(target_sd=("target_sd", "mean"), anchor_sd=("anchor_sd", "mean"),
                   TI=("ti", "mean"), MAE=("abs_error", "mean"))
              .sort_values("TI", ascending=False))
    print(summ.round(4).to_string())
    summ.to_csv(OUT / "ti_summary.csv")

    print("\n=== Phase 8: mean |estimate change| when target information is destroyed ===")
    piv = rem.pivot_table(index="estimator", columns="surrogate", values="shift_from_real")
    print(piv.round(4).to_string())
    piv.to_csv(OUT / "ti_removal_summary.csv")

    # ---- the decisive test: does TI predict error? ---------------------
    print("\n=== Q4: does TI predict estimator error? ===")
    q4 = []
    per_est = df.groupby("estimator").agg(TI=("ti", "mean"), MAE=("abs_error", "mean"))
    r_across = sps.spearmanr(per_est.TI, per_est.MAE)
    print(f"  across the 9 estimators (n=9): Spearman TI vs MAE = "
          f"{r_across.statistic:.3f} (p={r_across.pvalue:.3f})")
    q4.append({"level": "across_estimators", "n": len(per_est),
               "spearman": float(r_across.statistic), "p": float(r_across.pvalue)})
    for est, g in df.groupby("estimator"):
        if g.ti.notna().sum() > 3 and g.ti.std() > 1e-12:
            r = sps.spearmanr(g.ti, g.abs_error)
            q4.append({"level": f"within_{est}", "n": int(len(g)),
                       "spearman": float(r.statistic), "p": float(r.pvalue)})
    qdf = pd.DataFrame(q4)
    qdf.to_csv(OUT / "ti_vs_error.csv", index=False)
    print(qdf.round(3).to_string(index=False))

    write_experiment_manifest(
        EXPERIMENT_ID,
        params={"definition": "TI = Var_T / (Var_T + Var_S), a first-order Sobol' "
                              "index over two input groups",
                "resampling": "cluster bootstrap - photographs on target, leaves on source",
                "n_boot": args.n_boot, "n_models": len(models),
                "estimators": sorted(facs), "gdscore_mode": "eval (consistent with the rest)",
                "target_arm": "D", "label_set": "SHARED_3",
                "status": "NEW ESTIMATOR EXPERIMENT - no historical result read or modified"},
        inputs={"eval_cache": cache_fingerprint(sorted(paths.CACHE_DIR.glob("*.npz")))},
        outputs=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))],
        metrics={"spearman_TI_vs_MAE_across_estimators": float(r_across.statistic),
                 "p_value": float(r_across.pvalue)},
        runtime_seconds=time.time() - t0,
        notes="Phases 5-8. TI is a Sobol' index, established mathematics applied to a "
              "new object; it is not a new metric and is not claimed as one.",
        out_dir=OUT,
    )
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
