"""Candidate Family F implemented and benchmarked (Phases 9-13, 16-18).

The novelty audit concluded on paper that every candidate family reduces to
empirical-Bayes shrinkage, and that shrinkage responds to weak target
information by leaning *harder* on the source anchor - the opposite of what the
motivating observation asks for. Rejecting a candidate on paper is weaker than
testing it, so the strongest and most defensible family is implemented here and
run against the existing benchmark.

    TIShrinkage(base):   estimate = w * base(T) + (1 - w) * anchor
                         w = measured TI of `base` on this target

`w` is computed from the same cluster-bootstrap machinery as the audit, using
only unlabelled target data and source-side calibration. No target labels are
involved at any point.

We test whether this improves MAE over the base estimator and over the existing
nine. No outcome is predicted here.

Two ablations, per Phase 13:

    fixed-w      w frozen at 0.5, so any gain from *measuring* w is isolated
    anchor-only  w = 0, the pure source anchor

NEW ESTIMATOR EXPERIMENT. Historical results are read for comparison only and
are never rewritten.

    python scripts/ti_candidate.py
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
from estimators.methods import REGISTRY, GdScore
from estimators.ti_audit import anchor_sensitivity, target_sensitivity
from inference.cache import CacheKey, load_features, load_labels
from utils.manifest import cache_fingerprint, write_experiment_manifest

EXPERIMENT_ID = "ti_candidate"
OUT = paths.RESULTS_DIR / EXPERIMENT_ID
N_BOOT = 40


def shrunk_estimate(fac, source, s_groups, logits, features, t_groups,
                    seed: int, mode: str) -> tuple[float, float]:
    """Return (estimate, w). `mode` selects the candidate or an ablation."""
    base = float(fac().fit(source).estimate(logits, features))
    anchor = float(source.accuracy)          # source-side only, no target labels
    if mode == "anchor_only":
        return anchor, 0.0
    if mode == "fixed_w":
        w = 0.5
    else:                                     # "ti" - the candidate itself
        t_sd, _ = target_sensitivity(fac, source, logits, features, t_groups,
                                     N_BOOT, seed)
        a_sd, _ = anchor_sensitivity(fac, source, s_groups, logits, features,
                                     N_BOOT, seed)
        vt, va = t_sd ** 2, a_sd ** 2
        w = float(vt / (vt + va)) if (vt + va) > 0 else 0.0
    return float(np.clip(w * base + (1.0 - w) * anchor, 0.0, 1.0)), w


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    models = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if c["version"] in ("V2", "V9") and c["label_set"] == "SHARED_3":
            models.append(c)

    facs = {n: s["factory"] for n, s in REGISTRY.items()}
    facs["gdscore"] = lambda: GdScore()
    print(f"models: {len(models)} | bases: {len(facs)} | bootstraps: {N_BOOT}",
          flush=True)

    rows = []
    for i, c in enumerate(models, 1):
        sk = CacheKey(c["version"], c["backbone"], c["seed"], "source_val", c["label_set"])
        tk = CacheKey(c["version"], c["backbone"], c["seed"], "tomatovillage", c["label_set"])
        s, t = load_features(sk), load_features(tk)
        src = SourceStats(logits=s["logits"], labels=load_labels(sk), features=s["features"])
        truth = float((t["logits"].argmax(1) == load_labels(tk)).mean())  # POST-HOC only

        for name, fac in facs.items():
            seed = 1000 * i + abs(hash(name)) % 997
            base = float(fac().fit(src).estimate(t["logits"], t["features"]))
            rows.append({"backbone": c["backbone"], "seed": c["seed"], "base": name,
                         "variant": "base", "estimate": base, "w": np.nan,
                         "true_accuracy": truth, "abs_error": abs(base - truth)})
            for mode in ("ti", "fixed_w", "anchor_only"):
                est, w = shrunk_estimate(fac, src, s["group_ids"], t["logits"],
                                         t["features"], t["group_ids"], seed, mode)
                rows.append({"backbone": c["backbone"], "seed": c["seed"], "base": name,
                             "variant": mode, "estimate": est, "w": w,
                             "true_accuracy": truth, "abs_error": abs(est - truth)})
        print(f"  [{i}/{len(models)}] {c['backbone']}/seed{c['seed']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "candidate_per_model.csv", index=False)
    pd.set_option("display.width", 220)

    print("\n=== Table F: candidate vs base, MAE per base estimator ===")
    piv = df.pivot_table(index="base", columns="variant", values="abs_error")
    piv = piv[["base", "ti", "fixed_w", "anchor_only"]]
    piv["ti_minus_base"] = piv["ti"] - piv["base"]
    piv.index.name = "base_estimator"          # "base" is also a column here
    print(piv.round(4).sort_values(by="base").to_string())
    piv.to_csv(OUT / "candidate_summary.csv")

    print("\n=== Table G: paired test, candidate vs its own base (per base) ===")
    stats = []
    for b, g in df[df.variant.isin(["base", "ti"])].groupby("base"):
        w = g.pivot_table(index=["backbone", "seed"], columns="variant",
                          values="abs_error").dropna()
        if len(w) < 3:
            continue
        d = w["ti"] - w["base"]
        try:
            p = float(sps.wilcoxon(w["ti"], w["base"]).pvalue)
        except ValueError:
            p = float("nan")
        boot = [np.mean(d.sample(len(d), replace=True, random_state=k))
                for k in range(2000)]
        stats.append({"base": b, "mean_diff": float(d.mean()),
                      "median_diff": float(d.median()),
                      "ci_lo": float(np.percentile(boot, 2.5)),
                      "ci_hi": float(np.percentile(boot, 97.5)),
                      "wilcoxon_p": p, "n": int(len(d))})
    sdf = pd.DataFrame(stats).sort_values("mean_diff")
    m = len(sdf)
    sdf = sdf.sort_values("wilcoxon_p")
    sdf["holm_alpha"] = [0.05 / (m - i) for i in range(m)]
    sdf["survives_holm"] = sdf.wilcoxon_p < sdf.holm_alpha
    print(sdf.round(4).to_string(index=False))
    sdf.to_csv(OUT / "candidate_stats.csv", index=False)

    print("\n=== best overall, all variants ranked by MAE ===")
    allr = (df.assign(method=df.base + "/" + df.variant)
              .groupby("method").abs_error.mean().sort_values())
    print(allr.head(12).round(4).to_string())
    print(f"\n  best existing baseline (base variants only): "
          f"{piv['base'].min():.4f} ({piv['base'].idxmin()})")
    print(f"  best candidate variant                     : "
          f"{piv['ti'].min():.4f} ({piv['ti'].idxmin()})")

    write_experiment_manifest(
        EXPERIMENT_ID,
        params={"candidate": "Family F - TI-adaptive shrinkage toward the source anchor",
                "form": "w * base(T) + (1-w) * source_accuracy, w = measured TI",
                "ablations": ["fixed_w (w=0.5)", "anchor_only (w=0)"],
                "n_boot": N_BOOT, "n_models": len(models),
                "anchor": "source validation accuracy - no target labels",
                "status": "NEW ESTIMATOR EXPERIMENT - historical results read only"},
        inputs={"eval_cache": cache_fingerprint(sorted(paths.CACHE_DIR.glob("*.npz")))},
        outputs=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))],
        metrics={"best_base_mae": float(piv["base"].min()),
                 "best_candidate_mae": float(piv["ti"].min()),
                 "n_bases_improved": int((piv["ti_minus_base"] < 0).sum()),
                 "n_bases": int(len(piv))},
        runtime_seconds=time.time() - t0,
        notes="Family F implemented and benchmarked rather than rejected on paper. "
              "Paired Wilcoxon with Holm correction over the bases.",
        out_dir=OUT,
    )
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
