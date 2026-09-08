"""RQ5: how many labelled field photographs beat the label-free estimators?

Runs the experiment pre-registered in `docs/rq5_plan.md`. Outputs go to
`results/rq5/`; no existing table is touched, no model is retrained, no
estimator is added or modified.

Order of operations is deliberate:

1. **baseline reproduction** (brief section 29) - stop if it fails;
2. label-free estimates on the full unlabelled Arm-D pool;
3. Setting A - random labelled photograph samples;
4. Setting B - capture-day holdout;
5. manifest.

    python scripts/rq5_labelled_sample.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from data.splits import apply_exclusions, load_index
from estimators.base import SourceStats
from estimators.methods import ALL_ESTIMATORS, build
from inference.cache import CacheKey, load_features, load_labels
from sessions.labelled_sample import labelled_estimate, seed_for
from utils.manifest import ESTIMATOR_SUITE_VERSION, cache_fingerprint, write_experiment_manifest

EXPERIMENT_ID = "rq5"
OUT = paths.RESULTS_DIR / EXPERIMENT_ID
BUDGETS = [5, 10, 20, 30, 50, 75, 100]
FULL_BUDGET = 391
DAY_BUDGETS = [5, 10, 20, 30, 50]
R = 100
BASE_SEED = 20260909
REPRO_TOLERANCE = 1e-6


def clean_models() -> list[dict]:
    out = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        c = m["config"]
        if c["version"] in ("V2", "V9") and c["label_set"] == "SHARED_3":
            out.append({"version": c["version"], "backbone": c["backbone"],
                        "seed": c["seed"], "label_set": c["label_set"],
                        "checkpoint": m.get("checkpoint")})
    return out


def fitted_estimators(m: dict):
    """Fit the eight estimators on source validation, exactly as the pipeline does."""
    src_key = CacheKey(m["version"], m["backbone"], m["seed"], "source_val", m["label_set"])
    src = load_features(src_key)
    source = SourceStats(logits=src["logits"], labels=load_labels(src_key),
                         features=src["features"])
    return {n: build(n).fit(source) for n in ALL_ESTIMATORS}


# ---------------------------------------------------------------------------
# 1. baseline reproduction (brief section 29)
# ---------------------------------------------------------------------------


def reproduce_baselines(models: list[dict], n_focus: int = 100) -> pd.DataFrame:
    """Recompute logged Arm-D session MAE from the same cached inputs."""
    with np.load(paths.ESTIMATES_DIR / "sessions.npz") as z:
        store = {k: z[k] for k in z.files}

    # aggregate_metrics.parquet is aggregated over seeds, so the strictest
    # available comparison is per-seed MAE recomputed from per_session.parquet.
    ps = pd.read_parquet(paths.TABLES_DIR / "per_session.parquet")
    ps = ps[(ps.arm == "D") & (ps.n_requested == n_focus) & (ps.version.isin(["V2", "V9"]))]
    logged = (
        ps.assign(abs_err=ps.error.abs())
        .groupby(["version", "backbone", "seed", "estimator"], as_index=False)
        .abs_err.mean()
        .rename(columns={"abs_err": "mae"})
    )

    rows = []
    for m in models:
        fitted = fitted_estimators(m)
        key = CacheKey(m["version"], m["backbone"], m["seed"], "tomatovillage", m["label_set"])
        d = load_features(key)
        y = load_labels(key)
        correct = (d["logits"].argmax(1) == y).astype(float)
        store_key = f"{m['version']}|{m['backbone']}|{m['seed']}|{m['label_set']}|D|{n_focus}"
        if store_key not in store:
            continue
        for name, est in fitted.items():
            errs = []
            for padded in store[store_key]:
                idx = padded[padded >= 0]
                if len(idx) == 0:
                    continue
                e = est.estimate(d["logits"][idx], d["features"][idx])
                errs.append(abs(e - float(correct[idx].mean())))
            rows.append({"version": m["version"], "backbone": m["backbone"], "seed": m["seed"],
                         "estimator": name, "mae_recomputed": float(np.mean(errs))})
    rep = pd.DataFrame(rows)
    merged = rep.merge(
        logged[["version", "backbone", "seed", "estimator", "mae"]],
        on=["version", "backbone", "seed", "estimator"], how="inner")
    merged["abs_diff"] = (merged.mae_recomputed - merged.mae).abs()
    return merged


# ---------------------------------------------------------------------------
# 2. label-free estimates on the full unlabelled pool
# ---------------------------------------------------------------------------


def label_free_full_pool(models: list[dict]) -> pd.DataFrame:
    rows = []
    for m in models:
        fitted = fitted_estimators(m)
        key = CacheKey(m["version"], m["backbone"], m["seed"], "tomatovillage", m["label_set"])
        d = load_features(key)
        y = load_labels(key)                       # POST-HOC truth only
        pool_acc = float((d["logits"].argmax(1) == y).mean())
        for name, est in fitted.items():
            e = est.estimate(d["logits"], d["features"])   # no labels reach this call
            rows.append({
                "backbone": m["backbone"], "seed": m["seed"], "estimator": name,
                "estimate": e, "pool_accuracy": pool_acc, "abs_error": abs(e - pool_acc),
                "n_unlabelled_crops": int(len(y)),
                "n_labels_used": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Setting A - random labelled photograph samples
# ---------------------------------------------------------------------------


def setting_a(models: list[dict]) -> pd.DataFrame:
    rows = []
    for m in models:
        key = CacheKey(m["version"], m["backbone"], m["seed"], "tomatovillage", m["label_set"])
        d = load_features(key)
        y = load_labels(key)
        correct = (d["logits"].argmax(1) == y).astype(int)
        pool_acc = float(correct.mean())
        groups = d["group_ids"]
        for budget in BUDGETS + [FULL_BUDGET]:
            reps = 1 if budget == FULL_BUDGET else R
            for r in range(reps):
                est = labelled_estimate(correct, groups, budget, seed_for(r, BASE_SEED))
                rows.append({
                    "backbone": m["backbone"], "seed": m["seed"], "repetition": r,
                    "n_label_photographs": budget, "n_label_crops": est.n_crops,
                    "estimate": est.accuracy, "pool_accuracy": pool_acc,
                    "abs_error": abs(est.accuracy - pool_acc),
                    "ci_low": est.ci_low, "ci_high": est.ci_high,
                    "ci_covers_truth": bool(est.ci_low <= pool_acc <= est.ci_high),
                    "rng_seed": seed_for(r, BASE_SEED),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Setting B - capture-day holdout
# ---------------------------------------------------------------------------


def setting_b(models: list[dict]) -> pd.DataFrame:
    df, _ = apply_exclusions(load_index())
    tv = df[df.domain == "tomatovillage"]
    date_of = dict(zip(tv["path"], tv["capture_date"]))

    rows = []
    for m in models:
        key = CacheKey(m["version"], m["backbone"], m["seed"], "tomatovillage", m["label_set"])
        d = load_features(key)
        y = load_labels(key)
        correct = (d["logits"].argmax(1) == y).astype(int)
        days = np.array([date_of.get(p) for p in d["paths"]])
        fitted = fitted_estimators(m)

        for day in sorted({x for x in days if x}):
            msk = days == day
            day_correct = correct[msk]
            day_groups = d["group_ids"][msk]
            day_acc = float(day_correct.mean())
            n_photos = int(len(np.unique(day_groups)))

            # label-free reference for the same day, on unlabelled data only
            for name, est in fitted.items():
                e = est.estimate(d["logits"][msk], d["features"][msk])
                rows.append({
                    "backbone": m["backbone"], "seed": m["seed"], "capture_date": day,
                    "method": f"labelfree_{name}", "n_label_photographs": 0,
                    "n_label_crops": 0, "repetition": 0,
                    "estimate": e, "day_accuracy": day_acc, "abs_error": abs(e - day_acc),
                    "day_photographs": n_photos,
                })
            for budget in DAY_BUDGETS:
                if budget > n_photos:
                    continue
                for r in range(R):
                    est = labelled_estimate(day_correct, day_groups, budget,
                                            seed_for(r, BASE_SEED))
                    rows.append({
                        "backbone": m["backbone"], "seed": m["seed"], "capture_date": day,
                        "method": "labelled", "n_label_photographs": budget,
                        "n_label_crops": est.n_crops, "repetition": r,
                        "estimate": est.accuracy, "day_accuracy": day_acc,
                        "abs_error": abs(est.accuracy - day_acc),
                        "day_photographs": n_photos,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    models = clean_models()
    print(f"models: {len(models)} | backbones: {sorted({m['backbone'] for m in models})}",
          flush=True)

    print("\n[1/4] baseline reproduction (brief section 29) ...", flush=True)
    rep = reproduce_baselines(models)
    rep.to_csv(OUT / "rq5_baseline_reproduction.csv", index=False)
    worst = float(rep.abs_diff.max())
    print(f"  compared {len(rep)} (model, estimator) cells; worst abs diff = {worst:.3e}",
          flush=True)
    if worst > REPRO_TOLERANCE:
        print(f"  STOP: reproduction exceeds tolerance {REPRO_TOLERANCE}", flush=True)
        rep.sort_values("abs_diff", ascending=False).head(10).to_csv(
            OUT / "rq5_baseline_reproduction_FAILURES.csv", index=False)
        return 2
    print("  PASS", flush=True)

    print("[2/4] label-free estimates on the full unlabelled pool ...", flush=True)
    lf = label_free_full_pool(models)
    lf.to_csv(OUT / "rq5_labelfree_full_pool.csv", index=False)

    print("[3/4] Setting A - random labelled photograph samples ...", flush=True)
    a = setting_a(models)
    a.to_csv(OUT / "rq5_settingA_labelled.csv", index=False)

    print("[4/4] Setting B - capture-day holdout ...", flush=True)
    b = setting_b(models)
    b.to_csv(OUT / "rq5_settingB_capture_day.csv", index=False)

    # convergence check (brief section 28)
    full = a[a.n_label_photographs == FULL_BUDGET]
    conv = float(full.abs_error.max())
    print(f"\nlarge-N convergence: max |estimate - pool| at N={FULL_BUDGET} = {conv:.2e}",
          flush=True)
    if conv > 1e-12:
        print("  STOP: full-budget estimate does not equal the pool accuracy", flush=True)
        return 3

    outputs = [str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))]
    write_experiment_manifest(
        EXPERIMENT_ID,
        params={
            "research_question": "labelled field photographs needed to beat label-free estimators",
            "target_arm": "D", "label_set": "SHARED_3",
            "versions_used": ["V2", "V9"], "arm_B_excluded": True,
            "independent_sampling_unit": "photograph (group_id 'ts:IMG<date><time>')",
            "n_independent_photographs": 391, "n_crops": 1616,
            "cluster_sizes": {"healthy": 1, "blight": 8},
            "labelled_budgets_photographs": BUDGETS,
            "convergence_budget": FULL_BUDGET,
            "capture_day_budgets": DAY_BUDGETS,
            "repetitions": R,
            "seed_convention": f"seed = {BASE_SEED} + repetition_index",
            "seeds": [seed_for(r, BASE_SEED) for r in range(R)],
            "labelled_estimator": "ratio estimator: correct crops / total crops in sampled photographs",
            "label_free_estimators": ALL_ESTIMATORS,
            "estimator_suite_version": ESTIMATOR_SUITE_VERSION,
            "metrics": ["MAE", "median absolute error", "95th percentile absolute error"],
            "ci_method": "Wilson score on labelled crops (anti-conservative; coverage measured)",
            "aggregate_ci": "percentile bootstrap over models, 2000 resamples",
            "primary_unit": "trained model (n=20), paired",
            "multiplicity": "Holm across 8 baselines within each budget",
            "reproduction_tolerance": REPRO_TOLERANCE,
        },
        inputs={"feature_cache": cache_fingerprint(sorted(paths.CACHE_DIR.glob("*.npz"))),
                "sessions": "results/raw/estimates/sessions.npz",
                "aggregate_metrics": "results/tables/aggregate_metrics.parquet"},
        outputs=outputs,
        metrics={"baseline_reproduction_worst_abs_diff": worst,
                 "large_n_convergence_max_abs_error": conv,
                 "n_rows": {Path(o).name: int(len(pd.read_csv(ROOT / o))) for o in outputs}},
        runtime_seconds=time.time() - t0,
        notes="RQ5. Pre-registered in docs/rq5_plan.md (committed first). No retraining, "
              "no new estimator, no historical table modified.",
        out_dir=OUT,
    )
    print(f"\nwrote {len(outputs)} tables + manifest to {OUT} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
