"""Target sample-size sweep — the REQUIRED experiment (readiness §7).

The TI-versus-error sign reversal was demonstrated only at full pool size. A
deployed rover session is 25-400 photographs, so the question is whether the
reversal survives at deployment scale, weakens, disappears, or appears only at
some sizes. All of those are valid outcomes and none is assumed.

Design, fixed here and not altered after seeing results:

* **Sampling unit is the photograph**, drawn **without replacement**. Crops from
  one photograph move together. Sampling crops independently would treat
  correlated observations as independent and inflate everything downstream.
* **Estimand is the accuracy on the drawn sample.** A deployed estimator is
  asked about the session in front of it, not about a pool it never saw.
* **TI uses exactly the definition from the completed audit** -
  ``Var_T / (Var_T + Var_S)`` with cluster bootstraps on both sides, target
  bootstrap taken *within* the drawn sample. The definition is not changed.
* **N = 400 is not available on any arm** (source_test 361 photographs,
  plantdoc 262, tomatovillage 391). It is reported as unavailable, not faked.

Two repetition counts, because the costs differ by two orders of magnitude:
error metrics are one estimator pass per repetition, TI is ``2 * n_boot`` passes.

NEW EXPERIMENT. Historical results are neither read for writing nor modified.

    python scripts/ti_sample_size.py [--reps 50] [--ti-reps 6] [--n-boot 30]
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
from estimators.ti_audit import anchor_sensitivity, target_sensitivity
from inference.cache import CacheKey, load_features, load_labels
from utils.manifest import cache_fingerprint, write_experiment_manifest

EXPERIMENT_ID = "ti_sample_size"
OUT = paths.RESULTS_DIR / EXPERIMENT_ID
ARMS = ("source_test", "plantdoc", "tomatovillage")
SIZES = (25, 50, 100, 200, 400)
BASE_SEED = 20260910


def factories() -> dict:
    f = {n: s["factory"] for n, s in REGISTRY.items()}
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
    ap.add_argument("--reps", type=int, default=50, help="repetitions for error metrics")
    ap.add_argument("--ti-reps", type=int, default=6, help="repetitions for TI")
    ap.add_argument("--n-boot", type=int, default=30, help="cluster bootstraps per TI side")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    models, facs = clean_models(), factories()
    print(f"models {len(models)} | estimators {len(facs)} | reps {args.reps} | "
          f"ti-reps {args.ti_reps} | n_boot {args.n_boot}", flush=True)

    # ---- feasibility, recorded before anything is computed --------------
    feas = []
    probe = models[0]
    for arm in ARMS:
        k = CacheKey(probe["version"], probe["backbone"], probe["seed"], arm,
                     probe["label_set"])
        d = load_features(k)
        n_groups = int(len(np.unique(d["group_ids"])))
        for N in SIZES:
            feas.append({"arm": arm, "n_photographs_available": n_groups,
                         "requested_N": N, "feasible": bool(N <= n_groups)})
    fdf = pd.DataFrame(feas)
    fdf.to_csv(OUT / "feasibility.csv", index=False)
    print("\n=== feasibility (photograph-level, without replacement) ===")
    print(fdf.pivot(index="arm", columns="requested_N", values="feasible").to_string())
    unavailable = sorted(fdf[~fdf.feasible].requested_N.unique())
    print(f"  NOT AVAILABLE on any arm: N = {unavailable}")

    rows, ti_rows = [], []
    total = len(ARMS) * len(models)
    step = 0
    for arm in ARMS:
        for c in models:
            step += 1
            sk = CacheKey(c["version"], c["backbone"], c["seed"], "source_val",
                          c["label_set"])
            tk = CacheKey(c["version"], c["backbone"], c["seed"], arm, c["label_set"])
            s, t = load_features(sk), load_features(tk)
            src = SourceStats(logits=s["logits"], labels=load_labels(sk),
                              features=s["features"])
            y = load_labels(tk)                     # POST-HOC scoring only
            groups = t["group_ids"]
            uniq = np.unique(groups)
            fitted = {n: f().fit(src) for n, f in facs.items()}

            for N in SIZES:
                if N > len(uniq):
                    continue                        # recorded in feasibility.csv
                for rep in range(args.reps):
                    rng = np.random.default_rng(BASE_SEED + 7919 * rep + N)
                    pick = rng.choice(uniq, N, replace=False)
                    idx = np.flatnonzero(np.isin(groups, pick))
                    truth = float((t["logits"][idx].argmax(1) == y[idx]).mean())
                    for name, est in fitted.items():
                        try:
                            v = float(est.estimate(t["logits"][idx], t["features"][idx]))
                        except (ValueError, RuntimeError):
                            continue
                        rows.append({"arm": arm, "N": N, "rep": rep,
                                     "backbone": c["backbone"], "seed": c["seed"],
                                     "estimator": name, "estimate": v,
                                     "true_accuracy": truth, "error": v - truth,
                                     "abs_error": abs(v - truth),
                                     "n_crops": int(len(idx))})

                for rep in range(args.ti_reps):
                    rng = np.random.default_rng(BASE_SEED + 104729 * rep + N)
                    pick = rng.choice(uniq, N, replace=False)
                    idx = np.flatnonzero(np.isin(groups, pick))
                    truth = float((t["logits"][idx].argmax(1) == y[idx]).mean())
                    lg, ft, gp = t["logits"][idx], t["features"][idx], groups[idx]
                    for name, fac in facs.items():
                        sd = BASE_SEED + rep * 31 + N
                        t_sd, _ = target_sensitivity(fac, src, lg, ft, gp,
                                                     args.n_boot, sd)
                        a_sd, _ = anchor_sensitivity(fac, src, s["group_ids"], lg, ft,
                                                     args.n_boot, sd)
                        vt, va = t_sd ** 2, a_sd ** 2
                        try:
                            v = float(fac().fit(src).estimate(lg, ft))
                        except (ValueError, RuntimeError):
                            continue
                        ti_rows.append({"arm": arm, "N": N, "rep": rep,
                                        "backbone": c["backbone"], "seed": c["seed"],
                                        "estimator": name,
                                        "target_sd": t_sd, "anchor_sd": a_sd,
                                        "ti": float(vt / (vt + va)) if (vt + va) > 0
                                        else np.nan,
                                        "estimate": v, "true_accuracy": truth,
                                        "abs_error": abs(v - truth)})
            print(f"  [{step}/{total}] {arm}/{c['backbone']}/seed{c['seed']}", flush=True)

    df = pd.DataFrame(rows)
    tdf = pd.DataFrame(ti_rows)
    df.to_csv(OUT / "sample_size_errors.csv", index=False)
    tdf.to_csv(OUT / "sample_size_ti.csv", index=False)
    pd.set_option("display.width", 250)

    # ---- Table 1: MAE by estimator x arm x N ---------------------------
    t1 = df.pivot_table(index="estimator", columns=["arm", "N"], values="abs_error")
    t1.to_csv(OUT / "table1_mae.csv")
    print("\n=== Table 1: MAE by estimator x arm x N ===")
    print(t1.round(4).to_string())

    # ---- Table 2: TI by estimator x arm x N ----------------------------
    t2 = tdf.pivot_table(index="estimator", columns=["arm", "N"], values="ti")
    t2.to_csv(OUT / "table2_ti.csv")
    print("\n=== Table 2: TI by estimator x arm x N ===")
    print(t2.round(4).to_string())

    # ---- Table 3: Spearman(TI, |error|) by arm x N ---------------------
    # Per MODEL across the nine estimators, preserving the paired structure:
    # the 20 models are 4 backbones x 5 seeds and are NOT independent.
    per = []
    for (arm, N, bb, sd, rep), g in tdf.groupby(["arm", "N", "backbone", "seed", "rep"]):
        g = g.dropna(subset=["ti", "abs_error"])
        if len(g) >= 6 and g.ti.std() > 0:
            per.append({"arm": arm, "N": N, "backbone": bb, "seed": sd, "rep": rep,
                        "rho": float(sps.spearmanr(g.ti, g.abs_error).statistic)})
    pdf = pd.DataFrame(per)
    pdf.to_csv(OUT / "per_model_rho.csv", index=False)

    t3 = (pdf.groupby(["arm", "N"]).rho
            .agg(mean="mean", sd="std", n="count").reset_index())
    boot_lo, boot_hi = [], []
    for _, r in t3.iterrows():
        v = pdf[(pdf.arm == r.arm) & (pdf.N == r.N)].rho.to_numpy()
        b = [np.mean(np.random.default_rng(k).choice(v, len(v), replace=True))
             for k in range(2000)]
        boot_lo.append(np.percentile(b, 2.5))
        boot_hi.append(np.percentile(b, 97.5))
    t3["ci_lo"], t3["ci_hi"] = boot_lo, boot_hi
    t3.to_csv(OUT / "table3_rho_by_arm_N.csv", index=False)
    print("\n=== Table 3: mean per-model Spearman(TI, |error|) by arm x N ===")
    print(t3.round(4).to_string(index=False))

    # ---- Table 4: is the reversal present at each N? -------------------
    print("\n=== Table 4: zero-shift vs severe-shift at each N (paired by model) ===")
    comp = []
    for N in sorted(pdf.N.unique()):
        base = pdf[(pdf.arm == "source_test") & (pdf.N == N)]
        for other in ("plantdoc", "tomatovillage"):
            oth = pdf[(pdf.arm == other) & (pdf.N == N)]
            m = base.merge(oth, on=["backbone", "seed", "rep"],
                           suffixes=("_src", "_oth"))
            if len(m) < 5:
                continue
            d = m.rho_src - m.rho_oth
            try:
                p = float(sps.wilcoxon(m.rho_src, m.rho_oth).pvalue)
            except ValueError:
                p = float("nan")
            b = [np.mean(np.random.default_rng(k).choice(d.to_numpy(), len(d),
                                                         replace=True))
                 for k in range(2000)]
            comp.append({"N": N, "comparison": f"source_test - {other}",
                         "mean_diff": float(d.mean()),
                         "ci_lo": float(np.percentile(b, 2.5)),
                         "ci_hi": float(np.percentile(b, 97.5)),
                         "wilcoxon_p": p, "n_pairs": int(len(d)),
                         "reversal_present": bool(np.percentile(b, 2.5) > 0)})
    cdf = pd.DataFrame(comp)
    cdf.to_csv(OUT / "table4_reversal_by_N.csv", index=False)
    print(cdf.round(4).to_string(index=False))

    write_experiment_manifest(
        EXPERIMENT_ID,
        params={"sampling_unit": "photograph, without replacement",
                "estimand": "accuracy on the drawn sample",
                "arms": list(ARMS), "sizes_requested": list(SIZES),
                "sizes_unavailable": [int(x) for x in unavailable],
                "reps_error": args.reps, "reps_ti": args.ti_reps,
                "n_boot": args.n_boot, "base_seed": BASE_SEED,
                "ti_definition": "Var_T/(Var_T+Var_S), unchanged from ti_audit",
                "n_models": len(models), "label_set": "SHARED_3",
                "status": "NEW EXPERIMENT - no historical result read or modified"},
        inputs={"eval_cache": cache_fingerprint(sorted(paths.CACHE_DIR.glob("*.npz")))},
        outputs=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))],
        metrics={"n_error_rows": int(len(df)), "n_ti_rows": int(len(tdf)),
                 "n_per_model_rho": int(len(pdf))},
        runtime_seconds=time.time() - t0,
        notes="Required readiness experiment. N=400 unavailable on every arm and "
              "reported as such. The 20 models are 4 backbones x 5 seeds and are "
              "not independent; all tests are paired over models.",
        out_dir=OUT,
    )
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
