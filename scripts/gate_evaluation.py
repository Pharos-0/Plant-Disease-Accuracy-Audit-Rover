"""Does the session gate actually work? (Parts 9-10, 14)

`src/rover/session_gate.py` already implements a session-level OPERATE /
REVIEW / WITHHOLD decision, but its thresholds are admitted guesses and the gate
has never been evaluated against the truth. This script evaluates it.

The operational question, stated so it can be scored:

    A rover should operate autonomously on a session only if the classifier's
    TRUE accuracy on that session is at least `tau_safe`. The gate must decide
    this from an estimate alone.

That makes it a binary decision with a measurable confusion matrix, and the
error that matters is asymmetric:

    UNSAFE-OPERATE  gate says operate, truth is below tau_safe   <- the harm
    MISSED          gate withholds, truth was fine               <- the cost

Baselines required by Part 10, all evaluated on the same sessions:

    always_operate      trust the classifier unconditionally
    never_operate       the trivially safe policy
    confidence          mean max-softmax as the gate signal
    <each estimator>    the nine label-free estimators as gate signals
    oracle              uses the true accuracy; not deployable, an upper bound

**No threshold is tuned.** The full operating characteristic is swept and
reported, so there is no opportunity to select a threshold on the test data.

Sessions are drawn at photograph level without replacement, matching the
sample-size sweep. Nothing historical is read for writing or modified.

    python scripts/gate_evaluation.py [--reps 40]
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

from data import paths
from estimators.base import SourceStats
from estimators.methods import REGISTRY, GdScore
from inference.cache import CacheKey, load_features, load_labels
from utils.manifest import cache_fingerprint, write_experiment_manifest

EXPERIMENT_ID = "gate_evaluation"
OUT = paths.RESULTS_DIR / EXPERIMENT_ID
ARMS = ("source_test", "plantdoc", "tomatovillage")
SIZES = (25, 50, 100, 200)
TAU_SAFE = 0.80          # a session is "safe to operate on" if TRUE acc >= this
THRESHOLDS = np.round(np.arange(0.30, 1.001, 0.02), 3)
BASE_SEED = 20260911


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
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    models, facs = clean_models(), factories()
    print(f"models {len(models)} | signals {len(facs)} | reps {args.reps} | "
          f"tau_safe {TAU_SAFE}", flush=True)

    rows = []
    step, total = 0, len(ARMS) * len(models)
    for arm in ARMS:
        for c in models:
            step += 1
            sk = CacheKey(c["version"], c["backbone"], c["seed"], "source_val",
                          c["label_set"])
            tk = CacheKey(c["version"], c["backbone"], c["seed"], arm, c["label_set"])
            s, t = load_features(sk), load_features(tk)
            src = SourceStats(logits=s["logits"], labels=load_labels(sk),
                              features=s["features"])
            y = load_labels(tk)                       # POST-HOC scoring only
            groups = t["group_ids"]
            uniq = np.unique(groups)
            fitted = {n: f().fit(src) for n, f in facs.items()}

            for N in SIZES:
                if N > len(uniq):
                    continue
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
                                     "signal": name, "score": v,
                                     "true_accuracy": truth,
                                     "safe": bool(truth >= TAU_SAFE)})
            print(f"  [{step}/{total}] {arm}/{c['backbone']}/seed{c['seed']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gate_sessions.csv", index=False)
    pd.set_option("display.width", 240)

    print(f"\n=== session pool: {len(df):,} (signal, session) pairs ===")
    base = df[df.signal == "confidence"].drop_duplicates(
        subset=["arm", "N", "rep", "backbone", "seed"])
    print("fraction of sessions where the classifier is genuinely safe "
          f"(true acc >= {TAU_SAFE}):")
    print(base.groupby("arm").safe.mean().round(4).to_string())

    # ---- operating characteristic, no threshold tuned ------------------
    oc = []
    for (arm, signal), g in df.groupby(["arm", "signal"]):
        n_safe, n_unsafe = int(g.safe.sum()), int((~g.safe).sum())
        for thr in THRESHOLDS:
            op = g.score >= thr
            unsafe_operate = int((op & ~g.safe).sum())
            missed = int((~op & g.safe).sum())
            oc.append({"arm": arm, "signal": signal, "threshold": float(thr),
                       "coverage": float(op.mean()),
                       "unsafe_operate_rate": unsafe_operate / max(n_unsafe, 1),
                       "missed_rate": missed / max(n_safe, 1),
                       "n": int(len(g))})
    ocdf = pd.DataFrame(oc)
    ocdf.to_csv(OUT / "gate_operating_characteristic.csv", index=False)

    # ---- headline: coverage achievable at a fixed safety level ---------
    print("\n=== coverage achievable while holding unsafe-operate <= 5% ===")
    print("    (higher coverage is better; 0.000 means the signal cannot be used safely)")
    best = []
    for (arm, signal), g in ocdf.groupby(["arm", "signal"]):
        ok = g[g.unsafe_operate_rate <= 0.05]
        best.append({"arm": arm, "signal": signal,
                     "max_coverage": float(ok.coverage.max()) if len(ok) else 0.0})
    bdf = pd.DataFrame(best)
    piv = bdf.pivot(index="signal", columns="arm", values="max_coverage")
    # add the non-signal policies
    for arm in ARMS:
        sub = base[base.arm == arm]
        piv.loc["always_operate", arm] = 1.0
        piv.loc["never_operate", arm] = 0.0
        piv.loc["oracle", arm] = float(sub.safe.mean())
    print(piv.round(3).to_string())
    piv.to_csv(OUT / "gate_coverage_at_5pct.csv")

    print("\n=== what 'always operate' costs: unsafe-operate rate with no gate ===")
    for arm in ARMS:
        sub = base[base.arm == arm]
        print(f"  {arm:<15} {float((~sub.safe).mean()):.4f} of sessions would be "
              f"operated on unsafely")

    write_experiment_manifest(
        EXPERIMENT_ID,
        params={"question": "can any label-free signal gate autonomous operation",
                "tau_safe": TAU_SAFE, "thresholds_swept": len(THRESHOLDS),
                "threshold_tuning": "NONE - full operating characteristic reported",
                "arms": list(ARMS), "sizes": list(SIZES), "reps": args.reps,
                "sampling": "photograph level, without replacement",
                "base_seed": BASE_SEED, "n_models": len(models),
                "status": "NEW EXPERIMENT - no historical result read or modified"},
        inputs={"eval_cache": cache_fingerprint(sorted(paths.CACHE_DIR.glob("*.npz")))},
        outputs=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))],
        metrics={"n_rows": int(len(df))},
        runtime_seconds=time.time() - t0,
        notes="Deployment-gate evaluation with the Part 10 baselines. Session level "
              "only; no per-image correctness claim is made or tested.",
        out_dir=OUT,
    )
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
