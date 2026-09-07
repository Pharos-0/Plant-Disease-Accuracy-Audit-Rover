"""Phase 2 of the benchmark: score the estimates against the truth.

Runs strictly after ``run_estimates.py`` has written its output. This is the
only stage permitted to open target labels.

    python scripts/run_evaluation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from evaluation.metrics import (
    bootstrap_ci,
    error_metrics,
    gate_operating_characteristic,
    minimum_session_size,
)
from evaluation.shift import shift_report
from inference.cache import CacheKey, load_features, load_labels

CACHE_ARM_OF = {"A": "source_test", "B": "plantdoc", "C": "plantdoc", "D": "tomatovillage"}
MAE_TOLERANCES = [0.05, 0.10]


def actual_session_accuracies(est_df: pd.DataFrame, sessions: dict) -> pd.DataFrame:
    """True accuracy of every session. Opens target labels - permitted here."""
    keys = est_df[
        ["version", "backbone", "seed", "label_set", "arm", "n_requested"]
    ].drop_duplicates()

    records = []
    label_cache: dict[tuple, np.ndarray] = {}
    pred_cache: dict[tuple, np.ndarray] = {}

    for _, r in keys.iterrows():
        cache_arm = CACHE_ARM_OF[r.arm]
        ck = (r.version, r.backbone, int(r.seed), cache_arm, r.label_set)
        if ck not in label_cache:
            key = CacheKey(r.version, r.backbone, int(r.seed), cache_arm, r.label_set)
            label_cache[ck] = load_labels(key)
            pred_cache[ck] = load_features(key)["logits"].argmax(1)
        labels, preds = label_cache[ck], pred_cache[ck]
        correct = (preds == labels).astype(np.float64)

        store_key = (
            f"{r.version}|{r.backbone}|{int(r.seed)}|{r.label_set}|{r.arm}|{int(r.n_requested)}"
        )
        if store_key not in sessions:
            continue
        idx_mat = sessions[store_key]
        for si, padded in enumerate(idx_mat):
            idx = padded[padded >= 0]
            if len(idx) == 0:
                continue
            records.append(
                {
                    "version": r.version,
                    "backbone": r.backbone,
                    "seed": int(r.seed),
                    "label_set": r.label_set,
                    "arm": r.arm,
                    "n_requested": int(r.n_requested),
                    "session": si,
                    "actual": float(correct[idx].mean()),
                    "actual_n": int(len(idx)),
                }
            )
    return pd.DataFrame(records)


def measure_shift(est_df: pd.DataFrame) -> pd.DataFrame:
    """Frechet / MMD / label-shift between source and each target arm."""
    rows = []
    combos = est_df[["version", "backbone", "seed", "label_set", "arm"]].drop_duplicates()
    for _, r in combos.iterrows():
        src_key = CacheKey(r.version, r.backbone, int(r.seed), "source_val", r.label_set)
        tgt_key = CacheKey(r.version, r.backbone, int(r.seed), CACHE_ARM_OF[r.arm], r.label_set)
        if not (src_key.features_path.exists() and tgt_key.features_path.exists()):
            continue
        s, t = load_features(src_key), load_features(tgt_key)
        sl, tl = load_labels(src_key), load_labels(tgt_key)
        k = s["logits"].shape[1]
        rep = shift_report(s["features"], t["features"], sl, tl, n_classes=k, seed=int(r.seed))
        rep.update(
            {
                "version": r.version, "backbone": r.backbone, "seed": int(r.seed),
                "label_set": r.label_set, "arm": r.arm,
                "target_accuracy": float((t["logits"].argmax(1) == tl).mean()),
                "source_accuracy": float((s["logits"].argmax(1) == sl).mean()),
            }
        )
        rep.pop("source_prior", None)
        rep.pop("target_prior", None)
        rows.append(rep)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()

    est_path = paths.ESTIMATES_DIR / "estimates.parquet"
    if not est_path.exists():
        print("no estimates found - run scripts/run_estimates.py first")
        return 1

    est_df = pd.read_parquet(est_path)
    with np.load(paths.ESTIMATES_DIR / "sessions.npz") as z:
        sessions = {k: z[k] for k in z.files}
    print(f"{len(est_df):,} estimates over {len(sessions)} session groups", flush=True)

    print("computing true session accuracies ...", flush=True)
    actual = actual_session_accuracies(est_df, sessions)

    merged = est_df.merge(
        actual,
        on=["version", "backbone", "seed", "label_set", "arm", "n_requested", "session"],
        how="inner",
    )
    merged["error"] = merged["estimate"] - merged["actual"]
    merged.to_parquet(paths.TABLES_DIR / "per_session.parquet", index=False)
    print(f"  {len(merged):,} scored sessions", flush=True)

    # ---- aggregate ------------------------------------------------------
    group_cols = ["version", "backbone", "label_set", "arm", "n_requested", "estimator"]
    agg_rows = []
    for keys, g in merged.groupby(group_cols):
        m = error_metrics(g["estimate"].to_numpy(), g["actual"].to_numpy())
        lo, hi = bootstrap_ci(np.abs(g["error"].to_numpy()))
        m.update(dict(zip(group_cols, keys)))
        m["mae_ci_lo"], m["mae_ci_hi"] = lo, hi
        agg_rows.append(m)
    agg = pd.DataFrame(agg_rows)
    agg.to_parquet(paths.TABLES_DIR / "aggregate_metrics.parquet", index=False)
    agg.to_csv(paths.TABLES_DIR / "aggregate_metrics.csv", index=False)

    # ---- N* --------------------------------------------------------------
    nstar_rows = []
    for keys, g in agg.groupby(["version", "backbone", "label_set", "arm", "estimator"]):
        row = dict(zip(["version", "backbone", "label_set", "arm", "estimator"], keys))
        for tol in MAE_TOLERANCES:
            row[f"n_star_at_{tol:.2f}"] = minimum_session_size(g, tol)
        nstar_rows.append(row)
    pd.DataFrame(nstar_rows).to_csv(paths.TABLES_DIR / "n_star.csv", index=False)

    # ---- gate operating characteristic ----------------------------------
    gate_rows = []
    for keys, g in merged.groupby(["version", "backbone", "label_set", "arm", "estimator", "n_requested"]):
        oc = gate_operating_characteristic(g["estimate"].to_numpy(), g["actual"].to_numpy())
        for c, v in zip(
            ["version", "backbone", "label_set", "arm", "estimator", "n_requested"], keys
        ):
            oc[c] = v
        gate_rows.append(oc)
    if gate_rows:
        pd.concat(gate_rows, ignore_index=True).to_parquet(
            paths.TABLES_DIR / "gate_operating_characteristic.parquet", index=False
        )

    # ---- shift magnitude -------------------------------------------------
    print("measuring shift magnitude ...", flush=True)
    shift = measure_shift(est_df)
    if len(shift):
        shift.to_csv(paths.TABLES_DIR / "shift_magnitude.csv", index=False)

    print(f"\nwrote tables to {paths.TABLES_DIR}")
    print("\n--- MAE by estimator and arm (N=100, averaged over backbones/seeds) ---")
    view = agg[agg.n_requested == 100]
    if len(view):
        piv = view.pivot_table(index="estimator", columns="arm", values="mae", aggfunc="mean")
        print(piv.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
