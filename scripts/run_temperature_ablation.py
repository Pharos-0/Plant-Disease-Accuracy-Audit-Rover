"""Ablation: does calibrating the source model rescue the estimators?

The report's "what would change the conclusion" section names this as the first
thing to try, and `docs/estimator_reproducibility.md` records why it is open: the
ATC paper specifies calibrated probabilities, but the official release fits a
`TempScaling` object and then computes `softmax(inverse_softmax(probs))`, which is
the identity — so the released pipeline runs uncalibrated, and we followed it.

Here we fit a single temperature on the **source validation set** (source labels
only — the target stays unlabeled, so this remains label-free for deployment),
apply it to source and target logits alike, and re-run the whole estimator suite.

This is an **ablation reported alongside the pre-registered result**, not a change
to the protocol. Nothing in `src/estimators/` is modified and no threshold is
tuned toward a hypothesis (§7 anti-confirmation rule).

    python scripts/run_temperature_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from estimators.base import SourceStats, softmax
from estimators.methods import ALL_ESTIMATORS, build
from evaluation.metrics import error_metrics
from inference.cache import CacheKey, load_features, load_labels
from sessions.build import build_sessions

CACHE_ARM_OF = {"A": "source_test", "B": "plantdoc", "C": "plantdoc", "D": "tomatovillage"}
N_SESSIONS = 100
SESSION_N = 100


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Single temperature minimising NLL on the source validation set."""

    def nll(log_t: float) -> float:
        t = float(np.exp(log_t))  # keep T > 0
        p = softmax(logits / t)
        return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def expected_calibration_error(logits: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    p = softmax(logits)
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def arm_of(cache_arm: str, label_set: str) -> str:
    if cache_arm == "source_test":
        return "A"
    if cache_arm == "plantdoc":
        return "C" if label_set == "SHARED_3" else "B"
    return "D"


def main() -> int:
    paths.ensure_dirs()
    manifests = sorted(paths.RAW_RESULTS.glob("*/manifest_*.json"))
    if not manifests:
        print("no trained models found")
        return 1

    rows: list[dict] = []
    temps: list[dict] = []

    for mp in manifests:
        cfg = json.loads(mp.read_text(encoding="utf-8"))["config"]
        version, backbone, seed, label_set = (
            cfg["version"], cfg["backbone"], cfg["seed"], cfg["label_set"]
        )
        src_key = CacheKey(version, backbone, seed, "source_val", label_set)
        if not src_key.features_path.exists():
            continue

        src = load_features(src_key)
        src_labels = load_labels(src_key)  # SOURCE labels - permitted
        temperature = fit_temperature(src["logits"], src_labels)

        ece_before = expected_calibration_error(src["logits"], src_labels)
        ece_after = expected_calibration_error(src["logits"] / temperature, src_labels)
        temps.append(
            {
                "version": version, "backbone": backbone, "seed": seed,
                "label_set": label_set, "temperature": temperature,
                "source_ece_before": ece_before, "source_ece_after": ece_after,
            }
        )

        for calibrated in (False, True):
            t = temperature if calibrated else 1.0
            source = SourceStats(logits=src["logits"] / t, labels=src_labels)
            fitted = {n: build(n).fit(source) for n in ALL_ESTIMATORS}

            for cache_arm in ("source_test", "plantdoc", "tomatovillage"):
                key = CacheKey(version, backbone, seed, cache_arm, label_set)
                if not key.features_path.exists():
                    continue
                tgt = load_features(key)
                logits, feats, groups = tgt["logits"], tgt["features"], tgt["group_ids"]
                labels = load_labels(key)  # evaluation only, after estimates
                preds_correct = (logits.argmax(1) == labels).astype(float)
                arm = arm_of(cache_arm, label_set)

                plan = build_sessions(
                    groups, SESSION_N, arm=arm, seed=seed, n_sessions=N_SESSIONS
                )
                for name, est in fitted.items():
                    ests, acts = [], []
                    for idx in plan.indices:
                        if len(idx) == 0:
                            continue
                        ests.append(est.estimate(logits[idx] / t, feats[idx]))
                        acts.append(float(preds_correct[idx].mean()))
                    if not ests:
                        continue
                    m = error_metrics(np.array(ests), np.array(acts))
                    rows.append(
                        {
                            "version": version, "backbone": backbone, "seed": seed,
                            "label_set": label_set, "arm": arm, "estimator": name,
                            "calibrated": calibrated, "temperature": t,
                            "mae": m["mae"], "bias": m["bias"],
                            "pearson_r": m["pearson_r"], "std_estimate": m["std_estimate"],
                        }
                    )
        print(f"  {version}/{backbone}/seed{seed}/{label_set}: T={temperature:.3f} "
              f"ECE {ece_before:.4f} -> {ece_after:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(paths.TABLES_DIR / "temperature_ablation.csv", index=False)
    pd.DataFrame(temps).to_csv(paths.TABLES_DIR / "temperature_fitted.csv", index=False)

    print("\n=== temperatures fitted on source validation ===")
    tdf = pd.DataFrame(temps)
    print(tdf.groupby("backbone")[["temperature", "source_ece_before", "source_ece_after"]]
          .mean().round(4).to_string())

    print("\n=== MAE, uncalibrated vs temperature-scaled (N=100) ===")
    piv = df.pivot_table(index=["arm", "estimator"], columns="calibrated", values="mae")
    piv.columns = ["uncalibrated", "calibrated"]
    piv["change"] = piv["calibrated"] - piv["uncalibrated"]
    print(piv.round(4).to_string())

    print("\n=== correlation with truth, uncalibrated vs calibrated ===")
    pr = df.pivot_table(index=["arm", "estimator"], columns="calibrated", values="pearson_r")
    pr.columns = ["uncalibrated", "calibrated"]
    pr["change"] = pr["calibrated"] - pr["uncalibrated"]
    print(pr.round(4).to_string())

    print(f"\nwrote {paths.TABLES_DIR / 'temperature_ablation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
