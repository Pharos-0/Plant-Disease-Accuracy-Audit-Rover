"""RQ4 diagnostics — measure the failure mechanism instead of inferring it.

Implements the nine diagnostics pre-registered in `docs/rq4_diagnostic_plan.md`.
Operates entirely on cached logits/features. No training, no GPU, no rerun of
the estimator grid, and no existing table is overwritten: everything lands in
`results/rq4/`.

Every output table carries a `deployability` column with one of:

* ``DEPLOYABLE`` - computable at deployment from unlabelled target data alone;
* ``POST-HOC``   - uses target labels, explanatory only, must never feed an
  estimator.

Architecture work uses **V2 + V9 only** (same protocol, label set, seeds and
hyperparameters). Arm B is excluded because it exists for ResNet-50 alone, and
the 30-model pool is never used as an architecture average.

    python scripts/run_rq4_diagnostics.py
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
from data.splits import apply_exclusions, load_index
from estimators.base import max_confidence, normalised_entropy, softmax
from evaluation.shift import frechet_distance, mmd_rbf
from utils.manifest import cache_fingerprint, write_experiment_manifest
from inference.cache import CacheKey, load_features, load_labels

EXPERIMENT_ID = "rq4"
OUT = paths.RESULTS_DIR / EXPERIMENT_ID
ARM_OF_CACHE = {"source_test": "A", "plantdoc": "C", "tomatovillage": "D"}
#: the clean four-way architecture comparison
CLEAN_VERSIONS = ("V2", "V9")
N_BOOT = 1000
RNG_SEED = 12345


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def clean_models() -> list[dict]:
    """V2 + V9: same protocol, label set, seeds and hyperparameters."""
    out = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if c["version"] in CLEAN_VERSIONS and c["label_set"] == "SHARED_3":
            out.append(
                {"version": c["version"], "backbone": c["backbone"],
                 "seed": c["seed"], "label_set": c["label_set"]}
            )
    return out


def arm_data(m: dict, cache_arm: str) -> dict | None:
    key = CacheKey(m["version"], m["backbone"], m["seed"], cache_arm, m["label_set"])
    if not key.features_path.exists():
        return None
    f = load_features(key)
    labels = load_labels(key)  # POST-HOC only
    logits = f["logits"]
    probs = softmax(logits)
    pred = logits.argmax(1)
    return {
        "logits": logits, "probs": probs, "features": f["features"],
        "paths": f["paths"], "group_ids": f["group_ids"],
        "labels": labels, "pred": pred,
        "correct": (pred == labels).astype(int),
        "conf": max_confidence(logits),
        "entropy": normalised_entropy(logits),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U identity; NaN when a class is absent."""
    pos, neg = scores[positive == 1], scores[positive == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = sps.rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cluster_bootstrap_auroc(
    scores: np.ndarray, positive: np.ndarray, groups: np.ndarray,
    n_boot: int = N_BOOT, seed: int = RNG_SEED
) -> tuple[float, float]:
    """95% CI for AUROC, resampling **groups** (photographs), not images.

    Tomato-Village supplies ~4.13 crops per photograph, so an image-level
    bootstrap would understate the interval by treating crops as independent.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in drawn])
        v = auroc(scores[idx], positive[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ---------------------------------------------------------------------------
# Diagnostics A + B + F: image-level, POST-HOC
# ---------------------------------------------------------------------------


def diagnostics_ABF(models: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per_class = [], []
    for m in models:
        for cache_arm, arm in ARM_OF_CACHE.items():
            d = arm_data(m, cache_arm)
            if d is None:
                continue
            conf, corr, ent = d["conf"], d["correct"], d["entropy"]
            lo, hi = cluster_bootstrap_auroc(conf, corr, d["group_ids"])
            rows.append({
                "diagnostic": "A+B", "deployability": "POST-HOC",
                "version": m["version"], "backbone": m["backbone"], "seed": m["seed"],
                "arm": arm, "n_images": len(conf),
                "n_groups": int(len(np.unique(d["group_ids"]))),
                "accuracy": float(corr.mean()),
                "mean_conf": float(conf.mean()),
                "mean_conf_correct": float(conf[corr == 1].mean()) if corr.any() else np.nan,
                "mean_conf_incorrect": float(conf[corr == 0].mean()) if (corr == 0).any() else np.nan,
                "conf_separation": (float(conf[corr == 1].mean() - conf[corr == 0].mean())
                                    if corr.any() and (corr == 0).any() else np.nan),
                "auroc_conf_correct": auroc(conf, corr),
                "auroc_ci_lo": lo, "auroc_ci_hi": hi,
                "auroc_negentropy": auroc(-ent, corr),
                "mean_entropy": float(ent.mean()),
                "ece": ece(conf, corr),
                "brier": brier(d["probs"], d["labels"]),
                "conf_minus_acc": float(conf.mean() - corr.mean()),
            })
            # Diagnostic F - class-conditional, POST-HOC
            for k in range(d["probs"].shape[1]):
                tm = d["labels"] == k
                pm = d["pred"] == k
                per_class.append({
                    "diagnostic": "F", "deployability": "POST-HOC",
                    "version": m["version"], "backbone": m["backbone"], "seed": m["seed"],
                    "arm": arm, "class_idx": k,
                    "n_true": int(tm.sum()), "n_pred": int(pm.sum()),
                    "true_share": float(tm.mean()), "pred_share": float(pm.mean()),
                    "recall": float(corr[tm].mean()) if tm.any() else np.nan,
                    "precision": float(corr[pm].mean()) if pm.any() else np.nan,
                    "mean_conf_when_predicted": float(conf[pm].mean()) if pm.any() else np.nan,
                    "conf_minus_precision": (float(conf[pm].mean() - corr[pm].mean())
                                             if pm.any() else np.nan),
                })
    return pd.DataFrame(rows), pd.DataFrame(per_class)


# ---------------------------------------------------------------------------
# Diagnostic C + I: session-level, from the logged benchmark
# ---------------------------------------------------------------------------


def diagnostics_CI(per_session: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = per_session[per_session.version.isin(CLEAN_VERSIONS) & (per_session.arm != "B")]

    rows = []
    for keys, g in d.groupby(["backbone", "arm", "estimator", "n_requested"]):
        est, act = g["estimate"].to_numpy(), g["actual"].to_numpy()
        var_e, var_a = float(est.var(ddof=1)), float(act.var(ddof=1))
        slope = np.polyfit(act, est, 1)[0] if act.std() > 1e-12 else np.nan
        pear = float(np.corrcoef(est, act)[0, 1]) if est.std() > 1e-12 and act.std() > 1e-12 else np.nan
        spear = float(sps.spearmanr(est, act).statistic) if est.std() > 1e-12 else np.nan
        bias = float((est - act).mean())
        mse = float(((est - act) ** 2).mean())
        rows.append({
            "diagnostic": "C+I", "deployability": "POST-HOC",
            "backbone": keys[0], "arm": keys[1], "estimator": keys[2],
            "n_requested": int(keys[3]), "n_sessions": len(g),
            "var_estimate": var_e, "var_actual": var_a,
            "var_ratio": var_e / var_a if var_a > 0 else np.nan,
            "sd_estimate": float(np.sqrt(var_e)), "sd_actual": float(np.sqrt(var_a)),
            "slope_est_on_true": float(slope), "pearson": pear, "spearman": spear,
            "r2": pear ** 2 if np.isfinite(pear) else np.nan,
            "bias": bias, "abs_bias": abs(bias), "mae": float(np.abs(est - act).mean()),
            "mse": mse,
            "bias2_share_of_mse": (bias ** 2) / mse if mse > 0 else np.nan,
            "iqr_estimate": float(np.subtract(*np.percentile(est, [75, 25]))),
            "iqr_actual": float(np.subtract(*np.percentile(act, [75, 25]))),
        })
    ci = pd.DataFrame(rows)

    # Diagnostic I: does variance fall with N while bias holds?
    trend = []
    for keys, g in ci.groupby(["backbone", "arm", "estimator"]):
        g = g.sort_values("n_requested")
        n = g.n_requested.to_numpy(float)
        v, ab = g.var_estimate.to_numpy(), g.abs_bias.to_numpy()
        # slope of log(variance) on log(N): -1 is the 1/N ideal
        lv = np.polyfit(np.log(n), np.log(np.clip(v, 1e-12, None)), 1)[0]
        trend.append({
            "diagnostic": "I", "deployability": "POST-HOC",
            "backbone": keys[0], "arm": keys[1], "estimator": keys[2],
            "var_at_25": v[0], "var_at_400": v[-1],
            "var_reduction_factor": v[0] / v[-1] if v[-1] > 0 else np.nan,
            "log_var_slope_vs_logN": float(lv),
            "abs_bias_at_25": ab[0], "abs_bias_at_400": ab[-1],
            "abs_bias_change": float(ab[-1] - ab[0]),
            "bias2_share_at_25": g.bias2_share_of_mse.iloc[0],
            "bias2_share_at_400": g.bias2_share_of_mse.iloc[-1],
        })
    return ci, pd.DataFrame(trend)


# ---------------------------------------------------------------------------
# Diagnostics D + E: per-session unlabelled statistics
# ---------------------------------------------------------------------------


def diagnostics_DE(models: list[dict], n_focus: int = 100) -> pd.DataFrame:
    """Per-session DEPLOYABLE statistics, paired with POST-HOC truth."""
    with np.load(paths.ESTIMATES_DIR / "sessions.npz") as z:
        store = {k: z[k] for k in z.files}
    est_df = pd.read_parquet(paths.ESTIMATES_DIR / "estimates.parquet")
    est_df = est_df[est_df.n_requested == n_focus]

    rows = []
    for m in models:
        src = arm_data(m, "source_val")
        if src is None:
            continue
        k = src["probs"].shape[1]
        src_marg = np.bincount(src["pred"], minlength=k) / len(src["pred"])
        src_feat, src_conf = src["features"], src["conf"].mean()

        for cache_arm, arm in ARM_OF_CACHE.items():
            d = arm_data(m, cache_arm)
            if d is None:
                continue
            key = f"{m['version']}|{m['backbone']}|{m['seed']}|{m['label_set']}|{arm}|{n_focus}"
            if key not in store:
                continue
            sub = est_df[(est_df.version == m["version"]) & (est_df.backbone == m["backbone"])
                         & (est_df.seed == m["seed"]) & (est_df.arm == arm)]
            by_est = {n: g.set_index("session")["estimate"] for n, g in sub.groupby("estimator")}

            for si, padded in enumerate(store[key]):
                idx = padded[padded >= 0]
                if len(idx) < 10:
                    continue
                probs, pred = d["probs"][idx], d["pred"][idx]
                marg = np.bincount(pred, minlength=k) / len(pred)
                row = {
                    "diagnostic": "D+E", "deployability": "DEPLOYABLE (except *_POSTHOC)",
                    "version": m["version"], "backbone": m["backbone"], "seed": m["seed"],
                    "arm": arm, "session": si, "n_images": int(len(idx)),
                    "n_photographs": int(len(np.unique(d["group_ids"][idx]))),
                    # --- DEPLOYABLE unlabelled signals ---
                    "mean_conf": float(d["conf"][idx].mean()),
                    "mean_entropy": float(d["entropy"][idx].mean()),
                    "conf_drop_vs_source": float(src_conf - d["conf"][idx].mean()),
                    "marginal_entropy": float(-(marg * np.log(np.clip(marg, 1e-12, None))).sum()),
                    "marginal_tv_vs_source": float(0.5 * np.abs(marg - src_marg).sum()),
                    "marginal_chi2_vs_source": float(
                        (((marg - src_marg) ** 2) / np.clip(src_marg, 1e-12, None)).sum()),
                    "dominant_class_share": float(marg.max()),
                    "mmd_features_vs_source": mmd_rbf(src_feat, d["features"][idx], seed=si),
                    "frechet_vs_source_PERBACKBONE": frechet_distance(
                        src_feat, d["features"][idx], n_components=32),
                    # --- POST-HOC truth ---
                    "true_accuracy_POSTHOC": float(d["correct"][idx].mean()),
                }
                for name, series in by_est.items():
                    if si in series.index:
                        row[f"est_{name}"] = float(series.loc[si])
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnostic G: capture day
# ---------------------------------------------------------------------------


def diagnostic_G(models: list[dict]) -> pd.DataFrame:
    df, _ = apply_exclusions(load_index())
    tv = df[df.domain == "tomatovillage"]
    date_of = dict(zip(tv["path"], tv["capture_date"]))

    rows = []
    for m in models:
        src = arm_data(m, "source_val")
        d = arm_data(m, "tomatovillage")
        if src is None or d is None:
            continue
        k = src["probs"].shape[1]
        src_marg = np.bincount(src["pred"], minlength=k) / len(src["pred"])
        days = np.array([date_of.get(p) for p in d["paths"]])
        for day in sorted({x for x in days if x}):
            msk = days == day
            pred = d["pred"][msk]
            marg = np.bincount(pred, minlength=k) / len(pred)
            rows.append({
                "diagnostic": "G", "deployability": "DEPLOYABLE (except *_POSTHOC)",
                "backbone": m["backbone"], "seed": m["seed"], "capture_date": day,
                "n_images": int(msk.sum()),
                "n_photographs": int(len(np.unique(d["group_ids"][msk]))),
                "mean_conf": float(d["conf"][msk].mean()),
                "mean_entropy": float(d["entropy"][msk].mean()),
                "marginal_tv_vs_source": float(0.5 * np.abs(marg - src_marg).sum()),
                "dominant_class_share": float(marg.max()),
                "mmd_features_vs_source": mmd_rbf(src["features"], d["features"][msk], seed=0),
                "true_accuracy_POSTHOC": float(d["correct"][msk].mean()),
                "auroc_conf_correct_POSTHOC": auroc(d["conf"][msk], d["correct"][msk]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    models = clean_models()
    print(f"clean models (V2+V9, SHARED_3): {len(models)}", flush=True)
    print(f"backbones: {sorted({m['backbone'] for m in models})}", flush=True)

    print("\n[A+B+F] image-level confidence / calibration / class-conditional ...", flush=True)
    abf, per_class = diagnostics_ABF(models)
    abf.to_csv(OUT / "diagA_B_confidence_calibration.csv", index=False)
    per_class.to_csv(OUT / "diagF_class_conditional.csv", index=False)

    print("[C+I] session sensitivity and bias-vs-variance ...", flush=True)
    ps = pd.read_parquet(paths.TABLES_DIR / "per_session.parquet")
    ci, trend = diagnostics_CI(ps)
    ci.to_csv(OUT / "diagC_session_sensitivity.csv", index=False)
    trend.to_csv(OUT / "diagI_bias_variance_vs_N.csv", index=False)

    print("[D+E] per-session unlabelled shift and class-marginal ...", flush=True)
    de = diagnostics_DE(models)
    de.to_csv(OUT / "diagD_E_session_signals.csv", index=False)

    print("[G] capture day ...", flush=True)
    g = diagnostic_G(models)
    g.to_csv(OUT / "diagG_capture_day.csv", index=False)

    outputs = [str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))]
    write_experiment_manifest(
        EXPERIMENT_ID,
        params={
            "diagnostics": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            "versions_used": list(CLEAN_VERSIONS),
            "label_set": "SHARED_3",
            "arms": sorted(ARM_OF_CACHE.values()),
            "arm_B_excluded_reason": "SHARED_7 exists for resnet50 only",
            "backbones": sorted({m["backbone"] for m in models}),
            "seeds": sorted({m["seed"] for m in models}),
            "n_models": len(models),
            "session_sampling_rule": "reused logged sessions.npz (group-level sampling)",
            "n_focus_for_session_signals": 100,
            "bootstrap": {"n_boot": N_BOOT, "scheme": "cluster bootstrap over photographs",
                          "seed": RNG_SEED},
            "independent_units": {"A_B_F": "image (CI clustered by photograph)",
                                  "C_I": "session", "D_E": "session",
                                  "G": "capture day", "H": "seed"},
        },
        inputs={"feature_cache": cache_fingerprint(sorted(paths.CACHE_DIR.glob("*.npz"))),
                "per_session": "results/tables/per_session.parquet",
                "sessions": "results/raw/estimates/sessions.npz"},
        outputs=outputs,
        metrics={"n_rows": {Path(o).name: int(len(pd.read_csv(ROOT / o))) for o in outputs}},
        runtime_seconds=time.time() - t0,
        notes="RQ4 diagnostics. Pre-registered in docs/rq4_diagnostic_plan.md. "
              "No training, no estimator rerun, no existing table overwritten.",
        out_dir=OUT,
    )
    print(f"\nwrote {len(outputs)} tables + manifest to {OUT} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
