"""Phase 7b, post-hoc: *why* does the train/eval switch move GdScore so much?

**This analysis was not pre-registered.** `docs/rq7b_batchnorm_plan.md` fixed the
confirmatory comparison (MAE, rank, calibration r, Spearman, raw-score magnitude)
and `scripts/rq7b_batchnorm_test.py` runs it. This script is exploratory work done
*after* seeing that result, and everything it produces is reported as such.

It exists because the confirmatory numbers contain an oddity that would otherwise
be mis-read: the target raw score is almost **unchanged** between the two modes
(ratio 1.00), yet the calibrated estimate moves by ~0.41. So whatever the switch
did, it did not do it by changing GdScore's score on the field data.

Three measurements, no modelling:

1. **Calibration geometry** - where the target score sits relative to the source
   bootstrap distribution, and the fitted slope/intercept, in each mode.
2. **Decomposition** - `estimate = intercept + slope x score`, so the share of the
   estimate actually carried by the target data can be read off.
3. **Scope** - whether the other two source-calibrated estimators (nuclear norm,
   MaNo) behave the same way, or whether this is specific to GdScore.

Also records the *calibration anchor*: source-validation accuracy recomputed in
each forward mode. Source labels only - the target labels are touched nowhere
except to score an already-written estimate.

No GPU, no retraining, no existing result modified.

    python scripts/rq7b_mechanism.py
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
from estimators.base import SourceStats
from estimators.methods import GdScore, MaNo, NuclearNorm
from inference.cache import CacheKey, load_features, load_labels
from inference.train_mode_cache import (
    GD_BATCH_SIZE,
    TrainModeKey,
    load_train_mode,
    load_train_mode_labels,
)
from utils.manifest import write_experiment_manifest

EXPERIMENT_ID = "rq7b_batchnorm"
OUT = paths.RESULTS_DIR / EXPERIMENT_ID
GD_SEED, THRESHOLD, NORM_TYPE = 1, 0.5, 0.3          # identical to Phase 7


def clean_models() -> list[dict]:
    out = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if c["version"] in ("V2", "V9") and c["label_set"] == "SHARED_3":
            out.append(c)
    return out


def _load(cfg: dict, arm: str, mode: str):
    """Activations + labels for one arm in one forward mode."""
    if mode == "train":
        k = TrainModeKey(cfg["version"], cfg["backbone"], cfg["seed"], arm,
                         cfg["label_set"])
        return load_train_mode(k), load_train_mode_labels(k)
    k = CacheKey(cfg["version"], cfg["backbone"], cfg["seed"], arm, cfg["label_set"])
    return load_features(k), load_labels(k)


def calibration_geometry(models: list[dict]) -> pd.DataFrame:
    """Where does the target score fall relative to the source bootstraps?"""
    rows = []
    for cfg in models:
        for mode in ("train", "eval"):
            s, s_lab = _load(cfg, "source_val", mode)
            t, _ = _load(cfg, "tomatovillage", mode)
            src = SourceStats(logits=s["logits"], labels=s_lab, features=s["features"])
            est = GdScore(threshold=THRESHOLD, norm_type=NORM_TYPE,
                          batch_size=GD_BATCH_SIZE, gd_seed=GD_SEED).fit(src)

            # Replay the bootstrap draws the fit used, to recover the source
            # score distribution the linear map was built on.
            rng = np.random.default_rng(est.seed)
            n = len(src.logits)
            size = max(est.batch_size, min(est.subset_size, n))
            ss = np.asarray([
                est.raw_score(src.logits[i], src.features[i])
                for i in (rng.choice(n, size=size, replace=True)
                          for _ in range(est.n_calibration_subsets))
            ])
            tgt = est.raw_score(t["logits"], t["features"])
            raw_est = est.slope_ * tgt + est.intercept_
            rows.append({
                "backbone": cfg["backbone"], "seed": cfg["seed"], "mode": mode,
                "slope": est.slope_, "intercept": est.intercept_,
                "calibration_r": est.calibration_r_,
                "src_mean": ss.mean(), "src_sd": ss.std(),
                "src_min": ss.min(), "src_max": ss.max(), "tgt": tgt,
                "z_extrap": (tgt - ss.mean()) / max(ss.std(), 1e-12),
                "inside_src_range": bool(ss.min() <= tgt <= ss.max()),
                "raw_estimate": raw_est,
                "clipped_estimate": float(np.clip(raw_est, 0.0, 1.0)),
                "was_clipped": bool(raw_est < 0.0 or raw_est > 1.0),
            })
    return pd.DataFrame(rows)


def anchors(models: list[dict]) -> pd.DataFrame:
    """Source-validation accuracy in each forward mode - the calibration anchor.

    Also the true field accuracy, which is the eval-mode target accuracy and is
    POST-HOC evaluation only.
    """
    rows = []
    for cfg in models:
        r = {"backbone": cfg["backbone"], "seed": cfg["seed"]}
        for arm in ("source_val", "tomatovillage"):
            for mode in ("eval", "train"):
                d, y = _load(cfg, arm, mode)
                r[f"{arm}_{mode}_acc"] = float((d["logits"].argmax(1) == y).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def other_calibrated(models: list[dict]) -> pd.DataFrame:
    """Is the intercept collapse specific to GdScore, or general to our linear map?"""
    rows = []
    for cfg in models:
        s, s_lab = _load(cfg, "source_val", "eval")
        t, t_lab = _load(cfg, "tomatovillage", "eval")
        src = SourceStats(logits=s["logits"], labels=s_lab, features=s["features"])
        acc = float((t["logits"].argmax(1) == t_lab).mean())   # POST-HOC truth
        for cls in (NuclearNorm, MaNo):
            est = cls().fit(src)
            score = est._score_matrix(t["logits"])
            rows.append({
                "estimator": est.name, "backbone": cfg["backbone"], "seed": cfg["seed"],
                "intercept": est.intercept_, "contribution": est.slope_ * score,
                "estimate": est.estimate(t["logits"], t["features"]),
                "source_accuracy": src.accuracy, "pool_accuracy": acc,
                "calibration_r": est.calibration_r_,
            })
    df = pd.DataFrame(rows)
    df["data_share"] = df.contribution.abs() / df.estimate.abs()
    df["abs_error"] = (df.estimate - df.pool_accuracy).abs()
    return df


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    models = clean_models()
    pd.set_option("display.width", 250)
    print(f"models: {len(models)} | POST-HOC mechanism analysis, not pre-registered\n",
          flush=True)

    cal = calibration_geometry(models)
    cal.to_csv(OUT / "rq7b_calibration_probe.csv", index=False)
    anc = anchors(models)
    anc.to_csv(OUT / "rq7b_anchor_accuracy.csv", index=False)

    print("=== 1. calibration geometry (mean over 5 seeds) ===")
    print(cal.groupby(["backbone", "mode"])[
        ["src_mean", "src_sd", "tgt", "z_extrap", "slope", "intercept"]
    ].mean().to_string(float_format=lambda v: f"{v:,.4g}"))
    print("\n  target score, in source-bootstrap SDs from the source mean:")
    print(cal.groupby(["backbone", "mode"]).z_extrap.mean().unstack().round(1).to_string())
    print("\n  fraction of seeds where the target score is INSIDE the source range:")
    print(cal.groupby(["backbone", "mode"]).inside_src_range.mean().unstack().to_string())

    print("\n=== 2. the calibration anchor: source-val accuracy per forward mode ===")
    g = anc.groupby("backbone")[["source_val_eval_acc", "source_val_train_acc",
                                 "tomatovillage_eval_acc"]].mean()
    g.columns = ["src_val_EVAL", "src_val_TRAIN", "true_field_acc"]
    g["|src_TRAIN - field|"] = (g.src_val_TRAIN - g.true_field_acc).abs()
    g["|src_EVAL - field|"] = (g.src_val_EVAL - g.true_field_acc).abs()
    print(g.round(4).to_string())

    # ---- 3. decomposition -------------------------------------------------
    both = pd.read_csv(OUT / "rq7b_train_vs_eval.csv")
    dec = (both.merge(anc, on=["backbone", "seed"])
                .merge(cal[["backbone", "seed", "mode", "slope", "intercept",
                            "tgt", "z_extrap"]], on=["backbone", "seed", "mode"]))
    dec["anchor"] = np.where(dec["mode"] == "train",
                             dec.source_val_train_acc, dec.source_val_eval_acc)
    dec["contribution"] = dec.slope * dec.tgt
    dec["data_share"] = dec.contribution.abs() / dec.estimate.abs()
    dec["anchor_abs_error"] = (dec.anchor - dec.pool_accuracy).abs()
    dec.to_csv(OUT / "rq7b_decomposition.csv", index=False)

    print("\n=== 3. estimate = intercept + slope x score ===")
    print(dec.groupby(["backbone", "mode"])[
        ["intercept", "contribution", "estimate", "anchor", "pool_accuracy",
         "abs_error", "anchor_abs_error"]
    ].mean().to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  share of the estimate carried by the target data:")
    print(dec.groupby(["backbone", "mode"]).data_share.mean().unstack().round(4).to_string())

    r_anchor = float(np.corrcoef(dec[dec["mode"] == "train"].estimate,
                                 dec[dec["mode"] == "train"].source_val_train_acc)[0, 1])
    print(f"\n  GdScore train-mode estimate vs train-mode source accuracy: r = {r_anchor:.3f}")

    print("\n=== 4. GdScore MAE vs the trivial rule 'predict source-val accuracy' ===")
    piv = dec.pivot_table(index="backbone", columns="mode",
                          values=["abs_error", "anchor_abs_error"])
    print(piv.round(4).to_string())

    oth = other_calibrated(models)
    oth.to_csv(OUT / "rq7b_other_calibrated.csv", index=False)
    print("\n=== 5. scope: do nuclear_norm / MaNo collapse to their intercept too? ===")
    print(oth.groupby("estimator")[["intercept", "contribution", "estimate",
                                    "data_share", "calibration_r", "abs_error"]]
          .mean().to_string(float_format=lambda v: f"{v:.4f}"))

    outputs = [str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))]
    write_experiment_manifest(
        EXPERIMENT_ID + "_mechanism",
        params={
            "status": "POST-HOC - not pre-registered; exploratory, run after the "
                      "confirmatory Phase 7b comparison",
            "question": "the target raw score is unchanged between modes, so what "
                        "moves the estimate by 0.41?",
            "threshold": THRESHOLD, "norm_type": NORM_TYPE,
            "batch_size": GD_BATCH_SIZE, "gd_seed": GD_SEED,
            "n_models": len(models), "target_arm": "D", "label_set": "SHARED_3",
        },
        inputs={"confirmatory": "results/rq7b_batchnorm/rq7b_train_vs_eval.csv"},
        outputs=outputs,
        metrics={"gdscore_estimate_vs_source_accuracy_r": r_anchor,
                 "gdscore_mean_data_share": float(dec.data_share.mean()),
                 "nuclear_norm_mean_data_share":
                     float(oth[oth.estimator == "nuclear_norm"].data_share.mean()),
                 "mano_mean_data_share":
                     float(oth[oth.estimator == "mano"].data_share.mean())},
        runtime_seconds=time.time() - t0,
        notes="Post-hoc mechanism analysis for Phase 7b. Measurement only, no "
              "causal claim. Reported separately from the pre-registered result.",
        out_dir=OUT,
        # the confirmatory run owns manifest.json in this directory
        filename="manifest_mechanism.json",
    )
    print(f"\nwrote {len(outputs)} tables + manifest_mechanism.json "
          f"in {time.time()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
