"""Estimator error on the fourth target arm, against the three frozen arms.

Runs the eight cache-computable estimators on every frozen SHARED_3 classifier
and every arm, and reports mean absolute error against the true accuracy. The
oracle constant predictor - the single value minimising MAE over the models, not
deployable because it needs the labels - is included as the same diagnostic
ceiling the manuscript uses.

GdScore is absent by construction: it requires a fresh forward pass in train()
mode over the target images, not just cached logits, so it cannot be evaluated
here without re-running inference. Its absence is stated rather than papered
over, and it is why this table says eight rather than nine.

    python scripts/arm4_evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths                                        # noqa: E402
from estimators.base import SourceStats                       # noqa: E402
from estimators.methods import ALL_ESTIMATORS, build           # noqa: E402
from inference.cache import CacheKey, load_features, load_labels  # noqa: E402

ARMS = ("source_test", "plantdoc", "tomatovillage", "pakistan")
OUT = paths.RESULTS_DIR / "arm4_pakistan"


def frozen_models() -> list[dict]:
    out = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if c["label_set"] == "SHARED_3" and c["version"] in ("V2", "V9"):
            out.append(c)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    models = frozen_models()
    print(f"{len(models)} frozen classifiers x {len(ARMS)} arms x "
          f"{len(ALL_ESTIMATORS)} estimators\n")

    rows = []
    for c in models:
        sk = CacheKey(c["version"], c["backbone"], c["seed"], "source_val", c["label_set"])
        s = load_features(sk)
        source = SourceStats(logits=s["logits"], labels=load_labels(sk),
                             features=s["features"])
        for arm in ARMS:
            k = CacheKey(c["version"], c["backbone"], c["seed"], arm, c["label_set"])
            if not k.features_path.exists():
                continue
            d = load_features(k)
            y = load_labels(k)
            true_acc = float((d["logits"].argmax(1) == y).mean())
            for name in ALL_ESTIMATORS:
                est = build(name).fit(source)
                try:
                    v = est.estimate(d["logits"], d.get("features"))
                except Exception as exc:                 # record, never silently skip
                    rows.append({"arm": arm, "backbone": c["backbone"], "seed": c["seed"],
                                 "estimator": name, "estimate": np.nan,
                                 "true_accuracy": true_acc, "abs_error": np.nan,
                                 "error_note": f"{type(exc).__name__}: {exc}"})
                    continue
                rows.append({"arm": arm, "backbone": c["backbone"], "seed": c["seed"],
                             "estimator": name, "estimate": v,
                             "true_accuracy": true_acc, "abs_error": abs(v - true_acc),
                             "error_note": ""})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "arm4_estimator_errors.csv", index=False)
    failed = int(df.abs_error.isna().sum())
    if failed:
        print(f"{failed} estimator calls failed; see error_note in the CSV\n")

    print("MEAN ABSOLUTE ERROR BY ARM AND ESTIMATOR")
    piv = df.pivot_table(index="estimator", columns="arm", values="abs_error",
                         aggfunc="mean").reindex(columns=list(ARMS)).round(4)
    print(piv.to_string())

    # Oracle constant: the single prediction minimising MAE over the 20 models.
    print("\nORACLE CONSTANT PREDICTOR (needs the labels; a ceiling, not a method)")
    orc = {}
    for arm in ARMS:
        t = df[df.arm == arm].groupby(["backbone", "seed"]).true_accuracy.first().to_numpy()
        if not len(t):
            continue
        grid = np.linspace(0, 1, 2001)
        mae = np.abs(grid[:, None] - t[None, :]).mean(1)
        orc[arm] = (float(grid[mae.argmin()]), float(mae.min()))
        print(f"  {arm:15s} constant {orc[arm][0]:.4f}  MAE {orc[arm][1]:.4f}")

    print("\nDID ANY ESTIMATOR BEAT THE ORACLE CONSTANT?")
    for arm in ARMS:
        if arm not in orc:
            continue
        best = piv[arm].min()
        who = piv[arm].idxmin()
        verdict = "YES" if best < orc[arm][1] else "no"
        print(f"  {arm:15s} best {who} at {best:.4f} vs oracle {orc[arm][1]:.4f}  -> {verdict}")

    pd.DataFrame([{"arm": a, "oracle_constant": v[0], "oracle_mae": v[1]}
                  for a, v in orc.items()]).to_csv(OUT / "arm4_oracle.csv", index=False)
    piv.to_csv(OUT / "arm4_mae_by_arm.csv")
    print(f"\nwrote {OUT / 'arm4_estimator_errors.csv'}")
    print(f"wrote {OUT / 'arm4_mae_by_arm.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
