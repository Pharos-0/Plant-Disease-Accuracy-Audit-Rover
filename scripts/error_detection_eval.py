"""Validation step 3: does the classifier's own uncertainty identify its errors?

A system that becomes less confident under poor conditions has not thereby
shown that it knows WHEN it is wrong. That requires labels: for each image,
was the prediction actually wrong, and did the uncertainty score flag it?

Four per-image uncertainty scores, all computable from logits alone and so all
available on the rover at inference time:

  msp      maximum softmax probability         (high = confident)
  margin   top-1 minus top-2 probability       (high = confident)
  entropy  negative predictive entropy         (high = confident)
  energy   log-sum-exp of the logits           (high = confident)

For each frozen classifier and each target condition:

  AUROC    probability a random error scores less confident than a random
           correct prediction. 0.5 = the score knows nothing about errors.
  AUPRC    the same question weighted towards the errors; its chance level is
           the error rate, so it is reported beside that rate.
  risk-coverage   accuracy on the most-confident fraction of images, i.e. what
           abstaining on the least confident would buy.

Twenty classifiers are 4 architectures x 5 seeds, not twenty independent
subjects, so the spread reported is across classifiers and is not a confidence
interval for the population of models.

    python scripts/error_detection_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths                                             # noqa: E402
from inference.cache import CacheKey, load_features, load_labels   # noqa: E402

ARMS = ["source_test", "plantdoc", "tomatovillage", "pakistan"]
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]
OUT = paths.RESULTS_DIR / "error_detection"


def scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    ps = np.sort(p, 1)
    return {"msp": ps[:, -1],
            "margin": ps[:, -1] - ps[:, -2],
            "entropy": (p * np.log(np.clip(p, 1e-12, 1))).sum(1),
            "energy": np.log(np.exp(z).sum(1)) + logits.max(1)}


def frozen_models() -> list[dict]:
    out = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if c["label_set"] == "SHARED_3" and c["version"] in ("V2", "V9"):
            out.append(c)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, rc = [], []
    for c in frozen_models():
        for arm in ARMS:
            k = CacheKey(c["version"], c["backbone"], c["seed"], arm, c["label_set"])
            if not k.features_path.exists():
                continue
            lg = load_features(k)["logits"]
            y = load_labels(k)
            wrong = (lg.argmax(1) != y).astype(int)
            for name, s in scores(lg).items():
                if wrong.min() == wrong.max():           # all right or all wrong
                    auroc = auprc = np.nan
                else:
                    auroc = roc_auc_score(wrong, -s)       # low confidence -> error
                    auprc = average_precision_score(wrong, -s)
                rows.append({"arm": arm, "backbone": c["backbone"], "seed": c["seed"],
                             "score": name, "auroc": auroc, "auprc": auprc,
                             "error_rate": wrong.mean(), "n": len(y)})
                if name == "msp":
                    order = np.argsort(-s)
                    for cov in COVERAGES:
                        keep = order[:max(1, int(round(cov * len(y))))]
                        rc.append({"arm": arm, "backbone": c["backbone"], "seed": c["seed"],
                                   "coverage": cov, "accuracy": 1 - wrong[keep].mean()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "error_detection.csv", index=False)
    rcd = pd.DataFrame(rc)
    rcd.to_csv(OUT / "risk_coverage.csv", index=False)

    print("CAN THE CLASSIFIER'S UNCERTAINTY FIND ITS OWN ERRORS?")
    print("AUROC for flagging wrong predictions; 0.5 = no better than chance\n")
    piv = df.pivot_table(index="score", columns="arm", values="auroc",
                         aggfunc=["mean", "std"]).round(3)
    for arm in ARMS:
        line = "  ".join(f"{s}={piv[('mean', arm)][s]:.3f}±{piv[('std', arm)][s]:.3f}"
                         for s in ("msp", "margin", "entropy", "energy"))
        err = df[(df.arm == arm) & (df.score == "msp")].error_rate.mean()
        print(f"  {arm:14s} error rate {err:5.1%}   {line}")

    print("\nAUPRC (chance level = the error rate)")
    for arm in ARMS:
        g = df[(df.arm == arm) & (df.score == "msp")]
        print(f"  {arm:14s} msp AUPRC {g.auprc.mean():.3f}  vs chance {g.error_rate.mean():.3f}")

    print("\nWHAT ABSTAINING BUYS: accuracy on the most-confident fraction (msp)")
    t = rcd.groupby(["arm", "coverage"]).accuracy.mean().unstack("coverage")
    t = t[sorted(t.columns, reverse=True)].reindex(ARMS).round(3)
    print("  coverage      " + "  ".join(f"{c:>5.0%}" for c in t.columns))
    for arm, r in t.iterrows():
        print(f"  {arm:13s} " + "  ".join(f"{v:5.3f}" for v in r.values))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
