"""Screening audit — protocol section 8, plus a recall test.

Two checks, because the screening rules have already been wrong twice and both
errors pointed the same way (towards an emptier gap).

**1. Recall test (new).** A known-answer set of label-free accuracy-estimation
papers, assembled from works already read during this search. For each: is it in
the corpus at all, and if so did the screener mark it `include`? A screener that
throws out papers we *know* belong cannot be trusted to report a null. This is
the check that would have caught bug 1b immediately.

**2. Excluded-set sample (protocol section 8).** A reproducible random sample of
excluded records per exclusion reason, printed for reading. Seeded, so the same
sample comes back on a re-run and a reviewer can check the same records.

Reports only - changes nothing.

    python scripts/litsearch_audit.py
"""
from __future__ import annotations

import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from utils.manifest import write_experiment_manifest

OUT = paths.RESULTS_DIR / "litsearch"
SAMPLE_PER_REASON = 15
SEED = 20260909

#: Known-answer set: every one of these is a label-free accuracy/performance
#: estimation paper, confirmed by reading it during this search. Matched on a
#: distinctive title fragment. Not exhaustive - a recall probe, not a census.
KNOWN_RELEVANT = [
    ("AutoEval / Deng & Zheng", "Are Labels Always Necessary for Classifier Accuracy"),
    ("Platanios agreement", "Estimating Accuracy from Unlabeled Data"),
    ("Steinhardt & Liang", "Unsupervised Risk Estimation Using Only Conditional"),
    ("MaNo", "Exploiting Matrix Norm for Unsupervised Accuracy Estimation"),
    ("GdScore", "Leveraging Gradients for Unsupervised Accuracy Estimation"),
    ("Projection Norm", "Predicting Out-of-Distribution Error with the Projection Norm"),
    ("Distance-Matters", "Distance Matters For Improving Performance Estimation"),
    ("AETTA", "Label-Free Accuracy Estimation for Test-Time Adaptation"),
    ("DAPPER", "Label-Free Performance Estimation after Personalization"),
    ("Mandoline", "Mandoline: Model Evaluation under Distribution Shift"),
    ("ALSA", "Anchors in Logit Space"),
    ("IW-GAE", "Importance Weighted Group Accuracy Estimation"),
    ("Aries", "Labeling-Free Accuracy Estimation"),
    ("LF robustness / driving", "Label-Free Robustness Estimation of Object Detection"),
    ("Self-training ensembles", "Detecting Errors and Estimating Accuracy on Unlabeled Data"),
    ("PPI power analysis", "Power Analysis for Prediction-Powered Inference"),
    ("PPI risk monitoring", "Prediction-Powered Risk Monitoring"),
    ("Crop yield PPI", "Scalable Vision-Guided Crop Yield Estimation"),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main() -> int:
    t0 = time.time()
    src = OUT / "screened.csv"
    if not src.exists():
        print("run scripts/litsearch_screen.py first")
        return 2
    df = pd.read_csv(src).fillna({"title": "", "abstract": "", "reason": "",
                                  "decision": "", "databases": ""})
    df["_nt"] = df.title.map(norm)
    print(f"corpus: {len(df):,} unique works\n")

    # ---- 1. recall test -------------------------------------------------
    print("=" * 96)
    print("1. RECALL TEST - known label-free accuracy-estimation papers")
    print("=" * 96)
    rows = []
    for label, frag in KNOWN_RELEVANT:
        hit = df[df._nt.str.contains(re.escape(norm(frag)), regex=True)]
        if hit.empty:
            rows.append({"paper": label, "in_corpus": False, "decision": "-",
                         "reason": "-", "ok": None})
            print(f"  NOT RETRIEVED  {label:<26} (not in corpus at all)")
        else:
            r = hit.iloc[0]
            ok = r.decision in ("include", "adjacent")
            rows.append({"paper": label, "in_corpus": True, "decision": r.decision,
                         "reason": r.reason, "ok": ok})
            flag = "ok  " if ok else "MISS"
            print(f"  {flag}           {label:<26} -> {r.decision} ({r.reason})")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "audit_recall.csv", index=False)
    in_corpus = res[res.in_corpus]
    n_ok = int(in_corpus.ok.sum()) if len(in_corpus) else 0
    print(f"\n  retrieved by the search : {len(in_corpus)}/{len(res)}")
    print(f"  correctly kept          : {n_ok}/{len(in_corpus)}")
    missed = in_corpus[~in_corpus.ok.astype(bool)]
    if len(missed):
        print("\n  *** SCREENER MISSES - these are known-relevant and were excluded:")
        for _, r in missed.iterrows():
            print(f"      {r.paper}: {r.reason}")
        print("  *** Fix the rules before any null result is quoted.")
    else:
        print("  no screener misses among retrieved known-relevant papers")

    # ---- 2. excluded-set sample ----------------------------------------
    print("\n" + "=" * 96)
    print(f"2. EXCLUDED-SET SAMPLE (seed {SEED}, {SAMPLE_PER_REASON} per reason)")
    print("=" * 96)
    exc = df[df.decision == "exclude"]
    picks = []
    for reason, g in exc.groupby("reason"):
        n = min(SAMPLE_PER_REASON, len(g))
        s = g.sample(n, random_state=SEED)
        s = s.assign(audit_reason=reason)
        picks.append(s)
        print(f"\n--- {reason}  (n={len(g):,}, showing {n}) ---")
        for _, r in s.iterrows():
            print(f"  {str(r.title)[:100]}")
    pd.concat(picks)[["title", "year", "venue", "audit_reason", "found_by_query"]] \
        .to_csv(OUT / "audit_excluded_sample.csv", index=False)

    write_experiment_manifest(
        "litsearch_audit",
        params={"protocol": "docs/litsearch_protocol.md section 8",
                "checks": ["recall test against a known-answer set",
                           "seeded random sample of the excluded set"],
                "seed": SEED, "sample_per_reason": SAMPLE_PER_REASON,
                "n_known_relevant": len(KNOWN_RELEVANT)},
        inputs={"corpus": "results/litsearch/screened.csv"},
        outputs=["results/litsearch/audit_recall.csv",
                 "results/litsearch/audit_excluded_sample.csv"],
        metrics={"known_retrieved": int(len(in_corpus)),
                 "known_total": int(len(res)),
                 "screener_kept": n_ok,
                 "screener_misses": int(len(missed))},
        runtime_seconds=time.time() - t0,
        notes="Audit only; changes nothing. A screener that discards known-relevant "
              "papers cannot support a null result.",
        out_dir=OUT,
        filename="manifest_audit.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
