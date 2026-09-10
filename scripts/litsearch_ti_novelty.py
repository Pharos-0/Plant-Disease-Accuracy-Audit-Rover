"""Targeted novelty verification for the target-information audit.

**Supplementary and narrow.** The systematic search (222 pre-registered queries)
is complete and is not re-run. This asks one question the systematic search was
not designed to answer:

    Has anyone used target-data sensitivity, perturbation, ablation, or variance
    decomposition to audit whether a LABEL-FREE ACCURACY ESTIMATOR is genuinely
    responsive to the unlabelled target population?

Searching for "Sobol" alone would answer the wrong question - the mathematics is
known and that equivalence is already documented. What matters is the
**application**, so every query pairs a sensitivity/attribution concept with an
accuracy-estimation concept.

Queries are lists of individually-quoted phrases joined with AND, so a
multi-word concept is never collapsed into one impossible exact phrase - the bug
that produced a false null earlier in this project.

    python scripts/litsearch_ti_novelty.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from utils.manifest import write_experiment_manifest

sys.path.insert(0, str(ROOT / "scripts"))
from litsearch import search_crossref, search_openalex  # noqa: E402
from litsearch_followup import arxiv_query  # noqa: E402

OUT = paths.RESULTS_DIR / "litsearch"

#: sensitivity/attribution concept  x  accuracy-estimation concept
PAIRS: list[list[str]] = [
    ["sensitivity analysis", "accuracy estimation"],
    ["sensitivity", "unsupervised accuracy estimation"],
    ["target sensitivity", "distribution shift"],
    ["variance decomposition", "accuracy estimation"],
    ["variance decomposition", "performance prediction"],
    ["ablation", "unlabeled", "accuracy estimation"],
    ["perturbation", "accuracy estimation", "unlabeled"],
    ["permutation", "performance estimation", "unlabeled"],
    ["input attribution", "performance estimation"],
    ["how much", "target data", "accuracy estimation"],
    ["estimator", "responsive", "target distribution"],
    ["auditing", "performance estimation"],
    ["validity", "out-of-distribution performance prediction"],
    ["meta-evaluation", "accuracy estimation"],
    ["evaluating", "accuracy estimators", "distribution shift"],
    ["trivial baseline", "out-of-distribution performance"],
    ["constant predictor", "accuracy estimation"],
    ["source calibration", "dominates", "estimate"],
    ["anchor", "accuracy estimation", "distribution shift"],
    ["spurious", "performance estimation", "unlabeled"],
    ["shortcut", "accuracy estimation", "benchmark"],
    ["benchmark validity", "performance prediction"],
]


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, log = [], []
    for phrases in PAIRS:
        sq = " AND ".join(f'abs:"{p}"' for p in phrases)
        ax, ax_n, ax_s = arxiv_query(sq)
        oa, oa_n, oa_s = search_openalex(" AND ".join(phrases))
        cr, cr_n, cr_s = search_crossref(" ".join(phrases))
        for r in ax + oa + cr:
            r["found_by_query"] = " + ".join(phrases)
            r["found_by_search"] = "TI_NOVELTY"
            r["question"] = "ti_application_novelty"
        rows.extend(ax + oa + cr)
        log.append({"phrases": " + ".join(phrases), "query": sq,
                    "arxiv_total": ax_n, "arxiv_status": ax_s,
                    "openalex_total": oa_n, "openalex_status": oa_s,
                    "crossref_retrieved": len(cr), "crossref_status": cr_s})
        fmt = lambda v: "FAIL" if v is None else f"{v:>7}"   # noqa: E731
        print(f"  ax={fmt(ax_n)} oa={fmt(oa_n)}  {' + '.join(phrases)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ti_novelty_records.csv", index=False)
    pd.DataFrame(log).to_csv(OUT / "ti_novelty_query_log.csv", index=False)
    print(f"\n{len(df)} records -> results/litsearch/ti_novelty_records.csv")

    write_experiment_manifest(
        "litsearch_ti_novelty",
        params={"status": "SUPPLEMENTARY - targeted, narrow; the systematic search "
                          "is complete and was not re-run",
                "question": "has target-data sensitivity been used to audit label-free "
                            "accuracy estimators",
                "query_form": "individually quoted phrases joined by AND",
                "n_pairs": len(PAIRS),
                "databases": ["arxiv", "openalex", "crossref"]},
        inputs={"protocol": "docs/litsearch_protocol.md"},
        outputs=["results/litsearch/ti_novelty_records.csv",
                 "results/litsearch/ti_novelty_query_log.csv"],
        metrics={"n_records": int(len(df))},
        runtime_seconds=time.time() - t0,
        notes="A negative result here means 'no matching work identified in the "
              "searched sources', never 'first'.",
        out_dir=OUT,
        filename="manifest_ti_novelty.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
