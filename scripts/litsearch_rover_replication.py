"""Deep replication audit — agricultural robotics x reliability/performance monitoring.

The question this must answer honestly:

    Has someone already built an agricultural robot that decides whether to
    trust its own perception model, using label-free performance estimation or
    an equivalent runtime reliability signal?

If yes at Level 4-5, the rover research integration adds nothing and must be
demoted to a demonstration. That is a real possible outcome and the search is
built to find it, not to avoid it.

Queries are lists of individually-quoted phrases joined by AND, so a multi-word
concept is never collapsed into one impossible exact phrase.

OpenAlex is budget-limited (~100 requests/UTC day), so it is used only for the
highest-value pairings; arXiv and Crossref carry the rest.

    python scripts/litsearch_rover_replication.py
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

ROBOT = ["agricultural robot", "agricultural rover", "field robot",
         "autonomous spraying", "precision spraying", "farm robot"]
RELIAB = ["runtime monitoring", "runtime assurance", "safety monitor",
          "perception monitoring", "model monitoring", "reliability monitoring",
          "out-of-distribution detection", "failure detection",
          "selective prediction", "abstention", "uncertainty estimation"]
ESTIM = ["label-free accuracy estimation", "unsupervised accuracy estimation",
         "performance estimation", "accuracy estimation", "deployment gate"]

#: (phrases, use_openalex). OpenAlex is spent only where it is decisive.
def build() -> list[tuple[list[str], bool]]:
    q: list[tuple[list[str], bool]] = []
    # A. the direct replication question: robot x label-free estimation
    for r in ROBOT:
        for e in ESTIM[:3]:
            q.append(([r, e], True))
    # B. robot x runtime reliability (functional replication)
    for r in ROBOT[:4]:
        for s in RELIAB:
            q.append(([r, s], False))
    # C. plant disease x reliability
    for s in ["runtime monitoring", "out-of-distribution detection",
              "selective prediction", "uncertainty estimation",
              "performance estimation", "abstention"]:
        q.append((["plant disease", s], True))
    # D. autonomous action gated by estimated performance
    for pair in (["deployment", "label-free", "robot"],
                 ["autonomous", "accuracy estimation", "unlabeled"],
                 ["robot", "unsupervised accuracy estimation"],
                 ["perception", "functional insufficiency"],
                 ["safety cage", "machine learning", "perception"],
                 ["risk-aware autonomy", "perception"],
                 ["selective autonomy", "perception"],
                 ["test-time risk estimation", "deployment"]):
        q.append((pair, True))
    return q


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, log = [], []
    oa_used = 0
    for phrases, use_oa in build():
        sq = " AND ".join(f'abs:"{p}"' for p in phrases)
        ax, ax_n, ax_s = arxiv_query(sq)
        oa, oa_n, oa_s = ([], None, "not_requested")
        if use_oa and oa_used < 55:
            oa, oa_n, oa_s = search_openalex(" AND ".join(phrases))
            oa_used += 1
        cr, cr_n, cr_s = search_crossref(" ".join(phrases))
        for r in ax + oa + cr:
            r["found_by_query"] = " + ".join(phrases)
            r["found_by_search"] = "ROVER_REPLICATION"
            r["question"] = "rover_replication"
        rows.extend(ax + oa + cr)
        log.append({"phrases": " + ".join(phrases), "query": sq,
                    "arxiv_total": ax_n, "arxiv_status": ax_s,
                    "openalex_total": oa_n, "openalex_status": oa_s,
                    "crossref_retrieved": len(cr), "crossref_status": cr_s})
        f = lambda v: "  -  " if v is None else f"{v:>6}"   # noqa: E731
        print(f"  ax={f(ax_n)} oa={f(oa_n)}  {' + '.join(phrases)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rover_replication_records.csv", index=False)
    pd.DataFrame(log).to_csv(OUT / "rover_replication_query_log.csv", index=False)
    print(f"\n{len(df)} records | OpenAlex requests used: {oa_used}")

    write_experiment_manifest(
        "litsearch_rover_replication",
        params={"status": "SUPPLEMENTARY - targeted replication audit",
                "question": "has an agricultural robot already gated autonomous "
                            "action on label-free performance estimation",
                "levels": "0 none, 1 component, 2 conceptual, 3 methodological, "
                          "4 functional replication, 5 direct replication",
                "n_queries": len(log), "openalex_requests": oa_used},
        inputs={"protocol": "docs/litsearch_protocol.md"},
        outputs=["results/litsearch/rover_replication_records.csv",
                 "results/litsearch/rover_replication_query_log.csv"],
        metrics={"n_records": int(len(df))},
        runtime_seconds=time.time() - t0,
        notes="A negative result means 'no matching work identified in the sources "
              "searched', never 'first'.",
        out_dir=OUT,
        filename="manifest_rover_replication.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
