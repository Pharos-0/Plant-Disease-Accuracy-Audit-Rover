"""POST-HOC supplementary pass for the Paper 2 literature search.

NOT part of the pre-registered protocol. Added because the known-answer recall
check retrieved 1 of 3 papers: the two misses are WEED sprayers, and every
pre-registered family required a disease or reliability term. This pass asks one
narrow question of the weed-sprayer literature: does any targeted-spraying study
address classifier uncertainty, confidence thresholds or domain shift? It reuses
the main script's clients and first-pass rule; results are reported separately.

    python scripts/litsearch_paper2_supplement.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litsearch_paper2 as L  # noqa: E402

OUT = L.OUT / "supplement"
HEAD = ["weed"]
SPRAY = ["targeted spraying", "spot spraying", "precision spraying", "smart sprayer",
         "selective spraying", "site-specific weed"]
REL = ["uncertainty", "confidence threshold", "domain shift", "out-of-distribution",
       "generalization", "conformal prediction"]
RECALL = ["precision weed management smart sprayer artificial intelligence low-cost",
          "robotised patch sprayer autonomous precise spraying"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = []
    for r in REL:
        queries.append({"family": "S", "groups": [HEAD, SPRAY, [r]]})
    for i, x in enumerate(queries, 1):
        x["qid"] = f"S-{i:02d}"
        x["boolean"] = " AND ".join(L.q(g[0]) if len(g) == 1 else L.any_of(g) for g in x["groups"])
        x["plain"] = " ".join(t for g in x["groups"] for t in g)
    log, rows = [], []
    for qu in queries:
        for name, fn in (("openalex", L.openalex), ("arxiv", L.arxiv)):
            recs, total, status = fn(qu)
            log.append({"qid": qu["qid"], "database": name,
                        "query": qu["boolean"], "total_hits": total,
                        "retrieved": len(recs), "status": status})
            for rec in recs:
                rec.update({"qid": qu["qid"], "database": name})
                rows.append(rec)
            print(f"{qu['qid']} {name:9s} {status} hits={total} kept={len(recs)}", flush=True)
    # recall probe: are the two missed known papers findable at all with plain words?
    for j, words in enumerate(RECALL, 1):
        recs, total, status = L.crossref({"plain": words})
        log.append({"qid": f"R-{j}", "database": "crossref", "query": words,
                    "total_hits": total, "retrieved": len(recs), "status": status})
        for rec in recs:
            rec.update({"qid": f"R-{j}", "database": "crossref"})
            rows.append(rec)
        time.sleep(0.3)
    pd.DataFrame(log).to_csv(OUT / "query_log.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "raw_records.csv", index=False)
    main_keys = set(pd.read_csv(L.OUT / "unique_records.csv").doi.dropna())
    df["new"] = ~df.doi.isin(main_keys) | (df.doi == "")
    u = df.drop_duplicates(subset=["doi", "title"])
    u.to_csv(OUT / "unique_records.csv", index=False)
    found = L.KNOWN & set(df.doi)
    print(f"{len(u)} unique; {int(u.new.sum())} not in the main corpus")
    print(f"known papers now found: {sorted(found)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
