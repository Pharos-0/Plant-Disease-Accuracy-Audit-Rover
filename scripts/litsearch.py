"""Systematic literature search, per `docs/litsearch_protocol.md`.

Retrieval only in this script: it executes the pre-registered queries against
OpenAlex, Crossref and arXiv, records every query string verbatim with its count,
deduplicates by DOI then normalised title, and writes the unscreened corpus.
Screening happens in `scripts/litsearch_screen.py` so that retrieval can be
re-run without re-deciding anything.

The protocol's access limitation is enforced, not worked around: Scopus, Web of
Science and the IEEE/ACM native APIs are unavailable without institutional
credentials, so a null result here means "none found by this search".

    python scripts/litsearch.py
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from utils.manifest import write_experiment_manifest

OUT = paths.RESULTS_DIR / "litsearch"
MAILTO = "tjpharossophysamuel@gmail.com"
UA = {"User-Agent": f"mtech-thesis-litsearch/1.0 (mailto:{MAILTO})"}
PER_PAGE = 200
MAX_PER_QUERY = 600          # cap per query; counts are still recorded in full

# ---------------------------------------------------------------- blocks
# Fixed by the protocol. Do not edit after seeing results.
BLOCK_T = [
    "unsupervised accuracy estimation",
    "label-free accuracy estimation",
    "automatic model evaluation",
    "AutoEval",
    "predicting model performance",
    "performance estimation without labels",
    "estimating classifier accuracy",
    "unsupervised model evaluation",
    "out-of-distribution performance prediction",
    "accuracy prediction unlabeled",
    "unsupervised risk estimation",
]
BLOCK_D = [
    "plant disease", "crop disease", "leaf disease", "PlantVillage",
    "precision agriculture", "agricultural imagery", "agricultural vision",
    "crop monitoring",
]
BLOCK_S = [
    "distribution shift", "domain shift", "laboratory to field", "lab-to-field",
    "field conditions", "out-of-distribution",
]
BLOCK_M = [
    "average thresholded confidence", "difference of confidences",
    "confidence optimal transport", "MaNo", "GdScore",
    "nuclear norm accuracy estimation", "agreement-on-the-line",
    "projection norm", "Frechet distance accuracy estimation",
    "prediction-powered inference", "confidence-based performance estimation",
]
BLOCK_C = [
    "calibration accuracy estimation artifact",
    "intercept accuracy estimation",
    "source accuracy baseline accuracy estimation",
    "trivial baseline performance estimation",
]


class BudgetExhausted(RuntimeError):
    """A daily API quota ran out. Abort - never degrade into recording zeros."""


#: OpenAlex allows 1,000 requests/day and resets at midnight UTC. Retrying a 429
#: spends budget that does not exist and gets us nothing, so a quota refusal
#: aborts the run instead of being retried.
_REQUESTS_USED = {"openalex": 0, "crossref": 0, "arxiv": 0}
# MEASURED, not the documented figure. The `x-ratelimit-limit: 1000` header is
# misleading: the binding constraint is a monetary budget, reported by the API as
# "this request costs $0.001 but you only have $0 remaining", which works out to
# roughly 100 requests per UTC day. Two runs stopped at exactly query 100.
OPENALEX_DAILY_LIMIT = 100
OPENALEX_SAFETY_MARGIN = 2


def _get(url: str, params: dict | None = None, *, api: str,
         tries: int = 3) -> tuple[requests.Response | None, str]:
    """One polite GET. Returns (response, status) - NEVER silently a zero.

    ``status`` is "ok" or a description of the failure, and is recorded per query
    so that a zero count in the results is always distinguishable from a request
    that never succeeded. Conflating the two is how a search fabricates a null
    result, which is exactly what this function exists to prevent.
    """
    if api == "openalex" and _REQUESTS_USED["openalex"] >= (
            OPENALEX_DAILY_LIMIT - OPENALEX_SAFETY_MARGIN):
        raise BudgetExhausted(
            f"OpenAlex: {_REQUESTS_USED['openalex']} requests used, at the "
            f"{OPENALEX_DAILY_LIMIT}/day limit. Resume after midnight UTC.")

    last = "no attempt"
    for attempt in range(tries):
        try:
            _REQUESTS_USED[api] += 1
            r = requests.get(url, params=params, headers=UA, timeout=45)
            if r.status_code == 429:
                retry_after = int(r.headers.get("retry-after", 0) or 0)
                if retry_after > 300 or "Insufficient budget" in r.text:
                    raise BudgetExhausted(
                        f"{api}: quota exhausted, resets in {retry_after}s "
                        f"({retry_after/3600:.1f} h). {r.text[:160]}")
                time.sleep(min(30, 5 * (attempt + 1)))
                last = "rate_limited"
                continue
            if r.status_code in (500, 502, 503):
                time.sleep(3 * (attempt + 1))
                last = f"http_{r.status_code}"
                continue
            if not r.ok:
                return None, f"http_{r.status_code}"
            return r, "ok"
        except requests.RequestException as e:
            last = f"{type(e).__name__}"
            time.sleep(2 * (attempt + 1))
    return None, last


# ---------------------------------------------------------------- OpenAlex
def _openalex_abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as an inverted index; rebuild the text."""
    if not inv:
        return ""
    order: list[tuple[int, str]] = []
    for word, positions in inv.items():
        order.extend((p, word) for p in positions)
    return " ".join(w for _, w in sorted(order))


def search_openalex(query: str) -> tuple[list[dict], int | None, str]:
    """Title-and-abstract search. Returns (records, total_count_or_None, status).

    One request per query: `per-page=200` returns both the true `meta.count` and
    up to 200 records, which is ample for title/abstract screening. Deep
    pagination is what exhausted the daily quota on the first attempt.

    A failed request returns count ``None``, never ``0``.
    """
    rows: list[dict] = []
    r, status = _get("https://api.openalex.org/works", {
        "filter": f"title_and_abstract.search:{query}",
        "per-page": PER_PAGE, "mailto": MAILTO,
        "select": "id,doi,title,publication_year,abstract_inverted_index,"
                  "primary_location,type,cited_by_count,authorships",
    }, api="openalex")
    if r is None:
        return [], None, status
    js = r.json()
    total = js.get("meta", {}).get("count")
    for w in js.get("results", []):
        loc = (w.get("primary_location") or {}).get("source") or {}
        auth = [a.get("author", {}).get("display_name", "")
                for a in (w.get("authorships") or [])[:6]]
        rows.append({
            "database": "openalex",
            "id": w.get("id"),
            "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
            "title": w.get("title") or "",
            "year": w.get("publication_year"),
            "venue": loc.get("display_name") or "",
            "type": w.get("type") or "",
            "cited_by": w.get("cited_by_count"),
            "authors": "; ".join(a for a in auth if a),
            "abstract": _openalex_abstract(w.get("abstract_inverted_index")),
        })
    time.sleep(0.15)
    return rows, total, status


# ---------------------------------------------------------------- Crossref
# CAVEAT, recorded rather than worked around. Crossref's `query.bibliographic` is
# a relevance ranking over the whole index, not a boolean filter: it ignores the
# "AND" in a composed query and reports totals in the millions. Its `total` is
# therefore NOT a count of matching works and must never be quoted as one. The
# top-ranked records it returns are still screened like every other record, so
# the protocol is followed as written; the effect is simply that Crossref
# contributes many records that screening then excludes. Reported in the PRISMA
# flow instead of being hidden by re-tuning the query after seeing results.
def search_crossref(query: str, rows_wanted: int = 200) -> tuple[list[dict], int | None, str]:
    r, status = _get("https://api.crossref.org/works", {
        "query.bibliographic": query, "rows": min(rows_wanted, 100),
        "mailto": MAILTO, "select": "DOI,title,abstract,issued,container-title,type,author",
    }, api="crossref")
    if r is None:
        return [], None, status
    msg = r.json().get("message", {})
    out = []
    for it in msg.get("items", []):
        title = " ".join(it.get("title") or [])
        auth = [f"{a.get('given','')} {a.get('family','')}".strip()
                for a in (it.get("author") or [])[:6]]
        out.append({
            "database": "crossref",
            "id": it.get("DOI", ""),
            "doi": (it.get("DOI") or "").lower(),
            "title": title,
            "year": (it.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0],
            "venue": " ".join(it.get("container-title") or []),
            "type": it.get("type", ""),
            "cited_by": None,
            "authors": "; ".join(a for a in auth if a),
            "abstract": re.sub(r"<[^>]+>", " ", it.get("abstract") or ""),
        })
    time.sleep(0.2)
    return out, msg.get("total-results"), status


# ---------------------------------------------------------------- arXiv
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, max_results: int = 100) -> tuple[list[dict], int | None, str]:
    terms = " AND ".join(f'abs:"{t}"' for t in query.split(" AND "))
    r, status = _get("http://export.arxiv.org/api/query", {
        "search_query": terms, "max_results": max_results,
        "sortBy": "relevance", "sortOrder": "descending",
    }, api="arxiv")
    if r is None:
        return [], None, status
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return [], None, "xml_parse_error"
    total_el = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    total = int(total_el.text) if total_el is not None and total_el.text else 0
    out = []
    for e in root.findall("a:entry", ARXIV_NS):
        def txt(tag: str) -> str:
            el = e.find(f"a:{tag}", ARXIV_NS)
            return (el.text or "").strip() if el is not None else ""
        auth = [a.findtext("a:name", default="", namespaces=ARXIV_NS)
                for a in e.findall("a:author", ARXIV_NS)[:6]]
        doi_el = e.find("{http://arxiv.org/schemas/atom}doi")
        out.append({
            "database": "arxiv",
            "id": txt("id"),
            "doi": (doi_el.text or "").lower() if doi_el is not None else "",
            "title": re.sub(r"\s+", " ", txt("title")),
            "year": int(txt("published")[:4]) if txt("published")[:4].isdigit() else None,
            "venue": "arXiv",
            "type": "preprint",
            "cited_by": None,
            "authors": "; ".join(a for a in auth if a),
            "abstract": re.sub(r"\s+", " ", txt("summary")),
        })
    time.sleep(3.1)          # arXiv asks for one request per 3 s
    return out, total, status


# ---------------------------------------------------------------- driver
def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", (t or "").lower())
    return re.sub(r"[^a-z0-9]+", "", t)


def build_queries() -> list[dict]:
    """The pre-registered search set. Order and content fixed by the protocol."""
    q: list[dict] = []
    for t in BLOCK_T:
        q.append({"search": "S1", "question": "Q1", "query": t})
    for t in BLOCK_T:
        for d in BLOCK_D:
            q.append({"search": "S2", "question": "Q2", "query": f"{t} AND {d}"})
    for t in BLOCK_T[:5]:
        for d in BLOCK_D[:4]:
            for s in ("distribution shift", "domain shift", "laboratory to field"):
                q.append({"search": "S3", "question": "Q3", "query": f"{t} AND {d} AND {s}"})
    for d in BLOCK_D:
        for s in BLOCK_S:
            q.append({"search": "S4", "question": "context", "query": f"{d} AND {s}"})
    for m in BLOCK_M:
        q.append({"search": "S5", "question": "Q1", "query": m})
    for c in BLOCK_C:
        q.append({"search": "S6", "question": "Q4", "query": c})
    return q


def main(argv: list[str] | None = None) -> int:
    """Retrieval only.

    Deduplication and screening live in ``litsearch_screen.py``, which reads every
    ``raw_*.csv`` present - so retrieval can be run in stages (one database today,
    another when its quota resets) and the stages merge naturally.
    """
    ap = argparse.ArgumentParser(description="systematic literature search retrieval")
    ap.add_argument("--databases", default="openalex,crossref,arxiv",
                    help="comma-separated subset to query in this run")
    ap.add_argument("--searches", default="",
                    help="comma-separated search blocks to run (e.g. S5,S6). "
                         "The query SET is fixed by the protocol; this only "
                         "changes which part of it a budget-limited run spends on.")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-run queries already recorded as ok (default: skip them)")
    args = ap.parse_args(argv)
    want = {d.strip() for d in args.databases.split(",") if d.strip()}
    unknown = want - {"openalex", "crossref", "arxiv"}
    if unknown:
        print(f"unknown database(s): {sorted(unknown)}")
        return 2

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    queries = build_queries()
    tag = "_".join(sorted(want))

    if args.searches:
        keep = {s.strip() for s in args.searches.split(",") if s.strip()}
        queries = [q for q in queries if q["search"] in keep]
        print(f"restricted to search blocks {sorted(keep)}", flush=True)

    # ---- resume ---------------------------------------------------------
    # OpenAlex allows only ~100 requests/day, so a naive re-run spends the whole
    # budget repeating queries that already succeeded and never reaches the rest.
    # Skip any query already recorded ok for every requested database.
    prior_log = OUT / f"query_log_{tag}.csv"
    done: set[str] = set()
    if prior_log.exists() and not args.no_resume:
        pl = pd.read_csv(prior_log)
        ok_cols = [f"{db}_status" for db in sorted(want) if f"{db}_status" in pl.columns]
        if ok_cols:
            mask = pl[ok_cols].apply(lambda c: c.eq("ok")).all(axis=1)
            done = set(pl.loc[mask, "query"].astype(str))
    remaining = [q for q in queries if q["query"] not in done]
    if done:
        print(f"resuming: {len(done)} queries already recorded ok, "
              f"{len(remaining)} to run", flush=True)
    queries = remaining
    if not queries:
        print("nothing left to run for these databases and blocks.")
        return 0

    print(f"queries this run: {len(queries)} | databases: {sorted(want)}", flush=True)

    rows_by_db: dict[str, list[dict]] = {d: [] for d in want}
    log: list[dict] = []
    aborted = None
    for i, spec in enumerate(queries, 1):
        q = spec["query"]
        entry = dict(spec)
        try:
            for db in ("openalex", "crossref", "arxiv"):
                if db not in want:
                    entry[f"{db}_total"] = None
                    entry[f"{db}_retrieved"] = 0
                    entry[f"{db}_status"] = "not_requested"
                    continue
                # arXiv is 1 request per 3 s and indexes preprints, so it is not
                # used for the S4 agricultural-context sweep.
                if db == "arxiv" and spec["search"] == "S4":
                    entry["arxiv_total"] = None
                    entry["arxiv_retrieved"] = 0
                    entry["arxiv_status"] = "not_run_by_design"
                    continue
                fn = {"openalex": search_openalex, "crossref": search_crossref,
                      "arxiv": search_arxiv}[db]
                recs, total, status = fn(q)
                for r in recs:
                    r["found_by_search"] = spec["search"]
                    r["found_by_query"] = q
                    r["question"] = spec["question"]
                rows_by_db[db].extend(recs)
                entry[f"{db}_total"] = total
                entry[f"{db}_retrieved"] = len(recs)
                entry[f"{db}_status"] = status
        except BudgetExhausted as e:
            aborted = f"query {i}/{len(queries)}: {e}"
            print(f"\n*** ABORTED: {e}", flush=True)
            print(f"*** {i-1} queries completed. Whatever succeeded is written and "
                  f"marked; nothing that failed is recorded as a zero.", flush=True)
            break

        log.append(entry)

        def cell(db: str) -> str:
            status, total = entry[f"{db}_status"], entry[f"{db}_total"]
            if status in ("not_requested", "not_run_by_design"):
                return "   -   "
            return "  FAIL " if total is None else f"{total:>7}"

        print(f"  [{i}/{len(queries)}] {spec['search']} {q[:56]:<56} "
              f"oa={cell('openalex')} cr={cell('crossref')} ax={cell('arxiv')}",
              flush=True)

    if not log:
        print("\nNo query completed, so there is nothing to write and nothing to "
              "conclude. Existing results are left untouched.")
        if aborted:
            print(f"Reason: {aborted}")
        return 3

    # Accumulate across resumed runs rather than overwriting: a budget-limited
    # database is gathered over several days, and each day must add to what is
    # already on disk instead of replacing it.
    qlog = pd.DataFrame(log)
    if prior_log.exists() and not args.no_resume:
        qlog = (pd.concat([pd.read_csv(prior_log), qlog], ignore_index=True)
                  .drop_duplicates(subset="query", keep="last"))
    qlog.to_csv(OUT / f"query_log_{tag}.csv", index=False)

    total_rows = 0
    for db, rows in rows_by_db.items():
        if not rows:
            continue
        new = pd.DataFrame(rows)
        raw_path = OUT / f"raw_{db}.csv"
        if raw_path.exists() and not args.no_resume:
            new = pd.concat([pd.read_csv(raw_path), new], ignore_index=True)
            if {"id", "found_by_query"}.issubset(new.columns):
                new = new.drop_duplicates(subset=["id", "found_by_query"])
        new.to_csv(raw_path, index=False)
        total_rows += len(new)
        print(f"\n  raw_{db}.csv: {len(new):,} records "
              f"(+{len(rows):,} this run)")

    # ---- integrity report: a zero must never be a disguised failure ------
    ok_states = ("ok", "not_run_by_design", "not_requested")
    n_fail: dict[str, int] = {}
    print("\n=== query status (a FAILED query is NOT a zero result) ===")
    for db in ("openalex", "crossref", "arxiv"):
        col = f"{db}_status"
        n_fail[db] = int((~qlog[col].fillna("").isin(ok_states)).sum())
        vc = qlog[col].value_counts()
        print(f"  {db:<10} " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    if aborted:
        print(f"\n  *** RUN INCOMPLETE: {aborted}")
        print(f"  *** {len(qlog)}/{len(queries)} queries executed. Re-run to resume.")
    else:
        print(f"\n  run complete: {len(qlog)}/{len(queries)} queries")

    write_experiment_manifest(
        f"litsearch_{tag}",
        params={
            "protocol": "docs/litsearch_protocol.md",
            "executed": time.strftime("%Y-%m-%d"),
            "databases_this_run": sorted(want),
            "databases_unavailable": ["scopus", "web_of_science", "ieee_xplore_api",
                                      "acm_dl", "dblp", "semantic_scholar"],
            "unavailable_reason": "institutional credentials / bot challenge / rate "
                                  "limit; probed 2026-09-09",
            "n_queries_planned": len(queries),
            "deviation_openalex_boolean":
                "OpenAlex title_and_abstract.search ANDs the terms but does not "
                "enforce the quoted phrase, so each query is a SUPERSET of the "
                "phrase search written in the protocol. Recall is higher, precision "
                "lower; screening restores precision. A null result under the "
                "broader search is therefore stronger, not weaker, evidence.",
        },
        inputs={"query_blocks": {"T": len(BLOCK_T), "D": len(BLOCK_D),
                                 "S": len(BLOCK_S), "M": len(BLOCK_M),
                                 "C": len(BLOCK_C)}},
        outputs=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))],
        metrics={"records_retrieved": int(total_rows),
                 "queries_executed": int(len(qlog)),
                 "queries_planned": int(len(queries)),
                 "run_complete": aborted is None,
                 "aborted_reason": aborted,
                 "failed_queries_by_db": n_fail,
                 "requests_used": dict(_REQUESTS_USED)},
        runtime_seconds=time.time() - t0,
        notes="Retrieval only; dedup and screening are in litsearch_screen.py, which "
              "merges every raw_*.csv so retrieval can run in stages. Coverage is NOT "
              "exhaustive - see protocol section 3. A failed request records a null "
              "count and a status string, never a zero: conflating the two would "
              "fabricate a null result. Check `run_complete` before quoting a count.",
        out_dir=OUT,
        filename=f"manifest_retrieval_{tag}.json",
    )
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
