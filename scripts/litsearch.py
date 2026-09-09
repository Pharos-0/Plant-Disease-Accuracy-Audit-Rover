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


def _get(url: str, params: dict | None = None, tries: int = 4) -> requests.Response | None:
    """One polite GET with backoff. Returns None rather than raising."""
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=45)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(3 * (attempt + 1))
                continue
            return r if r.ok else None
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


# ---------------------------------------------------------------- OpenAlex
def _openalex_abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as an inverted index; rebuild the text."""
    if not inv:
        return ""
    order: list[tuple[int, str]] = []
    for word, positions in inv.items():
        order.extend((p, word) for p in positions)
    return " ".join(w for _, w in sorted(order))


def search_openalex(query: str) -> tuple[list[dict], int]:
    """Title-and-abstract search. Returns (records, total_count_reported)."""
    rows, cursor, total = [], "*", 0
    while cursor and len(rows) < MAX_PER_QUERY:
        r = _get("https://api.openalex.org/works", {
            "filter": f"title_and_abstract.search:{query}",
            "per-page": PER_PAGE, "cursor": cursor, "mailto": MAILTO,
            "select": "id,doi,title,publication_year,abstract_inverted_index,"
                      "primary_location,type,cited_by_count,authorships",
        })
        if r is None:
            break
        js = r.json()
        total = js.get("meta", {}).get("count", total)
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
        cursor = js.get("meta", {}).get("next_cursor")
        time.sleep(0.15)
    return rows, total


# ---------------------------------------------------------------- Crossref
# CAVEAT, recorded rather than worked around. Crossref's `query.bibliographic` is
# a relevance ranking over the whole index, not a boolean filter: it ignores the
# "AND" in a composed query and reports totals in the millions. Its `total` is
# therefore NOT a count of matching works and must never be quoted as one. The
# top-ranked records it returns are still screened like every other record, so
# the protocol is followed as written; the effect is simply that Crossref
# contributes many records that screening then excludes. Reported in the PRISMA
# flow instead of being hidden by re-tuning the query after seeing results.
def search_crossref(query: str, rows_wanted: int = 200) -> tuple[list[dict], int]:
    r = _get("https://api.crossref.org/works", {
        "query.bibliographic": query, "rows": min(rows_wanted, 100),
        "mailto": MAILTO, "select": "DOI,title,abstract,issued,container-title,type,author",
    })
    if r is None:
        return [], 0
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
    return out, msg.get("total-results", 0)


# ---------------------------------------------------------------- arXiv
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, max_results: int = 100) -> tuple[list[dict], int]:
    terms = " AND ".join(f'abs:"{t}"' for t in query.split(" AND "))
    r = _get("http://export.arxiv.org/api/query", {
        "search_query": terms, "max_results": max_results,
        "sortBy": "relevance", "sortOrder": "descending",
    })
    if r is None:
        return [], 0
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return [], 0
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
    return out, total


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


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    queries = build_queries()
    print(f"pre-registered queries: {len(queries)}", flush=True)

    all_rows: list[dict] = []
    log: list[dict] = []
    for i, spec in enumerate(queries, 1):
        q = spec["query"]
        oa, oa_n = search_openalex(q)
        cr, cr_n = search_crossref(q)
        # arXiv is rate-limited to ~1 req/3 s, so it is used for the searches
        # where preprints matter most rather than for the S4 context sweep.
        if spec["search"] in ("S1", "S2", "S3", "S5", "S6"):
            ax, ax_n = search_arxiv(q)
        else:
            ax, ax_n = [], 0
        for r in oa + cr + ax:
            r["found_by_search"] = spec["search"]
            r["found_by_query"] = q
            r["question"] = spec["question"]
        all_rows.extend(oa + cr + ax)
        log.append({**spec, "openalex_total": oa_n, "openalex_retrieved": len(oa),
                    "crossref_total": cr_n, "crossref_retrieved": len(cr),
                    "arxiv_total": ax_n, "arxiv_retrieved": len(ax)})
        print(f"  [{i}/{len(queries)}] {spec['search']} {q[:62]:<62} "
              f"oa={oa_n:>7} cr={cr_n:>7} ax={ax_n:>5}", flush=True)

    raw = pd.DataFrame(all_rows)
    raw.to_csv(OUT / "raw_all.csv", index=False)
    for db, g in raw.groupby("database"):
        g.to_csv(OUT / f"raw_{db}.csv", index=False)
    qlog = pd.DataFrame(log)
    qlog.to_csv(OUT / "query_log.csv", index=False)

    # ---- deduplicate: DOI first, then normalised title -----------------
    raw["norm_title"] = raw.title.map(norm_title)
    raw = raw[raw.norm_title.str.len() > 8]
    raw["dedup_key"] = raw.doi.where(raw.doi.str.len() > 0, raw.norm_title)
    # keep the richest record per work: prefer one that has an abstract
    raw["_has_abs"] = (raw.abstract.fillna("").str.len() > 40).astype(int)
    dedup = (raw.sort_values(["_has_abs", "cited_by"], ascending=False)
                .groupby("dedup_key", as_index=False).first())
    hits = (raw.groupby("dedup_key")
               .agg(n_queries=("found_by_query", "nunique"),
                    searches=("found_by_search", lambda s: ",".join(sorted(set(s)))),
                    databases=("database", lambda s: ",".join(sorted(set(s))))))
    dedup = dedup.merge(hits, on="dedup_key", how="left").drop(columns=["_has_abs"])
    dedup.to_csv(OUT / "deduplicated.csv", index=False)

    print(f"\nidentified   : {len(raw):,} records")
    print(f"deduplicated : {len(dedup):,} unique works")
    print(f"with abstract: {(dedup.abstract.fillna('').str.len() > 40).sum():,}")
    print("\nper search:")
    print(raw.groupby("found_by_search").dedup_key.nunique().to_string())

    write_experiment_manifest(
        "litsearch",
        params={"protocol": "docs/litsearch_protocol.md",
                "executed": "2026-09-09",
                "databases_searched": ["openalex", "crossref", "arxiv"],
                "databases_unavailable": ["scopus", "web_of_science",
                                          "ieee_xplore_api", "acm_dl", "dblp",
                                          "semantic_scholar"],
                "unavailable_reason": "institutional credentials / bot challenge / "
                                      "rate limit; probed 2026-09-09",
                "n_queries": len(queries), "max_per_query": MAX_PER_QUERY},
        inputs={"query_blocks": {"T": len(BLOCK_T), "D": len(BLOCK_D),
                                 "S": len(BLOCK_S), "M": len(BLOCK_M),
                                 "C": len(BLOCK_C)}},
        outputs=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.csv"))],
        metrics={"records_identified": int(len(raw)),
                 "unique_works": int(len(dedup))},
        runtime_seconds=time.time() - t0,
        notes="Retrieval only. Screening is a separate script so retrieval can be "
              "re-run without re-deciding anything. Coverage is NOT exhaustive - see "
              "protocol section 3.",
        out_dir=OUT,
    )
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
