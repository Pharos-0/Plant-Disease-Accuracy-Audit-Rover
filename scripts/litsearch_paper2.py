"""Retrieval for the Paper 2 literature search, per docs/litsearch_paper2_protocol.md.

Runs the 34 pre-registered queries against OpenAlex, Crossref and arXiv, logs every
query verbatim with its hit count and HTTP status, deduplicates by DOI then by
normalised title, and applies the protocol's automatic first-pass rule. Manual
screening is a separate, recorded step.

No personal details are sent: requests carry a generic User-Agent and no mailto.

    python scripts/litsearch_paper2.py
"""
from __future__ import annotations

import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "litsearch_paper2"
UA = {"User-Agent": "mtech-thesis-litsearch-paper2/1.0"}

# ---------------------------------------------------------------- protocol §4
S = ["selective spraying", "targeted spraying", "spot spraying", "precision spraying",
     "site-specific spraying", "variable rate spraying", "smart sprayer", "robotic sprayer",
     "spraying robot", "patch spraying"]
D = ["plant disease", "disease detection", "leaf disease", "crop disease"]
R2 = ["uncertainty", "confidence", "domain shift", "out-of-distribution"]
F3_TERMS = ["uncertainty", "abstention", "reject option", "out-of-distribution", "domain shift"]
F3_TAIL = ["spraying", "pesticide", "treatment"]
F4_HEAD = ["agricultural robot", "spraying robot", "agricultural sprayer"]
F4_TERMS = ["finite state machine", "formal verification", "runtime monitoring",
            "digital twin", "fault injection", "safety"]
F5_HEAD = ["selective spraying", "spot spraying", "targeted spraying"]
F5_TERMS = ["hit rate", "missed", "spray coverage"]
KNOWN = {"10.1016/j.biosystemseng.2015.12.004", "10.1016/j.biosystemseng.2015.12.018",
         "10.1016/j.compag.2018.12.048"}


def q(term: str) -> str:
    return f'"{term}"'


def any_of(terms: list[str]) -> str:
    return "(" + " OR ".join(q(t) for t in terms) + ")"


def build_queries() -> list[dict]:
    out = []
    for s in S:
        out.append({"family": "F1", "groups": [[s], D]})
    for s in S:
        out.append({"family": "F2", "groups": [[s], R2]})
    for t in F3_TERMS:
        out.append({"family": "F3", "groups": [["plant disease"], [t], F3_TAIL]})
    for t in F4_TERMS:
        out.append({"family": "F4", "groups": [F4_HEAD, [t]]})
    for t in F5_TERMS:
        out.append({"family": "F5", "groups": [F5_HEAD, [t]]})
    for i, x in enumerate(out, 1):
        x["qid"] = f"{x['family']}-{i:02d}"
        x["boolean"] = " AND ".join(q(g[0]) if len(g) == 1 else any_of(g) for g in x["groups"])
        x["plain"] = " ".join(t for g in x["groups"] for t in g)
    return out


# ---------------------------------------------------------------- clients
def get(url: str, params: dict) -> tuple[requests.Response | None, str]:
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
        except requests.RequestException as e:
            time.sleep(3 * (attempt + 1))
            last = f"error:{type(e).__name__}"
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            last = f"http:{r.status_code}"
            continue
        return r, f"http:{r.status_code}"
    return None, last


def inverted_to_text(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def openalex(query: dict) -> tuple[list[dict], int | None, str]:
    r, status = get("https://api.openalex.org/works",
                    {"search": query["boolean"], "per-page": 200,
                     "select": "doi,title,abstract_inverted_index,publication_year,"
                               "primary_location,open_access,type,cited_by_count"})
    time.sleep(0.2)
    if r is None or r.status_code != 200:
        return [], None, status
    js = r.json()
    recs = []
    for w in js.get("results", []):
        loc = w.get("primary_location") or {}
        src = (loc.get("source") or {}).get("display_name", "")
        recs.append({"doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
                     "title": w.get("title") or "", "year": w.get("publication_year"),
                     "venue": src, "abstract": inverted_to_text(w.get("abstract_inverted_index")),
                     "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
                     "type": w.get("type") or "", "cited_by": w.get("cited_by_count")})
    return recs, js.get("meta", {}).get("count"), status


def crossref(query: dict) -> tuple[list[dict], int | None, str]:
    r, status = get("https://api.crossref.org/works",
                    {"query.bibliographic": query["plain"], "rows": 100,
                     "select": "DOI,title,abstract,issued,container-title,type,is-referenced-by-count"})
    time.sleep(0.3)
    if r is None or r.status_code != 200:
        return [], None, status
    js = r.json()["message"]
    recs = []
    for it in js.get("items", []):
        ab = re.sub(r"<[^>]+>", " ", it.get("abstract") or "")
        recs.append({"doi": (it.get("DOI") or "").lower(), "title": (it.get("title") or [""])[0],
                     "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                     "venue": (it.get("container-title") or [""])[0], "abstract": ab,
                     "oa_url": "", "type": it.get("type") or "",
                     "cited_by": it.get("is-referenced-by-count")})
    return recs, js.get("total-results"), status


def arxiv(query: dict) -> tuple[list[dict], int | None, str]:
    parts = []
    for g in query["groups"]:
        alts = [f'abs:"{t}"' for t in g]
        parts.append(alts[0] if len(alts) == 1 else "(" + " OR ".join(alts) + ")")
    r, status = get("http://export.arxiv.org/api/query",
                    {"search_query": " AND ".join(parts), "max_results": 100})
    time.sleep(3.1)                     # arXiv asks for one request per 3 s
    if r is None or r.status_code != 200:
        return [], None, status
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    root = ET.fromstring(r.text)
    total = root.find("o:totalResults", ns)
    recs = []
    for e in root.findall("a:entry", ns):
        doi = e.find("{http://arxiv.org/schemas/atom}doi")
        recs.append({"doi": (doi.text if doi is not None else "").lower(),
                     "title": " ".join((e.findtext("a:title", "", ns)).split()),
                     "year": (e.findtext("a:published", "", ns) or "")[:4],
                     "venue": "arXiv", "abstract": " ".join(e.findtext("a:summary", "", ns).split()),
                     "oa_url": e.findtext("a:id", "", ns), "type": "preprint", "cited_by": None})
    return recs, int(total.text) if total is not None else None, status


# ---------------------------------------------------------------- first pass (§7.1)
SPRAY = re.compile(r"spray|pesticid|fungicid|herbicid|agrochemical|treatment|plant protection", re.I)
VISION = re.compile(r"vision|camera|image|imaging|detect|classif|deep learning|neural|CNN|YOLO|"
                    r"segment|recogni", re.I)
SAFETY = re.compile(r"safety|safe |fail-safe|verif|state machine|runtime monitor|digital twin|"
                    r"fault", re.I)


def first_pass(row) -> str:
    text = f"{row.title} {row.abstract}"
    if not SPRAY.search(text):
        return ""
    rules = []
    if VISION.search(text):
        rules.append("spray+vision")
    if SAFETY.search(text):
        rules.append("spray+safety")
    return ";".join(rules)


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = build_queries()
    assert len(queries) == 34, len(queries)
    log, rows = [], []
    for qu in queries:
        for name, fn in (("openalex", openalex), ("crossref", crossref), ("arxiv", arxiv)):
            recs, total, status = fn(qu)
            log.append({"qid": qu["qid"], "family": qu["family"], "database": name,
                        "query": qu["plain"] if name == "crossref" else qu["boolean"],
                        "total_hits": total, "retrieved": len(recs), "status": status,
                        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            for rec in recs:
                rec.update({"qid": qu["qid"], "family": qu["family"], "database": name})
                rows.append(rec)
            print(f"{qu['qid']:6s} {name:9s} {status:9s} hits={total} kept={len(recs)}", flush=True)
    pd.DataFrame(log).to_csv(OUT / "query_log.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "raw_records.csv", index=False)
    df["key"] = [d if d else "t:" + norm_title(t) for d, t in zip(df.doi, df.title)]
    agg = (df.groupby("key")
             .agg(doi=("doi", "first"), title=("title", "first"), year=("year", "first"),
                  venue=("venue", "first"),
                  abstract=("abstract", lambda s: max(s.fillna(""), key=len)),
                  oa_url=("oa_url", lambda s: next((x for x in s if x), "")),
                  cited_by=("cited_by", "max"),
                  databases=("database", lambda s: ";".join(sorted(set(s)))),
                  qids=("qid", lambda s: ";".join(sorted(set(s)))))
             .reset_index(drop=True))
    # second pass of dedup by normalised title (a DOI and a DOI-less copy of one work)
    agg["nt"] = agg.title.map(norm_title)
    agg = agg.sort_values("doi", ascending=False).drop_duplicates("nt").drop(columns="nt")
    agg["first_pass"] = agg.apply(first_pass, axis=1)
    agg.to_csv(OUT / "unique_records.csv", index=False)
    cand = agg[agg.first_pass != ""]
    cand.to_csv(OUT / "first_pass.csv", index=False)

    found = KNOWN & set(agg.doi)
    print(f"\n{len(df)} raw records, {len(agg)} unique, {len(cand)} pass the first rule")
    print(f"known-answer recall: {len(found)}/{len(KNOWN)} {sorted(KNOWN - found)}")
    failed = [x for x in log if not x["status"].startswith("http:200")]
    print(f"failed requests: {len(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
