"""Targeted follow-up searches — **supplementary, not the pre-registered set**.

`docs/litsearch_protocol.md` fixed 222 queries in advance and those remain the
systematic search. This script runs *additional* targeted queries to settle
questions the systematic search raised but could not answer, plus a **correction
re-run** of one search block that the systematic pass executed incorrectly.
Results are written to their own files and never merged into the pre-registered
corpus.

Supplementary searching after a systematic pass is normal practice; passing it
off as part of the pre-registered set would not be. Hence the separate script,
the separate outputs, and the label on every row.

    SQ1  Is the *architecture dependence* of label-free accuracy estimators
         already documented? This is the one open row in the gap table that
         could REMOVE a claimed contribution, so it gets the most queries.
    SQ2  Is the calibration-anchor / intercept-dominance pathology described
         anywhere?
    SQ3  Retrieve AutoEval (Deng & Zheng, CVPR 2021) by name - an abstract
         search missed it because the term is overloaded by "AutoEval-Video".
    S6c  **Correction.** The pre-registered S6 block (block C of the protocol)
         was written as concept *combinations* but executed as single exact
         phrases, because `search_arxiv` wraps a query with no " AND " in it in
         one `abs:"..."`. `abs:"calibration accuracy estimation artifact"` can
         never match anything. All four S6 zeros were therefore artefacts, not
         evidence, and Q4 is re-run here as proper conjunctions.

Queries are written as **lists of phrases** and joined with AND, so this class
of error cannot recur: each phrase is quoted individually.

OpenAlex is deliberately not used - its daily quota is reserved for stage 2 of
the systematic search, which matters more.

    python scripts/litsearch_followup.py
"""
from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from utils.manifest import write_experiment_manifest

sys.path.insert(0, str(ROOT / "scripts"))
from litsearch import UA, _get, search_crossref  # noqa: E402

OUT = paths.RESULTS_DIR / "litsearch"
NS = {"a": "http://www.w3.org/2005/Atom"}

#: Each entry is a list of phrases, ANDed. Each phrase is quoted on its own, so
#: a multi-word concept is never collapsed into one impossible exact phrase.
FOLLOWUP: dict[str, list[list[str]]] = {
    "SQ1_architecture_dependence": [
        ["unsupervised accuracy estimation", "architecture"],
        ["accuracy estimation", "vision transformer"],
        ["out-of-distribution error", "architecture"],
        ["performance prediction", "architecture", "distribution shift"],
        ["accuracy estimation", "batch normalization"],
        ["accuracy estimation", "backbone"],
        ["estimating accuracy", "architectures"],
        ["unsupervised accuracy estimation", "fails"],
    ],
    "SQ2_anchor_pathology": [
        ["accuracy estimation", "trivial baseline"],
        ["performance estimation", "trivial baseline"],
        ["accuracy estimation", "pitfalls"],
        ["unsupervised accuracy estimation", "evaluation protocol"],
        ["accuracy estimation", "source accuracy"],
        ["accuracy estimation", "calibration", "distribution shift"],
        ["performance estimation", "shortcut"],
        ["accuracy estimation", "regression to the mean"],
    ],
    "S6c_correction_anchor_pathology": [
        ["accuracy estimation", "calibration", "artifact"],
        ["accuracy estimation", "intercept"],
        ["accuracy estimation", "constant predictor"],
        ["performance estimation", "baseline", "unlabeled"],
    ],
}
#: title searches, for works whose term is overloaded in abstracts
TITLE_QUERIES = [
    ("SQ3_autoeval_by_name", 'ti:"Are Labels Always Necessary"'),
    ("SQ3_autoeval_by_name", 'ti:"Classifier Accuracy Evaluation"'),
    ("SQ1_architecture_dependence", 'ti:"accuracy estimation" AND abs:"architecture"'),
]


def arxiv_query(search_query: str) -> tuple[list[dict], int | None, str]:
    """Run a raw arXiv search_query string, returning (records, total, status)."""
    r, status = _get("http://export.arxiv.org/api/query",
                     {"search_query": search_query, "max_results": 60,
                      "sortBy": "relevance", "sortOrder": "descending"},
                     api="arxiv")
    if r is None:
        return [], None, status
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return [], None, "xml_parse_error"
    tot_el = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    total = int(tot_el.text) if tot_el is not None and tot_el.text else 0
    out = []
    for e in root.findall("a:entry", NS):
        out.append({
            "database": "arxiv",
            "id": (e.findtext("a:id", default="", namespaces=NS) or "").strip(),
            "doi": "",
            "title": " ".join((e.findtext("a:title", default="",
                                          namespaces=NS) or "").split()),
            "year": (e.findtext("a:published", default="", namespaces=NS) or "")[:4],
            "venue": "arXiv", "type": "preprint", "cited_by": None,
            "authors": "; ".join(a.findtext("a:name", default="", namespaces=NS)
                                 for a in e.findall("a:author", NS)[:6]),
            "abstract": " ".join((e.findtext("a:summary", default="",
                                             namespaces=NS) or "").split()),
        })
    time.sleep(3.1)
    return out, total, status


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, log = [], []

    for sq, phrase_lists in FOLLOWUP.items():
        print(f"\n=== {sq} ===", flush=True)
        for phrases in phrase_lists:
            sq_str = " AND ".join(f'abs:"{p}"' for p in phrases)
            ax, ax_n, ax_s = arxiv_query(sq_str)
            cr, cr_n, cr_s = search_crossref(" ".join(phrases))
            for r in ax + cr:
                r.update(followup_question=sq, found_by_query=sq_str,
                         found_by_search="FOLLOWUP", question=sq)
            rows.extend(ax + cr)
            log.append({"followup_question": sq, "query": sq_str,
                        "phrases": " | ".join(phrases),
                        "arxiv_total": ax_n, "arxiv_status": ax_s,
                        "crossref_retrieved": len(cr), "crossref_status": cr_s})
            print(f"  arXiv={'FAIL' if ax_n is None else f'{ax_n:>5}'}  "
                  f"{' + '.join(phrases)}", flush=True)

    print("\n=== title searches ===", flush=True)
    for sq, sq_str in TITLE_QUERIES:
        ax, ax_n, ax_s = arxiv_query(sq_str)
        for r in ax:
            r.update(followup_question=sq, found_by_query=sq_str,
                     found_by_search="FOLLOWUP_TITLE", question=sq)
        rows.extend(ax)
        log.append({"followup_question": sq, "query": sq_str, "phrases": "",
                    "arxiv_total": ax_n, "arxiv_status": ax_s,
                    "crossref_retrieved": 0, "crossref_status": "not_run"})
        print(f"  arXiv={'FAIL' if ax_n is None else f'{ax_n:>5}'}  {sq_str}",
              flush=True)

    df = pd.DataFrame(rows)
    # Deliberately NOT named raw_*.csv: litsearch_screen.py globs that pattern,
    # and these supplementary records must not silently join the corpus.
    df.to_csv(OUT / "followup_records.csv", index=False)
    pd.DataFrame(log).to_csv(OUT / "followup_query_log.csv", index=False)
    print(f"\n{len(df)} records -> results/litsearch/followup_records.csv")

    write_experiment_manifest(
        "litsearch_followup",
        params={"status": "SUPPLEMENTARY - not part of the pre-registered 222",
                "why": "settles questions the systematic search raised, and "
                       "corrects the S6 block which ran as impossible exact phrases",
                "databases": ["arxiv", "crossref"],
                "openalex": "deliberately not used; quota reserved for stage 2",
                "query_form": "lists of individually-quoted phrases joined by AND",
                "questions": list(FOLLOWUP) + ["SQ3_autoeval_by_name"]},
        inputs={"protocol": "docs/litsearch_protocol.md"},
        outputs=["results/litsearch/followup_records.csv",
                 "results/litsearch/followup_query_log.csv"],
        metrics={"n_queries": sum(len(v) for v in FOLLOWUP.values())
                 + len(TITLE_QUERIES),
                 "n_records": int(len(df))},
        runtime_seconds=time.time() - t0,
        notes="Supplementary targeted search, reported separately from the "
              "pre-registered corpus and never merged into it. Includes the S6 "
              "correction re-run.",
        out_dir=OUT,
        filename="manifest_followup.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
