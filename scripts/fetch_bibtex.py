"""Fetch BibTeX for a list of DOIs from the DOI registry (Crossref / DataCite).

Content negotiation on https://doi.org returns the registry's own record, so
author names, titles, venues and pages are the publisher's, not retyped. Each
entry's key is replaced with the one given, and the raw record is kept.

    python scripts/fetch_bibtex.py OUT.bib key=doi key=doi ...
"""
from __future__ import annotations

import re
import sys
import time

import requests

UA = {"User-Agent": "mtech-thesis-refcheck/1.0", "Accept": "application/x-bibtex; charset=utf-8"}
DROP = {"keywords", "copyright", "issn", "month", "publisher"}


def _fields(body: str) -> list[tuple[str, str]]:
    """Split `name = {value}, name = number, ...` respecting nested braces."""
    out, i = [], 0
    while i < len(body):
        m = re.compile(r"\s*,?\s*([A-Za-z_]+)\s*=\s*").match(body, i)
        if not m:
            break
        name, i = m.group(1).lower(), m.end()
        if i < len(body) and body[i] == "{":
            depth, j = 0, i
            while j < len(body):
                depth += {"{": 1, "}": -1}.get(body[j], 0)
                if depth == 0:
                    break
                j += 1
            out.append((name, body[i + 1:j]))
            i = j + 1
        else:
            m2 = re.compile(r"[^,}]+").match(body, i)
            out.append((name, m2.group(0).strip()))
            i = m2.end()
    return out


def normalise(bib: str, key: str) -> str:
    """One field per line, the key replaced, noise fields dropped - the layout the
    Word converter parses - and HTML entities from the registry undone."""
    m = re.match(r"\s*@(\w+)\s*\{[^,]*,(.*)\}\s*$", bib, re.S)
    kind, body = m.group(1).lower(), m.group(2)
    lines = [f"@{kind}{{{key},"]
    for name, val in _fields(body):
        if name in DROP:
            continue
        val = val.replace("&amp;", "\\&").replace("–", "--")
        lines.append(f"  {name} = {{{val}}},")
    lines[-1] = lines[-1].rstrip(",")
    return "\n".join(lines) + "\n}"


def main() -> int:
    out, pairs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    entries, failed = [], []
    for key, doi in pairs:
        r = requests.get(f"https://doi.org/{doi}", headers=UA, timeout=60)
        time.sleep(0.5)
        if r.status_code != 200 or not r.text.lstrip().startswith("@"):
            failed.append((key, doi, r.status_code))
            continue
        entries.append(normalise(r.content.decode("utf-8").strip(), key))
        print(f"ok   {key:28s} {doi}")
    head = ("%% FETCHED from the DOI registry by scripts/fetch_bibtex.py; "
            "records are the publishers' own.\n\n")
    with open(out, "w", encoding="utf-8") as f:
        f.write(head + "\n\n".join(entries) + "\n")
    for k, d, s in failed:
        print(f"FAIL {k:28s} {d} (HTTP {s})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
