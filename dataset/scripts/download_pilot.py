"""Step 6: download a SMALL pilot from licence-verified sources only.

The licence gate is enforced here rather than left to discipline. A source whose
row in `license_registry.csv` does not read VERIFIED / download_permitted=YES is
refused by this script, and the refusal is printed with its reason. "Publicly
downloadable" is not "freely redistributable", and the registry is the only
thing allowed to decide which is which.

The pilot exists to test the METHOD, not to build the dataset: a few dozen
images per source, enough to measure rover-relevance, check for duplicates
against what we already hold, and estimate the full download size before
committing to it.

    python dataset/scripts/download_pilot.py --per-source 40
    python dataset/scripts/download_pilot.py --dry-run     # list only, fetch nothing
"""
from __future__ import annotations

import argparse
import hashlib
import io
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = yaml.safe_load((ROOT / "dataset" / "config.yaml").read_text(encoding="utf-8"))
META = ROOT / "dataset" / "metadata"

UA = {"User-Agent": "rover-tomato-dataset/0.1 (academic research; contact via repository)"}
TIMEOUT = 60
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class Blocked(RuntimeError):
    """A source was refused by the licence gate. Never downgrade this to a warning."""


# ---------------------------------------------------------------- licence gate
def approved_sources() -> pd.DataFrame:
    lic = pd.read_csv(META / "license_registry.csv")
    src = pd.read_csv(META / "source_registry.csv")
    m = lic.merge(src[["source_id", "dataset_doi", "status"]], on="source_id", how="left")
    ok = m[(m.license_status == "VERIFIED") & (m.download_permitted == "YES")]
    blocked = m[~m.source_id.isin(ok.source_id)]
    print("LICENCE GATE")
    for _, r in ok.iterrows():
        print(f"  [ALLOW] {r.source_id} {r['name'][:52]:52s} {r.license_name}")
    for _, r in blocked.iterrows():
        print(f"  [BLOCK] {r.source_id} {r['name'][:52]:52s} {r.download_permitted}")
    print()
    return ok


# ------------------------------------------------------------ Mendeley Data API
def mendeley_files(doi: str) -> list[dict]:
    """List the files of a Mendeley Data deposit from its public API."""
    if not doi.startswith("10.17632/"):
        raise Blocked(f"not a Mendeley DOI: {doi}")
    tail = doi.split("/", 1)[1]
    ds_id, _, version = tail.partition(".")
    url = (f"https://data.mendeley.com/public-api/datasets/{quote(ds_id)}/"
           f"files?folder_id=root&version={version or '1'}")
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def flatten(entries: list[dict]) -> list[dict]:
    """Keep only image-like leaf files, with a download URL and a size."""
    out = []
    for e in entries:
        name = e.get("filename") or e.get("name") or ""
        cd = e.get("content_details") or {}
        url = cd.get("download_url") or e.get("download_url")
        size = cd.get("size") or e.get("size") or 0
        if url and Path(name).suffix.lower() in IMAGE_EXTS:
            out.append({"filename": name, "url": url, "size": int(size)})
    return out


def zenodo_record(doi: str) -> dict:
    """Fetch a Zenodo record by DOI. The API states the licence, so the gate can
    be re-checked against the server rather than trusted from our own CSV."""
    if "zenodo." not in doi:
        raise Blocked(f"not a Zenodo DOI: {doi}")
    rec_id = doi.rsplit("zenodo.", 1)[1]
    r = requests.get(f"https://zenodo.org/api/records/{quote(rec_id)}",
                     headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def zenodo_pilot(src, per_source: int, raw_root: Path, dry_run: bool) -> tuple[list, dict]:
    """Download one Zenodo archive and extract up to `per_source` images."""
    rec = zenodo_record(src.dataset_doi)
    served = ((rec["metadata"].get("license") or {}).get("id") or "UNKNOWN").lower()
    expected = str(src.license_name).lower().replace(" ", "-").replace("cc-by-4.0", "cc-by-4.0")
    if "cc-by" not in served:
        raise Blocked(f"Zenodo serves licence {served!r}, registry says "
                      f"{src.license_name!r}; refusing")
    files = rec.get("files", [])
    total_mb = sum(f["size"] for f in files) / 1e6
    est = {"source_id": src.source_id, "n_files": len(files),
           "total_mb": round(total_mb, 1), "note": f"zenodo licence served: {served}"}
    print(f"    Zenodo licence served as {served}; {len(files)} file(s), "
          f"{total_mb:,.1f} MB")
    if dry_run:
        print(f"    dry run: would fetch the archive and extract <= {per_source} images")
        return [], est

    rows = []
    dest = raw_root / src.source_id
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        url = f["links"]["self"]
        name = f["key"]
        if not name.lower().endswith(".zip"):
            print(f"    skipping {name}: only .zip is extracted automatically here")
            continue
        print(f"    fetching {name} ({f['size']/1e6:,.1f} MB)")
        resp = requests.get(url, headers=UA, timeout=TIMEOUT * 10)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            members = [m for m in zf.namelist()
                       if Path(m).suffix.lower() in IMAGE_EXTS and not m.endswith("/")]
            print(f"    archive holds {len(members)} image members; "
                  f"extracting {min(per_source, len(members))}")
            for m in members[:per_source]:
                body = zf.read(m)
                digest = sha256_bytes(body)
                out = dest / f"{digest[:16]}{Path(m).suffix.lower()}"
                if not out.exists():
                    out.write_bytes(body)
                rows.append({
                    "source_id": src.source_id,
                    "original_filename": m,
                    "stored_path": str(out),
                    "source_url": f"https://doi.org/{src.dataset_doi}",
                    "original_image_url": f"{url}#{m}",
                    "license": src.license_name,
                    "license_url": src.license_url,
                    "download_date": time.strftime("%Y-%m-%d"),
                    "bytes": len(body),
                    "sha256": digest,
                })
            est["note"] += f"; {len(members)} image members in archive"
    return rows, est


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int,
                    default=CFG["pilot"]["max_images_per_source"])
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be fetched and estimate size; fetch nothing")
    args = ap.parse_args()

    raw_root = Path(CFG["paths"]["raw"])
    ok = approved_sources()
    if ok.empty:
        print("no licence-verified source: nothing to download")
        return 1

    budget_mb = CFG["pilot"]["max_total_download_mb"]
    spent_mb = 0.0
    rows, estimates = [], []

    for _, src in ok.iterrows():
        sid, doi = src.source_id, src.dataset_doi
        print(f"--- {sid}  {src['name'][:60]}")
        if isinstance(doi, str) and "zenodo." in doi:
            try:
                zrows, zest = zenodo_pilot(src, args.per_source, raw_root, args.dry_run)
            except Exception as exc:
                print(f"    FAILED: {type(exc).__name__}: {exc}")
                print("    recorded as FAILED, not as zero\n")
                estimates.append({"source_id": sid, "n_files": None, "total_mb": None,
                                  "note": f"failed: {type(exc).__name__}"})
                continue
            rows.extend(zrows)
            estimates.append(zest)
            spent_mb += sum(r["bytes"] for r in zrows) / 1e6
            print(f"    extracted {len(zrows)} images\n")
            continue

        if not isinstance(doi, str) or not doi.startswith("10.17632/"):
            print(f"    no automated lister for {doi}; fetch manually and record "
                  f"provenance by hand\n")
            estimates.append({"source_id": sid, "n_files": None, "total_mb": None,
                              "note": "no automated lister"})
            continue
        try:
            files = flatten(mendeley_files(doi))
        except Exception as exc:
            print(f"    LIST FAILED: {type(exc).__name__}: {exc}")
            print("    recorded as FAILED, not as zero\n")
            estimates.append({"source_id": sid, "n_files": None, "total_mb": None,
                              "note": f"list failed: {type(exc).__name__}"})
            continue

        total_mb = sum(f["size"] for f in files) / 1e6
        mean_mb = total_mb / len(files) if files else 0.0
        print(f"    {len(files)} image files, {total_mb:,.0f} MB total, "
              f"{mean_mb:.2f} MB mean")
        estimates.append({"source_id": sid, "n_files": len(files),
                          "total_mb": round(total_mb, 1), "note": ""})

        take = files[:args.per_source]
        if args.dry_run:
            print(f"    dry run: would fetch {len(take)} "
                  f"(~{sum(f['size'] for f in take)/1e6:.1f} MB)\n")
            continue

        dest = raw_root / sid
        dest.mkdir(parents=True, exist_ok=True)
        got = 0
        for f in take:
            if spent_mb > budget_mb:
                print(f"    pilot budget {budget_mb} MB reached; stopping")
                break
            try:
                resp = requests.get(f["url"], headers=UA, timeout=TIMEOUT)
                resp.raise_for_status()
            except Exception as exc:
                print(f"    FAILED {f['filename']}: {type(exc).__name__}")
                continue
            body = resp.content
            digest = sha256_bytes(body)
            out = dest / f"{digest[:16]}{Path(f['filename']).suffix.lower()}"
            if not out.exists():                  # never overwrite
                out.write_bytes(body)
            spent_mb += len(body) / 1e6
            got += 1
            rows.append({
                "source_id": sid,
                "original_filename": f["filename"],
                "stored_path": str(out),
                "source_url": f"https://doi.org/{doi}",
                "original_image_url": f["url"],
                "license": src.license_name,
                "license_url": src.license_url,
                "download_date": time.strftime("%Y-%m-%d"),
                "bytes": len(body),
                "sha256": digest,
            })
            time.sleep(0.2)                       # be a polite client
        print(f"    fetched {got} files, {spent_mb:.1f} MB spent of {budget_mb} MB\n")

    pd.DataFrame(estimates).to_csv(META / "download_size_estimates.csv", index=False)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(META / "pilot_downloads.csv", index=False)
        print(f"wrote {len(df)} rows to dataset/metadata/pilot_downloads.csv")
    print("wrote dataset/metadata/download_size_estimates.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
