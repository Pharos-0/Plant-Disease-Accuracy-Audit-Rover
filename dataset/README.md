# Rover-View Tomato Dataset

**This dataset consists of real images obtained from publicly available sources and/or
permitted repositories. Images have not been represented as rover-captured unless they
actually were.**

Status: **BUILT.** 3,994 images retained from 5 sources, 3,875 in verified leakage-free
splits, 10/10 quality gates passing. 3,153 are CC BY 4.0 and redistributable. The
**rover-view core** — WIDE viewpoint and top relevance — is **218 images**, 139 of them
genuinely rover-captured.

One source carries a contested provenance finding: SRC004's advertised 7,200 "field" images
are its augmented folder, and 311 of its 830 raw images are byte-identical to PlantDoc. See
`documentation/PILOT_REPORT.md` §16 before using or redistributing it.

Read `documentation/PILOT_REPORT.md` before using it: §11 records that the rover-captured
source contains only **two capture sessions on one day in one greenhouse**, which is the
binding limitation and is not fixable by adding frames.

## What this is

Real tomato imagery whose *viewpoint and visual conditions* are relevant to a low-mounted
ground-rover camera. No AI-generated images. No synthetic images. Every image keeps its true
provenance.

## The one distinction that governs everything here

| | |
|---|---|
| REAL FIELD IMAGE | ≠ REAL ROVER IMAGE |
| ROVER-VIEW PROXY | ≠ ROVER-CAPTURED DATA |
| PROCESSED REAL IMAGE | ≠ SYNTHETIC IMAGE |
| PUBLICLY ACCESSIBLE | ≠ FREELY REDISTRIBUTABLE |
| MACHINE-GENERATED LABEL | ≠ GROUND TRUTH |

`rover_relevance` is a **proxy score**, 0–3, for how useful a real image is as a stand-in for
what a rover camera might see. It is never a claim about how the image was captured.

Exactly one source in this registry is genuinely rover-captured: **SRC014 AgRobTomato**, in
which a mobile robot was driven along greenhouse inter-rows with a ZED camera.

## Layout

```
dataset/
├── config.yaml              paths, thresholds, pilot budget - nothing hardcoded
├── metadata/                source & licence registries, features, provenance
├── documentation/           rubric, pilot report, dataset card, sample grids
├── splits/                  train/val/test, grouped to prevent leakage
├── annotations/             source labels kept separate from any of our own
└── scripts/
    ├── rover_relevance.py   the measurable rubric
    ├── inventory_local.py   local inventory + rubric calibration
    ├── download_pilot.py    licence-gated downloader
    ├── analyse_pilot.py     features, dedup, sample grids
    └── overlap_test.py      known-answer validation of the dedup pipeline
```

Image bytes live outside this repository, on the dataset volume named in `config.yaml`.

## Reproduce

```bash
python dataset/scripts/inventory_local.py      # inventory + calibrate the rubric
python dataset/scripts/download_pilot.py --dry-run
python dataset/scripts/download_pilot.py --per-source 60
python dataset/scripts/analyse_pilot.py
python dataset/scripts/overlap_test.py         # known-answer dedup validation
python dataset/scripts/build_dataset.py        # full ingest, score, dedup
python dataset/scripts/create_splits.py        # grouped splits + leakage verification
python dataset/scripts/dataset_stats.py        # statistics, plots, sample grids
python dataset/scripts/validate_dataset.py     # the ten quality gates
```

## Licences

Every downloaded image is CC BY 4.0 and carries its licence and source DOI in
`metadata/pilot_downloads.csv`. The downloader refuses any source not marked VERIFIED in
`metadata/license_registry.csv` — the gate is enforced in code, not by convention.

Three datasets already on this machine (PlantVillage, PlantDoc, Tomato-Village) have
**unverified licences** and are marked `ALREADY_LOCAL_DO_NOT_REDISTRIBUTE`. PlantDoc is
web-scraped, so its underlying images carry third-party rights independent of any dataset
licence; it is the highest-risk case in the registry.

## Attribution

CC BY 4.0 requires credit. Cite each source dataset by the DOI recorded in
`metadata/source_registry.csv`, notably:

- **SRC014** AgRobTomato — Magalhães, Moreira, dos Santos, Cunha (2021), `10.5281/zenodo.5596799`
- **SRC015** Greenhouse tomatoes — XiaozhiYang (2023), `10.5281/zenodo.8429227`
