# Dataset Card — Rover-View Tomato Dataset (PILOT)

**Version:** pilot, 2026-09-28 · **Status:** not complete, not released

## Motivation

A prototype ground rover carries a low-mounted USB camera through a tomato crop. Public
tomato imagery is dominated by specimen photography — a single leaf on a backdrop — which
does not resemble what such a camera sees. This dataset assembles *real* imagery whose
viewpoint and visual conditions are relevant to that platform, with provenance and licence
preserved for every image.

The need is measured, not assumed: of 23,431 tomato images already held, **3.2%** are
in-the-wild. See `documentation/PILOT_REPORT.md` §1.

## Intended use

Evaluating computer vision under a ground-rover viewpoint, and studying the visual gap
between specimen imagery and deployment imagery.

**Not intended for:** any claim that a model validated here is safe to act on in a field. No
autonomous action should be gated on this dataset.

## Composition (built 2026-09-28)

| | |
|---|---:|
| Images ingested | 2,669 |
| Duplicates dropped (perceptual hash) | 200 |
| **Retained** | **2,469** |
| Guard-band frames (retained, excluded from splits) | 119 |
| In train/val/test | 2,350 |
| Redistributable (CC BY 4.0) | 1,628 |
| Local-only (licence unverified) | 722 |
| Synthetic images | **0** — none generated, none accepted |
| **Rover-view core** (WIDE viewpoint, relevance 3) | **180** |

| source | images | share | WIDE | mean relevance |
|---|---:|---:|---:|---:|
| SRC015 handheld, greenhouse | 1,301 | 55.4% | 28 | 2.22 |
| SRC002 PlantDoc (local only) | 722 | 30.7% | 25 | 1.89 |
| **SRC014 rover-captured** | **327** | **13.9%** | **307** | **2.36** |

The largest source is 55.4% of the dataset; the genuinely rover-captured source is 13.9%
while supplying 307 of the 360 WIDE images. Size and value point in opposite directions, and
no source is allowed to stand for the dataset on the strength of its size.

## Sources

| id | name | licence | setting | capture |
|---|---|---|---|---|
| SRC014 | AgRobTomato | CC BY 4.0 | greenhouse, Portugal | **ground robot AgRob v16, ZED camera, driven along inter-rows** |
| SRC015 | Greenhouse tomatoes | CC BY 4.0 | greenhouse, Chengdu, China | handheld mobile phone, in-canopy |

## Known biases and limitations

**Geographic.** Two greenhouses, Portugal and China. No open-field imagery downloaded yet.
No Indian imagery, which is where the rover will operate.

**Environmental.** Greenhouse only. No rain, no strong sun, no dust, no dusk.

**Viewpoint.** Addressed since the pilot. `half_detail_scale`, a granulometry-based structure
scale, now separates a travelling camera from a close-up: SRC014 measures 0.125, SRC015 0.482.
Folding it in moved SRC015's mean score from 2.75 to 2.02 while leaving SRC014 at 2.57, which
is the error it was built to correct. It is consulted only where a scene is already present,
because for a specimen on a backdrop the distance question is meaningless.

**Capture sessions, not image count, is the binding constraint.** See the split section.

**A person appears in at least one image** (TOM_000321, the robot's human operator). These are
the source authors' own CC BY 4.0 release; no face-obscuring has been applied, and anyone
redistributing should consider whether that is appropriate for their use.

**Class balance.** Neither downloaded source is a leaf-disease dataset; both are fruit /
ripeness oriented. The disease-relevant sources (SRC004, SRC005) are blocked by a bot wall.

**Source imbalance.** SRC015 is 3.3× the size of SRC014. No source may be allowed to dominate
by size alone; any final composition must state its balance explicitly.

**Overlap.** 164 images appear in both sources at Hamming 0. Any merge must deduplicate.

## Annotations

| source | annotation | provenance |
|---|---|---|
| SRC014 | Pascal VOC object detection | **SOURCE_LABELS** — from the dataset authors |
| SRC015 | not yet inspected | UNKNOWN |
| all | `rover_relevance` 0–3 | **OUR_LABELS** — machine-computed, rule-based, not human-verified |

`rover_relevance` is machine-generated and is **not ground truth**. It has been calibrated
against three datasets whose acquisition protocol was established by direct visual
inspection, and it fails in the documented way above.

No labels have been invented. No automated model has been used to pre-annotate anything.

## Preprocessing

None applied to `raw/`. Raw images are stored byte-identical to the archive member, named by
their SHA-256, and are never overwritten. Any rover-view proxy transformation
(resize to 320×240, mild motion blur, compression) would be written to `processed/` and
recorded per image in `processing_applied`, with the link to the original preserved.

**No transformed image is ever described as rover-captured.**

## Train/test methodology

Split by **789 leakage groups**, derived from what each source's filenames encode rather than
assumed: contiguous 40-frame temporal blocks with a 5-frame guard band for the video-derived
SRC014; consecutive-index chunks for SRC015, whose index order is a stated heuristic because
no session metadata exists; one group per image for PlantDoc's unrelated web stills.

**Verified, not assumed.** Every train×test, val×test and train×val pair was compared by
perceptual hash: **0 violations**.

**The limit this exposes.** SRC014 contains only **two capture sessions, both on 2020-08-06,
in one greenhouse**. Its blocks are segments of two traverses, not independent samples. The
split prevents frame-level leakage; it cannot manufacture independence. Any result on the
rover-view core rests on effective n = 2 sessions.

## Ethical and legal

All downloaded images are CC BY 4.0, permitting redistribution with attribution. Three
locally held datasets have unverified licences and are excluded from any redistribution.
Sources with unknown licences are blocked in code, not by convention. No access control,
paywall or bot protection was bypassed; where one was encountered, the source was recorded as
blocked.

## Reproducibility

Every step is a script in `dataset/scripts/`, driven by `dataset/config.yaml`. The dedup
pipeline is validated against a known answer (the declared SRC014→SRC015 reuse) rather than
assumed correct.
