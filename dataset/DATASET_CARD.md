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

## Composition (pilot)

| | |
|---|---|
| Images downloaded | 120 (60 per source) |
| Sources downloaded | 2, both CC BY 4.0 |
| Full archive size of those sources | 1,923 image members → **1,759 distinct** after dedup |
| Synthetic images | **0** — none generated, none accepted |
| Images represented as rover-captured | only SRC014, which genuinely is |

## Sources

| id | name | licence | setting | capture |
|---|---|---|---|---|
| SRC014 | AgRobTomato | CC BY 4.0 | greenhouse, Portugal | **ground robot AgRob v16, ZED camera, driven along inter-rows** |
| SRC015 | Greenhouse tomatoes | CC BY 4.0 | greenhouse, Chengdu, China | handheld mobile phone, in-canopy |

## Known biases and limitations

**Geographic.** Two greenhouses, Portugal and China. No open-field imagery downloaded yet.
No Indian imagery, which is where the rover will operate.

**Environmental.** Greenhouse only. No rain, no strong sun, no dust, no dusk.

**Viewpoint — the most important limitation.** The rubric measures whether a scene continues
past the subject, which is necessary for rover-relevance but **not sufficient**. It cannot
currently distinguish a camera travelling past a plant from a camera held 10 cm from a fruit.
SRC015 scores higher than SRC014 despite SRC014 being the genuinely rover-captured source.
Reported in full in `documentation/PILOT_REPORT.md` §5.2 rather than tuned away.

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

Not yet split. When split, grouping must be by capture session and source, not by individual
image — SRC014 is video-derived, so consecutive frames are near-identical and random
splitting would leak directly.

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
