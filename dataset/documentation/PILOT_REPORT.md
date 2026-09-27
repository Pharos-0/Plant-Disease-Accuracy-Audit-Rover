# Pilot Report — Rover-View Tomato Dataset

**2026-09-28.** Steps 1–7 of the build plan: local inventory, rubric calibration, source
discovery, licence verification, and a 120-image pilot from two licence-clear sources.
**No bulk download has been performed.** This report is the input to that decision.

---

## 1. What was already on this machine

| source | tomato images on disk | measured `border_scene_frac` | rover-relevant (score 2–3) |
|---|---:|---:|---:|
| PlantVillage (tomato) | 18,160 | 0.151 | 14% |
| Tomato-Village (Variant-a) | 4,525 | 0.028 | 2% |
| **PlantDoc (tomato)** | **746** | **0.653** | **86%** |
| total | 23,431 | | |

**The headline is the imbalance.** 23,431 tomato images were already here, and only 746 —
**3.2%** — are in-the-wild imagery. The other 97% are single specimens on a backdrop. That is
the gap this dataset exists to close, and it is now measured rather than asserted.

## 2. The rubric, and the fact that it failed first

`rover_relevance.py` scores 0–3 on one question: **does the scene continue past the
subject?** A rover travelling a crop row sees a plant embedded in surroundings that run to
the frame edge. A specimen photograph does not.

The first version tested for a backdrop by intensity uniformity and **failed calibration**.
Inspecting the images showed why: PlantVillage leaves sit on a **coarse speckled grey
surface**, so its border intensity spread is high and its Canny edge density is the *highest*
of the three datasets — the detector fires on background grit. Smoothness was the wrong test.
What grey, white and black backdrops share is that they are **achromatic**.

A second apparent failure was not one. PlantDoc appeared to fail because only 55% scored 2–3
— but PlantDoc is web-scraped and genuinely mixed, containing field shots with straw and soil
*alongside* potted seedlings against black cloth. The expectation was wrong, not the
measurement. The rubric is therefore calibrated on **separation and ordering**, never on one
label per source, because a per-source label invites fitting thresholds until they pass.

A third feature was deleted. `green_component_count` was meant to count leaves, but
contiguous canopy foliage forms **one** connected component, exactly as an isolated leaf
does. It measured nothing. It is still recorded so the claim can be re-checked, but it takes
no part in scoring.

**After the fix, all four calibration criteria pass:** 86% and 98% of the two specimen
protocols score 0–1, 86% of the in-the-wild source scores 2–3, and the three sources order
correctly by mean score without the rubric seeing any dataset label.

## 3. Sources discovered

18 registered in `metadata/source_registry.csv`. Licence status in
`metadata/license_registry.csv`.

| status | n | |
|---|---:|---|
| licence VERIFIED, download permitted | 4 | SRC004, SRC005, SRC014, SRC015 |
| licence VERIFIED, deferred on size | 2 | SRC016 (2.4 GB .rar), SRC017 (10.3 GB) |
| licence VERIFIED, rejected on content | 1 | SRC018 — sensor CSV, no imagery |
| already local, licence never verified | 3 | SRC001–003 — usable locally, **not redistributable** |
| **licence UNKNOWN, blocked** | 1 | SRC007 BUTom21 — see §6 |
| candidate, not yet verified | 7 | SRC008–013, SRC006 partially |

### The best source found

**SRC014 AgRobTomato** (CC BY 4.0, `10.5281/zenodo.5596799`) is *actual rover-captured data*:
> "Mobile robot AgRob v16, controlled by a human operator, was guided through the greenhouse
> **inter-rows** and captured RGB images of the tomato plants using a ZED camera."

A ground robot, driving crop rows, with a camera. That is the target viewpoint, not a proxy
for it. 449 images, Barroselas, Portugal, 6 and 8 August 2020.

## 4. Pilot result — 120 images, 2 sources

| source | n | score 2–3 | mean score | `border_scene_frac` |
|---|---:|---:|---:|---:|
| SRC014 AgRobTomato | 60 | 97% | 2.57 | 0.451 |
| SRC015 Greenhouse tomatoes (Chengdu) | 60 | 95% | 2.75 | 0.653 |

Both are overwhelmingly rover-relevant by the rubric. Archive totals: **449** and **1,474**
image members.

## 5. Two findings the pilot produced that the plan did not anticipate

### 5.1 The dedup pipeline passes a known-answer test — and the overlap is large

SRC015's authors state part of their dataset was taken from SRC014. Hashing **every** member
of both archives (1,923 images, not the biased first-60 slice):

- **170 cross-source pairs**, **164 distinct images on each side**
- **all 164 at Hamming distance 0** — exact perceptual matches
- that is **36.5% of SRC014** reappearing inside SRC015, and **11.1% of SRC015**

Naively merging the two sources would have double-counted 164 images and, worse, scattered
identical images across train and test. Distinct contribution is **449 + 1,310 = 1,759**, not
1,923.

### 5.2 The rubric has a real blind spot: it does not measure camera distance

Look at the sample grid. SRC014's frames are wide, low, looking **along** the inter-row, with
whole plants, support wires and ground — the travelling-rover view. SRC015's are handheld
phone close-ups **inside** the canopy, some nearly macro on a single fruit.

Yet SRC015 scores *higher* (2.75 vs 2.57), because a macro shot deep in the canopy has
vegetation at every border, which is exactly what `border_scene_frac` rewards.

**"Embedded in a scene" is necessary for rover-relevance but not sufficient.** A distance or
object-scale feature is missing, and the rubric currently cannot separate "camera travelling
past a plant" from "camera held 10 cm from a fruit". This is a genuine limitation of the
current rubric and is **not** fixed by tuning a threshold.

Candidate additions, none yet implemented: dominant object scale relative to the frame;
vanishing-structure or row-line detection; depth-of-field proxy via a sharpness gradient from
centre to border; horizon or ground-plane presence.

## 6. Stop conditions triggered

Three, per §27 of the build plan.

**Mendeley returns HTTP 403 on every documented API endpoint** (`api.data.mendeley.com`,
`data.mendeley.com/api/...`, `public-api/...`), for SRC004 and SRC005 — both CC BY 4.0 and
both high-value. This is a bot wall, not a licence restriction, and **it was not worked
around**. These two sources need a manual browser download, which is entirely legitimate for
a CC BY 4.0 dataset; the files can then be registered with `download_pilot.py`'s provenance
schema by hand.

**SRC007 BUTom21 has no licence at all.** No LICENSE file and no licence statement in the
repository. It is the most on-target source found after AgRobTomato — a Bonn agricultural
robotics dataset — but publicly downloadable is not freely redistributable, so it is blocked
pending the Dataverse terms. Note also that BUTom-ST21's annotations are partly **machine
generated** (Mask2Former / YOLO) and must never be treated as ground truth.

**Two verified sources are deferred on size**: SRC016 at 2.4 GB in `.rar` (needs an
extractor), SRC017 at 10.3 GB.

## 7. Storage

320 MB used so far (two cached archives plus 120 extracted images). Disk has 197 GB free on
C: and 139 GB on D:, so storage does not constrain any decision here.

| if we take | images | download |
|---|---:|---:|
| SRC014 + SRC015, fully, deduplicated | 1,759 | 312 MB (already cached) |
| + SRC016 | +600 | +2.4 GB |
| + SRC017 | unknown | +10.3 GB |
| + SRC004 + SRC005 (manual) | +8,228 | unknown |

## 8. Honest answer to "does this approximate a rover viewpoint?"

**For SRC014, yes — because it is not an approximation.** It is a ground robot driving
greenhouse inter-rows. That is 449 images.

**For SRC015, partly.** Same environment and clutter, but handheld and much closer than a
travelling rover would be.

**Everything else found so far is a proxy at best**, and the 23,431 images already on disk
are 97% specimen photography.

The honest position is that the rover-view corpus currently stands at roughly **449 genuinely
rover-captured + 746 in-the-wild PlantDoc + 1,310 in-canopy handheld**, and that scaling past
that depends on resolving the three stop conditions in §6.
