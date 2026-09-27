# Pilot Report — Rover-View Tomato Dataset

**2026-09-28.** Part I (§1–8) covers steps 1–7 of the build plan: local inventory, rubric
calibration, source discovery, licence verification, and a 120-image pilot. Part II (§9–13),
written after the pilot was reviewed, covers the camera-distance feature and the scale-up to
**2,469 retained images**.

Part I is left as written, including the limitation in §5.2 that Part II then fixes, because
the sequence is the point: the pilot existed to find that limitation before it was scaled.

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

---

# Part II — Distance Feature and Scale-Up (2026-09-28)

## 9. The distance feature

§5.2 recorded a blind spot: the rubric could not tell a camera travelling past a plant from
one held 10 cm from a fruit. It is now measured, by **morphological granulometry**.

`half_detail_scale` is the smallest opening radius capturing half the image's residual
detail, `r(k) = mean |g − opening(g, k)|`, normalised. Fine repeated canopy texture saturates
at small radii; one large smooth fruit does not.

Measured **before** it was wired into any decision:

| source | `half_detail_scale` | character |
|---|---:|---|
| SRC014 rover-captured, inter-row | **0.125** | finest texture, widest view |
| PlantVillage specimen | 0.269 | |
| PlantDoc mixed | 0.377 | |
| Tomato-Village specimen | 0.426 | |
| SRC015 handheld in-canopy | **0.482** | coarsest, closest |

Two corrections were needed to get there.

**`_standardise` only ever downscaled**, so a 48-pixel structuring element covered most of a
256 px PlantVillage image but a corner of a 4000 px photograph — the feature was measuring
file size, not viewpoint. It now resizes to a fixed long side in both directions. Spearman
correlation between the feature and stored megapixels is **−0.131**.

**The feature is consulted only where a scene is present.** For a specimen on a backdrop the
distance question is moot, and the measurement is confounded — PlantVillage's speckled grey
stone is genuinely fine texture, for reasons having nothing to do with camera distance.

### It corrected the exact error it was built for

| | before | after |
|---|---:|---:|
| SRC014 (rover-captured) mean score | 2.57 | **2.57** |
| SRC015 (handheld close-ups) mean score | 2.75 | **2.02** |

SRC015's score-3 count fell from 49 to 5. Calibration against the three local sources still
passes on all four criteria.

## 10. The scaled dataset

| | |
|---|---:|
| Images ingested | 2,669 |
| Duplicates dropped | 200 |
| **Retained** | **2,469** |
| Guard-band frames (retained, excluded from splits) | 119 |
| In splits | 2,350 |

| source | images | share | leakage groups | WIDE | mean relevance |
|---|---:|---:|---:|---:|---:|
| SRC015 handheld | 1,301 | 55.4% | 53 | 28 | 2.22 |
| SRC002 PlantDoc (local only) | 722 | 30.7% | 722 | 25 | 1.89 |
| **SRC014 rover-captured** | **327** | **13.9%** | **14** | **307** | **2.36** |

**Source imbalance, stated rather than smoothed:** the largest source is 55.4% of the
dataset, but the genuinely rover-captured source is only 13.9% — while contributing **307 of
the 360 WIDE images**. Size and value point in opposite directions here.

**The rover-view core is 180 images** (WIDE viewpoint and relevance 3): 139 from SRC014, 27
from SRC015, 14 from PlantDoc.

Licences: **1,628 of 2,350 are CC BY 4.0 and redistributable**; the 722 PlantDoc images are
usable locally only.

## 11. Leakage control, and the limit it exposes

Grouping is derived from what the filenames encode, not assumed:

- **SRC014** — `tomate_barroselas_20200806_0000.jpg`. Contiguous temporal blocks of 40
  frames, with a **5-frame guard band dropped at every boundary** so no two frames straddling
  a split share content.
- **SRC015** — `image/1.jpg`…`image/1474.jpg`. No session metadata exists; the index is
  *assumed* to be acquisition order, a heuristic stated as such, and chunked in 25s.
- **SRC002** — unrelated web stills, one group each.

789 groups. **Verified, not assumed**: every train×test, val×test and train×val pair was
compared by perceptual hash — **0 violations**.

### The hard limit

**SRC014 contains only two capture sessions, both on 2020-08-06, in one greenhouse.** Its
temporal blocks are segments of two traverses, not independent samples. The split prevents
frame-level leakage; it cannot manufacture independence. Any result computed on the
rover-view core rests on **two traverses of one greenhouse on one day** — effective n = 2, in
the same way the thesis's 1,616 crops were really 391 photographs.

This is the single most important limitation of the dataset and it is not fixable by
collecting more of the same source.

## 12. Quality gates

**10 pass, 0 fail, 0 unevaluable.** Gate 5 became evaluable once splits existed and passes on
verification rather than on the grouping looking sensible.

## 13. What would actually improve this

1. **More capture sessions**, not more frames. A second greenhouse, a second day, a second
   platform would do more than ten thousand extra frames of the same two traverses.
2. **Open-field imagery.** Everything rover-captured here is greenhouse.
3. **Disease-labelled rover-view imagery.** The two blocked Mendeley sources are the
   disease-relevant ones; the rover-view sources are fruit/ripeness oriented.
4. Resolve SRC007 BUTom21's licence — agricultural-robotics provenance, currently blocked.
