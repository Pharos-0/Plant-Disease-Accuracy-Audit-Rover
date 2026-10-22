# Laptop-Camera Experiment — Protocol

**Written 2026-10-13, before any real leaf was photographed.** The decision rule, conditions
and metrics below are fixed. A dry run on edited field photographs (§8) found one design
flaw, which was corrected *before* this protocol was finalised; both runs are recorded.

**Revised 2026-10-14, still before any real leaf was photographed** (§10): the sample is
specified in plants rather than leaves; each diagnosis must be confirmed independently and the
confirmation recorded; intervals resample whole plants; and the primary outcome is now stated as
the paired difference in missed treatment between the two policies on the same frames. The
decision rule and conditions are unchanged.

**Addendum 2026-10-20, still before any real leaf was photographed** (§11): the first real
session will be a **single-plant pilot** captured through the **rover's camera**. What it will
and will not report is fixed in §11 now, before any frame exists. The decision rule, thresholds
and conditions are unchanged.

---

## 1. What this experiment can and cannot show

| Claim | Supported? |
|---|---|
| Live camera inference works end to end | **Yes** |
| Lighting, distance, angle, occlusion and background change predictions | **Yes**, measured against true labels |
| Confidence identifies wrong predictions | **Yes**, as a measured AUROC |
| The decision rule never leaves a visible diseased leaf unsprayed | **Yes**, the missed-treatment rate is measured directly |
| Letting the classifier withhold spray misses more diseased leaves than spraying all leaf tissue | **Yes**, as a paired difference on the same frames, with plant-level intervals |
| Disease-recognition accuracy in a real field | **No** — a few leaves on a desk are not a field sample |
| Rover motors, sensors or spray hardware work | **No** |
| Real pesticide savings | **No** |

## 2. What you need

- A laptop with a webcam (or a phone — see §7).
- **Real tomato leaves from several independent plants**: healthy, early blight, late blight.
  Leaves of one plant are not independent samples, any more than frames of one leaf are. **Aim
  for at least 6 diseased plants and 4 healthy plants, 2–3 leaves from each** — about 25 leaves.
  Ten leaves from two plants would be a pilot, not evidence: even zero missed treatments on them
  would say little.
- **An independent confirmation of each diagnosis.** A leaf's appearance alone does not establish
  its disease. Have each plant's diagnosis confirmed by an agronomist, a plant-protection or
  extension officer, or a laboratory test, and record how (`--diagnosis-source`, §5). A plant
  whose diagnosis cannot be confirmed is **left out**, not labelled by best judgement. If only a
  visual diagnosis is available, record `visual only`; the analysis reports the sources, and the
  results must then be described as against visual labels.
- With only healthy plants, false alarms (wasted spray) can be measured but disease detection
  cannot.
- A lamp or phone torch, sheets of white paper and dark cloth, and some soil or plant clutter.

## 3. Conditions — 5 frames each, per leaf

| key | condition | how to set it up |
|---|---|---|
| `1` | **normal** | ordinary room light, leaf filling about half the frame, plain background |
| `2` | **low light** | room lights off, only indirect light from a window or screen |
| `3` | **harsh light** | phone torch close to the leaf, at an angle, casting a hard shadow |
| `4` | **distance / angle** | leaf held twice as far away, and tilted about 45° |
| `5` | **occlusion** | about a third of the leaf covered by a finger or paper |
| `6` | **background** | the leaf on dark cloth, then on soil or among other plants |

Between the 5 frames, move the leaf slightly so they are not identical — but they are still
**one leaf** from **one plant**, and the analysis treats them so.

**Low light is the priority condition**: the dry run's largest difference between the policies
was there (§9). If time is short, record *normal* and *low light* for every leaf first.

## 4. The decision rule (fixed before any real frame)

The demo runs the deployable classifier (MobileNetV3, the model practical on a Raspberry Pi),
and decides per frame:

1. **Is there leaf tissue?** Foliage of any plant colour — green, yellow, orange-brown,
   red-brown — in each of 4 nozzle zones (≥ 15% of the zone).
2. **Is the frame too dark to judge?** If median brightness is below 32/255 (128 divided by
   the tissue test's maximum brightness gain of 4; it was 35 before the v3 revision in §9), the frame is
   **deferred**: no spray, flagged, and counted separately — never silently treated as a
   decision.
3. **Spray, unless confidently healthy.** A visible leaf is left unsprayed only if the session
   gate is open *and* the classifier is confident on this frame (max-softmax ≥ τ) *and* it
   predicts healthy. **Uncertainty never reduces treatment.** τ = 0.888, fixed as the smallest
   threshold at which confident predictions on the source validation split are ≥ 99% accurate.

The session gate is a dataset-level estimate over the last 25 frames. On one leaf held in front
of a camera it demonstrates the mechanism; it is not a meaningful accuracy estimate, because the
frames are not different plants.

## 5. Running it

```bash
python demo/live_sprayer.py --camera 0 --diagnosis-source "agronomist"
```

| key | action |
|---|---|
| `h` / `e` / `l` | set the true label: healthy / early blight / late blight |
| `1`–`6` | set the condition |
| `n` | next leaf **from the same plant** |
| `p` | next **plant** (also starts its first leaf) |
| **SPACE** | save this frame with its labels |
| `q` | quit |

Every saved frame records its plant, leaf, condition, true label, diagnosis source, and **both**
policies' decisions, so the two policies are always compared on exactly the same frames.

**Nothing is saved except on SPACE.** Frames and a log go to `camera_sessions/<date_time>/`.

## 6. Analysis

```bash
python scripts/analyse_camera_session.py camera_sessions/<date_time>
```

**Primary outcome.** Per condition, the share of diseased frames missed by the
**classifier-gated** policy and by the **spatial-only** policy, and their **difference** (gated
minus spatial-only) on the same frames, with a 95% interval (`policy_comparison.csv`). The
claim to be tested is that the difference is above zero in low light.

Secondary, per condition: accuracy, confidence, abstention, spray rate, and three outcomes for
every diseased leaf — **sprayed**, **missed** (the system decided not to spray), **deferred** (it
refused to decide). Healthy leaves give the **wasted-spray** rate. Pooled: confidence as an
error detector (AUROC). Paired: each condition against *normal* for the same leaf.

**Every interval is a cluster bootstrap over plants** when every frame carries a plant
identifier, and over leaves otherwise; the analysis prints which. Frames of one leaf, and
leaves of one plant, are not independent samples.

**Safety criterion: missed treatment = 0.** Deferred leaves are reported beside it, and
"untreated" = missed + deferred.

## 7. Using a phone instead

Take the photos with a phone, copy them to the laptop, and list them in a CSV:

```
path,truth,plant_id,leaf_id,condition,diagnosis_source
C:/photos/IMG_001.jpg,healthy,P01,P01L01,normal,agronomist
C:/photos/IMG_002.jpg,healthy,P01,P01L01,low_light,agronomist
C:/photos/IMG_003.jpg,late_blight,P02,P02L01,low_light,lab test
```

`plant_id` and `diagnosis_source` are optional columns, but without `plant_id` on every row the
intervals fall back to resampling leaves.

```bash
python demo/live_sprayer.py --manifest photos.csv --out camera_sessions/phone_run
```

## 8. Dry run — what it found, and the fix

Before any real leaf, the whole pipeline was run on **edited real field photographs**: 360
crops from 80 Tomato-Village photographs (Rajasthan, India) with true labels, each dimmed,
over-exposed, occluded, and downscaled-and-rotated (`scripts/make_perturbation_manifest.py`).
These are edited photos, not camera captures, and they preview behaviour rather than measure
it.

**Version 1 — green-only leaf detection — failed the safety criterion.** 24% of diseased
leaves were missed in normal light and 64% in low light. 499 of the 558 misses came from the
plant-detection step, not the classifier: **late-blight leaves average only 0.20 green, and
only 62% of them showed any zone ≥ 15% green.** A green test is blind to the most severe
disease — the sicker the leaf, the less likely to be sprayed. The sprayer simulations could not
reveal this, because their disease labels were borrowed from another dataset and did not change
a plant's colour.

**Fixed before this protocol was finalised**: leaf detection now accepts any plant colour, and
too-dark frames are deferred. The version-2 results are in §9.

The dry run also showed the thesis finding live: **under low light the classifier became more
confident (+0.136) while becoming less accurate (−9.7 points).**

## 9. Dry run, version 2

**Version 2 — any plant colour + too-dark deferral.** Missed treatment fell sharply in every
condition except low light (normal 24.1% → 2.8%; harsh 22.8% → 3.4%; distance/angle
26.9% → 7.5%; occlusion 36.6% → 6.9%), but **low light still missed 46.2%**, from two causes:

- **76 frames:** the tissue test needed V ≥ 25, so a dimmed leaf vanished while the frame
  (median brightness 56) was not dark enough to be deferred — the two thresholds disagreed.
- **84 frames:** dimming made the classifier **more confident**, the label-free gate opened,
  and confident-but-wrong *healthy* calls withheld treatment from diseased leaves. **The gate
  was fooled by exactly the condition that broke the classifier** — the thesis finding, live.

**Version 3 — exposure-invariant tissue test, and a second policy logged beside the first.**
The tissue test now judges a brightness-normalised image (gain capped at 4×; anything darker
is deferred). And every frame records two decisions: **gated** (the classifier may withhold
spray) and **spatial-only** (spray all detected leaf tissue; a classifier call never withholds).

| condition | gated: missed | gated: wasted | **spatial-only: missed** | spatial-only: wasted |
|---|---:|---:|---:|---:|
| normal | 3.8% | 100% | **1.9%** | 100% |
| **low light** | **26.6%** | 90% | **1.6%** | 100% |
| harsh light | 3.4% | 100% | 3.4% | 100% |
| distance / angle | 7.5% | 100% | **0.9%** | 100% |
| occlusion | 6.6% | 85% | 6.2% | 85% |

**Letting the disease classifier withhold spray cost safety and saved almost nothing.** In low
light the gated policy withheld treatment from 26.6% of *diseased* leaves but only 10% of
*healthy* ones — worse than chance. Confidence did not identify errors (AUROC 0.512, interval
0.477–0.547). On these images the classifier almost never calls a leaf healthy, so healthy
leaves are sprayed under either policy; whether that holds for real leaves under a laptop
camera is one of the things the real session will show.

**This was the last revision made on the dry run.** The edited photographs were used to find
and fix design faults; they are not the test. The real camera session is, and it will report
both policies.

### What the dry run implies for the design

Safe savings come from **where** to spray — leaf versus soil versus weed — not from **whether**
the classifier believes a plant is diseased. That is consistent with every other result in the
project: label-free estimators cannot verify the classifier under field shift (the thesis
research), confidence barely flags errors in the field (AUROC 0.57–0.67 across target
conditions), and the 62.5% open-field saving came almost entirely from spatial targeting.

## 10. Revision of 2026-10-14 — before any real leaf

Made after an external review of Paper 2 and before any real frame was recorded. Nothing in the
decision rule, thresholds or conditions changed.

| What | Before | After | Why |
|---|---|---|---|
| Sample | ≥ 10 leaves (5 healthy, 5 diseased) | ≥ 6 diseased and ≥ 4 healthy **plants**, 2–3 leaves each | leaves of one plant are not independent; ten leaves is a pilot |
| Labels | best judgement, noted | independent confirmation (agronomist, extension officer or lab), recorded per frame; unconfirmed plants left out | appearance alone does not establish a diagnosis |
| Resampling unit | leaf | plant (leaf if any frame lacks a plant id) | uncertainty at the level of independent units |
| Primary outcome | missed treatment per policy | **paired difference**, gated minus spatial-only, same frames | the claim is about the policies, so compare them directly |

Re-run on the dry-run data, the new analysis reproduces §9 and adds the paired difference: in low
light the gated policy missed **25.0 percentage points** more diseased frames than spatial-only
(95% interval 16.7–33.6, resampling the 40 diseased photographs). Tomato-Village does not record
which plant a photograph came from, so the photographs are not known to be independent plants:
if several share a plant, that interval is too narrow. It is a dry-run estimate on edited
images, not biological replication. That is the
effect the real session tests on actual captures.

## 11. Addendum of 2026-10-20 — the first session is a single-plant pilot

Written the night before the first real session, **before any real leaf was photographed**.
Nothing in §4 (decision rule, τ, thresholds) or §3 (conditions) changes.

### 11.1 Why a pilot, stated in advance

The first available material is **one plant, bought from a shop, very likely healthy, with a
visual diagnosis only**. Against §2 and §10 (≥ 6 diseased and ≥ 4 healthy plants, independently
confirmed) that is a pilot, and two consequences follow and are accepted now:

- **The primary outcome cannot be estimated.** The gated-minus-spatial difference in missed
  treatment (§6, §10) is defined on *diseased* frames. A healthy plant has none. The session will
  not report it, and the analysis prints `PRIMARY OUTCOME NOT ESTIMABLE`.
- **No interval is estimable.** One plant is one resampling unit; there is no between-plant
  variation, so every interval is reported as n/a, never as a narrow interval.

### 11.2 What the pilot will report — fixed now

| # | Quantity | Measurable on one healthy plant because |
|---|---|---|
| P1 | The capture chain runs end to end on a real plant through the **rover camera** (frames logged, both policies recorded per frame) | it is a property of the software and camera, not of disease |
| P2 | **False alarms on real healthy leaves**, per condition: share of frames predicted diseased, share abstained, and **wasted spray** under the gated and the spatial-only policy | the truth (healthy) is what the plant provides; this answers the open question in §9 — whether the classifier "almost never calls a leaf healthy" holds for real leaves |
| P3 | **Tissue detection on real foliage**, per condition: share of leaf-in-view frames on which the spatial-only policy fires, and the deferral (too-dark) rate | the spatial decision never consults the classifier, so its tissue detection can be measured on any leaf. **Caveat, fixed now:** a healthy leaf is green; this is not a measurement on brown, diseased tissue and does not substitute for missed treatment on diseased leaves |
| P4 | **Confidence under shift on real leaves**: mean max-softmax per condition, and the paired low-light-minus-normal change on the same leaf (direction and size only, no interval) | it is the dry run's live observation (§8: more confident, less accurate in low light), checked on a real capture |
| P5 | The **§16 artefacts**: one saved rover-camera frame of the real plant, and the camera lens height above the ground **measured with a tape** (with the lens-to-plant distance beside it) | required by `docs/rover_hardware_test_report.md` §3 |
| P6 | The **rover's on-board AUTO** on the real plant: fires on the plant, defers in darkness, relay actuation observed by the operator (tank empty, or dyed water only) | the on-board decision is the spatial-only policy itself (`rover/spatial_policy.py`, checked frame for frame against this demo) |

`scripts/analyse_camera_session.py` prints P2–P4 per condition under **PILOT QUANTITIES** and
writes them to `pilot_report.csv` whenever a session is below the §2 minimum; P5 comes from
`scripts/capture_rover_artifact.py`, P6 from the operator's observation, recorded verbatim.

**How it is described.** Every number from this session is reported as a **pilot on one
plant**, with its diagnosis source recorded exactly (`--diagnosis-source "visual only (nursery
stock)"`, or the confirming authority if one is obtained). It is not evidence for or against the
primary claim, and the thesis and Paper 2 say so beside it.

**Inclusion later.** The plant may join the full sample (§2) only if its diagnosis is later
confirmed independently and that confirmation recorded; its frames are then analysed together
with the others, plant-level resampling unchanged.

### 11.3 Procedure

Frames are captured through the rover's camera with the unchanged decision rule:

```bash
.venv/Scripts/python.exe demo/live_sprayer.py --camera 192.168.137.167:8080 --diagnosis-source "visual only (nursery stock)"
```

*(Added 2026-10-21, at the session.)* The project's OpenCV has no window support, so on the
laptop used the capture runs as a browser page instead — the same pipeline, overlay and log
record, through `demo/web_capture.py`: add `--web 8090` and open `http://127.0.0.1:8090/`. The
page also refuses a second diagnosis for one leaf and asks before a new plant, after the first
capture showed both errors. The results are in `docs/pilot_results_20261021.md`.

Keys, conditions and frame counts are those of §3 and §5: **normal and low light first** (the
priority conditions), 5 frames per leaf per condition, 2–3 leaves, the rest of the conditions if
time allows. Each record carries the camera source, including whether the clean `/raw` frame or
a cropped `/snapshot` was used (the rover servers draw a status banner that must not reach the
classifier; see `demo/rover_camera.py`).

### 11.4 Software changes made before this session (none touches §3 or §4)

| Change | Why | Effect on the dry-run results |
|---|---|---|
| Analysis reports **n/a**, not 0%, for a quantity with no frames to judge; intervals from a single unit are n/a; a session below the §2 minimum is labelled **PILOT**; the verdict is written to `analysis_status.json` | a healthy-only session printed "missed treatment 0.0%" — the safety criterion appearing met when it was never measured — and one plant printed zero-width intervals | none: re-run on `dryrun_v3`, stdout and both CSVs are byte-identical |
| When no diseased frame is missed, an **exact 95% upper bound** (Clopper–Pearson at zero) on the share of diseased plants with any miss is printed beside the pre-registered bootstrap, labelled supplementary | the percentile bootstrap cannot exceed 0 when no miss occurs, so it prints [0%, 0%]; with 6 diseased plants and no misses the exact bound is still 39% | none: no zero-miss condition in the dry run |
| `demo/live_sprayer.py --camera` accepts the **rover camera** and reads it banner-free | the protocol could only use a laptop webcam; the rover servers stamp a banner (black, with text that can be green) on `/snapshot` | none: the decision rule is untouched |
| The rover's **AUTO** now runs the spatial-only policy of §4 exactly (any of 4 zones ≥ 15% tissue; too-dark frames deferred) | it used a whole-frame 15% test with no deferral: it missed a plant filling one zone, and would have sprayed a frame too dark to judge | brings the robot into line with §4 |
| *(2026-10-21, at the start of the session, still before any real leaf frame)* A **stuck-sensor frame is a camera fault and is deferred**, beside §4's blank-frame check: grey variation within rows (pooled) < 0.5 **and** between row means > 10 | on the rover the camera's sensor stuck and delivered static uniform-row stripes — s.d. ≈ 80, so not "blank" — which the tissue test read as 86% leaf: AUTO would have sprayed on a frozen image. A power-cycle of the Pi cleared it; a USB reset did not | none: of the 1,800 dry-run frames, none trips it (tested). A first version, using the *median* row, would have deferred 90 real harsh-light leaves on white and was rejected before use |
| *(2026-10-21, after the pilot)* A frame **missing a colour channel** is also a camera fault: brightest channel's 99th percentile > 100 and the weakest < ¼ of it | after another reopen the camera lost its blue channel (a white panel read 4/154/122 in BGR) and the tissue test called 97% of the frame leaf; a power-cycle cleared it | none: 0 of the 1,800 dry-run frames and 0 of the 32 pilot frames trip it (tested); weakest/brightest is 0.108 on the broken frames and ≥ 0.529 on every real frame bright enough to matter |

All are covered by tests (`tests/test_camera_analysis.py`, `test_rover_camera.py`,
`test_spatial_policy.py`, `test_rover_control_server.py`).

## 12. Prospective power of the planned session — a simulation (2026-10-23)

**Not a result of the session, which has not been run.** No plant was photographed for this and
no diagnosis is assumed for any real plant. `scripts/simulate_ten_plant_session.py` assembled
1,000 simulated sessions of exactly the §2/§10 size (6 diseased + 4 healthy plants, 2–3 leaves,
≤ 5 frames per leaf per condition) from the dry run's real, dataset-labelled photographs and the
decisions already logged for them (`dryrun_v3`), and ran the pre-registered analysis
(`policy_comparison`, plant-level intervals) on each. The photographs carry no plant identity, so
two bounds: a plant's leaves are different photographs (optimistic) or one photograph repeated
(pessimistic).

| Low light (the pre-registered claim) | Median gated − spatial | 95% of sessions | Power: interval above 0 | Median interval width |
|---|---:|---|---:|---:|
| independent leaves | +25.0 points | 14.3 – 37.3 | **100%** | 26 points |
| identical leaves | +24.0 points | 6.7 – 48.0 | **92%** | 42 points |

Other conditions: distance/angle +5–6 points (power 34–70%); normal, harsh light and occlusion
≈ 0 (no policy difference to detect). **Reading:** if real captures behave like the dry run, the
planned session is adequately powered for its primary claim. If they do not, that is what the
session is for. Results: `results/simulation_ten_plant/`.
