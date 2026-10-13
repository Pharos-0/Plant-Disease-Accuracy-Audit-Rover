# Laptop-Camera Experiment — Protocol

**Written 2026-10-13, before any real leaf was photographed.** The decision rule, conditions
and metrics below are fixed. A dry run on edited field photographs (§8) found one design
flaw, which was corrected *before* this protocol was finalised; both runs are recorded.

**Revised 2026-10-14, still before any real leaf was photographed** (§10): the sample is
specified in plants rather than leaves; each diagnosis must be confirmed independently and the
confirmation recorded; intervals resample whole plants; and the primary outcome is now stated as
the paired difference in missed treatment between the two policies on the same frames. The
decision rule and conditions are unchanged.

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
(95% interval 16.7–33.6, resampling photographs, which play the role of plants there). That is the
effect the real session tests on actual captures.
