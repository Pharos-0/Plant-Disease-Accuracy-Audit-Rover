# Open-Field Sprayer Experiment — Pre-Registered Protocol

**Written and committed 2026-10-13, BEFORE the data was downloaded.** Every choice below is
fixed. If the run requires a change, the change is recorded in §9 with its reason, and the
result under the original protocol is reported alongside it.

Why pre-register: the request for this experiment asked for a drastic reduction in pesticide
use. A protocol written after seeing the data could be tuned to deliver that. This one cannot.

---

## 1. Data

**GROUNDBASED_WEED** — Moreno, Rivera & Andújar, *Data in Brief* 2025,
`10.1016/j.dib.2025.112249`; deposit `10.20350/digitalCSIC/17475`, CC BY 4.0. Open-field
tomato, Santa Amalia, Badajoz, Spain, May–June 2021/2022. Handheld camera **1–1.5 m above
the canopy, zenithal** — the viewpoint of a downward spray boom. Early growth stage.

Classes (EPPO codes): `LYPES` = tomato crop; `SOLNI`, `POROL`, `SETVE`, `ECHCG`, `CYPRO` =
weeds; `NR` = not recognised.

**Only `TOMATO_1` (2021) is used.** It holds all 1,075 tomato boxes; `TOMATO_2` has none, so a
crop-targeted sprayer cannot be evaluated on it.

## 2. Split

`TOMATO_1` images in filename order, cut into consecutive blocks of 20 images. Blocks are
assigned 70% train / 30% test with seed 20261013, so that overlapping consecutive handheld
shots stay on one side. The split is then verified by perceptual hash; any cross-split pair
within Hamming 6 is reported.

## 3. Spray grid

Each image is divided into a **16 × 12 grid of sub-cells**. Nozzle zones are formed by
OR-aggregating sub-cells: a nozzle fires if any sub-cell in its zone is targeted.

| layout | zones per image | primary? |
|---|---|---|
| **4 × 3** | 12 | **yes — 4 nozzles, matching the rover** |
| 8 × 6 | 48 | sensitivity |
| 16 × 12 | 192 | sensitivity |

Finer layouts save more simply by being finer, so all three are reported and the primary is
the coarsest, which is the conservative choice.

## 4. Ground truth per sub-cell

- **crop**: a `LYPES` box intersects the sub-cell over ≥ 5% of the sub-cell area, or the
  box centre lies inside it.
- **weed**: the same rule for any weed box.
- **bare**: neither.

## 5. Disease state

The dataset has no disease labels. Each tomato box is paired with a randomly drawn plant from
**Tomato-Village (Jodhpur and Jaipur, Rajasthan, India)**, taking that plant's true label and
the real cached output of one of the twenty frozen classifiers. A crop cell **needs spray** if
it contains a tomato whose paired plant is truly diseased. False-positive cells (detector fires,
no tomato) receive a randomly drawn classifier output. Ten random pairings per classifier.

This composition is a modelling assumption — disease prevalence independent of plant position —
and is stated on every artefact.

## 6. Policies — all through the same FSM controller

| policy | targets |
|---|---|
| **Manual (blanket)** | every zone |
| **Vision-only** | zones whose HSV vegetation fraction ≥ **0.15** (the threshold fixed before the earlier greenhouse run); this sprays weeds too |
| **Proposed** | zones the **tomato detector** marks crop (P ≥ **0.5**); when the reliability gate is open, additionally require the India classifier to call the plant diseased (P ≥ **0.5**); when shut, fall back to crop-only |
| **Ceiling** | true crop zones on truly diseased plants — needs labels, not deployable |

Gate: label-free ATC-MC estimate over plants encountered so far ≥ **0.80**, after **25** plants.

## 7. The tomato detector

Frozen ImageNet ResNet-50 (`timm resnet50.a1_in1k`, cached), global-pooled features of each
sub-cell resized to 224 px, logistic regression with balanced class weights, threshold 0.5.
Trained on train-split sub-cells only. Its own precision and recall on test sub-cells are
reported separately before any spray result.

## 8. Metrics — reported for every policy and layout, on test images only

| metric | meaning |
|---|---|
| **pesticide use** | zones sprayed ÷ all zones; manual = 100% |
| **spray precision** | of zones sprayed, the share that truly needed it — *how accurate the spraying is* |
| **disease coverage** | of zones that needed spray, the share that got it — *what was missed* |
| off-target: weeds | share of sprayed zones containing only weeds |
| off-target: bare soil | share of sprayed zones containing nothing |

95% bootstrap intervals over test-image blocks. A reduction is only reported together with
coverage: consumption can always be cut by missing plants.

## 9. Deviations

*(none yet — any change after the data arrives is recorded here with its reason)*
