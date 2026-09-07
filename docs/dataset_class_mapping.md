# Dataset Class Mapping — Tomato Laboratory → Field

**Status:** verified 2026-09-06 from official repositories via GitHub API. Image counts for
PlantDoc and Tomato-Village are **exact** (full file-tree enumeration). PlantVillage counts are
from its own `leaf_grouping/filtered_leafmaps/` CSVs and cover a **subset** of some classes —
re-verify on download.

Rule (brief §6): never map two biologically different classes because their names look similar.

---

## 1. Sources

| Dataset | Role | Official source | Domain |
|---|---|---|---|
| PlantVillage | **Source** | `github.com/spMohanty/PlantVillage-Dataset`, `raw/color/Tomato___*` | Laboratory — single detached leaf, uniform background |
| PlantDoc | Target A | `github.com/pratikkayal/PlantDoc-Dataset` (CODS-COMAD 2020) | Web-scraped, in-the-wild, cluttered |
| Tomato-Village | Target B | `github.com/mamta-joshi-gehlot/Tomato-Village`, `Variant-a(Multiclass Classification)` | **Real Indian field** — Jodhpur/Jaipur, Rajasthan, Mar 2022 |
| Rover frames | Target C | To be collected | Robot deployment |

Tomato-Village's own README states the motivation directly: PlantVillage "was created in a
lab/controlled environment, and models trained on it do not perform well on real-world images."
That is this thesis's premise, asserted by the target dataset's authors.

---

## 2. Class inventories (verbatim folder names)

**PlantVillage — 10 tomato classes.** `Tomato___Bacterial_spot`, `Tomato___Early_blight`,
`Tomato___Late_blight`, `Tomato___Leaf_Mold`, `Tomato___Septoria_leaf_spot`,
`Tomato___Spider_mites Two-spotted_spider_mite`, `Tomato___Target_Spot`,
`Tomato___Tomato_Yellow_Leaf_Curl_Virus`, `Tomato___Tomato_mosaic_virus`, `Tomato___healthy`

**PlantDoc — 9 tomato classes** (of 28 total). Counts exact, train+test:

| PlantDoc class | Images |
|---|---:|
| Tomato Septoria leaf spot | 151 |
| Tomato leaf late blight | 111 |
| Tomato leaf bacterial spot | 110 |
| Tomato mold leaf | 91 |
| Tomato Early blight leaf | 88 |
| Tomato leaf yellow virus | 76 |
| Tomato leaf *(= healthy)* | 63 |
| Tomato leaf mosaic virus | 54 |
| Tomato two spotted spider mites leaf | **2** ← unusable |
| **Total** | **746** |

**Tomato-Village Variant-a — 8 classes, 4,526 images.** Counts exact:

| Class | train | val | test | Total |
|---|---:|---:|---:|---:|
| Leaf Miner | 716 | 204 | 104 | 1024 |
| Magnesium Deficiency | 655 | 187 | 95 | 937 |
| Late_blight | 632 | 180 | 92 | 904 |
| Spotted Wilt Virus | 361 | 103 | 53 | 517 |
| Early_blight | 347 | 99 | 50 | 496 |
| Nitrogen Deficiency | 251 | 72 | 37 | 360 |
| Healthy | 151 | 43 | 22 | 216 |
| Pottassium Deficiency | 50 | 14 | 8 | 72 |
| **Total** | 3163 | 902 | 461 | **4526** |

---

## 3. Mapping decisions

### 3.1 PlantVillage ↔ PlantDoc — 8 defensible pairs

| PlantVillage | PlantDoc | Verdict |
|---|---|---|
| `Tomato___healthy` | `Tomato leaf` | ✅ PlantDoc's unqualified "Tomato leaf" is its healthy class |
| `Tomato___Early_blight` | `Tomato Early blight leaf` | ✅ *Alternaria solani* |
| `Tomato___Late_blight` | `Tomato leaf late blight` | ✅ *Phytophthora infestans* |
| `Tomato___Bacterial_spot` | `Tomato leaf bacterial spot` | ✅ *Xanthomonas* spp. |
| `Tomato___Leaf_Mold` | `Tomato mold leaf` | ✅ *Passalora fulva* |
| `Tomato___Septoria_leaf_spot` | `Tomato Septoria leaf spot` | ✅ *Septoria lycopersici* |
| `Tomato___Tomato_mosaic_virus` | `Tomato leaf mosaic virus` | ✅ ToMV |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | `Tomato leaf yellow virus` | ✅ TYLCV |
| `Tomato___Spider_mites …` | `Tomato two spotted spider mites leaf` | ⚠️ Same pest, but **n=2**. Excluded — cannot support any session |
| `Tomato___Target_Spot` | — | ❌ No PlantDoc counterpart |

### 3.2 PlantVillage ↔ Tomato-Village — only 3 defensible pairs

| PlantVillage | Tomato-Village | Verdict |
|---|---|---|
| `Tomato___healthy` | `Healthy` | ✅ |
| `Tomato___Early_blight` | `Early_blight` | ✅ |
| `Tomato___Late_blight` | `Late_blight` | ✅ |

**Explicitly rejected mappings — do not "fix" these later:**

| Tomato-Village class | Tempting map | Why rejected |
|---|---|---|
| `Spotted Wilt Virus` | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | **Different pathogens.** TSWV is a *tospovirus* (thrips-vectored); TYLCV is a *begomovirus* (whitefly-vectored). Different symptoms, different biology. Name similarity only. |
| `Leaf Miner` | `Tomato___Spider_mites …` | Different pest orders (*Liriomyza* dipteran leaf mining vs. tetranychid mite stippling) |
| `Magnesium / Nitrogen / Pottassium Deficiency` | any | **Abiotic** nutrient disorders. PlantVillage contains no abiotic class at all |

PlantVillage-only: Bacterial_spot, Leaf_Mold, Septoria_leaf_spot, Spider_mites, Target_Spot, TYLCV, ToMV.

---

## 4. Final shared label sets

```python
SHARED_3 = ["healthy", "early_blight", "late_blight"]          # all domains
SHARED_8 = SHARED_3 + ["bacterial_spot", "leaf_mold",
                       "septoria_leaf_spot", "mosaic_virus",
                       "yellow_leaf_curl_virus"]               # PlantVillage ↔ PlantDoc only
```

Available target images:

| Arm | Target | Label space | Target images |
|---|---|---|---:|
| B | PlantDoc | `SHARED_8` | 744 |
| C | PlantDoc | `SHARED_3` | 262 |
| D | Tomato-Village | `SHARED_3` | **1,616** (LB 904 / EB 496 / H 216) |

---

## 5. STOP CONDITION A — assessed, not tripped

Three classes is a **small but scientifically valid** overlap: {healthy, early blight, late
blight} is a biologically clean, well-separated 3-way task, and both diseases are economically
major. The condition is **not** tripped. But it constrains the design, and the constraints are
binding:

1. **N = 1000 sessions are impossible** on Tomato-Village's 1,616 images. Ladder capped at N ≤ 400.
2. **Sessions at N ≥ 200 must overlap.** Bootstrap resampling required; overlap must be reported, never hidden.
3. **Healthy is the bottleneck** at 216 images — it alone limits class-balanced session construction.
4. PlantDoc `SHARED_3` (262 images) supports N ≤ 100 only. Use it as a *control*, not a main arm.

---

## 6. Two near-duplicate leakage traps — both datasets, both with metadata fixes

This is the most consequential finding in this document.

### 6.1 PlantVillage: exactly 4 photographs per physical leaf

`leaf_grouping/filtered_leafmaps/*.csv` maps every image to a physical leaf ID. The ratio is
uniformly **4.0 images per leaf**:

| PlantVillage class | Images | Distinct leaves |
|---|---:|---:|
| Tomato___Bacterial_spot | 2128 | 532 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 2740 | 685 |
| Tomato___Early_blight | 1000 | 250 |
| Tomato___Septoria_leaf_spot | 1000 | 250 |
| Tomato___healthy | 1000 | 250 |
| Tomato___Leaf_Mold | 952 | 238 |
| Tomato___Late_blight | 920 | 230 |
| Tomato___Tomato_mosaic_virus | **no leafmap** | — |

**The `SHARED_3` source set is not ~2,900 independent images. It is 730 independent leaves.**

A random image-level train/val/test split places the *same physical leaf* in training and test.
This is very likely a large part of why PlantVillage-trained models report >99% and then
collapse in the field — and it would silently corrupt the source-validation accuracy that
**ATC and DoC calibrate their thresholds against**. An inflated source accuracy propagates
directly into a biased target estimate.

**Mandatory:** split PlantVillage by `Leaf #`, never by image. Caveats: `filtered_leafmaps`
coverage is partial (Late blight 920 of ~1909; ToMV has none), so classes without a leafmap
must either be excluded or handled with a documented fallback.

### 6.2 Tomato-Village: multiple crops per source photograph

Filenames are `IMG<YYYYMMDD><HHMMSS>[_n].jpg` — e.g. `IMG20220323122542_1.jpg` and
`IMG20220323122542_2.jpg` are two crops of **one photograph**. All 4,525 files carry this
pattern, across three capture days:

| Capture date | Images |
|---|---:|
| 2022-03-23 | 1475 |
| 2022-03-24 | 1658 |
| 2022-03-25 | 1392 |

**Mandatory:** group sessions by base timestamp, never by individual crop. Random sampling
would place sibling crops of one photo in a session, making it artificially easy and biasing
every accuracy estimate optimistically — precisely the failure brief §8 warns about.

**Silver lining:** the three capture days are a *natural, defensible* basis for the easy /
medium / hard session diagnostic in brief §19 — real acquisition conditions rather than
manufactured difficulty.

---

## 7. Label shift is guaranteed

Source and target class priors differ sharply. In `SHARED_3`, PlantVillage is roughly balanced
(250/230/250 leaves) while Tomato-Village is 56% late blight, 31% early blight, 13% healthy.
Per `novelty_verification.md` §4, prevalence shift is the documented failure mode of ATC and
DoC. **This is a pre-registered hypothesis, not a surprise** — see the protocol.
