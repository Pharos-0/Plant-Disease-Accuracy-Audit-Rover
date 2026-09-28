# Arm 4 — A Fourth Target Dataset

**2026-09-28.** The manuscript's largest stated limitation is dataset-level sample size:

> *One crop, one primary source dataset, three target conditions. Dataset-level evidence is
> effectively $n=3$… **We cannot answer this with current data**.*

A fourth target dataset became available, so part of that can now be answered. This document
records what was asked, what was found, and what the answer does **not** establish.

---

## 1. Why the obvious experiment was not run

The request that prompted this was to train a model on the **rover-view core** (218 images,
WIDE viewpoint, top relevance). That is not possible, and the reason is a finding:

| source | in core | disease labels |
|---|---:|---|
| SRC014 | 139 | **none** — its Pascal VOC boxes are fruit *ripeness* |
| SRC015 | 27 | **none** — flat `image/N.jpg` |
| SRC005 | 38 | binary healthy/diseased |
| SRC002 | 14 | multi-class disease |

**166 of 218 (76%) carry no disease label**, and the labelled 52 use two incompatible label
spaces. Beyond that, SRC014's 139 frames come from **two capture sessions**, so a train/test
split of them would report a number about two traverses rather than about generalisation —
precisely the error this project exists to warn against.

**No labelled rover-view disease imagery was found in any source.** Training on the core
would have required inventing labels.

## 2. What was run instead

The raw folder of Mendeley deposit `10.17632/3mbnb82mxd.2`, restricted to the three
`SHARED_3` classes, gives a fourth target arm:

| class | n |
|---|---:|
| healthy | 115 |
| early_blight | 81 |
| late_blight | 47 |
| **total** | **243** |

This is after the two removals in `dataset/documentation/PILOT_REPORT.md` §16: the deposit's
7,200-image augmented folder, and the 311 images byte-identical to PlantDoc.

## 3. Two near-misses that would have destroyed frozen work

**`data/index.parquet` is hashed into all 47 experiment manifests** by `utils/manifest.py`.
Adding a fourth domain to `build_all()` and regenerating the index would have changed that
hash and invalidated the provenance chain of every frozen result. The edit was made, caught,
and reverted; `build_all` carries a comment saying why `"pakistan"` is deliberately absent
from it. Arm 4 is built on demand through `BUILDERS["pakistan"]`, written to its own
`data/index_arm4.parquet`, and cached under a new `CacheKey` arm name.

**`scripts/ti_audit.py` writes to `results/ti_audit/`**, which holds the frozen three-arm
results. Running it with a different arm set would have overwritten them — the same overwrite
trap this project hit once before. An `--out` flag was added, the directory backed up first,
and the four-arm run written to `results/ti_audit_4arm/`.

Both were verified afterwards: `index.parquet` SHA-256 unchanged at `9c8bcb346dcb213b…`,
`results/ti_audit/` byte-identical to its backup, and **27/27 validations still pass**.

## 4. Classifier accuracy — 20 frozen models

| arm | mean | sd | range | n |
|---|---:|---:|---|---:|
| `source_test` | 0.9618 | 0.031 | 0.917–1.000 | 436 |
| `plantdoc` | 0.5034 | 0.056 | 0.428–0.641 | 262 |
| `tomatovillage` | 0.4999 | 0.075 | 0.407–0.636 | 1,616 |
| **`pakistan`** | **0.3601** | 0.032 | 0.313–0.412 | 243 |

Arm 4's majority-class baseline is **0.4733**. The classifiers score **0.3601** — *below* a
constant "always healthy" predictor, and barely above the 0.333 chance line for three classes.
A sharper collapse than either existing field arm.

## 5. Estimator error — eight estimators × four arms

| estimator | `source_test` | `plantdoc` | `tomatovillage` | **`pakistan`** |
|---|---:|---:|---:|---:|
| cot | 0.0324 | 0.1127 | 0.1339 | **0.1493** |
| entropy | 0.1316 | 0.1510 | 0.1337 | 0.3076 |
| nuclear_norm | 0.0089 | 0.3058 | 0.3089 | 0.3966 |
| atc_ne | 0.0104 | 0.3408 | 0.3109 | 0.5165 |
| atc_mc | 0.0084 | 0.3569 | 0.3268 | 0.5325 |
| doc | 0.0083 | 0.3684 | 0.3407 | 0.5253 |
| mano | 0.0098 | 0.4267 | 0.4165 | **0.5736** |

Oracle constant (needs the labels; a ceiling, not a method): 0.0272 / 0.0424 / 0.0610 /
**0.0264**. **No estimator beats it on any shifted arm.** On arm 4 the best is COT at 0.1493
against 0.0264 — a 5.7× gap.

The anchor-dominated estimators behave exactly as the account predicts. The anchor–truth gap
on arm 4 is $0.9618 - 0.3601 = 0.602$, and their errors (0.52–0.57) approach it: they are
reporting source calibration, so the worse the target, the worse they get.

GdScore is absent from this table by construction — it needs a fresh forward pass in `train()`
mode, not cached logits. It **is** present in the TI audit below, which runs that pass.

## 6. The reversal replicates

Per-classifier Spearman between target informativeness and absolute error, averaged over the
20 classifiers, nine estimators per classifier:

| arm | mean ρ | sd | 95% CI | Wilcoxon p |
|---|---:|---:|---|---:|
| `source_test` (zero shift) | **+0.466** | 0.292 | [+0.331, +0.583] | 0.0003 |
| `plantdoc` | −0.428 | 0.259 | [−0.534, −0.313] | 0.0000 |
| `tomatovillage` | −0.526 | 0.257 | [−0.631, −0.416] | 0.0001 |
| **`pakistan` (new)** | **−0.273** | 0.432 | [−0.445, −0.074] | 0.0137 |

Paired against the zero-shift control, same classifier on both arms, 20 pairs:

| arm | contrast | 95% CI | p |
|---|---:|---|---:|
| `plantdoc` | +0.894 | [+0.690, +1.080] | 0.0000 |
| `tomatovillage` | +0.992 | [+0.822, +1.160] | 0.0001 |
| **`pakistan`** | **+0.738** | [+0.494, +0.978] | 0.0003 |

**The sign flips on the fourth arm too, and the interval excludes zero.**

**The honest qualification:** arm 4's effect is *weaker* (−0.273 vs −0.43 and −0.53) and
*noisier* (sd 0.432 vs ~0.26). With 243 images against 262 and 1,616, that is expected, but it
should be reported as a weaker replication rather than an equal one.

## 7. What this does and does not establish

**Does:** the reversal is not an artefact of the two target datasets on which it was found. A
fourth dataset, which played no part in establishing it, shows the same sign with an interval
excluding zero. Dataset-level evidence moves from $n=3$ to $n=4$.

**Does not:** arm 4's independence is **partial and unestablished**. 1,528 filenames in the
original archive carry a Roboflow `.rf.<hex>` export signature and several are URL-encoded web
filenames, so overlap with public datasets not held here is possible and was not ruled out.
The 311 detected PlantDoc duplicates were removed, but only those detectable against holdings
we have.

This is therefore **corroboration on partially independent data**, not clean replication, and
the manuscript must say so in those words. The crop is still tomato, the label set is still
`SHARED_3`, and the source dataset is still PlantVillage.

## 8. Reproduce

```bash
python scripts/arm4_pakistan.py --cache          # 20 models over arm 4
python scripts/arm4_evaluate.py                  # accuracy + estimator MAE
python scripts/ti_audit.py --arms source_test,plantdoc,tomatovillage,pakistan \
                           --n-boot 100 --out ti_audit_4arm
```

Outputs in `results/arm4_pakistan/` and `results/ti_audit_4arm/`. Nothing writes to
`data/index.parquet`, `results/ti_audit/`, or any existing cache key.
