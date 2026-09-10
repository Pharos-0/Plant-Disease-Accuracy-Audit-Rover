# Repository Audit — Target-Informativeness Research

**Phase 1.** Written 2026-09-10 before any new code. Every number below was read from a
file on disk; the file is named. Nothing here is inferred.

---

## 1. Structure

| directory | files | size | role |
|---|---:|---:|---|
| `results/` | 483 | 2.74 GB | caches, manifests, result tables — **historical, do not overwrite** |
| `src/` | 58 | 0.3 MB | library code |
| `scripts/` | 23 | 0.3 MB | experiment entry points |
| `tests/` | 16 | 0.3 MB | test suite (**104 passing**, executed 2026-09-10) |
| `docs/` | 24 | 0.2 MB | protocols, plans, results |
| `rover/` | 5 | 0.1 MB | hardware probes — out of scope here |
| `configs/`, `experiments/`, `notebooks/` | 0 | — | empty |

## 2. Estimator inventory and — critically — how each maps a score to an accuracy

This is the decisive structural fact for the proposed research, so it is set out in full.
Read from `src/estimators/methods.py`.

| estimator | `needs_source` | what the source contributes | form of the estimate |
|---|---|---|---|
| **confidence** | **False** | *nothing* | `mean(max softmax(T))` |
| **entropy** | **False** | *nothing* | `mean(normalised entropy(T))` |
| **ATC-MC / ATC-NE** | True | a **threshold** `t` fitted so source below-threshold count = source errors | `mean(score(T) >= t)` |
| **DoC** | True | `source_acc`, `source_conf` | `source_acc - (source_conf - mean conf(T))` |
| **COT** | True | the source **class marginal** `q` | `1 - OT cost(softmax(T), q)` |
| **nuclear_norm** | True | fitted `slope`, `intercept` | `slope * score(T) + intercept` |
| **MaNo** | True | fitted `slope`, `intercept` | `slope * score(T) + intercept` |
| **GdScore** | True | fitted `slope`, `intercept` | `slope * rawscore(T) + intercept` |

### 2.1 The finding that constrains the whole research direction

The Phase-7b decomposition — *estimate = intercept + slope × score*, with the intercept
share measured at 93–98% — **applies structurally to only three of the nine estimators**:
`nuclear_norm`, `MaNo`, `GdScore`. Those three share `_SourceCalibratedScore`
(`methods.py:148`), and **that linear calibration is our own wrapper, not part of the
published methods.** It is documented as ours in `docs/estimator_reproducibility.md`.

Consequences that must not be glossed over:

1. **`confidence` and `entropy` have no source anchor at all.** Any additive
   anchor/target decomposition gives them a target share of exactly 1 by construction.
   That is a definitional artefact, not a finding.
2. **`ATC` and `COT` take source information as a *parameter*** (a threshold; a class
   marginal), not as an additive term. `intercept + slope·score` is not merely
   uninformative for them — it is undefined.
3. **`DoC` is the one estimator whose anchor is intrinsic and additive**:
   `source_acc - source_conf` is a constant fixed at fit time, and the published method
   is built that way. Anchor dominance in DoC would be a property of the *method*, not of
   our wrapper.

> **Therefore an additive decomposition cannot be the general definition of target
> informativeness.** A general measure must be **perturbation-based** — how much the
> output moves when `T` is changed versus when `S` is changed — because that is defined
> for all nine regardless of internal form. This is recorded before any experiment is run,
> so the choice cannot be presented later as a result.

## 3. Verified existing results

Executed 2026-09-10, read from the named files.

**Full unlabelled Arm-D pool, 20 models** — `results/rq7_gdscore/rq7_comparison_full_pool.csv`:

| estimator | MAE | sd | n |
|---|---:|---:|---:|
| entropy | 0.1337 | 0.1078 | 20 |
| cot | 0.1339 | 0.1026 | 20 |
| gdscore | 0.1599 | 0.1614 | 20 |
| nuclear_norm | 0.3089 | 0.0971 | 20 |
| atc_ne | 0.3109 | 0.0858 | 20 |
| confidence | 0.3214 | 0.0945 | 20 |
| atc_mc | 0.3268 | 0.0751 | 20 |
| doc | 0.3407 | 0.0770 | 20 |
| mano | 0.4165 | 0.0680 | 20 |

Matches the figures in the brief. **VERIFIED.**

**GdScore train vs eval MAE** — `results/rq7b_batchnorm/rq7b_train_vs_eval.csv`: matches
the brief exactly (EfficientNet 0.0736/0.4878, MobileNet 0.1021/0.4715, ResNet
0.0520/0.4456, ViT 0.4120/0.4122). **VERIFIED.**

**Target-data share** — `results/rq7b_batchnorm/rq7b_decomposition.csv`: per-backbone means
0.0157–0.0663, i.e. the "1.6–6.6%" in the brief. **Correction to the brief:** across
individual models the full range is **0.0025 to 0.1222**, so 1.6–6.6% is the range of
per-backbone *means*, not of the raw values. **VERIFIED with that clarification.**

## 4. Reusable assets

| asset | location | notes |
|---|---|---|
| eval-mode logits + penultimate features | `results/raw/cache_*/*.npz` | 20 models × arms; the basis of "expensive once, cheap many times" |
| train-mode cache (GdScore only) | `results/raw/cache_trainmode/` | batch-size-128 dependent |
| labels, physically separated | `*__labels.npz` | evaluation only |
| 20 frozen checkpoints | `results/raw/<version>/` | V2 + V9, SHARED_3, 4 backbones × 5 seeds |
| estimator suite | `src/estimators/methods.py` | consistent `fit`/`estimate` interface |
| leakage firewall + tests | `tests/test_leakage_firewall.py` | AST import test, signature restriction |
| photograph-level cluster sampling | `src/sessions/labelled_sample.py` | RQ5; ratio estimator + Wilson interval |
| manifest writer | `src/utils/manifest.py` | git commit, seeds, package versions, checksums |

## 5. Must not be modified

- `results/rq4/`, `results/rq5/`, `results/rq7_gdscore/`, `results/rq7b_batchnorm/` and
  their manifests — historical, pre-registered, reported.
- `docs/rq4_*`, `docs/rq5_*`, `docs/rq7*` — the corresponding reports.
- `src/estimators/methods.py` **existing** classes — a new estimator is added alongside;
  no existing implementation is replaced or "improved".
- The trained checkpoints and both caches.

## 6. Reproducibility risks

| risk | severity | mitigation |
|---|---|---|
| **TF32 numerical noise** (`allow_tf32=True`) makes bitwise reproduction impossible | low | compare semantically (predictions, relative tolerance), as in Phase 7 |
| **Train-mode cache is batch-size-128 dependent** | medium | do not reuse it at other batch sizes; GdScore only |
| **`aggregate_metrics.parquet` has no `seed` column** | low | recompute per-seed from `per_session.parquet` |
| **n = 5 seeds per backbone** | **high** | architecture-level claims are underpowered; state as such |
| **20 models is a small population for MAE-over-models** | **high** | true accuracy sd is 0.075; any MAE near that is uninformative |
| Smart App Control intermittently blocks unsigned scipy DLLs | low | transient; re-run |
| OpenAlex ~100 requests/UTC-day | n/a here | literature search is complete |

## 7. What is available for the new work, without retraining

- 20 frozen classifiers, 4 architectures, 5 seeds.
- Cached logits and penultimate features for source-val and target arms.
- Nine implemented estimators with a uniform interface.
- A verified historical benchmark to compare against.
- Photograph-level grouping for cluster-correct statistics.

**No retraining and no GPU work is required** for the perturbation experiments in Phases
6–8. That is what makes the plan feasible.

## 8. Status

Phase 1 complete. Nothing modified. Test suite: **104 passing** (executed 2026-09-10).
