# Label-Free Accuracy Estimation for Plant Disease under Lab-to-Field Shift

Plant-disease classifiers report >99% on laboratory datasets and collapse in the
field. Everyone who has measured that collapse used **labeled** field test sets —
which a deployed machine never has. This project asks whether the collapse can be
**estimated from unlabeled field images alone**, benchmarks six label-free accuracy
estimators across target domains representing different kinds and magnitudes of real
distribution shift, asks **how many unlabeled observations are needed before the
estimate is trustworthy**, and demonstrates the estimate as a session-level gate on a
sprayer rover.

**Contribution type: empirical benchmark. Not a new method.** Every estimator here is
imported, not invented.

---

## The one constraint that shapes everything

ATC, DoC, MaNo and COT estimate accuracy over a **dataset or session**. They do not
produce a per-image correctness probability.

```
session of N unlabeled images → classifier → logits/features
    → dataset-level estimator → estimated session accuracy
    → gate → operate / withhold / request human review
```

The existing green-region rule on the rover decides *where* to spray. The session
estimate decides *whether the rover should be operating at all*. `SessionGate`
raises `PerImageQueryError` if asked about a single image, because conflating the two
is selective prediction — a different literature with different guarantees.

---

## Headline findings

Based on **30 trained models** (4 backbones × 5 seeds, plus the V1/V2 leakage ablation)
and **720,000 scored sessions** across four target arms.

| | |
|---|---|
| Lab→field drop is real | source 0.985 → Tomato-Village 0.503 (**48 pp**); Arm A control drops 0.9 pp |
| Arm A control passes | ATC/DoC reach MAE ≈ 0.013 with no shift — the estimators are correctly implemented |
| Estimators break under field shift | 6 of 8 over-estimate by >15 points; MaNo by +0.46 |
| **They stop ranking sessions at all** | correlation with true session accuracy collapses from 0.53 (arm A) to ≈0.07 (arm D) |
| **The gate carries no information** | precision-of-operate equals the base rate exactly (lift 1.00×) for every confidence-based estimator |
| **N\* essentially does not exist** | only **1 of 96** shifted (arm, estimator, backbone) cells reaches MAE ≤ 0.05 at any N, against 41 of 48 with no shift |
| Not architecture-dependent | source accuracy spans 0.923–0.997 across four backbones; arm-D bias stays +0.25 to +0.33 throughout |
| A wider label space is worse | 3 → 7 classes degrades 6 of 8 estimators (nuclear norm by +0.374) |
| The trivial baselines win | mean entropy and COT are far better calibrated than every sophisticated method |
| Shift is not a severity ladder | arms C and D have near-identical covariate shift but very different label shift |
| **H4 falsified** | leaf leakage changes nothing: source accuracy 0.9804 vs 0.9812 over 5 seeds, and 0/8 estimators shift significantly |

**Verdict.** On this evidence, tomato field accuracy **cannot** be reliably estimated
from unlabeled images with the current generation of confidence-based estimators. They
are correctly implemented, they fail identically across architectures, and they fail in
the unsafe direction. COT — the one method that uses the class marginal rather than the
confidence distribution — is the only one retaining useful signal, and it is
threshold-fragile.

Full detail: [`reports/phase1_feasibility_report.md`](reports/phase1_feasibility_report.md),
and [`docs/stop_condition_F_investigation.md`](docs/stop_condition_F_investigation.md)
for why the estimators go constant.

---

## Layout

```
src/
  data/         index, canonical class mapping, verification, V1/V2 splits
  models/       timm backbone factory
  training/     dataset, transforms, AMP training loop
  inference/    Stage 2 — cache logits + penultimate features
  estimators/   the estimator suite. NEVER sees a target label.
  sessions/     session construction and overlap accounting
  evaluation/   scoring, shift magnitude, gate. The ONLY consumer of target labels.
  visualization/figures, generated from logged tables only
  rover/        session gate, leaf-crop proposal, hardware isolation tests
scripts/        the pipeline drivers, in execution order
docs/           protocol, class mapping, audits, reproducibility, verification
results/        raw/ (manifests, checkpoints, caches, estimates) tables/ figures/
tests/          leakage firewall, reference-implementation checks, gate behaviour
```

## The leakage firewall

```
src/estimators/   MAY see: source val labels, source+target logits/features. NEVER target labels.
src/evaluation/   MAY see: target labels. Runs strictly AFTER estimates are written to disk.
```

Enforced three ways, not by discipline:

1. An AST test rejects any import in `src/estimators/` that could reach a label store,
   verified with a negative control (injecting `import pandas` fails the suite).
2. `Estimator.estimate` takes only target logits and features — there is no parameter
   through which a label could arrive.
3. Labels are cached in a **separate** `.npz` from logits and features, so the split
   is physical.

## Running it

```bash
python src/data/verify.py                      # re-verify every dataset count
python scripts/run_training.py --stage v1v2    # V1/V2 x 5 seeds
python scripts/run_caching.py                  # Stage 2 — the load-bearing step
python scripts/run_estimates.py                # phase 1: estimates, no target labels
python scripts/run_evaluation.py               # phase 2: score against the truth
python scripts/run_capture_day_diagnostic.py   # real acquisition days as sessions
python src/visualization/figures.py
python scripts/make_report.py
```

Run training **exclusively** — the 8 GB GPU will not hold a training run and a caching
run at once, and the collision surfaces as a confusing `CUDNN_STATUS_EXECUTION_FAILED`
followed by cascading OOMs.

## Environment

Python 3.12, torch 2.11.0+cu128 on an RTX 5060 Laptop (Blackwell, **sm_120**). A wheel
without sm_120 kernels installs and imports cleanly and only fails when a CUDA kernel
actually runs, so verify with a real operation:

```bash
python -c "import torch; x=torch.randn(4096,4096,device='cuda'); print((x@x).sum().item())"
```

Datasets live at `C:\ds\` — a short root, because class names like
`Tomato___Spider_mites Two-spotted_spider_mite` nest deeply enough to hit `MAX_PATH`.

**PlantDoc cannot be `git checkout`ed on Windows**: 20 of its tomato filenames are
scraped web URLs containing `? & =`, which NTFS rejects, and git aborts the entire
worktree. Use `scripts/fetch_plantdoc.py`, which pulls the zipball and sanitises names
while recording a `filename_map.csv` for provenance.

## Reproducibility

Every run writes `results/raw/<version>/manifest_*.json` with the seed, full config,
git commit, package versions, dataset checksums and metrics. **Figures and the report
are generated from those tables only** — no number in either is typed by hand.

Estimator reproduction levels are recorded honestly in
[`docs/estimator_reproducibility.md`](docs/estimator_reproducibility.md): ATC, DoC and
COT are verified against official implementations (L3); nuclear norm and MaNo are our
implementations of the published scores (L2). AETTA and AutoEval are argued
**ineligible** for a frozen classifier on unlabeled deployment data, and excluded.

## Status

**Arms A, B, C and D are complete** — 30 models, 5 seeds each, 720,000 scored sessions,
with figures and the Phase 1 report generated from the logged tables.

Arm B uses `SHARED_7` rather than `SHARED_8`: PlantVillage ships no leafmap for
`Tomato_mosaic_virus`, so none of its 373 images can enter a leakage-free leaf-grouped
split. Training `SHARED_8` on the grouped pool would leave that head with zero examples
and make PlantDoc's 54 ToMV images guaranteed errors. We checked whether leaf identity
could be recovered from filename numbering instead; the rule reproduces 100% of healthy
leaves but only 19% of late blight, so it is not usable. See `data.mapping.SHARED_7`.

**Arm E (rover) is not started.** The frozen `rover/original/web_rover.py` baseline was
not delivered with the handoff bundle, and the physical rover has not been inspected —
every row of `docs/rover_hardware_inventory.md` is `UNVERIFIED`. The thesis is designed
to remain complete if that chapter is deleted.

## Never claim

- ✗ "First to apply label-free accuracy estimation to a new domain" — medical imaging
  did it (arXiv:2507.22776, 2025).
- ✗ "First robot that knows when to distrust itself" — introspective perception,
  Daftry 2016.
- ✗ "Novel accuracy estimation method" — these are imported.
- ✗ Any per-image correctness claim.
- ✗ Any number from the delivered third-party paper draft. The delivered code contains
  no logging of any kind and could not have produced its Table I.
