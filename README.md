# Unsupervised Accuracy Estimation under Agricultural Laboratory-to-Field Shift

*Also called label-free accuracy estimation. We lead with "unsupervised" because in
agriculture "label-free" denotes dye-free biosensing — see `docs/litsearch_scopus.md`.*

M.Tech research project. **Status: research complete, ready for thesis writing.**

---

## What this asks

A plant-disease classifier trained on laboratory imagery is ~96% accurate there and ~50%
accurate on real field photographs. You cannot measure that drop without field labels, which
is exactly what deployment lacks. **Label-free accuracy estimation** claims to estimate a
fixed model's accuracy on unlabelled data. Does it work under real agricultural shift — and
how much of such an estimate actually depends on the field data at all?

## Why it matters

An autonomous rover deciding whether to spray needs to know whether to trust its classifier.
If the estimate is really an echo of laboratory calibration, trusting it is unsafe.

## Datasets

| arm | role | photographs | crops | classifier accuracy |
|---|---|---:|---:|---:|
| PlantVillage (source) | training / calibration | — | — | — |
| `source_test` | zero-shift control | 109 | 435 | **0.962** |
| PlantDoc | severe shift (web imagery) | 262 | 262 | **0.503** |
| Tomato-Village | severe shift (real Indian field) | 391 | 1,616 | **0.500** |

Label set `SHARED_3`. Sampling is **always at photograph level** — 1,616 crops come from
only 391 photographs, and treating crops as independent would inflate everything downstream.

## Models

20 frozen classifiers: **4 architectures × 5 seeds** — ResNet-50, EfficientNet-B0,
MobileNetV3-Large, ViT-S/16. They are not 20 independent datasets.

## Estimators

Nine: `confidence`, `entropy`, `atc_mc`, `atc_ne`, `doc`, `cot`, `nuclear_norm`, `mano`,
`gdscore`. This is a **sample** of the field, not the state of the art — at least twenty
methods exist. Selection is justified in `docs/estimator_selection.md`, which separates
principled exclusions from ignorance.

## Key findings

1. **They do not work well here.** Best MAE **0.1337** (entropy); none beats an oracle
   constant at **0.0610**.
2. **Most are not listening to the field.** Six of nine move **4–14× more** under source
   resampling than target resampling; four change by only 0.06–0.11 when the target is
   replaced by random logits.
3. **The relationship between target responsiveness and error reverses with shift
   severity** — +0.40/+0.44/+0.45 on the zero-shift arm versus −0.28 to −0.51 on both field
   arms, at every feasible sample size (paired *p* = 1e-4 to 1e-6).
4. **More unlabelled data does not help under shift** — Tomato-Village +0.0137 MAE going
   from 25 to 200 photographs, 2 of 9 estimators improved; under zero shift, 8 of 8 improved.
5. **No signal ranks field sessions usefully** for a fixed model (best Spearman 0.22), so a
   deployment gate cannot enable autonomy.
6. **~10–50 labelled photographs beat every label-free estimator** in this setting.

## The rover

A **deployment case study**, not a contribution. Runtime perception monitoring is
established in automotive and rail. Two modes:

- **NORMAL** — classifier → downstream autonomous action; works with the research module off.
- **RESEARCH** — session gate → OPERATE / REVIEW / WITHHOLD; refuses per-image queries by
  construction.

**Deployment recommendation: do not enable autonomous spraying under this shift.**

## Reproduce

**Fast validation** (~1 minute, no GPU) — verifies committed results and runs the test suite:

```bash
python scripts/run_all_validations.py
```

**Rover demonstration** (~10 seconds, replays cached real data):

```bash
python scripts/run_rover_demo.py
```

**Regenerate figures and tables** (~1 minute):

```bash
python scripts/make_thesis_figures.py && python scripts/make_thesis_tables.py
```

**Full reproduction** (hours, GPU required) — training, caching, then the experiments:

```bash
python scripts/run_training.py --stage v1v2
python scripts/run_caching.py
python scripts/run_estimates.py && python scripts/run_evaluation.py
python scripts/ti_audit.py --n-boot 100 --arms source_test,plantdoc,tomatovillage
python scripts/ti_sample_size.py && python scripts/gate_evaluation.py
```

Every experiment writes a manifest with git commit, seeds, package versions, hardware and
cache fingerprints. **Cached results are never presented as freshly reproduced.**

## Expected outputs

`results/figures/` — 31 figures · `results/tables/thesis/` — 12 tables (CSV + Markdown) ·
`results/<experiment>/manifest*.json` — provenance · **136 passing tests**.

## Limitations

One source dataset, one crop, three target arms, 20 models, 5 seeds per architecture.
**N = 400 does not exist on any arm**; `source_test` has no N = 200; GdScore needs ≥128
crops so some cells are absent. TI is a first-order Sobol index — **not a new metric** — and
is **not comparable across sample sizes**. Scopus, Web of Science, IEEE Xplore and ACM DL
were **not searched** (no institutional access). No validated disease→chemical mapping. No
field trial; no demonstrated safe autonomous spraying. No new estimator — one was built and
rejected on its own evidence.

## Never claimed

A new estimator · a new metric · "first" anything · per-image correctness · that label-free
estimation fails in general · universal agricultural performance · validated treatment
mapping · safe autonomous spraying.

## Where to look

| question | file |
|---|---|
| Thesis outline, abstract, RQ answers | `docs/thesis/THESIS_OUTLINE_AND_ABSTRACT.md` |
| Every claim and its evidence | `docs/CLAIM_EVIDENCE_TABLE.md` |
| What was tried and rejected | `docs/WHAT_WE_TRIED_AND_REJECTED.md` |
| Final contribution statement | `docs/THESIS_CONTRIBUTION_FINAL.md` |
| Rover evaluation | `docs/ROVER_RESEARCH_INTEGRATION.md` |
| Literature search | `docs/litsearch_results.md` |
| Viva preparation | `docs/VIVA_QUESTIONS_AND_ANSWERS.md` |
| Project status | `docs/PROJECT_COMPLETION_STATUS.md` |
