# RQ4 Diagnostic Plan — pre-registered

**Written before any diagnostic was run.** Committed ahead of the results so the
analysis is confirmatory rather than a search for the smallest p-value.

**RQ4:** What measurable failure modes cause existing label-free accuracy estimators to
become unreliable under agricultural laboratory-to-field distribution shift?

The benchmark establishes **that** the estimators fail. It does not establish **why**.
The current explanation — *confidence becomes decoupled from correctness* — is an
inference from symptoms. This phase either measures it or withdraws it.

---

## 1. Rules binding this phase

1. **No retraining, no rerun of the 720k grid, no new estimators, no new method.**
   Diagnostics only, on cached logits/features.
2. **Every output is labelled `DEPLOYABLE` or `POST-HOC`.**
   - `DEPLOYABLE` — computable at deployment from unlabelled target data alone.
   - `POST-HOC` — uses target labels; explanatory only, and may never feed an estimator.
3. **Independent units are declared per analysis** and never mixed:
   - Arm D image-level statistics → **photograph** (391 in `SHARED_3`, not 1,616 crops);
   - across-backbone comparisons → **seed** (5 per backbone);
   - session-level statistics → **session**, with overlap reported.
4. **Architecture comparisons use only V2 + V9** (`v2_leaf`, `SHARED_3`, seeds 0–4,
   identical hyperparameters). **Arm B is excluded** — it is ResNet-50 only. The 30-model
   pool is never used as an architecture average; it is ResNet-weighted.
5. **Effect sizes and intervals over p-values.** Where many tests are unavoidable, report
   the effect and a bootstrap interval; apply Holm correction within a diagnostic family.
6. **Nothing is tuned against target accuracy.**
7. Results land in `results/rq4/` under a new experiment namespace. **No existing table
   is overwritten.**

---

## 2. The diagnostics

Nine, fixed in advance. Each states what would **support** and what would **weaken** the
current interpretation, so the outcome is decidable rather than narratable.

### A — Confidence vs correctness `POST-HOC`

*Does confidence still discriminate correct from incorrect predictions under shift?*

Per (backbone, arm): mean confidence of correct vs incorrect predictions, their
separation, **AUROC of confidence for predicting correctness**, AUPRC, and confidence
densities. Arms A / C / D.

- **Supports decoupling:** AUROC falls substantially from arm A to arm D (toward 0.5).
- **Weakens it:** AUROC stays high on arm D — confidence would still rank errors, and
  the failure would lie elsewhere (e.g. in the threshold or the aggregation).

*Independent unit: image, within a fixed arm; backbone spread reported over 5 seeds.*

### B — Calibration and overconfidence `POST-HOC`

*Is the model systematically overconfident under field shift?*

ECE (15 bins, reusing the ablation's implementation), Brier score, mean confidence,
true accuracy, and **confidence − accuracy** per arm and backbone.

- **Supports overconfidence as the mechanism:** confidence − accuracy is large and
  positive on arms C/D, near zero on arm A.
- **Weakens it:** the gap is small, meaning the estimators fail for a reason other than
  a simple confidence offset.

### C — Session sensitivity `POST-HOC` (diagnosis) / `DEPLOYABLE` (the variance term)

*Have the estimators lost between-session sensitivity, as distinct from being offset?*

Per (estimator, arm, backbone): variance of estimates, variance of true accuracy, their
ratio, covariance, Pearson and Spearman correlation, OLS slope of estimate on truth, R².

The **variance of the estimates** is deployable. The comparison against true variance is
post-hoc.

- **Supports:** slope ≪ 1 and variance ratio ≪ 1 on arm D — the estimator is close to a
  constant.
- **Weakens:** slope ≈ 1 with an offset — a pure bias problem, correctable in principle.

*Independent unit: session; overlap (Jaccard 0.009–0.142) reported.*

### D — Logit and feature shift `DEPLOYABLE`

*Does measurable unlabelled shift track estimator failure?*

Reusing `src/evaluation/shift.py` rather than importing a new metric: MMD (bounded,
comparable across backbones) and label-prior TV on **predicted** marginals; per-session
mean confidence and entropy; Fréchet distance **per backbone only** — the audit showed it
is not comparable across architectures (arm D reads ~48 for ResNet-50 against ~14,500 for
EfficientNet-B0).

Correlate each with estimator bias and with true session accuracy.

- **Key discrimination:** a statistic that correlates with true accuracy *and* is
  computable without labels is a candidate deployable signal (→ Outcome 1). One that
  only correlates once labels are used is post-hoc (→ Outcome 2).

### E — Class-marginal effect and COT `DEPLOYABLE` (signal) / `POST-HOC` (bias)

*Does COT's architecture-dependent bias relate to its use of the class marginal?*

The audit found COT's arm-D bias spans +0.031 (ViT) to +0.235 (EfficientNet) — a 7-fold
spread no other estimator shows. COT is also the only estimator using the class marginal.
**The connection is currently untested.**

Per (backbone, session): predicted class marginal, its entropy, TV and χ² distance from
the source marginal, dominant-class share. Then test association with COT's bias across
backbones — **n = 4 architectures**, so this is descriptive with intervals, not a
significance test.

- **Supports:** backbones whose predicted marginal departs further from source show
  larger COT bias, ordered consistently.
- **Weakens:** no ordering, or the same pattern appears for confidence-only estimators
  (which would make it a general shift effect, not a marginal effect).

**Explicitly underpowered. Any finding here is an association at n = 4, and must be
reported as hypothesis-generating.**

### F — Class-conditional failure `POST-HOC`

*Is aggregate confidence misleading because particular classes fail while confidence
stays high?*

Per class: accuracy, mean confidence, confidence − accuracy, predicted vs true
distribution, confusion structure. Source vs arm D.

- **Supports:** one or two classes collapse while retaining high confidence, and the
  aggregate over-estimate is arithmetically attributable to them.

### G — Capture-day effect `DEPLOYABLE` + `POST-HOC`

*Can any unlabelled signal distinguish three real acquisition conditions whose true
accuracy differs by 24 points (0.604 / 0.491 / 0.369)?*

Per day: true accuracy `POST-HOC`; mean confidence, entropy, predicted marginal, MMD
against source, estimator outputs, n images and n photographs `DEPLOYABLE`.

- **The strongest possible positive result in this phase:** an unlabelled statistic that
  moves monotonically with true accuracy across the three days.
- Three days is n = 3. Direction and magnitude only — no significance claim.

### H — Architecture dependence `POST-HOC` for bias, `DEPLOYABLE` for the signals

Across the four backbones (V2 + V9, arms A/C/D): source and target accuracy, estimator
bias, mean confidence, entropy, ECE, confidence-correctness AUROC, shift statistics.

Classify the failure as architecture-independent, architecture-dependent,
estimator-specific, or an estimator × architecture interaction.

*Independent unit: seed (5 per backbone).*

### I — Bias versus variance across N `POST-HOC`

*Does increasing N reduce random variation while leaving systematic bias intact?*

Per (estimator, arm, backbone) at N ∈ {25, 50, 100, 200, 400}: mean estimate, variance of
the estimate, signed bias, |bias|, MAE, and the **share of MSE attributable to bias²**.

- **Supports "bias-dominated":** variance falls roughly as 1/N while |bias| is flat, and
  bias² dominates MSE at every N.
- **Weakens it:** |bias| also falls with N, which would make the failure partly a
  small-sample artefact.

This is the diagnostic that either earns or removes the phrase *bias-dominated*, which
the thesis currently uses.

---

## 3. Decision gate

At the end, exactly one outcome is declared:

| Outcome | Condition | Consequence |
|---|---|---|
| **1** | A clear mechanism exists **and** is measurable without target labels | Candidate for a corrected estimator (Phase 9) |
| **2** | A clear mechanism exists but needs labels to measure | **Do not** invent a correction; frame as a limitation of label-free estimation |
| **3** | No single mechanism explains the failure | Report as multifactorial; force no causal story |
| **4** | The decoupling interpretation is not supported | Correct the thesis framing |

All four are acceptable. **Outcome 2 is the most likely a priori** and is not a
disappointing result: "the thing that would tell you the estimate is wrong is itself
unobservable without labels" is a substantive and defensible finding about the ceiling on
label-free estimation.

---

## 4. What this phase cannot establish

- **Causation.** Every association here is observational over 30 fixed models. Nothing
  is intervened upon, so "explains" means "accounts for variance in", never "causes".
- **Generality beyond the tested conditions** — one source dataset, three target arms,
  four architectures, three capture days.
- **Anything per-image.** Diagnostics A/B/F compute image-level statistics *in aggregate*
  to explain session-level failure. That is not a per-image correctness claim and must
  not be written as one.
