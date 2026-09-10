# Feasibility Report — Target-Informativeness Direction

**Phases 5–13 executed 2026-09-10.** What was tested, what was not, and what each
experiment can and cannot support.

Evidence status: **VERIFIED** = executed, output file named. **NOT RUN** = not executed.

---

## 1. Experiments executed

| phase | experiment | script | output | status |
|---|---|---|---|---|
| 5 | synthetic validation of TI against a known answer | `scripts/ti_synthetic.py` | `results/ti_synthetic/v1_known_alpha.csv` | **VERIFIED** |
| 5 | synthetic shift severity × target size, exact truth | `scripts/ti_synthetic.py` | `results/ti_synthetic/v2_shift_and_size.csv` | **VERIFIED** |
| 6 | source-anchor perturbation, 9 estimators × 20 models | `scripts/ti_audit.py` | `results/ti_audit/ti_per_model.csv` | **VERIFIED** |
| 7 | target perturbation, same grid | `scripts/ti_audit.py` | `results/ti_audit/ti_per_model.csv` | **VERIFIED** |
| 8 | target-information removal (4 surrogates) | `scripts/ti_audit.py` | `results/ti_audit/ti_removal.csv` | **VERIFIED** |
| 9–13 | candidate Family F implemented, benchmarked, ablated | `scripts/ti_candidate.py` | `results/ti_candidate/candidate_summary.csv` | **VERIFIED** |
| 11 | unit tests + leakage firewall | `tests/test_ti_audit.py` | 21 tests | **VERIFIED**, suite 126 passing |

## 2. Experiments **not** run, and why

| not run | reason |
|---|---|
| Target sample-size sweep on the **agricultural** data (N = 25…400) | The TI audit uses the full 1,616-crop pool. A cluster bootstrap at N = 25 photographs would confound TI with sampling noise. Synthetic N = 100/400 was run instead. **Reported as a limitation, not as a result.** |
| Families A–E implemented | All reduce to shrinkage or ensembling (novelty audit §5). Family F, the most defensible, was implemented and **empirically rejected**; implementing four further variants of a rejected mechanism was judged waste, not rigour. **This is a decision, and it is stated rather than hidden.** |
| Non-agricultural target domains | Out of scope for compute. The framework is domain-independent by construction (it consumes only logits/features), but that is an **INFERENCE**, not a demonstration. |
| Scopus/WoS/IEEE/ACM verification of the Sobol equivalence | Requires institutional access. arXiv-only targeted check was run. |
| Rover-session demonstration | Depends on hardware not exercised here. |

## 3. Measurement caveats found during execution

**3a. The first synthetic world was degenerate — caught and fixed.** With class-mean
spread 1.5 the synthetic classifier reached accuracy 1.000 and the source accuracy had
bootstrap sd **0.0000**, so `V_S = 0` and TI collapsed to 1.0 for *every* probe including
`alpha = 0`. That was a measurement artefact. Spread was reduced to 0.35 (source accuracy
≈ 0.72, bootstrap sd 0.0194) and V1 then passed. **VERIFIED.**

This is also a **genuine limitation of TI**: the measure depends on how precisely the
anchor is determined, which is a property of the source sample size and where the accuracy
sits — not only of the estimator. An estimator audited against a very large or very
accurate source set will show a higher TI for that reason alone.

**3b. TI is a variance share, not a magnitude share.** On the known-α probe, `alpha = 0.5`
gives **TI = 0.078**, not 0.5, because TI compares variances. So a reported TI of 0.06 does
**not** mean "6% of the output comes from the target". The Phase-7b "93–98% intercept
share" was a *magnitude* decomposition and is a **different quantity**; the two must not be
quoted as if they measure the same thing.

**3c. "Shuffled = 0.0000" is not bitwise zero.** float32 summation is not associative; a
reordered mean differs by ~6e-8. Tests use a 1e-6 tolerance accordingly.

## 4. Leakage and validity

- The audit functions take **no argument** through which a target label could arrive;
  enforced by `test_audit_functions_never_take_target_labels`. **VERIFIED.**
- `src/estimators/ti_audit.py` passes the repository's existing AST import firewall
  (`tests/test_leakage_firewall.py`), which initially **rejected** it for importing
  `estimators.base` absolutely; fixed to a relative import. **VERIFIED.**
- The candidate's anchor is **source-validation accuracy**, computed from source labels
  only, which the protocol permits.
- True target accuracy is read **only** to score an estimate already produced.
- No hyperparameter was selected on target performance. The candidate's single free
  quantity, `w`, is *measured*, not tuned; the ablations fix it at 0.5 and 0.

## 5. What the executed work can support

- A validated measure of anchor-versus-target reliance for any label-free estimator
  (V1 passes: Spearman(α, TI) = 1.000, TI(0) = 0.0000, TI(1) = 1.0000).
- A measured audit of nine estimators on real agricultural lab→field data.
- A decisive **negative** result for TI-adaptive shrinkage.
- A mechanism explaining why the synthetic and real conclusions differ.

## 6. What it cannot support

- Any claim that TI is a **new metric** — it is a first-order Sobol' index.
- Any claim that TI predicts estimator reliability — **it does not**, and the sign of its
  association with error reverses between the two settings tested.
- Any claim about estimators outside the nine, domains outside agriculture, or target
  sample sizes below the full pool on real data.
- Any architecture-level claim beyond n = 5 seeds per backbone.
