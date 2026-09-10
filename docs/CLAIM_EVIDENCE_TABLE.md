# Claim–Evidence Provenance Table

**2026-09-11.** Every major thesis claim, its evidence, its source, its status, and the
wording permitted. Status vocabulary: **VERIFIED EXPERIMENTALLY** · **SOURCE-SUPPORTED** ·
**MODEL INFERENCE** · **HYPOTHESIS** · **UNKNOWN**.

| # | Claim | Evidence | Experiment / source | Status | Allowed wording |
|---|---|---|---|---|---|
| 1 | Classifier accuracy collapses lab→field | source_test 0.962 → PlantDoc 0.497, Tomato-Village 0.500 | `results/gate_evaluation/gate_sessions.csv` | **VERIFIED** | "accuracy fell from ≈0.96 to ≈0.50 on the two field arms tested" |
| 2 | Best estimator MAE is 0.134 on the full pool | entropy 0.1337, cot 0.1339 | `results/rq7_gdscore/rq7_comparison_full_pool.csv` | **VERIFIED** | "the best of the nine achieved MAE 0.134 in this setting" |
| 3 | No estimator beats an oracle constant | oracle constant 0.0610 vs min estimator 0.1337 | `docs/rq7_gdscore_results.md` §6 | **VERIFIED** | "none outperformed an oracle constant, which is a diagnostic ceiling, not a deployable baseline" |
| 4 | Six of nine estimators are anchor-dominated | anchor/target sd ratio 4–14× | `results/ti_audit/ti_summary.csv` | **VERIFIED** | "moved 4–14× more under source resampling than target resampling" |
| 5 | Some estimators barely react to a random target | COT 0.062, MaNo 0.065, GdScore 0.073 | `results/ti_audit/ti_removal_summary.csv` | **VERIFIED** | "changed by only 0.06–0.11 when the target was replaced by random logits" |
| 6 | **The TI–error relationship reverses with shift severity** | +0.400/+0.436/+0.453 vs −0.28 to −0.51; paired p = 1e-4 to 1e-6, n = 100/cell | `results/ti_sample_size/table3_*.csv`, `table4_*.csv` | **VERIFIED** | "the sign of the relationship reversed between the zero-shift and both severe-shift arms, at every feasible N" |
| 7 | The reversal holds at deployment-scale N | present at N = 25, 50, 100 (and 200 where available) | `table4_reversal_by_N.csv` | **VERIFIED** | "present at every feasible sample size, including N = 25" |
| 8 | More unlabelled data does not help under shift | Tomato-Village +0.0137, 2/9 improved; zero-shift −0.0110, 8/8 | `results/ti_sample_size/sample_size_errors.csv` | **VERIFIED** | "additional unlabelled target data produced little reduction in error under the severe-shift conditions tested" |
| 9 | TI is not comparable across N | target sd falls 2.5–3.0×, anchor sd flat (1.00–1.17×) | `t07_target_vs_anchor_sensitivity.csv` | **VERIFIED** | "TI must be compared only at fixed N" |
| 10 | TI recovers a known answer | Spearman(α,TI)=1.000; TI(0)=0.0000, TI(1)=1.0000 | `results/ti_synthetic/v1_known_alpha.csv` | **VERIFIED** | "the measure recovered a known target dependence exactly" |
| 11 | **TI is a first-order Sobol' index** | derivation | `docs/NEW_ESTIMATOR_NOVELTY_AUDIT.md` §4 | **SOURCE-SUPPORTED** (Sobol' 1993) | "a first-order Sobol' index over two input groups — established mathematics, not a new metric" |
| 12 | TI does not predict reliability | sign reverses; no significant association within arms | `results/ti_sample_size/`, `ti_audit/ti_vs_error.csv` | **VERIFIED** | "TI did not predict estimator reliability and cannot serve as a deployment gate" |
| 13 | TI-adaptive shrinkage fails | 0/9 improved, 7 significantly harmed (Wilcoxon, Holm) | `results/ti_candidate/candidate_stats.csv` | **VERIFIED** | "the candidate correction improved none of the nine bases and significantly harmed seven" |
| 14 | No signal ranks field sessions usefully | best per-model ρ = 0.224 (entropy); GdScore 0.076 | `results/gate_evaluation/`, `t10_session_ranking.csv` | **VERIFIED** | "no tested signal usefully ranked field sessions within a fixed model" |
| 15 | The pooled ranking result is a Simpson artefact | pooled GdScore ρ = 0.696 vs paired 0.076; entropy beats it, p = 8.3e-05 | `docs/ROVER_RESEARCH_INTEGRATION.md` §2 | **VERIFIED** | "pooling across models inverted the ranking; the per-model paired result is the one that answers the deployment question" |
| 16 | A gate cannot enable autonomy here | at τ=0.80, 0.2% (PlantDoc) / 0.1% (Tomato-Village) of sessions safe | `t11_fraction_sessions_safe.csv` | **VERIFIED** | "essentially no field session met a plausible operating threshold, so the limiting factor is the classifier, not the gate" |
| 17 | Domain-shift detection is easy | pooled AUC 0.80–0.98 | `results/gate_evaluation/` | **VERIFIED** | "separating laboratory from field sessions was easy — but that is domain-shift detection, not accuracy estimation" |
| 18 | ~20–50 labelled photographs beat label-free here | 10 photos → MAE 0.1255 < 0.1337; 50 → 0.0513 | `results/rq5/rq5_settingA_labelled.csv` | **VERIFIED** | "practical evidence for this setting, not a universal labelling requirement" |
| 19 | The labelling result is consistent with PPI theory | saving scales with R²; measured R² = 0.026 | *Power Analysis for PPI* (2026) + `rq7_correlation.csv` | **SOURCE-SUPPORTED + VERIFIED** | "consistent with prediction-powered-inference theory given the measured R²" |
| 20 | GdScore did not transfer here | full-pool MAE 0.1599, calibration r ≈ 0, R² = 0.026 | `docs/rq7_gdscore_results.md` | **VERIFIED** | "GdScore did not transfer reliably **in this experimental setting**" — never "GdScore is poor" |
| 21 | GdScore's behaviour is consistent with BatchNorm train-mode | train 0.052–0.102 vs eval 0.446–0.488 on BN backbones; ViT control unchanged (0.0011) | `results/rq7b_batchnorm/` | **VERIFIED (consistency only)** | "consistent with the hypothesis that train-mode BatchNorm statistics contribute" — never "causes" |
| 22 | No prior target-responsiveness audit found | 22 paired queries, 3 databases, 45 candidates read | `results/litsearch/ti_novelty_*.csv` | **VERIFIED (search), UNKNOWN (existence)** | "no prior work … was identified in the sources searched" — never "first" |
| 23 | No agricultural robot gates autonomy on label-free estimation | 46 paired queries; arXiv 0 for every "agricultural rover" × monitoring pairing | `docs/ROVER_REPLICATION_AUDIT.md` | **VERIFIED (search), UNKNOWN (existence)** | as above |
| 24 | Runtime perception monitoring is established elsewhere | 15 arXiv records (automotive/rail) | `docs/ROVER_REPLICATION_AUDIT.md` §2 | **SOURCE-SUPPORTED** | "the concept is established in automotive and rail; our mechanism and domain differ" |
| 25 | Camera probe passed a blind camera | 60 frames, per-frame std 0.0000, every pixel = 2 | measured 2026-09-10; fixed, `tests/test_camera_blank_detection.py` | **VERIFIED** | "a measured defect, now fixed and regression-tested" |
| 26 | Disease → chemical treatment | — | none | **UNKNOWN** | **no claim permitted**; downstream action stays abstract |
| 27 | Safe autonomous spraying demonstrated | — | no field trial run | **UNKNOWN** | **no claim permitted** |

## Claims with no evidence, and therefore barred

- ✗ a new estimator · ✗ a new metric · ✗ "first" anything · ✗ per-image correctness
- ✗ "label-free estimation fails in general" (COT reaches 0.038 on ViT)
- ✗ universal agricultural performance from one crop · ✗ validated treatment mapping
- ✗ TI as a safe deployment gate · ✗ "first systematic evaluation" (ODP-Bench exists)
