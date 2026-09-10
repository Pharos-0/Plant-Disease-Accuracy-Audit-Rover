# Can This Project Reach 9/10?

**2026-09-11.** Scored against executed evidence only.

---

## 1. Dimension scores

| # | dimension | score | basis |
|---|---|---:|---|
| A | Research question quality | **8** | well-posed, answerable without labels, practically motivated |
| B | Literature gap | **6** | real but narrow: the genre exists in 5 other domains, ODP-Bench exists, runtime perception monitoring is established in automotive |
| C | Novelty | **5** | no new estimator, no new metric (TI is a Sobol' index); the *application* of the audit is unmatched in sources searched |
| D | Experimental rigor | **9** | 9 estimators × 4 architectures × 5 seeds × 3 arms × 4 sample sizes; pre-registration; cluster sampling; 4 self-caught bugs |
| E | Statistical rigor | **8** | paired Wilcoxon, bootstrap CIs, Holm; capped by n = 3 datasets |
| F | Technical depth | **7** | L3 GdScore reproduction verified against autograd, train-mode cache with buffer restoration, AST leakage firewall |
| G | Reproducibility | **9** | manifests with git commit/seeds/checksums, 126 passing tests, cached artefacts, one-line commands |
| H | Agricultural relevance | **7** | real Indian field imagery, real deployment question |
| I | Robotics/deployment relevance | **4** | the gate demonstrably cannot enable autonomy here |
| J | Practical impact | **6** | the negative results are actionable: don't trust these estimators; more unlabelled data won't help; label 20–50 photographs |
| K | Thesis coherence | **8** | one question, consistent methodology, honest narrative |
| L | Publication potential | **5** | a workshop/short paper on the audit + negative results is plausible; not a top venue |

**CURRENT PROJECT SCORE: 7.0/10**
**CURRENT NOVELTY SCORE: 5/10**
**CURRENT THESIS QUALITY: 7.5/10**
**CURRENT ROVER QUALITY: 4/10**

## 2. Candidate paths to 9/10

| # | candidate | novelty | difficulty | cost | risk | data available? | likely meaningful? | obvious-combination risk | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Reliability-aware deployment framework | low-med | med | med | **high** | yes | **no** — §Rover integration shows the gate cannot work | **HIGH** | **reject** |
| 2 | Robust model-selection protocol | low | low | low | med | yes | maybe | high | reject |
| 3 | Multi-signal reliability gate | low | med | med | high | yes | no — signals are individually weak (best ρ 0.22) | **HIGH** | reject |
| 4 | Failure-aware decision framework | low | med | med | high | yes | no | high | reject |
| 5 | Target-shift severity estimation | med | med | med | med | yes | **plausible** — shift severity governs the reversal | med | **keep as secondary** |
| 6 | Label-free deployment-readiness assessment | med | med | med | med | yes | partly — reduces to shift detection | med-high | reject as primary |
| 7 | Human-verification fallback | very low | low | low | low | yes | no | very high | reject |
| 8 | Selective autonomy | low | med | med | high | yes | no | high | reject |
| 9 | **Cross-domain reliability benchmark** | med | **low** | **low** | **low** | **yes** | **yes** | med | **best available** |
| 10 | New estimator | — | — | — | — | — | **already failed**: 0/9 improved, 7 harmed | — | **reject** |

## 3. The honest answer

**9/10 is not reachable from here without new data or a working method.**

The ceiling is set by three things that no amount of additional analysis fixes:

1. **Novelty is capped at ~5–6.** No new estimator (one was tried and failed decisively), no
   new metric (TI is a Sobol' index), and the deployment-monitoring concept is established
   in automotive/rail.
2. **The deployment story is negative.** The gate cannot enable autonomy because the
   classifier is ≈0.50 in the field. That is an honest and useful finding, but it does not
   produce a strong positive contribution.
3. **External validity is n = 3 arms, one crop, one source dataset.** Every architecture-level
   claim rests on 5 seeds.

**Maximum defensible score: ≈8/10**, reached by consolidating what exists rather than adding
anything.

## 4. What would actually be needed for 9/10

Both of these, not either:

- **A second source dataset and a second crop** — so the reversal and the anchor-reliance
  findings are shown to generalise beyond tomato/PlantVillage. This is the single largest
  lever on external validity, and it requires training a new model family (GPU cost:
  substantial; the existing 20 models took multiple stages).
- **A reliability signal that works within a domain.** The best current within-field ranking
  is ρ = 0.22. Something reaching ρ ≈ 0.5 would make a deployment gate meaningful. There is
  no evidence any existing estimator does this, and building one has already failed once.

Neither is a small addition, and the second may not be achievable at all. **Pursuing them to
chase a score is not recommended.**

## 5. Recommended path: consolidate to ~8/10

Add nothing. Reframe what exists as what it is — a rigorous negative characterisation with
three concrete, transferable findings:

1. The TI–error relationship **reverses with shift severity**, at every feasible N
   (p ≈ 1e-4 to 1e-6, paired, n = 100 per cell).
2. **More unlabelled target data does not help under shift** (+0.0137 MAE on Tomato-Village,
   2/9 estimators improved) while it helps everything under zero shift (8/8).
3. **No label-free signal ranks field sessions usefully** for a fixed model (best ρ = 0.22),
   so a deployment gate is not currently viable — and the deployable alternative is a small
   labelled sample.

Plus the methodological record: four self-caught bugs, three Simpson reversals, a failed
candidate estimator reported in full.
