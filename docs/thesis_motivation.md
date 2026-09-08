# Thesis Motivation and Research Framing (corrected)

**Supersedes the motivation used at handoff.** The original framing rested on a claim
about FBR-UDA that has been checked and found **false** (`docs/section16_verification.md`).
This document states the motivation without it.

Every statement below is tagged by evidence class, per the claim-discipline rule:

| Tag | Meaning |
|---|---|
| **[A]** | directly measured in this project |
| **[B]** | derived from measured results |
| **[C]** | interpretation |
| **[E]** | literature claim, with source |
| **[F]** | future work |

---

## 1. What was wrong with the previous motivation

The handoff framing asserted that FBR-UDA *"could not report field performance for lack
of labels."* **This is false. [E]**

Jeon, Kim, Choi & Yang, *"Bridging the Lab-to-Field Gap in Plant Disease Diagnosis
Through Unsupervised Domain Adaptation Enhanced by Background Recomposition"*
(Ecological Informatics, 2025) reports field accuracy throughout: a lab-trained baseline
at 47.1–89.3% across crops, CDAN+E with FBR at 91.1%, and a **Real-to-Real benchmark at
96.1–98.8%**. None of those numbers is obtainable without labelled real-field test data.

The phrase that was misread — adaptation "without requiring labelled real-field data" —
refers to **training**. That is simply what the *unsupervised* in unsupervised domain
adaptation means. It was never a statement about evaluation.

*Evidence limitation:* ScienceDirect and SSRN both returned HTTP 403, so this rests on
abstract and results text retrieved consistently across four independent searches, not
on a verbatim page-cited quotation. The evidence is one-directional and not close, but
it is not the citation standard originally sought. **[E]**

**Consequence:** the claim must not appear anywhere in the thesis. It is not a small
citation error — it was the load-bearing sentence of the motivation.

---

## 2. The corrected motivation

The replacement is stronger than the original, because it is *true* and because it
describes a structural problem rather than an oversight by one research group.

### 2.1 The deployment problem

A plant-disease classifier trained on laboratory imagery can report very high source
accuracy and still perform far worse in a real field. **[A]** In this project, over 30
trained models:

| Setting | Source accuracy | Target accuracy |
|---|---:|---:|
| PlantVillage held-out (no shift, control) | 0.972 | **0.969** |
| PlantDoc, 3 classes | 0.968 | **0.501** |
| Tomato-Village, real Indian field | 0.972 | **0.472** |
| PlantDoc, 7 classes | 0.991 | **0.281** |

The control arm establishes that the drop is a property of the domain shift and not of
the training pipeline. **[B]**

### 2.2 The measurement problem — the actual gap

Here is the point the false claim was reaching for, stated correctly:

> **The lab-to-field collapse is well documented, and every documented measurement of it
> required labelled field data. A deployed machine does not have labelled field data.**

This is not a criticism of prior work. Measuring a collapse *requires* ground truth, and
the field has done that properly. FBR-UDA is a good example rather than a
counter-example: it needed labelled field data at three separate points — to quantify
the baseline collapse, to demonstrate its method helped, and to construct a Real-to-Real
ceiling. **[E]**

But that creates an asymmetry with practical consequences: **the community knows the
collapse happens, and a deployed sprayer has no way to know whether it is happening
today, on this field, in this light.** The instrument used to establish the phenomenon
is precisely the instrument unavailable at deployment.

### 2.3 The research opportunity

A separate literature — label-free / unsupervised accuracy estimation (ATC, DoC, COT,
MaNo, nuclear-norm and similar) — estimates a fixed model's accuracy on an *unlabelled*
target set. **[E]** These methods are established on general vision benchmarks and, more
recently, on medical imaging (arXiv:2507.22776). **[E]**

The question this thesis asks is therefore an **applicability question**, not an
invention question:

> Do these estimators work well enough, under real agricultural laboratory-to-field
> shift and at deployment-session sample sizes, to tell a machine whether to trust
> itself today?

### 2.4 Why the answer matters either way

- If **yes**: a deployed sprayer gains a label-free self-check, and the estimate can gate
  operation.
- If **no**: practitioners are warned off a class of methods that look applicable and are
  not — and the failure mode itself becomes the scientific content.

**On the current evidence the answer is no. [B]** That is the thesis's principal
finding, and it is reported as such rather than reframed.

---

## 3. The gap statement, worded carefully

Novelty is claimed at the level the evidence supports, and no higher.

**What is claimed:**

> To the extent verified in the reviewed literature, existing label-free accuracy
> estimators have not been systematically benchmarked on plant-disease classifiers under
> real laboratory-to-field distribution shift, at deployment-session sample sizes, with
> an explicit analysis of failure modes and architecture dependence.

**What is NOT claimed:**

- ✗ that no one has ever applied label-free accuracy estimation to agriculture — the
  search performed here is a **web screen, not a Scopus/IEEE/ACM database search**, so
  completeness cannot be asserted;
- ✗ first application of label-free accuracy estimation to a new domain — medical
  imaging did this in 2025 (arXiv:2507.22776); **[E]**
- ✗ a new estimator, confidence score, or classifier;
- ✗ first reliability-aware or self-assessing agricultural robot — introspective
  perception predates this by a decade (Daftry 2016). **[E]**

**Contribution type: empirical benchmark.** Every estimator is imported.

### 3.1 Position relative to the nearest neighbour

arXiv:2507.22776 (chest X-ray) is the closest work and makes the same *genre* move.
Differences established by reading it in full: **[E]**

| | arXiv:2507.22776 | This thesis |
|---|---|---|
| Domain | chest X-ray | tomato leaf disease |
| Prevalence shift | simulated | naturally occurring |
| Evaluation-set size | 1,000–22,000 images, never varied | **N = 25–400, swept** |
| Deployment sessions | not considered | the unit of analysis |
| Decision gating | not considered | operate/withhold, lift over base rate |

Its finding that all estimators struggled under prevalence shift **independently
corroborates** what is observed here in an unrelated domain. **[E]**

---

## 4. Research questions

**RQ1.** How accurately can existing label-free accuracy estimators estimate the
performance of plant-disease classifiers under laboratory-to-field distribution shift?

**RQ2.** How does estimator reliability vary across target domains, classifier
architectures, and types of agricultural shift?

**RQ3.** How does the number of unlabelled deployment observations affect the error and
stability of the estimate?

**RQ4.** What measurable failure modes cause these estimators to become unreliable under
agricultural shift?

**RQ5.** How much labelled field data would be required to beat the best label-free
estimate? **[F — not yet run]**

*RQ6 (a corrected estimator) is deliberately omitted.* It will be added only if the RQ4
diagnostics identify a specific systematic failure mode addressable without target
labels. It is not carried as an aspiration.

---

## 5. The operational abstraction

This constrains what may be claimed anywhere in the thesis.

```
N unlabelled field observations
        ↓
frozen disease classifier
        ↓
logits / probabilities / features
        ↓
label-free accuracy estimator
        ↓
estimated SESSION accuracy
        ↓
deployment gate: operate / withhold / request review
```

**Never:** `single image → estimator → "this image is correct"`.

These estimators are dataset-level. Per-image correctness is selective prediction — a
different literature with different guarantees. The code enforces this: `SessionGate`
raises `PerImageQueryError` on any per-image query. **[A]**

---

## 6. Honest statement of the finding

For use in the abstract and conclusion, worded to survive a viva:

> Under the evaluated agricultural laboratory-to-field shifts, the tested confidence-based
> label-free accuracy estimators did not reliably track plant-disease classifier accuracy.
> On the no-shift control they were accurate to within 0.013 MAE, confirming the
> implementations are correct; on real field data they over-estimated accuracy by up to
> 0.46 and their correlation with true session accuracy fell close to zero. The failure
> persisted across four classifier architectures, was not repaired by increasing the
> number of unlabelled observations within the evaluated range, and was not repaired by
> source-fitted temperature scaling. **[B]**

Note the qualifiers, all load-bearing: *evaluated* shifts, *tested* estimators,
*within the evaluated range*, *four architectures*. The result is a strong empirical
finding about specific methods under specific conditions — not a universal impossibility
proof, and it must never be written as one.
