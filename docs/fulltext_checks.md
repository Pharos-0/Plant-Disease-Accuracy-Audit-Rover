# Full-Text Checks on the Two Papers That Threatened the Framing

**2026-09-28.** The campus-network handoff named two unread papers as the highest-value
remaining pre-submission work: ODP-Bench, "the paper that most threatens the framing", and
GBCE, an unread Scopus record that might sit inside our protocol. Both are now checked.
This document records what was actually read, and what could not be.

---

## 1. ODP-Bench — FULL TEXT READ, framing survives

**Yu, Li, Li, He, Zhang & Cui, "ODP-Bench: Benchmarking Out-of-Distribution Performance
Prediction", ICCV 2025, pp. 1846–1858.** Also arXiv:2510.27263.

Obtained the arXiv PDF and extracted text from all **17 pages** (71,140 characters). The
ICCV Open Access PDF and the ACM mirror both returned HTTP 403; arXiv served the same
paper, title and author list matching.

### Keyword census over the complete text

| term | occurrences |
|---|---:|
| `sobol` | **0** |
| `variance decomposition` | **0** |
| `sensitivity analys` | **0** |
| `decompos` | **0** |
| `ablation` | **0** |
| `agricultur` / `plant` / `crop` / `leaf` | **0** / **0** / **0** / **0** |
| `MAE` / `mean absolute error` | **0** / **0** |
| `rank correlation` | **25** |
| `labeled sample` / `labelled sample` / `label budget` | **0** / **0** / **0** |

### What this settles

**The anchor-versus-target decomposition is not in ODP-Bench.** This was the one check the
risk assessment said "could still affect the lead contribution". It does not. There is no
decomposition, no variance-based sensitivity analysis, and no ablation of what an estimate
responds to, anywhere in the paper.

**ODP-Bench measures ordering, not distance.** Verbatim: *"The adopted metric is Spearman's
rank correlation between predicted scores and ground truth performances."* It also
explicitly declines the alternative: *"We do not choose coefficient of determination R² as
the main evaluation metric since it is quite sensitive to outliers and it cannot address the
nonlinear correlation."* So ODP-Bench asks whether an estimator ranks models and datasets
correctly. We ask how far an estimate is from the truth, in absolute error, against an
oracle constant. These are different questions, and the difference is now quotable.

**No agricultural data of any kind.** Zero hits for agriculture, plant, crop or leaf across
17 pages. It uses CIFAR-10, CIFAR-100-C, ImageNet and related benchmarks.

**Its algorithm set overlaps but does not coincide with ours.** From Figure 1: ATC, Nuclear
Norm, DOC, NI, MaNo, Dispersion, Agreement, COT, COTT. Ours: ATC-MC, ATC-NE, confidence,
COT, DoC, entropy, GdScore, MaNo, nuclear norm. Overlap is substantial — which is
reassuring, since it means we drew from the field-standard set rather than an idiosyncratic
one — but ODP-Bench does not include GdScore, and we do not include COTT, NI or Agreement.

**Their headline finding is consistent with ours and worth citing as such**: *"most of
current algorithms do well in synthetic corruptions, but generally fail in natural shifts."*
Our laboratory-to-field collapse is a natural shift, and our estimators fail there too. That
is independent corroboration from a benchmark built on entirely different data.

### Manuscript consequence

The distinguishing sentence is now written into Section II of both versions, replacing the
weaker unverified claim that ODP-Bench "does not cover" our territory. The full version also
records the metric difference and the GdScore exclusion.

---

## 2. GBCE — INELIGIBLE, full text not obtainable

**Matsuno, Ryuta. "GBCE: Enhanced Training Loss to Estimate Accuracy of Models in
Production." Advances in Knowledge Discovery and Data Mining (PAKDD 2025), LNCS,
pp. 29–40. Springer Nature Singapore.** DOI `10.1007/978-981-96-8180-8_3`.

### What was verified, and how

| item | source | status |
|---|---|---|
| Author, title, series, pages 29–40, year 2025, publisher | Crossref API | **VERIFIED** |
| DBLP key `conf/pakdd/Matsuno25`, PAKDD 2025 Sydney | Semantic Scholar Graph API | **VERIFIED** |
| Abstract | Semantic Scholar: `null`, *"elided by the publisher"* | **NOT OBTAINED from publisher** |
| Full text | Springer → `idp.springer.com` auth redirect; ACM DL 403; ResearchGate 403 | **NOT READ** |
| Abstract content | as surfaced by web search only | **UNVERIFIED SECONDHAND** |

### Eligibility classification: INELIGIBLE

On the abstract content available, GBCE proposes a *guided binary cross-entropy loss* for
training a dedicated **check model** whose job is to check predictions in order to estimate
accuracy, in a production setting where **target labels arrive with delay**. Two independent
grounds place it outside this study's protocol:

1. **It is a training-time intervention.** Our protocol takes an already-frozen,
   off-the-shelf classifier and touches nothing about how it was trained. A method requiring
   a new loss and an auxiliary trained model cannot be applied post hoc.
2. **Its setting is delayed labels, not absent labels.** Labels eventually arrive in its
   setting, which is the regime that makes a check model trainable at all. In agricultural
   deployment the labels never arrive.

This is the same class of exclusion already applied to AETTA (assumes a test-time-adapting
model) and Mandoline (assumes slice specifications). It is recorded in
`docs/estimator_selection.md` terms as a **principled exclusion, not ignorance**.

**The honest limitation:** classification rests on the abstract, not the full text. The
manuscript says so — *"We were unable to obtain its full text"* — and the bib entry carries
the same note. If institutional access becomes available, read it; nothing in the paper
depends on the outcome, because it is cited only to be excluded.

---

## 3. One paper newly cited as a result of this check

**Deng, Tu, Radwan, Abu Alsheikh, Gould & Zheng, "Confidence and Dispersity as Signals:
Unsupervised Model Evaluation and Ranking", arXiv:2510.02956, 3 October 2025.** Verified
through the arXiv API: id, title, six authors, date and abstract all retrieved.

It unifies two settings: estimating one model's accuracy across many unlabelled test sets,
and **ranking candidate models on a single unlabelled test set**. The second is the mirror
image of our session-ranking analysis, where one fixed model is applied to many unlabelled
subsets. It reports the nuclear norm of the prediction matrix as the most robust signal.

Two reasons this had to be cited rather than ignored. First, a reviewer who knows it would
otherwise ask why our ranking result is not positioned against it. Second, it reaches a more
optimistic conclusion than we do, and the manuscript now says so and attributes the
difference to shift severity rather than claiming a disagreement about the signals. Our
per-model paired session-ranking correlation is 0.076 at worst and 0.224 at best; theirs are
strong. Under a shift that takes accuracy to 0.50, ranking does not survive.

---

## 4. What remains unread

| paper | why it matters | blocker |
|---|---|---|
| GBCE full text | confirm the eligibility exclusion | closed access |
| Tomato-Village source paper | acquisition conditions | Springer paywall (`10.1007/s00530-023-01158-y`) |
| The 11 IEEE Class-A papers, full text | the same decomposition check ODP-Bench just passed | metadata-scope search only; full texts not retrieved |

The third is the one that still has research value. ODP-Bench was the strongest single
threat and it is now cleared; the residual risk is spread thinly across eleven papers rather
than concentrated in one.
