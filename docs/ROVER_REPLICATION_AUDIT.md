# Rover Replication Audit

**2026-09-11.** Deep search for functional equivalents of "agricultural robot that gates its
own autonomous action on an estimate of its perception model's reliability".

Search: `scripts/litsearch_rover_replication.py`, 46 paired queries across arXiv, OpenAlex
and Crossref, 10,033 records; plus four sharper arXiv probes. Log:
`results/litsearch/rover_replication_query_log.csv`.

Levels: 0 none · 1 component · 2 conceptual · 3 methodological · 4 functional · 5 direct.

---

## 1. Headline

**No Level 4 or Level 5 replication identified in the sources searched.**
**A genuine Level 2–3 body of work exists — but in automotive and rail, not agriculture.**

## 2. Findings by level

### Level 2–3 — runtime perception monitoring (automotive / rail)

arXiv `"runtime monitoring" AND "perception" AND "autonomous"` → **15 records**, including:

| work | year | what it does | why it is not us |
|---|---|---|---|
| *Monitoring of Perception Systems: Deterministic, Probabilistic, and Learning-based Fault Detection* | 2022 | fault detection for perception pipelines | fault/consistency detection, not accuracy estimation on unlabelled data |
| *PerceMon: Online Monitoring for Perception Systems* | 2021 | temporal-logic monitors over perception streams | formal specification monitoring; no performance estimate |
| *Certified Control: An Architecture for Verifiable Safety of Autonomous Vehicles* | 2021 | a checker that certifies each control action | per-action certification via a verifiable checker, not a session-level accuracy estimate |
| *A Method for the Runtime Validation of AI-based Environment Perception in Automated Driving* | 2024 | runtime validation of perception | driving domain; validation against redundant sensing, not label-free estimation |

**This is the strongest conceptual prior art.** The idea "monitor the perception model at
runtime and restrict autonomy when it looks unreliable" is **established** in safety
engineering. Our distinguishing features are the *mechanism* (dataset-level label-free
accuracy estimation) and the *domain* (agriculture) — not the idea itself.

### Level 2 — agricultural cross-domain reliability

*From vineyard to vision: Multi-domain analysis and mitigation of grape cluster detection
failures* (Results in Engineering, 2025, 28 cites). Cross-dataset evaluation of six
detectors across vineyards in China, Brazil and Italy; introduces a False Classification
Rate argued to reflect reliability better than mAP; supplies a decision matrix linking
vineyard tasks to detector choice.

**Closest agricultural work.** But every metric it uses — F1, mAP, FCR — **requires
labels**. It is labelled cross-domain evaluation plus deployment guidance. It does not
estimate performance without target labels. **Level 2, not a replication.**

### Level 0–1 — everything else

`"safe autonomy" AND "agriculture"` → **0** on arXiv. `"agricultural robot" AND
"label-free accuracy estimation"` → 0 on arXiv, 2 on OpenAlex (both unrelated on reading).
`"agricultural rover"` × {runtime monitoring, runtime assurance, safety monitor, perception
monitoring} → **0 on arXiv for every pairing**. The broad ROBOT × AGRI × RELIABILITY screen
returned 55 candidates, of which the overwhelming majority are irrelevant on reading (a
water-treatment handbook, Holland's genetic-algorithms book, GNSS/RTK localisation, yield
forecasting, harvesting arms).

## 3. Verdict

| question | answer |
|---|---|
| Direct replication (L5) | **No** — none identified |
| Functional replication (L4) | **No** — none identified |
| Methodological overlap (L3) | **Yes** — automotive/rail runtime perception monitoring |
| Conceptual overlap (L2) | **Yes** — the same, plus agricultural cross-domain reliability with labels |
| Obvious-combination risk | **MEDIUM-HIGH** — "take runtime perception monitoring, apply it to an agricultural robot, use a label-free estimator as the signal" is a combination a reviewer can describe in one sentence |

**Wording this supports:** *"No prior work gating agricultural autonomous action on
label-free accuracy estimation was identified in the sources searched."* Not "first", and
not "novel concept" — the concept is established elsewhere.

## 4. Limitation

Web of Science and the ACM Digital Library were **not searched** (IEEE Xplore and Scopus have
since been searched, 2026-09-11 and 2026-09-18) — no
institutional access. IEEE in particular is where agricultural-robotics and safety
engineering publish heavily, so this audit's coverage of exactly the relevant venues is its
weakest point. Recorded, not glossed.
