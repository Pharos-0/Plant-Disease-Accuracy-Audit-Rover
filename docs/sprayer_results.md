# Autonomous Selective Pesticide Sprayer — Implementation and Measured Results

**2026-10-13.** The sprayer is built as working, tested software: an FSM controller with
sensor fusion and a vision-guided selective spray policy, run against real imagery. Every
number below is read from `results/sprayer/`. **This is a simulation over real imagery, not a
field trial.**

---

## 1. Why the target figures are not reproduced

The requested chart and table carried 92.4% navigation accuracy, 89.1% spraying precision,
41.7% pesticide reduction and 18.6 Wh/run. Those are the figures from the third-party paper
that `docs/rover_code_audit.md` established had **no code path that could have produced any
of them**, and that `literature/final_audit/REFRAME.md` lists under REMOVE. They are not used.

What follows is what the system actually does when measured.

## 2. What was built — the five contributions as implemented

| # | Contribution | Where | Evidence |
|---|---|---|---|
| 1 | Integrated autonomous sprayer | `rover/sprayer_fsm.py`, `rover/rover_v1.py` | runs end to end in the digital twin |
| 2 | FSM control architecture | `State`, `TRANSITIONS`, `SPRAY_OK` | spraying is permitted in exactly **one** of nine states; illegal transitions raise |
| 3 | Multi-sensor fusion | `FusedReading` | four physically different inputs, each with its own failure response — a stale ultrasonic reading stops the platform, a stale vision reading only disables spraying |
| 4 | Vision-guided selective spraying | `nozzle_decision` | per-nozzle canopy detection, disease classification, and a reliability gate that degrades to vision-only rather than to blanket |
| 5 | Modular architecture | controller consumes a fused reading and emits a command, with no branch on whether it runs on hardware or in simulation | the same controller drives the real GPIO layer and the twin |

## 3. How the measurement is composed

No available dataset holds both rover-viewpoint geometry and disease labels, so two real
sources are combined, and the combination is a modelling assumption:

- **Traverse geometry** — AgRobTomato (Zenodo `10.5281/zenodo.5596799`): **446 frames captured
  by a ground robot driven along greenhouse inter-rows**. Canopy in each of four nozzle zones
  is measured from these real frames.
- **Disease state and classifier behaviour** — Tomato-Village, **Jodhpur and Jaipur districts,
  Rajasthan, India**: true labels and the real outputs of the twenty frozen classifiers.

Each frame is paired with a randomly drawn India plant, so the plant in the frame is not the
plant whose disease state is used. Twenty classifiers × ten random pairings = **200
traversals per policy**.

## 4. Table I — measured

| Parameter | Blanket (manual) | Vision-only autonomous | **Proposed selective** |
|---|---:|---:|---:|
| Pesticide use, % of blanket | 100.0 | 96.6 | **95.5** [95.1, 95.8] |
| Pesticide reduction | 0.0% | 3.4% | **4.5%** |
| Diseased zones that received spray | 100.0% | 100.0% | **98.9%** [98.5, 99.2] |
| Spray landing on non-target zones | 16.3% | 13.4% | 13.2% |
| Navigation accuracy | not measured | not measured | not measured |
| Energy per run (Wh) | not measured | not measured | not measured |
| Autonomous return | operator | FSM: 500/500 docked\* | FSM: 500/500 docked\* |
| Fault handling | operator | FSM safe stop: 600/600\* | FSM safe stop: 600/600\* |

95% bootstrap intervals in brackets. \*Vision-only and the proposed policy share one
controller, so these rows are identical by construction; they are properties of the control
software in simulation and say nothing about whether a physical relay opens.

**Navigation accuracy and energy are not measured** because the infrared row sensor and the
battery are simulated. A number for either would describe the simulator.

## 5. Why the reduction is 4.5%, not 41.7%

A selective sprayer saves pesticide in exactly two ways: by not spraying bare ground, and by
not spraying healthy plants. Both are small here.

- **96.6% of nozzle zones contain canopy.** The traverse is a dense greenhouse row; there is
  almost no bare ground to skip. That is the whole of vision-only's 3.4%.
- **86.6% of the India plants are diseased.** There are few healthy plants to skip.

Feeding the controller the **true** disease state gives the ceiling — the most any selective
sprayer could save on this traverse: **16.3%**. The proposed sprayer realises **4.5% of that
16.3%, or 28% of what was achievable**.

**41.7% is not reachable on this data by any sprayer, however good.**

## 6. The skips are genuinely targeted

Of the zones the proposed policy skipped that vision-only would have sprayed, **52.3% truly
did not need spray**. Random skipping would give the healthy prevalence, **13.4%**. The
classifier's spray/no-spray decisions are **3.9× better than chance** — a real positive result,
and a more useful one than the 3-class accuracy of roughly 0.50 suggests, because most of that
classifier's errors confuse early and late blight, which does not change the decision to spray.

The cost is visible in panel (b) of Fig. 11: coverage falls from 100% to 98.9%. A sprayer that
saves pesticide by missing disease is not better, and the figure reports both so that neither
hides the other.

## 7. Safety campaign

| Test | Result |
|---|---|
| Randomised fuzz over fused readings | **0 violations** in 160,000 commands, of which **5,998 were spray commands** and **1,497 met an obstacle mid-spray** |
| Fault injected while spraying | **600/600** reached FAULT_SAFE_STOP with pump off and stayed latched |
| Low resource during a run | **500/500** returned and docked without spraying on the way |

The fuzz's first version produced only 372 spray commands, because faults fired so often that
nearly every run latched within a few steps. A zero over that denominator says little, so the
fault rate was lowered until the dangerous states were actually exercised.

## 8. What would change these numbers

The reduction is governed by the bare-ground fraction, and this traverse is a dense greenhouse
row. An open-field tomato row early in the season has substantial bare soil between plants,
which is where commercial spot-spraying savings come from. The right data for measuring that
is already registered: **SRC006 GROUNDBASED_WEED** — open-field tomato in Spain, photographed
**top-down from 1–1.5 m**, the viewpoint a downward spray boom actually has, CC BY 4.0 — but it
is about 10 GB and has not been downloaded. Running the same experiment on it would measure the
open-field figure. Whatever it returns is the number to report.

## 9. Reproduce

```bash
python scripts/run_sprayer_experiment.py   # 800 traversals incl. the oracle ceiling
python scripts/run_sprayer_safety.py       # fuzz, fault injection, autonomous return
python scripts/make_sprayer_figure.py      # Fig. 11
python -m pytest tests/test_sprayer_fsm.py -q
```

---

## 10. Relation to `rover-v2/` — two simulations, and why they disagree

A separate workstream built `rover-v2/`, a field-row simulation with navigation physics,
deposition modelling and 20 paired seeds per condition. Its `results/tables/TABLE_I_MEASURED.md`
reports the proposed v2 system at **+28.5% ± 4.8** pesticide reduction against manual. This
document reports **4.5%**. Both are honest measurements of different things, and they should
not be merged or averaged.

| | `rover-v2` | this document |
|---|---|---|
| Imagery | **rendered** field row, flat colours | **real** rover-captured frames (AgRobTomato) |
| Disease decision | HSV gate on rendered plants | the **twenty frozen classifiers**, on India (Tomato-Village) plants |
| Geometry | one 10 m open row with gaps | dense greenhouse inter-row, 96.6% canopy |
| Covers | navigation, deposition, energy model, collision, return | pesticide use, disease coverage, FSM safety |

The gap is the informative part. `rover-v2/docs/METHOD.md` §5 item 3 already warns that its
vision results come from rendered images on which the HSV gate "almost certainly performs better
than it will on a real plant", and asks readers to *"treat every detection-derived figure as an
upper bound."* This experiment is an independent check of that warning on real imagery with a
real classifier, and the saving does shrink — mostly because a dense real canopy leaves little
bare ground to skip, which is the dominant source of spot-spraying savings.

**Recommended use:** `rover-v2`'s Table I is the system table — it is the only one covering
every row. This document's result belongs beside it as the real-imagery reality check, and
§5's ceiling argument explains the difference.

---

## Added 2026-10-13: a bias this simulation could not see

A later dry run with **real diseased leaves** (`docs/camera_experiment_protocol.md` §8) found
that a green-pixel plant test is **blind to severe disease**: late-blight leaves average only
0.20 green, and 38% of them showed no zone ≥ 15% green. The sicker the plant, the less likely
a green test is to spray it.

This simulation could not reveal that, because its disease state was borrowed from another
dataset and did not change any plant's colour. **Wherever a result here relies on green
detection — the vision-only baseline, and canopy detection — real missed treatment of
diseased plants would be higher than the coverage figures above.** The camera demo now detects
leaf tissue of any plant colour; the simulations have not been re-run with that change.
