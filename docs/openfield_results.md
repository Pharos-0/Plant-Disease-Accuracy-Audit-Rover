# Open-Field Sprayer — Results

**2026-10-13.** Run under the protocol in `docs/openfield_protocol.md`, committed at 11:28
**before the data existed**, with deviations D1–D7 committed before any spray result. Every
number below is read from `results/openfield/`. **Simulation over real imagery — not a field
trial.**

---

## 1. Headline (pre-registered 4-nozzle layout, 60 held-out images)

| | Manual (blanket) | Vision-only (green) | **Proposed (detector)** | Ceiling (perfect) |
|---|---:|---:|---:|---:|
| **Pesticide use** | 100.0% | 9.2% [6.2, 12.1] | **37.5%** [28.3, 47.0] | 9.0% |
| **Reduction vs manual** | — | 90.8% | **62.5%** | 91.0% |
| **Spray precision** | 9.0% | 55.4% | **23.5%** [18.8, 28.9] | 100% |
| **Precision vs manual** | 1.0× | 6.2× | **2.6×** | 11.2× |
| **Diseased plants reached** | 100.0% | **56.6%** [42.0, 71.6] | **98.4%** [95.6, 100] | 100% |
| Spray landing on weeds | 53.3% | 37.9% | 45.9% | 0% |
| Spray landing on bare soil | 36.5% | 0.0% | 27.4% | 0% |
| Navigation accuracy, energy | not measured | not measured | not measured | — |
| Autonomous return (sim) | operator | 500/500 | 500/500 | — |
| Fault → safe stop (sim) | operator | 600/600 | 600/600 | — |

95% bootstrap intervals over test images. *Spray precision* = of the zones sprayed, the share
that held a diseased tomato plant. *Manual* here is blanket broadcast coverage of the ground,
which is how boom and broadcast-knapsack spraying apply chemical; a careful operator
spot-spraying by eye would do better than this baseline.

## 2. What it means

**The proposed system cuts pesticide by 62.5% while still reaching 98.4% of diseased plants,
and it is 2.6× as precise as manual spraying.** That is a large reduction with almost nothing
missed, and it holds within its confidence interval (28–47% use, i.e. a 53–72% reduction).

**Vision-only saves more chemical, but it is not a safe alternative for disease control: it
misses 43% of diseased plants.** It fires only where a zone is at least 15% green, so a small
or partly-visible tomato plant in a mostly-bare zone is skipped — while large weeds are
sprayed. Cutting chemical by missing the disease is the failure the coverage panel exists to
expose.

## 3. Where the proposed system's chemical goes, and the lever to improve it

Of everything the proposed system sprays, **27.4% lands on bare soil and 45.9% on weeds.**
The cause is measurable. The tomato detector, on held-out sub-cells:

| precision | recall | average precision | tomato prevalence |
|---:|---:|---:|---:|
| 0.289 | 0.831 | 0.706 | 4.2% |

A nozzle zone fires if **any** of its sub-cells is called tomato, so each false positive turns
a whole zone on. The detector is the lever: better precision would move the proposed system
toward the ceiling without giving up coverage. That is the next experiment, and it would need
its own pre-registration — the 0.5 threshold was not tuned here.

## 4. Nozzle resolution trades chemical against coverage

| layout | proposed use | reduction | precision | coverage |
|---|---:|---:|---:|---:|
| **4 × 3 (primary)** | 37.5% | 62.5% | 23.5% | 98.4% |
| 8 × 6 | 22.5% | 77.5% | 20.8% | 93.6% |
| 16 × 12 | 11.9% | 88.1% | 25.3% | 82.7% |

Finer nozzles spray less, but miss more, because with small zones the detector's 17% miss rate
is no longer covered by a neighbouring detection in the same zone. The coarse primary layout is
the conservative one, chosen before the data was seen.

## 5. The bias in the sample, sized

Only 193 of 938 images could be downloaded (D1), and they are not typical (D2): they hold
about **half the tomato density** of the full field. The ceiling — the most any sprayer could
save — computed from annotations on both:

| layout | crop zones, 193 images | crop zones, all 938 |
|---|---:|---:|
| 4 × 3 | 9.4% | 18.6% |
| 8 × 6 | 5.8% | 12.7% |
| 16 × 12 | 4.3% | 10.3% |

So **the 62.5% reduction overstates what the full field would give.** With twice the plants,
the proposed system has twice as many true targets to spray. A decomposition makes the size
of that explicit:

- on the sample, true hits are 8.8% of zones, and the system fires on **31.5%** of zones that
  hold no diseased plant;
- on the full field, zones needing spray rise to 18.6% × 0.866 = **16.1%**;
- keeping 98.4% coverage and the same 31.5% false-positive rate: use = 15.8% + 26.4% =
  **42.3%**, a reduction of **about 58%**.

That is an estimate under a stated assumption — that the false-positive rate carries over —
and not a measurement. The full field has fewer weeds per plant, which would lower the
false-positive rate and push the figure up again. The measured result is 62.5%, on the sample.

## 6. What is not established

- **Disease is composed, not observed.** The Spanish field has no disease labels; each tomato
  plant is paired with a random plant from Tomato-Village (Rajasthan, India) and that plant's
  real classifier output. Disease prevalence there is 86.6%, so the disease filter rarely
  removes a plant — almost all of the saving comes from not spraying soil and weeds.
- **No pump, nozzle or field was involved.** A zone "sprayed" is a control decision.
- **Navigation accuracy and energy are not measured**: the IR sensor and battery are simulated.
- **One field, early season, one year (2021).**

## 7. Reproduce

```bash
python scripts/recover_truncated_zip.py <TOMATO_1 partial> C:/ds/groundbased_weed/recovered
python scripts/openfield_sprayer.py all
python scripts/make_openfield_figure.py
python -m pytest tests/test_openfield_geometry.py tests/test_sprayer_fsm.py -q
```

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
