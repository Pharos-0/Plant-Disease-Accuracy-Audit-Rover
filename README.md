# Plant-Disease Accuracy Audit Rover

**From laboratory images to a sprayer rover.** This project audits nine label-free accuracy
estimators across 20 plant-disease classifiers under laboratory-to-field distribution shift,
using 5,039 labeled tomato images. It also includes a Raspberry Pi 5 sprayer rover, fail-safe
control software, hardware tests and a digital twin for evaluating controller behavior.

M.Tech research project, Department of Data Science and Business Systems, SRM Institute of
Science and Technology (SRM IST), Kattankulathur.
Author: **Pharos Sophy Samuel T J**. Guide: **Dr. S. Ganesh Kumar**.

---

## Publication

A manuscript on this work is under submission. It is not in this repository while it is
under review; this repository holds the code, results and rover software behind it.

## The question

A plant-disease classifier trained on laboratory photographs is about 96% accurate there and
about 50% accurate on real field photographs. Measuring that drop needs labeled field images,
which a deployed system does not have. **Unsupervised (label-free) accuracy estimators** claim
to estimate a classifier's accuracy from unlabeled images alone. This project asks whether
they work under a real agricultural shift. It then asks what a sprayer should do if they don't.

## Data and models

| dataset | role | images analyzed |
|---|---|---|
| PlantVillage, tomato | source: training, calibration, laboratory test | 2,918 images of 730 leaves |
| PlantDoc | field target (scene shift) | 262 |
| Tomato-Village | field target (acquisition shift, Rajasthan) | 1,616 crops from 391 photographs |
| Pakistan field set | fourth target, added after the main study | 243 |
| Rover camera | deployment test on one tomato plant | 92 frames |

- **Classes:** three, the only ones that mean the same disease in every dataset: healthy,
  early blight and late blight.
- **Splits and sampling:** splits are by physical leaf, and sampling is always by photograph.
- **Leakage control:** enforced in code. No estimator can reach a target label.
- **Classifiers:** 20 = 4 architectures × 5 seeds. The architectures are ResNet-50,
  EfficientNet-B0, MobileNetV3-Large and ViT-S/16.
- **Estimators:** 9, covering every main family. They are Confidence, Entropy, ATC-MC,
  ATC-NE, DoC, COT, nuclear norm, MaNo and GdScore. In total 94,160 estimates were scored.

## Key results

1. **The shift is severe.** Accuracy falls from 0.962 in the laboratory to 0.503 (PlantDoc),
   0.500 (Tomato-Village) and 0.360 (Pakistan).
2. **No estimator beats an oracle constant.** On Tomato-Village the best estimator errs by
   0.134, against 0.061 for the constant.
3. **Most estimators are not reading the field.** A Sobol' variance audit shows six of the nine
   responding 4–14 times more to their laboratory calibration than to the field images.
4. **Responsiveness and error reverse their relation under shift**, at every sample size.
5. **More unlabeled data does not help, and neither does per-image confidence.** Confidence
   flags field errors only weakly (AUROC 0.57–0.67).
6. **Ten labeled field photographs beat every estimator.**

## The rover

A Raspberry Pi 5 rover with a USB camera, an ultrasonic range sensor, a relay-switched pump and
four motors. The control software is in [`rover/`](rover/).

- **The spray rule:** it sprays wherever the camera sees plant tissue, and never lets an
  unverified classifier withhold spray.
- **Fail-safes:** forward motion is refused near an obstacle or when the range sensor is lost.
  The relay is released at power-on and on STOP. Faulty camera frames are rejected.
- **Tests on the robot:** 29 of 29 automatic spray commands reached the relay. Forward motion
  was refused 1.09 s after an injected sensor loss.
- **Dry run on 1,800 edited photographs:** letting the classifier withhold spray missed 26.6% of
  diseased leaves in low light. The rover's rule missed 1.6%.
- **Digital twin** ([`rover/digital_twin.py`](rover/digital_twin.py)): a nine-state controller
  design held its safety invariants over 160,000 fuzzed commands and 600 injected faults. A
  label-free gate inside it never withheld, even as true accuracy fell to 0.533.

**Environmental benefit (estimated).** In simulation over real open-field images, a
crop-targeting controller design cut estimated pesticide use by 62.5% (95% interval
53.0–71.7%). It still treated 98.4% of the zones designated diseased, so less chemical would
reach soil and water.

**Not yet shown.** The rover is operator-driven. It has pumped water but no pesticide, and it
has not been field-tested. Spray dose per pulse, currents and battery life are not measured.
The full controller has run only in the twin.

## Repository map

| folder | contents |
|---|---|
| `src/` | data layer, estimators, sessions, evaluation, training, inference |
| `rover/` | on-robot control software, spray rule, controller, digital twin, hardware tests |
| `scripts/` | experiments, analyses, figures and tables |
| `dataset/` | dataset scripts, metadata and documentation (no images) |
| `demo/` | camera and live-sprayer demos |
| `results/` | committed results with provenance manifests |
| `camera_sessions/` | logs and analyses of the rover-camera sessions (frames stay local) |
| `docs/` | protocols, audits, claim-evidence table, dataset class mapping |
| `tests/` | test suite (`pytest`); tests that need the datasets or trained weights skip without them |

## Reproduce

Fast validation (about a minute, no GPU) checks the committed results against their sources:

```bash
python scripts/run_all_validations.py
python -m pytest -q
```

Full reproduction takes hours and needs a GPU: training, caching, then the experiments.

```bash
python scripts/run_training.py --stage v1v2
python scripts/run_caching.py
python scripts/run_estimates.py && python scripts/run_evaluation.py
python scripts/ti_audit.py --n-boot 100 --arms source_test,plantdoc,tomatovillage
python scripts/ti_sample_size.py && python scripts/gate_evaluation.py
```

Every experiment writes a manifest with the git commit, seeds, package versions and cache
fingerprints.

## Data availability

The image datasets are **not redistributed** here. Download them from their publishers:
PlantVillage, PlantDoc, Tomato-Village and the Pakistani field deposit (Mendeley Data,
doi:10.17632/3mbnb82mxd.2). Their sources and the class mapping are in
[`docs/dataset_class_mapping.md`](docs/dataset_class_mapping.md). Code, results, manifests and
dataset metadata are included. Rover-camera frames and the literature-search records (which
contain publishers' abstracts) are kept private.

## Limitations

- One crop, three classes and one source dataset.
- Four targets, one of them only partially independent.
- The 20 classifiers are 4 architectures × 5 seeds, not 20 independent models.
- The rover case study rests on one plant without a confirmed diagnosis.
- The pesticide savings are simulated decisions, not measured chemical use.
