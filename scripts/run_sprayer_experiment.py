"""Measure what the selective sprayer actually saves, and what it actually misses.

THE QUESTION
------------
Three spray policies run through the same FSM controller over the same traverse:

  BLANKET      every nozzle, every frame - the manual / boom-sprayer baseline
  VISION_ONLY  fire a nozzle only where its zone contains canopy
  SELECTIVE    canopy AND classified diseased AND the reliability gate open

For each we measure two things, and the second is the one a spray-reduction
chart usually leaves out:

  pesticide use      nozzle firings as a fraction of blanket
  disease coverage   of the nozzle zones that truly needed spray (canopy present
                     on a truly diseased plant), the fraction that received it

A policy that saves 90% of the chemical by missing 90% of the disease is not a
better sprayer. Reporting use without coverage is how that gets hidden.

HOW THE SIMULATION IS COMPOSED - READ THIS BEFORE QUOTING ANY NUMBER
--------------------------------------------------------------------
No available dataset has both rover-viewpoint geometry and disease labels, so
two real sources are composed:

  * TRAVERSE GEOMETRY from AgRobTomato (Zenodo 10.5281/zenodo.5596799): 446
    frames captured by a ground robot driven along greenhouse inter-rows in
    Portugal. Canopy per nozzle zone is MEASURED from those real frames by the
    same vegetation segmentation the dataset rubric uses.
  * PLANT DISEASE STATE AND CLASSIFIER BEHAVIOUR from Tomato-Village (Jodhpur
    and Jaipur, Rajasthan, India): true labels and the real cached outputs of
    the twenty frozen classifiers.

Each traverse frame is paired with one India plant drawn at random. The plant
in the frame is therefore NOT the plant whose disease state is used. That is a
modelling assumption - that disease prevalence is independent of where canopy
falls in the frame - and every artefact says so.

WHAT IS NOT MEASURED
--------------------
Navigation accuracy and energy per run. The infrared row sensor and the battery
are simulated, so any number produced for them would be a property of the
simulator. They are reported as NOT MEASURED rather than estimated.

    python scripts/run_sprayer_experiment.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rover"))
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))

from data import paths                                             # noqa: E402
from data.mapping import LABEL_SETS                                # noqa: E402
from estimators.base import SourceStats                            # noqa: E402
from estimators.methods import build                               # noqa: E402
from inference.cache import CacheKey, load_features, load_labels   # noqa: E402
from rover_relevance import _standardise, _vegetation              # noqa: E402
from sprayer_fsm import (FusedReading, SprayerController,          # noqa: E402
                         SprayPolicy, State, CANOPY_THRESHOLD)

OUT = paths.RESULTS_DIR / "sprayer"
N_NOZZLES = 4
TAU_SAFE = 0.80          # the manuscript's operating threshold; not tuned here
MIN_SESSION = 25
GATE_ESTIMATOR = "atc_mc"
N_PAIRINGS = 10          # random frame-to-plant pairings per classifier
SEED = 20261013
SIM_TAG = "SIMULATED"


# --------------------------------------------------------------------------
# 1. canopy per nozzle zone, measured from real rover frames
# --------------------------------------------------------------------------
def measure_canopy() -> pd.DataFrame:
    cache = OUT / "traverse_canopy.csv"
    if cache.exists():
        return pd.read_csv(cache)
    meta = pd.read_csv(ROOT / "dataset" / "metadata" / "metadata.csv")
    tr = meta[meta.source_id == "SRC014"].copy()
    tr["fname"] = tr.original_filename.astype(str).str.split("/").str[-1]
    tr["session"] = tr.fname.str.replace(r"_?\d+\.jpg$", "", regex=True)
    tr["idx"] = tr.fname.str.extract(r"(\d+)\.jpg$")[0].astype(int)
    tr = tr.sort_values(["session", "idx"]).reset_index(drop=True)

    rows = []
    for _, r in tr.iterrows():
        img = cv2.imdecode(np.fromfile(r.stored_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        hsv = cv2.cvtColor(_standardise(img), cv2.COLOR_BGR2HSV)
        veg = _vegetation(hsv[..., 0], hsv[..., 1], hsv[..., 2])
        # Boom sections: the frame width split into equal vertical bands.
        zones = np.array_split(veg, N_NOZZLES, axis=1)
        rows.append({"session": r.session, "idx": int(r.idx),
                     **{f"z{k}": float(z.mean()) for k, z in enumerate(zones)}})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# --------------------------------------------------------------------------
# 2. one traversal under one policy
# --------------------------------------------------------------------------
def traverse(policy: SprayPolicy, canopy: np.ndarray, p_diseased: np.ndarray,
             truly_diseased: np.ndarray, logits: np.ndarray, estimator,
             tank_start: float = 1.0, gate_override: bool | None = None) -> dict:
    ctl = SprayerController(policy=policy, n_nozzles=N_NOZZLES)
    n = len(canopy)
    fired = np.zeros((n, N_NOZZLES), dtype=bool)
    gate_states = []
    tank = tank_start
    battery = 1.0
    session: list[np.ndarray] = []

    for i in range(n):
        session.append(logits[i])
        if gate_override is not None:
            gate_open = gate_override             # oracle only
        elif len(session) < MIN_SESSION:
            gate_open = False
        else:
            # Label-free, exactly as deployment would have it.
            gate_open = estimator.estimate(np.asarray(session)) >= TAU_SAFE
        gate_states.append(gate_open)

        reading = FusedReading(
            range_cm=150.0, range_age=0,
            row_offset=0.0, row_age=0,
            canopy=tuple(float(x) for x in canopy[i]),
            disease_confidence=float(p_diseased[i]), vision_age=0,
            tank_fraction=tank, battery_fraction=battery, resource_age=0)
        cmd = ctl.step(reading, gate_open=gate_open)
        if i == 0:                          # IDLE -> TRAVERSING consumes a cycle
            cmd = ctl.step(reading, gate_open=gate_open)
        fired[i] = cmd.nozzles
        tank -= 0.0004 * sum(cmd.nozzles)   # bookkeeping only; see module notes
        battery -= 0.0005

    has_canopy = canopy >= CANOPY_THRESHOLD
    need = has_canopy & truly_diseased[:, None]
    use = fired.mean()
    coverage = (fired & need).sum() / max(1, need.sum())
    wasted = (fired & ~need).sum() / max(1, fired.sum())
    return {"use": use, "coverage": coverage, "wasted": wasted,
            "gate_open_frac": float(np.mean(gate_states)),
            "final_state": ctl.state.name, "fired": fired, "need": need}


# --------------------------------------------------------------------------
# 3. the experiment
# --------------------------------------------------------------------------
def frozen_models() -> list[dict]:
    out = []
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if c["label_set"] == "SHARED_3" and c["version"] in ("V2", "V9"):
            out.append(c)
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("SELECTIVE SPRAYER EXPERIMENT - simulation over real imagery")
    print("geometry: AgRobTomato rover traverse | disease: Tomato-Village, India")
    print("=" * 76)

    cdf = measure_canopy()
    canopy = cdf[[f"z{k}" for k in range(N_NOZZLES)]].to_numpy()
    n = len(canopy)
    print(f"\ntraverse: {n} real rover frames, {N_NOZZLES} nozzle zones each")
    print(f"zones with canopy >= {CANOPY_THRESHOLD}: "
          f"{(canopy >= CANOPY_THRESHOLD).mean():.1%}")

    healthy = LABEL_SETS["SHARED_3"].index("healthy")
    rng = np.random.default_rng(SEED)
    rows = []
    for c in frozen_models():
        k = CacheKey(c["version"], c["backbone"], c["seed"], "tomatovillage", c["label_set"])
        d = load_features(k)
        y = load_labels(k)
        lg = d["logits"]
        z = lg - lg.max(1, keepdims=True)
        prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        p_dis_all = 1.0 - prob[:, healthy]

        sk = CacheKey(c["version"], c["backbone"], c["seed"], "source_val", c["label_set"])
        s = load_features(sk)
        est = build(GATE_ESTIMATOR).fit(
            SourceStats(logits=s["logits"], labels=load_labels(sk), features=s["features"]))

        for rep in range(N_PAIRINGS):
            pick = rng.choice(len(y), n, replace=True)
            truth = y[pick] != healthy
            runs = {}
            for pol in SprayPolicy:
                runs[pol.value] = traverse(pol, canopy, p_dis_all[pick], truth,
                                           lg[pick], est)
            # Ceiling: true disease state, gate forced open. Needs labels.
            runs["oracle"] = traverse(SprayPolicy.SELECTIVE, canopy,
                                      truth.astype(float), truth, lg[pick], est,
                                      gate_override=True)
            # Were the classifier's skips targeted, or no better than random?
            skipped = runs["vision_only"]["fired"] & ~runs["selective"]["fired"]
            n_skip = int(skipped.sum())
            skip_prec = (float((skipped & ~runs["selective"]["need"]).sum() / n_skip)
                         if n_skip else float("nan"))
            healthy_prev = float((~truth).mean())
            for name, res in runs.items():
                rec = {k: v for k, v in res.items() if k not in ("fired", "need")}
                rows.append({"backbone": c["backbone"], "seed": c["seed"], "rep": rep,
                             "policy": name, "skip_precision": skip_prec,
                             "n_skipped": n_skip, "healthy_prevalence": healthy_prev,
                             **rec})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sprayer_runs.csv", index=False)

    def ci(x):
        b = [rng.choice(x, len(x), replace=True).mean() for _ in range(2000)]
        return np.percentile(b, [2.5, 97.5])

    summary = []
    for pol in ("blanket", "vision_only", "selective", "oracle"):
        g = df[df.policy == pol]
        u_lo, u_hi = ci(g.use.to_numpy())
        c_lo, c_hi = ci(g.coverage.to_numpy())
        summary.append({"policy": pol,
                        "pesticide_use_pct": 100 * g.use.mean(),
                        "use_ci_lo": 100 * u_lo, "use_ci_hi": 100 * u_hi,
                        "reduction_pct": 100 * (1 - g.use.mean()),
                        "disease_coverage_pct": 100 * g.coverage.mean(),
                        "cov_ci_lo": 100 * c_lo, "cov_ci_hi": 100 * c_hi,
                        "wasted_spray_pct": 100 * g.wasted.mean(),
                        "gate_open_pct": 100 * g.gate_open_frac.mean(),
                        "runs": len(g)})
    s = pd.DataFrame(summary).set_index("policy")
    s.to_csv(OUT / "sprayer_summary.csv")

    print(f"\n{len(df)} traversals = 20 classifiers x {N_PAIRINGS} pairings x "
          f"{df.policy.nunique()} policies (3 deployable + oracle ceiling)\n")
    print(f"{'policy':12s} {'pesticide use':>16s} {'reduction':>10s} "
          f"{'disease coverage':>18s} {'wasted':>8s}")
    for pol, r in s.iterrows():
        print(f"{pol:12s} {r.pesticide_use_pct:6.1f}% "
              f"[{r.use_ci_lo:4.1f},{r.use_ci_hi:4.1f}] {r.reduction_pct:8.1f}% "
              f"{r.disease_coverage_pct:8.1f}% [{r.cov_ci_lo:4.1f},{r.cov_ci_hi:4.1f}] "
              f"{r.wasted_spray_pct:6.1f}%")
    print(f"\nselective gate open on {s.loc['selective', 'gate_open_pct']:.1f}% "
          "of frames (label-free ATC-MC estimate >= 0.80)")

    sel = df[df.policy == "selective"]
    sp = sel.skip_precision.dropna()
    print("\nWERE THE CLASSIFIER'S SKIPS TARGETED?")
    print(f"  zones skipped by selective but sprayed by vision-only: "
          f"{sel.n_skipped.mean():.0f} per traversal")
    print(f"  of those, truly not needing spray: {100 * sp.mean():.1f}%")
    print(f"  healthy-plant prevalence (what random skipping would give): "
          f"{100 * sel.healthy_prevalence.mean():.1f}%")
    verdict = ("targeted" if sp.mean() > sel.healthy_prevalence.mean() + 0.05
               else "NO BETTER THAN RANDOM")
    print(f"  -> skips are {verdict}")
    s.loc["selective", "skip_precision_pct"] = 100 * sp.mean()
    s.loc["selective", "healthy_prevalence_pct"] = 100 * sel.healthy_prevalence.mean()
    s.to_csv(OUT / "sprayer_summary.csv")

    o = s.loc["oracle"]
    v = s.loc["selective"]
    print("\nCEILING")
    print(f"  most any selective sprayer could save here: {o.reduction_pct:.1f}%")
    print(f"  the deployed selective sprayer saved:       {v.reduction_pct:.1f}%")
    print(f"  share of the achievable saving realised:    "
          f"{100 * v.reduction_pct / o.reduction_pct:.0f}%")

    (OUT / "sprayer_meta.json").write_text(json.dumps({
        "hardware": SIM_TAG,
        "geometry_source": "AgRobTomato, Zenodo 10.5281/zenodo.5596799, 446 rover frames",
        "disease_source": "Tomato-Village, Jodhpur and Jaipur, Rajasthan, India",
        "pairing": "each frame paired with a random India plant; independence assumed",
        "not_measured": ["navigation accuracy", "energy per run (Wh)"],
        "canopy_threshold": CANOPY_THRESHOLD, "tau_safe": TAU_SAFE,
        "gate_estimator": GATE_ESTIMATOR, "n_nozzles": N_NOZZLES,
        "pairings_per_classifier": N_PAIRINGS, "seed": SEED,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote results/sprayer/  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
