"""Drive the real rover control software against a simulated rover.

WHAT THIS IS
------------
The control code under test is the real `rover_v1.RoverHardware`. Only the
peripherals are simulated: GPIO, camera, ultrasonic. The classifier outputs are
real cached activations from the frozen experiments, so the decisions taken here
are the decisions the software would take on that data.

WHAT THIS IS NOT
----------------
It is **not** a hardware test and **not** a field trial. Nothing here says
anything about whether the Pi, camera, motor driver, relay, pump or battery
work, or exist. Every artefact is stamped `hardware: SIMULATED` and the writer
refuses to emit a record without that field.

Timing is reported twice and never merged: virtual seconds, a property of the
scenario, and wall seconds, a property of whatever machine ran it. Neither is a
Raspberry Pi measurement.

FOUR SCENARIOS
--------------
  S1  zero-shift control   the gate should authorise operation
  S2  severe field shift   the gate should withhold; this is the thesis finding
  S3  camera fault         a dead camera returning constant frames
  S4  crash mid-traversal  the relay must still be released

    python scripts/run_digital_twin.py
    python scripts/run_digital_twin.py --scenario S2 --frames 120
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rover"))

from data import paths                                          # noqa: E402
from digital_twin import (HARDWARE_TAG, ObstacleProfile, SimulatedCamera,   # noqa: E402
                          TraversalResult, TwinClock, TwinGPIO, _Seq,
                          check_invariants)
from estimators.base import SourceStats                           # noqa: E402
from estimators.methods import build                              # noqa: E402
from inference.cache import CacheKey, load_features, load_labels  # noqa: E402
from rover_v1 import RoverHardware                              # noqa: E402

OUT = paths.RESULTS_DIR / "digital_twin"

#: Session-level accuracy at or above which the gate authorises operation. This
#: is the manuscript's operating threshold; it is NOT tuned here.
TAU_SAFE = 0.80
#: Frames accumulated before the gate will answer at all.
MIN_SESSION = 25
#: Stop distance in cm.
STOP_CM = 20.0
#: The label-free estimator the gate runs. Deployment has no labels, so the gate
#: cannot use true accuracy; it must use an estimate like any other. ATC-MC is
#: representative of the anchor-dominated family the study characterises.
GATE_ESTIMATOR = "atc_mc"

SCENARIOS = {
    "S1": {"arm": "source_test", "blank_every": None, "obstacles": {},
           "crash_at": None, "why": "zero-shift control: the gate should authorise"},
    "S2": {"arm": "tomatovillage", "blank_every": None, "obstacles": {40: 12.0},
           "crash_at": None, "why": "severe field shift: the gate should withhold"},
    "S3": {"arm": "tomatovillage", "blank_every": 7, "obstacles": {},
           "crash_at": None, "why": "camera fault: constant frames must be rejected"},
    "S4": {"arm": "source_test", "blank_every": None, "obstacles": {},
           "crash_at": 30, "why": "crash mid-traversal: the relay must be released"},
}


class InjectedFault(RuntimeError):
    """Raised on purpose, to prove the cleanup path runs."""


def frame_is_blank(frame: np.ndarray, threshold: float = 0.5) -> bool:
    """The check the camera probe was fixed to perform. A dead sensor returns a
    constant image, which passes a naive 'did we receive bytes' test."""
    return bool(np.asarray(frame).std() < threshold)


def pick_model() -> dict:
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if (c["label_set"] == "SHARED_3" and c["version"] in ("V2", "V9")
                and c["backbone"] == "resnet50" and c["seed"] == 0):
            return c
    raise SystemExit("no frozen SHARED_3 resnet50 seed0 model found")


def gate_state(n_seen: int, running_acc_estimate: float) -> str:
    """Session gate. Dataset-level only; it never answers about one image."""
    if n_seen < MIN_SESSION:
        return "INSUFFICIENT"
    if running_acc_estimate >= TAU_SAFE:
        return "OPERATE"
    if running_acc_estimate >= TAU_SAFE - 0.15:
        return "REVIEW"
    return "WITHHOLD"


def run(scenario: str, n_frames: int) -> TraversalResult:
    spec = SCENARIOS[scenario]
    cfg = pick_model()
    key = CacheKey(cfg["version"], cfg["backbone"], cfg["seed"], spec["arm"],
                   cfg["label_set"])
    d = load_features(key)
    y = load_labels(key)
    logits = d["logits"]
    img_paths = [str(p) for p in d["paths"]]

    # The gate's estimator is fitted on the SOURCE validation split, which is
    # all a deployed rover would have. No target label is ever passed to it.
    sk = CacheKey(cfg["version"], cfg["backbone"], cfg["seed"], "source_val",
                  cfg["label_set"])
    s = load_features(sk)
    estimator = build(GATE_ESTIMATOR).fit(
        SourceStats(logits=s["logits"], labels=load_labels(sk),
                    features=s["features"]))

    n = min(n_frames, len(logits))
    clock = TwinClock()
    obstacles = ObstacleProfile(distances=spec["obstacles"])
    seq = _Seq()
    gpio = TwinGPIO(clock, distance_source=obstacles.current, seq=seq)
    cam = SimulatedCamera(img_paths[:n], clock, blank_every=spec["blank_every"])

    frames: list[dict] = []
    session: list[np.ndarray] = []
    wall0 = time.perf_counter()
    correct = 0
    scored = 0

    try:
        with RoverHardware(gpio=gpio, clock=clock.now) as rover:
            rover.forward()
            for i in range(n):
                ok, injected, path = cam.read()
                if not ok:
                    break
                obstacles.step()
                dist = rover.get_distance()
                dist_cm = obstacles.current() if dist is None else dist

                if injected is not None and frame_is_blank(injected):
                    frames.append({"seq": seq.next(), "t": clock.now(), "frame": i,
                                   "path": path, "blank_frame": True, "gate_state": "INSUFFICIENT",
                                   "decision": "SKIP_BLANK_FRAME",
                                   "obstacle_cm": dist_cm})
                    continue

                # Real cached activation for this frame: no inference is faked.
                lg = logits[i]
                pred = int(lg.argmax())
                conf = float(np.exp(lg - lg.max()).max() / np.exp(lg - lg.max()).sum())
                session.append(lg)
                scored += 1

                # LABEL-FREE: the estimator sees only the session's logits.
                est = float(estimator.estimate(np.asarray(session)))
                state = gate_state(scored, est)

                # ORACLE, for contrast only. A deployed rover cannot compute
                # this, and nothing in the control path may read it.
                correct += int(pred == int(y[i]))
                oracle_acc = correct / scored
                oracle_state = gate_state(scored, oracle_acc)
                # Sequence the DECISION before any actuation it authorises, so
                # the invariant check reads the state actually in force. Taking
                # it after the relay call made the check read the previous
                # frame's state and flag a false violation at every boundary
                # where the gate first reached OPERATE.
                frame_seq = seq.next()

                if dist_cm < STOP_CM:
                    rover.stop()
                    decision = "STOP"
                elif state == "OPERATE":
                    rover.relay_on()
                    rover.relay_off()
                    decision = "ACTUATE"
                else:
                    decision = f"HOLD_{state}"

                frames.append({"seq": frame_seq, "t": clock.now(), "frame": i,
                               "path": path, "blank_frame": False, "pred": pred,
                               "confidence": round(conf, 4),
                               "estimated_accuracy": round(est, 4),
                               "oracle_accuracy": round(oracle_acc, 4),
                               "oracle_gate_state": oracle_state,
                               "gate_state": state, "decision": decision,
                               "obstacle_cm": round(float(dist_cm), 1)})

                if spec["crash_at"] is not None and i == spec["crash_at"]:
                    raise InjectedFault(f"injected fault at frame {i}")
    except InjectedFault as exc:
        frames.append({"seq": seq.next(), "t": clock.now(), "frame": -1,
                       "path": None, "blank_frame": False, "gate_state": "FAULT",
                       "decision": f"ABORT: {exc}", "obstacle_cm": 999})

    wall = time.perf_counter() - wall0
    rel = gpio.relay_history()
    return TraversalResult(frames=frames, relay_history=rel,
                           virtual_seconds=clock.now(), wall_seconds=wall,
                           violations=check_invariants(frames, rel))


def write(scenario: str, res: TraversalResult) -> Path:
    rec = {
        "hardware": res.hardware,
        "scenario": scenario,
        "why": SCENARIOS[scenario]["why"],
        "arm": SCENARIOS[scenario]["arm"],
        "frames": len(res.frames),
        "virtual_seconds": round(res.virtual_seconds, 3),
        "wall_seconds_on_this_machine": round(res.wall_seconds, 3),
        "machine": f"{platform.system()} {platform.machine()}",
        "not_a_raspberry_pi_measurement": True,
        "relay_transitions": len(res.relay_history),
        "invariant_violations": res.violations,
        "gate_estimator": GATE_ESTIMATOR,
        "gate_states": {s: sum(1 for f in res.frames if f["gate_state"] == s)
                        for s in sorted({f["gate_state"] for f in res.frames})},
        "oracle_gate_states": {s: sum(1 for f in res.frames
                                      if f.get("oracle_gate_state") == s)
                               for s in sorted({f.get("oracle_gate_state")
                                                for f in res.frames} - {None})},
        "unsafe_actuations": sum(
            1 for f in res.frames
            if f.get("decision") == "ACTUATE" and f.get("oracle_gate_state") != "OPERATE"),
        "decisions": {s: sum(1 for f in res.frames if f["decision"] == s)
                      for s in sorted({f["decision"] for f in res.frames})},
        "trace": res.frames,
    }
    if rec.get("hardware") != HARDWARE_TAG:
        raise RuntimeError("refusing to write a record without the SIMULATED tag")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"twin_{scenario}.json"
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    ap.add_argument("--frames", type=int, default=120)
    a = ap.parse_args()

    names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
    print("=" * 74)
    print("DIGITAL TWIN - peripherals simulated, control software real")
    print("NOT a hardware test. NOT a field trial. Establishes nothing about")
    print("the Pi, the camera, the motor driver, the relay, the pump or power.")
    print("=" * 74)

    total_violations = 0
    for s in names:
        res = run(s, a.frames)
        p = write(s, res)
        states = {}
        for f in res.frames:
            states[f["gate_state"]] = states.get(f["gate_state"], 0) + 1
        print(f"\n{s}  {SCENARIOS[s]['why']}")
        print(f"   arm {SCENARIOS[s]['arm']}, {len(res.frames)} frames, "
              f"{res.virtual_seconds:.1f} virtual s, "
              f"{res.wall_seconds*1000:.0f} ms wall on this machine")
        print(f"   gate: {states}")
        acted = sum(1 for f in res.frames if f["decision"] == "ACTUATE")
        unsafe = sum(1 for f in res.frames if f.get("decision") == "ACTUATE"
                     and f.get("oracle_gate_state") != "OPERATE")
        print(f"   actuations: {acted}   relay transitions: {len(res.relay_history)}")
        print(f"   of those, UNSAFE (oracle would not have authorised): {unsafe}")
        if res.violations:
            total_violations += len(res.violations)
            print(f"   INVARIANT VIOLATIONS ({len(res.violations)}):")
            for v in res.violations:
                print(f"     - {v}")
        else:
            print("   invariants: all held")
        print(f"   wrote {p.relative_to(ROOT)}")

    print("\n" + "=" * 74)
    if total_violations:
        print(f"{total_violations} invariant violation(s). This is a FAILURE.")
        return 2
    print("All safety invariants held across every scenario.")
    print("That is a statement about the CONTROL SOFTWARE only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
