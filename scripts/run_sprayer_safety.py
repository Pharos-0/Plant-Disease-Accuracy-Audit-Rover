"""Safety campaign for the sprayer FSM: the measured numbers behind Table I's
"Autonomous Return" and "Fault Handling" rows.

A fuzz test that never sprays passes trivially, so this script reports how
often the dangerous paths were actually exercised - spray commands issued,
obstacles encountered while spraying, faults injected mid-spray - alongside the
violation count. A zero is only meaningful next to a large denominator.

These are properties of the CONTROL SOFTWARE in simulation. They say nothing
about whether a physical relay opens or a physical pump stops.

    python scripts/run_sprayer_safety.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rover"))
sys.path.insert(0, str(ROOT / "src"))

from data import paths  # noqa: E402
from sprayer_fsm import (OBSTACLE_STOP_CM, FusedReading, SprayerController,  # noqa: E402
                         SprayPolicy, State)

SEED = 20261013
OUT = paths.RESULTS_DIR / "sprayer"


def reading(**kw) -> FusedReading:
    base = dict(range_cm=150.0, range_age=0, row_offset=0.0, row_age=0,
                canopy=(0.5, 0.5, 0.5, 0.5), disease_confidence=0.9,
                vision_age=0, tank_fraction=1.0, battery_fraction=1.0,
                resource_age=0, at_dock=False)
    base.update(kw)
    return FusedReading(**base)


def fuzz(n_runs: int, steps: int, rng) -> dict:
    """Latching faults are kept rare (0.3% per input per step). At the first
    version's rate - about one step in five - nearly every run latched into
    FAULT_SAFE_STOP within a few steps, and only 372 of 160,000 commands were
    spray commands: a zero-violation result over that denominator says little.
    """
    cmds = spray = violations = obstacle_while_spraying = 0
    for _ in range(n_runs):
        c = SprayerController(policy=rng.choice(list(SprayPolicy)))
        prev_spraying = False
        for _ in range(steps):
            r = reading(
                range_cm=float(rng.choice([5.0, 30.0, 150.0, 150.0])),
                range_age=int(5 if rng.random() < 0.003 else 0),
                row_age=int(rng.choice([0, 0, 5])),
                canopy=tuple(float(x) for x in rng.random(4)),
                disease_confidence=float(rng.random()),
                vision_age=int(rng.choice([0, 0, 0, 5])),
                tank_fraction=float(rng.choice([1.0, 1.0, 0.5, 0.05])),
                battery_fraction=float(rng.choice([1.0, 1.0, 0.5, 0.1])),
                resource_age=int(5 if rng.random() < 0.003 else 0),
                at_dock=bool(rng.random() < 0.1))
            cmd = c.step(r, gate_open=bool(rng.random() < 0.5))
            cmds += 1
            if prev_spraying and r.range_cm < OBSTACLE_STOP_CM:
                obstacle_while_spraying += 1
            if cmd.spraying:
                spray += 1
                if c.state is not State.SPRAYING or r.range_cm < OBSTACLE_STOP_CM:
                    violations += 1
            prev_spraying = cmd.spraying
    return {"runs": n_runs, "commands": cmds, "spray_commands": spray,
            "obstacles_met_while_spraying": obstacle_while_spraying,
            "violations": violations}


def fault_injection(trials: int) -> dict:
    faults = {"stale ultrasonic": dict(range_age=10),
              "stale resource telemetry": dict(resource_age=10),
              "wrong vision zone count": dict(canopy=(0.5, 0.5))}
    out = {}
    for name, kw in faults.items():
        safe = 0
        for _ in range(trials):
            c = SprayerController()
            c.step(reading(), gate_open=True)
            c.step(reading(), gate_open=True)            # spraying when it hits
            assert c.state is State.SPRAYING
            cmd = c.step(reading(**kw), gate_open=True)
            latched = all(not c.step(reading(), gate_open=True).spraying
                          for _ in range(3))
            if (c.state is State.FAULT_SAFE_STOP and not cmd.spraying
                    and cmd.drive == "stop" and latched):
                safe += 1
        out[name] = {"injected": trials, "safe_stops": safe}
    return out


def autonomous_return(trials: int, rng) -> dict:
    docked = 0
    for _ in range(trials):
        c = SprayerController()
        c.step(reading(), gate_open=True)
        c.step(reading(), gate_open=True)
        trigger = rng.choice(["tank", "battery"])
        kw = {"tank_fraction": 0.05} if trigger == "tank" else {"battery_fraction": 0.1}
        sprayed_on_way = False
        for _ in range(int(rng.integers(3, 20))):
            sprayed_on_way |= c.step(reading(**kw), gate_open=True).spraying
        c.step(reading(**kw, at_dock=True), gate_open=True)
        if c.state is State.DOCKED and not sprayed_on_way:
            docked += 1
    return {"trials": trials, "docked_without_spraying": docked}


def main() -> int:
    rng = np.random.default_rng(SEED)
    res = {"hardware": "SIMULATED",
           "fuzz": fuzz(2000, 80, rng),
           "fault_injection": fault_injection(200),
           "autonomous_return": autonomous_return(500, rng)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "safety.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    f = res["fuzz"]
    print("FUZZ  (random fused readings, all three policies)")
    print(f"  {f['commands']:,} commands over {f['runs']:,} runs")
    print(f"  {f['spray_commands']:,} spray commands issued   <- the denominator")
    print(f"  {f['obstacles_met_while_spraying']:,} obstacles met while spraying")
    print(f"  {f['violations']} violations (spray outside SPRAYING or at an obstacle)")
    print("\nFAULT INJECTION  (fault arrives mid-spray)")
    tot_i = tot_s = 0
    for k, v in res["fault_injection"].items():
        print(f"  {k:26s} {v['safe_stops']}/{v['injected']} safe stops")
        tot_i += v["injected"]
        tot_s += v["safe_stops"]
    print(f"  {'total':26s} {tot_s}/{tot_i}")
    a = res["autonomous_return"]
    print(f"\nAUTONOMOUS RETURN  {a['docked_without_spraying']}/{a['trials']} "
          "docked, none spraying on the way")
    print("\nSoftware properties in simulation. Not a test of any physical device.")
    return 0 if f["violations"] == 0 and tot_s == tot_i else 2


if __name__ == "__main__":
    raise SystemExit(main())
