"""Prospective power of the protocol's ten-plant camera session, by simulation.

THIS IS A SIMULATION OF THE PLANNED SESSION, NOT A RESULT OF IT. No plant was
photographed for it and no diagnosis is assumed for any real plant.

The real session (docs/camera_experiment_protocol.md, sections 2 and 10) needs at least
6 diseased and 4 healthy plants, 2-3 leaves each, 5 frames per leaf per condition,
analysed with intervals that resample whole plants. It has not been run. This script
asks what such a session could show if real captures behaved like the dry run: it
assembles many simulated sessions of exactly that size from the dry run's real,
dataset-labelled field photographs (camera_sessions/dryrun_v3, the decisions already
logged for them), and runs the pre-registered analysis itself
(scripts/analyse_camera_session.policy_comparison) on each.

The dry-run photographs carry no plant identity, so how alike a plant's leaves are is
unknown. Two bounds are reported:
  independent - a plant's leaves are different photographs (optimistic: real leaves of
                one plant resemble each other, which widens intervals);
  identical   - a plant's leaves are copies of one photograph (pessimistic).
The real session would fall between them, before the further uncertainty that the dry
run used edited photographs, not camera captures.

    python scripts/simulate_ten_plant_session.py --sessions 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_camera_session import load, policy_comparison  # noqa: E402

DRY = ROOT / "camera_sessions" / "dryrun_v3"
N_DISEASED, N_HEALTHY, LEAVES, FRAMES = 6, 4, (2, 3), 5


def one_session(df: pd.DataFrame, leaves_by: dict, rng, mode: str) -> pd.DataFrame:
    rows = []
    for kind, n_plants in (("diseased", N_DISEASED), ("healthy", N_HEALTHY)):
        pool = list(leaves_by[kind])
        k = rng.integers(LEAVES[0], LEAVES[1] + 1, n_plants)
        if mode == "independent":
            chosen = rng.choice(len(pool), int(k.sum()), replace=False)
            groups = np.split(chosen, np.cumsum(k)[:-1])
        else:                                   # identical: one photograph per plant
            one = rng.choice(len(pool), n_plants, replace=False)
            groups = [np.repeat(o, kk) for o, kk in zip(one, k)]
        for p, g in enumerate(groups):
            plant = f"{kind[0].upper()}{p:02d}"
            for li, idx in enumerate(g):
                leaf = df[df.leaf_id == pool[idx]]
                for cond, cg in leaf.groupby("condition"):
                    take = cg.sample(min(FRAMES, len(cg)), random_state=int(rng.integers(1 << 31)))
                    take = take.assign(unit=f"sim/{plant}", leaf=f"sim/{plant}L{li}")
                    rows.append(take)
    s = pd.concat(rows, ignore_index=True)
    s.attrs["unit_kind"] = "plant"
    return s


def simulate(n_sessions: int, seed: int) -> pd.DataFrame:
    df = load([DRY])
    leaves_by = {"diseased": sorted(df[df.diseased].leaf_id.unique()),
                 "healthy": sorted(df[~df.diseased].leaf_id.unique())}
    rng = np.random.default_rng(seed)
    out = []
    for mode in ("independent", "identical"):
        for i in range(n_sessions):
            s = one_session(df, leaves_by, rng, mode)
            pc = policy_comparison(s, rng)
            nh = max(1, int((~s.diseased).sum()))
            for _, r in pc.iterrows():
                out.append({"mode": mode, "session": i, "condition": r.condition,
                            "difference": r.difference, "lo": r.difference_lo,
                            "hi": r.difference_hi, "missed_gated": r.missed_gated,
                            "missed_spatial": r.missed_spatial,
                            "wasted_gated": float(s.wasted.sum() / nh),
                            "wasted_spatial": float(s.wasted_spatial.sum() / nh)})
    return pd.DataFrame(out)


def summarise(sim: pd.DataFrame) -> pd.DataFrame:
    g = sim.groupby(["mode", "condition"])
    return pd.DataFrame({
        "median_difference": g.difference.median(),
        "difference_2.5%": g.difference.quantile(0.025),
        "difference_97.5%": g.difference.quantile(0.975),
        "power_lo_above_0": g.lo.apply(lambda x: float((x > 0).mean())),
        "median_interval_width": (g.hi.median() - g.lo.median()),
        "median_missed_gated": g.missed_gated.median(),
        "median_missed_spatial": g.missed_spatial.median(),
    }).round(3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "simulation_ten_plant")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sim = simulate(a.sessions, a.seed)
    sim.to_csv(a.out / "sessions.csv", index=False)
    summ = summarise(sim)
    summ.to_csv(a.out / "power.csv")
    (a.out / "README.md").write_text(
        "# Prospective power of the ten-plant session - SIMULATION, not a result\n\n"
        "Produced by `scripts/simulate_ten_plant_session.py` from the dry run's real, "
        "dataset-labelled photographs; see that script's docstring for what it does and "
        "does not show. `power.csv` per bound and condition; `sessions.csv` every simulated "
        "session.\n", encoding="utf-8")
    pd.set_option("display.width", 160)
    print(f"{a.sessions} simulated sessions per bound "
          f"({N_DISEASED} diseased + {N_HEALTHY} healthy plants, {LEAVES[0]}-{LEAVES[1]} leaves, "
          f"<= {FRAMES} frames per leaf per condition)\n")
    print(summ.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
