"""Analyse labelled camera sessions from demo/live_sprayer.py.

Frames of the same leaf are NOT independent - photographing one leaf ten times
under ten lamps is one leaf, not ten samples - and leaves of the same plant are
not independent either. Every interval here is therefore a cluster bootstrap
over PLANTS when every frame carries a plant_id, and over leaves otherwise; the
unit used is printed. Paired comparisons are made within a leaf.

Outputs, per condition:
  accuracy            of the disease prediction against the recorded truth
  confidence          mean max-softmax
  abstention rate     frames below tau
  spray rate          frames the system decided to spray
  MISSED TREATMENT    truly diseased leaf, not sprayed   <- the safety criterion
  wasted spray        truly healthy leaf, sprayed        <- the chemical cost

Plus, pooled: whether confidence identifies wrong predictions (AUROC), and for
every leaf seen under both 'normal' and another condition, the paired change in
accuracy and confidence.

And the primary comparison: on the SAME frames, the share of diseased frames
missed by the classifier-gated policy and by the spatial-only policy, and their
difference (gated minus spatial-only), per condition, with plant-level
intervals. Written to policy_comparison.csv.

A quantity that the session cannot measure is reported as n/a, never as a number:
missed treatment with no diseased frames is not 0%, it is unmeasured, and an interval
from a single plant is not a narrow interval, it is no interval. A session below the
protocol's minimum of 6 diseased and 4 healthy plants is labelled a PILOT, and the
verdict is also written to analysis_status.json beside the CSVs.

    python scripts/analyse_camera_session.py camera_sessions/20261013_160000
    python scripts/analyse_camera_session.py camera_sessions/*      # pooled
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 20261013
B = 500
PLURAL = {"plant": "plants", "leaf": "leaves"}
#: The protocol's sample (docs/camera_experiment_protocol.md, sections 2 and 10).
MIN_DISEASED_PLANTS, MIN_HEALTHY_PLANTS = 6, 4


def _f(x: float, spec: str, width: int) -> str:
    """Format a number, or 'n/a' for a quantity the session could not measure."""
    return "n/a".rjust(width) if pd.isna(x) else format(x, spec)


def _share(num: pd.Series, den: pd.Series) -> float:
    """A share whose denominator may be empty: no frames to judge is NaN, not 0."""
    n = int(den.sum())
    return float(num.sum() / n) if n else np.nan


def zero_event_upper(units: int, alpha: float = 0.05) -> float:
    """Exact one-sided (1 - alpha) upper bound on a per-unit event probability when
    none of `units` independent units had the event (Clopper-Pearson at zero).

    Supplementary to the pre-registered bootstrap, which cannot exceed zero when no
    event occurs and so prints a [0, 0] interval that overstates certainty."""
    return 1.0 - alpha ** (1.0 / units) if units else np.nan


def zero_miss_notes(df: pd.DataFrame, policies: dict[str, str]) -> list[str]:
    """For each condition and policy with diseased frames but no miss, the exact
    upper bound on the share of diseased units that could still have a miss."""
    plural = PLURAL[df.attrs["unit_kind"]]
    notes = []
    for cond, g in df[df.diseased].groupby("condition"):
        k = g.unit.nunique()
        for name, col in policies.items():
            if col in g and not g[col].any():
                notes.append(f"  {cond} / {name}: 0 missed across {k} diseased {plural} - "
                             f"the bootstrap interval is degenerate at 0; exact 95% upper "
                             f"bound on the share of diseased {plural} with any miss: "
                             f"{zero_event_upper(k):.0%}")
    return notes


def load(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        f = d / "log.jsonl"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    r["session"] = d.name
                    rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no labelled frames found")
    df = df[df.truth.astype(str).str.len() > 0].copy()
    df["leaf"] = df.session + "/" + df.leaf_id.astype(str)
    # The resampling unit: the plant if every frame names one, else the leaf.
    plant = df.get("plant_id", pd.Series("", index=df.index)).fillna("").astype(str)
    if (plant.str.len() > 0).all():
        df["unit"] = df.session + "/" + plant
        df.attrs["unit_kind"] = "plant"
    else:
        df["unit"] = df.leaf
        df.attrs["unit_kind"] = "leaf"
    df["diseased"] = df.truth != "healthy"
    df["correct"] = df.truth == df.pred
    if "deferred" not in df:
        df["deferred"] = False
    df["deferred"] = df.deferred.fillna(False).astype(bool)
    # Three outcomes for a diseased leaf, kept apart: sprayed, MISSED (the system
    # decided not to), DEFERRED (it refused to decide). Untreated = both.
    df["missed"] = df.diseased & ~df.spray & ~df.deferred
    df["deferred_dis"] = df.diseased & df.deferred
    if "spray_spatial" in df:
        df["missed_spatial"] = df.diseased & ~df.spray_spatial.astype(bool) & ~df.deferred
        df["wasted_spatial"] = ~df.diseased & df.spray_spatial.astype(bool)
    df["wasted"] = ~df.diseased & df.spray
    return df


def cluster_ci(df: pd.DataFrame, fn, rng) -> tuple[float, float, float]:
    """Point estimate and a 95% cluster-bootstrap interval over units. With fewer
    than two units there is no between-unit variation to resample, so the interval
    is NaN - resampling one plant returns that plant every time, and the resulting
    zero-width interval would look precise while meaning nothing."""
    units = df.unit.unique()
    point = fn(df)
    if len(units) < 2:
        return point, np.nan, np.nan
    groups = {k: g for k, g in df.groupby("unit")}
    vals = []
    for _ in range(B):
        pick = rng.choice(units, len(units), replace=True)
        vals.append(fn(pd.concat([groups[k] for k in pick])))
    vals = np.asarray(vals, dtype=float)
    if np.isnan(vals).all():
        return point, np.nan, np.nan
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, lo, hi


def rate(col):
    return lambda d: float(d[col].mean()) if len(d) else np.nan


def conditional(col, mask):
    def f(d):
        s = d[d[mask]] if mask in d else d
        return float(s[col].mean()) if len(s) else np.nan
    return f


def policy_comparison(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Missed treatment of diseased frames under each policy, same frames, with a
    cluster bootstrap over units; the difference is paired by construction."""
    rows = []
    dis = df[df.diseased]
    for cond, g in dis.groupby("condition"):
        per = g.groupby("unit").agg(n=("missed", "size"), gated=("missed", "sum"),
                                    spatial=("missed_spatial", "sum")).to_numpy(float)
        point = per[:, 1:].sum(0) / per[:, 0].sum()
        if len(per) >= 2:
            idx = rng.integers(0, len(per), (B, len(per)))
            boot = per[idx, 1:].sum(1) / per[idx, 0].sum(1)[:, None]
            diff = boot[:, 0] - boot[:, 1]
            q = lambda a, p: float(np.percentile(a, p))        # noqa: E731
        else:   # one unit: nothing to resample, so no interval (see cluster_ci)
            boot = np.full((1, 2), np.nan)
            diff = np.full(1, np.nan)
            q = lambda a, p: np.nan                             # noqa: E731
        rows.append({"condition": cond, "diseased_frames": len(g),
                     "units": len(per), "unit_kind": df.attrs["unit_kind"],
                     "missed_gated": point[0],
                     "missed_gated_lo": q(boot[:, 0], 2.5),
                     "missed_gated_hi": q(boot[:, 0], 97.5),
                     "missed_spatial": point[1],
                     "missed_spatial_lo": q(boot[:, 1], 2.5),
                     "missed_spatial_hi": q(boot[:, 1], 97.5),
                     "difference": point[0] - point[1],
                     "difference_lo": q(diff, 2.5),
                     "difference_hi": q(diff, 97.5)})
    return pd.DataFrame(rows)


def pilot_report(df: pd.DataFrame) -> pd.DataFrame:
    """The quantities a single-plant pilot can measure (protocol section 11.2), per
    condition. None depends on the plant being diseased: P2 false alarms on healthy
    leaves, P3 the spatial policy's tissue detection and deferral on real foliage,
    P4 confidence. Every frame is assumed to show a leaf, as the protocol requires."""
    def per(g: pd.DataFrame) -> dict:
        h = g[~g.diseased]
        spatial = g.spray_spatial.astype(bool) if "spray_spatial" in g else None
        return {"frames": len(g), "healthy_frames": len(h),
                "p2_pred_diseased_on_healthy": _share(h.pred != "healthy", ~h.diseased),
                "p2_abstain_on_healthy": _share(h.abstain.astype(bool), ~h.diseased),
                "p2_wasted_gated": _share(h.spray.astype(bool), ~h.diseased),
                "p2_wasted_spatial": (_share(h.spray_spatial.astype(bool), ~h.diseased)
                                      if spatial is not None else np.nan),
                "p3_spatial_fires": float(spatial.mean()) if spatial is not None else np.nan,
                "p3_deferred": float(g.deferred.mean()),
                "p4_confidence": float(g.msp.mean())}
    return pd.DataFrame({c: per(g) for c, g in df.groupby("condition")}).T


def session_status(df: pd.DataFrame) -> dict:
    """What this session can and cannot support, decided from its own frames."""
    kind = df.attrs["unit_kind"]
    st = {"frames": int(len(df)), "leaves": int(df.leaf.nunique()),
          "units": int(df.unit.nunique()), "unit_kind": kind,
          "diseased_frames": int(df.diseased.sum()),
          "healthy_frames": int((~df.diseased).sum()),
          "intervals_estimable": bool(df.unit.nunique() >= 2),
          "primary_outcome_estimable": bool(df.diseased.any() and "missed_spatial" in df),
          "wasted_spray_estimable": bool((~df.diseased).any())}
    if kind == "plant":
        sick = df.groupby("unit").diseased.any()
        st["diseased_plants"], st["healthy_plants"] = int(sick.sum()), int((~sick).sum())
        st["pilot"] = bool(st["diseased_plants"] < MIN_DISEASED_PLANTS
                           or st["healthy_plants"] < MIN_HEALTHY_PLANTS)
    else:   # plant count unknown, so the protocol's minimum cannot be checked
        st["diseased_plants"] = st["healthy_plants"] = None
        st["pilot"] = None
    return st


def main() -> int:
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        print(__doc__)
        return 1
    df = load(dirs)
    rng = np.random.default_rng(SEED)
    n_plants = df.unit.nunique() if df.attrs["unit_kind"] == "plant" else "?"
    print(f"{len(df)} labelled frames, {df.leaf.nunique()} leaves, {n_plants} plants, "
          f"{df.session.nunique()} session(s)")
    print(f"truth: {df.truth.value_counts().to_dict()}")
    print(f"intervals resample whole {PLURAL[df.attrs['unit_kind']]}"
          + ("" if df.attrs["unit_kind"] == "plant" else
             "  (no plant_id on every frame: leaves of one plant treated as independent)"))
    src = df.get("diagnosis_source", pd.Series("", index=df.index)).fillna("").astype(str)
    src = src.where(src.str.len() > 0, "NOT RECORDED")
    print(f"diagnosis confirmed by: {src.value_counts().to_dict()}")
    st = session_status(df)
    if st["unit_kind"] == "plant":
        print(f"plants: {st['diseased_plants']} with diseased frames, "
              f"{st['healthy_plants']} healthy")
    if st["pilot"]:
        print(f"PILOT - below the protocol's minimum of {MIN_DISEASED_PLANTS} diseased and "
              f"{MIN_HEALTHY_PLANTS} healthy plants (docs/camera_experiment_protocol.md "
              f"sections 2 and 10). Report these numbers as a pilot, not as evidence.")
    if not st["intervals_estimable"]:
        print(f"only one {st['unit_kind']}: no confidence interval is estimable, so every "
              f"interval below is n/a.")
    if not st["diseased_frames"]:
        print("no diseased frames: missed treatment and the gated-vs-spatial comparison "
              "are NOT MEASURED (shown n/a), which is not the same as 0%.")
    if not st["healthy_frames"]:
        print("no healthy frames: wasted spray is NOT MEASURED (shown n/a).")
    print()

    rows = []
    for cond, g in df.groupby("condition"):
        rec = {"condition": cond, "frames": len(g), "leaves": g.leaf.nunique(),
               "units": g.unit.nunique()}
        for name, fn in (("accuracy", rate("correct")), ("confidence", rate("msp")),
                         ("abstain", rate("abstain")), ("spray", rate("spray")),
                         ("missed_treatment", lambda d: _share(d.missed, d.diseased)),
                         ("deferred", lambda d: _share(d.deferred_dis, d.diseased)),
                         ("wasted_spray", lambda d: _share(d.wasted, ~d.diseased))):
            p, lo, hi = cluster_ci(g, fn, rng)
            rec[name], rec[name + "_lo"], rec[name + "_hi"] = p, lo, hi
        rows.append(rec)
    t = pd.DataFrame(rows).set_index("condition")

    print(f"{'condition':16s} {'n/leaves':>9s} {'accuracy':>20s} {'conf':>6s} "
          f"{'abstain':>8s} {'spray':>6s} {'MISSED':>7s} {'defer':>6s} {'wasted':>7s}")
    for c, r in t.iterrows():
        print(f"{c:16s} {int(r.frames):4d}/{int(r.leaves):<4d} "
              f"{_f(r.accuracy, '6.1%', 6)} [{_f(r.accuracy_lo, '4.0%', 4)},"
              f"{_f(r.accuracy_hi, '4.0%', 4)}] "
              f"{r.confidence:6.2f} {r.abstain:8.1%} {r.spray:6.1%} "
              f"{_f(r.missed_treatment, '7.1%', 7)} {_f(r.deferred, '6.1%', 6)} "
              f"{_f(r.wasted_spray, '7.1%', 7)}")

    wrong = (~df.correct).astype(int)
    if 0 < wrong.sum() < len(df):
        auc, lo, hi = cluster_ci(
            df, lambda d: (roc_auc_score((~d.correct).astype(int), -d.msp)
                           if 0 < (~d.correct).sum() < len(d) else np.nan), rng)
        print(f"\nconfidence as an error detector: AUROC {auc:.3f} "
              f"[{_f(lo, '.3f', 3)}, {_f(hi, '.3f', 3)}]"
              "  (0.5 = no better than chance)")
    else:
        print("\nconfidence as an error detector: not estimable "
              "(every frame right, or every frame wrong)")

    if "normal" in df.condition.values:
        print("\npaired against 'normal', same leaf:")
        base = df[df.condition == "normal"].groupby("leaf")[["correct", "msp"]].mean()
        for cond in sorted(set(df.condition) - {"normal"}):
            other = df[df.condition == cond].groupby("leaf")[["correct", "msp"]].mean()
            both = base.join(other, lsuffix="_n", rsuffix="_c", how="inner")
            if both.empty:
                continue
            da = (both.correct_c - both.correct_n).to_numpy()
            dm = (both.msp_c - both.msp_n).to_numpy()
            # resample units, keeping each unit's leaves together
            unit_of = df.drop_duplicates("leaf").set_index("leaf").unit.reindex(both.index)
            by_unit = {u: da[(unit_of == u).to_numpy()] for u in unit_of.unique()}
            keys = list(by_unit)
            if len(keys) >= 2:
                bs = [np.concatenate([by_unit[k] for k in rng.choice(keys, len(keys))]).mean()
                      for _ in range(B)]
                ci = f"[{np.percentile(bs, 2.5):+.1%},{np.percentile(bs, 97.5):+.1%}]"
            else:
                ci = "[n/a]"
            print(f"  {cond:16s} {len(both)} leaves: accuracy change {da.mean():+.1%} "
                  f"{ci}, confidence change {dm.mean():+.3f}")

    out = dirs[0] if len(dirs) == 1 else dirs[0].parent
    if st["primary_outcome_estimable"]:
        pc = policy_comparison(df, rng)
        print(f"\nTWO POLICIES, same frames - diseased frames missed "
              f"(95% CI over {PLURAL[df.attrs['unit_kind']]}):")
        print(f"  {'condition':16s} {'gated':>20s} {'spatial-only':>20s} "
              f"{'gated - spatial':>22s}")
        for _, r in pc.iterrows():
            print(f"  {r.condition:16s} "
                  f"{_f(r.missed_gated, '6.1%', 6)} [{_f(r.missed_gated_lo, '5.1%', 5)},"
                  f"{_f(r.missed_gated_hi, '5.1%', 5)}] "
                  f"{_f(r.missed_spatial, '6.1%', 6)} [{_f(r.missed_spatial_lo, '5.1%', 5)},"
                  f"{_f(r.missed_spatial_hi, '5.1%', 5)}] "
                  f"{_f(r.difference, '+6.1%', 6)} [{_f(r.difference_lo, '+5.1%', 5)},"
                  f"{_f(r.difference_hi, '+5.1%', 5)}]")
        print(f"  wasted spray on healthy frames, all conditions: gated "
              f"{_f(_share(df.wasted, ~df.diseased), '.1%', 3)}, spatial-only "
              f"{_f(_share(df.wasted_spatial, ~df.diseased), '.1%', 3)}")
        notes = zero_miss_notes(df, {"gated": "missed", "spatial-only": "missed_spatial"})
        if notes:
            print("  zero misses are not proof of zero risk (supplementary, exact):")
            print("\n".join(notes))
        st["zero_miss_notes"] = [n.strip() for n in notes]
        pc.to_csv(out / "policy_comparison.csv", index=False)
    else:
        print("\nTWO POLICIES: PRIMARY OUTCOME NOT ESTIMABLE - "
              + ("no diseased frames; the gated-vs-spatial missed-treatment difference "
                 "needs diseased plants." if not st["diseased_frames"]
                 else "these frames carry no spatial-only decision (older log format)."))
        if "wasted_spatial" in df and st["healthy_frames"]:
            print(f"  wasted spray on healthy frames, all conditions: gated "
                  f"{_f(_share(df.wasted, ~df.diseased), '.1%', 3)}, spatial-only "
                  f"{_f(_share(df.wasted_spatial, ~df.diseased), '.1%', 3)}")

    if st["pilot"]:
        pr = pilot_report(df)
        print("\nPILOT QUANTITIES (protocol section 11.2; one-plant numbers, no intervals):")
        print(f"  {'condition':16s} {'frames':>6s} {'P2 called':>9s} {'P2 abst':>8s} "
              f"{'P2 wasted g/s':>14s} {'P3 fires':>9s} {'P3 dark':>8s} {'P4 conf':>8s}")
        for c, r in pr.iterrows():
            print(f"  {c:16s} {int(r.frames):6d} "
                  f"{_f(r.p2_pred_diseased_on_healthy, '9.0%', 9)} "
                  f"{_f(r.p2_abstain_on_healthy, '8.0%', 8)} "
                  f"{_f(r.p2_wasted_gated, '6.0%', 6)}/{_f(r.p2_wasted_spatial, '<6.0%', 6)} "
                  f"{_f(r.p3_spatial_fires, '9.0%', 9)} {_f(r.p3_deferred, '8.0%', 8)} "
                  f"{r.p4_confidence:8.2f}")
        print("  P2 called = healthy-leaf frames the classifier called diseased; P3 fires = "
              "frames the spatial-only policy sprays (tissue detected); P3 dark = deferred.\n"
              "  P3 counts every frame whatever its label: it shows the tissue test finds "
              "these leaves, not missed treatment on confirmed diseased leaves.")
        pr.to_csv(out / "pilot_report.csv")

    t.to_csv(out / "analysis_by_condition.csv")
    (out / "analysis_status.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    print(f"\nMISSED TREATMENT is the safety criterion: a diseased leaf the system "
          f"decided not to spray. wrote {out / 'analysis_by_condition.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
