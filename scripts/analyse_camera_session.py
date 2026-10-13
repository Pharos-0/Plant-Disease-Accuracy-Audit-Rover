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
    units = df.unit.unique()
    groups = {k: g for k, g in df.groupby("unit")}
    point = fn(df)
    vals = []
    for _ in range(B):
        pick = rng.choice(units, len(units), replace=True)
        vals.append(fn(pd.concat([groups[k] for k in pick])))
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
        idx = rng.integers(0, len(per), (B, len(per)))
        boot = per[idx, 1:].sum(1) / per[idx, 0].sum(1)[:, None]
        diff = boot[:, 0] - boot[:, 1]
        rows.append({"condition": cond, "diseased_frames": len(g),
                     "units": len(per), "unit_kind": df.attrs["unit_kind"],
                     "missed_gated": point[0],
                     "missed_gated_lo": np.percentile(boot[:, 0], 2.5),
                     "missed_gated_hi": np.percentile(boot[:, 0], 97.5),
                     "missed_spatial": point[1],
                     "missed_spatial_lo": np.percentile(boot[:, 1], 2.5),
                     "missed_spatial_hi": np.percentile(boot[:, 1], 97.5),
                     "difference": point[0] - point[1],
                     "difference_lo": np.percentile(diff, 2.5),
                     "difference_hi": np.percentile(diff, 97.5)})
    return pd.DataFrame(rows)


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
    print(f"diagnosis confirmed by: {src.value_counts().to_dict()}\n")

    rows = []
    for cond, g in df.groupby("condition"):
        rec = {"condition": cond, "frames": len(g), "leaves": g.leaf.nunique(),
               "units": g.unit.nunique()}
        for name, fn in (("accuracy", rate("correct")), ("confidence", rate("msp")),
                         ("abstain", rate("abstain")), ("spray", rate("spray")),
                         ("missed_treatment", lambda d: float(d.missed.sum() /
                                                              max(1, d.diseased.sum()))),
                         ("deferred", lambda d: float(d.deferred_dis.sum() /
                                                      max(1, d.diseased.sum()))),
                         ("wasted_spray", lambda d: float(d.wasted.sum() /
                                                          max(1, (~d.diseased).sum())))):
            p, lo, hi = cluster_ci(g, fn, rng)
            rec[name], rec[name + "_lo"], rec[name + "_hi"] = p, lo, hi
        rows.append(rec)
    t = pd.DataFrame(rows).set_index("condition")

    print(f"{'condition':16s} {'n/leaves':>9s} {'accuracy':>20s} {'conf':>6s} "
          f"{'abstain':>8s} {'spray':>6s} {'MISSED':>7s} {'defer':>6s} {'wasted':>7s}")
    for c, r in t.iterrows():
        print(f"{c:16s} {int(r.frames):4d}/{int(r.leaves):<4d} "
              f"{r.accuracy:6.1%} [{r.accuracy_lo:4.0%},{r.accuracy_hi:4.0%}] "
              f"{r.confidence:6.2f} {r.abstain:8.1%} {r.spray:6.1%} "
              f"{r.missed_treatment:7.1%} {r.deferred:6.1%} {r.wasted_spray:7.1%}")

    wrong = (~df.correct).astype(int)
    if 0 < wrong.sum() < len(df):
        auc, lo, hi = cluster_ci(
            df, lambda d: (roc_auc_score((~d.correct).astype(int), -d.msp)
                           if 0 < (~d.correct).sum() < len(d) else np.nan), rng)
        print(f"\nconfidence as an error detector: AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]"
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
            bs = [np.concatenate([by_unit[k] for k in rng.choice(keys, len(keys))]).mean()
                  for _ in range(B)]
            print(f"  {cond:16s} {len(both)} leaves: accuracy change {da.mean():+.1%} "
                  f"[{np.percentile(bs, 2.5):+.1%},{np.percentile(bs, 97.5):+.1%}], "
                  f"confidence change {dm.mean():+.3f}")

    out = dirs[0] if len(dirs) == 1 else dirs[0].parent
    if "missed_spatial" in df and df.diseased.any():
        pc = policy_comparison(df, rng)
        print(f"\nTWO POLICIES, same frames - diseased frames missed "
              f"(95% CI over {PLURAL[df.attrs['unit_kind']]}):")
        print(f"  {'condition':16s} {'gated':>20s} {'spatial-only':>20s} "
              f"{'gated - spatial':>22s}")
        for _, r in pc.iterrows():
            print(f"  {r.condition:16s} "
                  f"{r.missed_gated:6.1%} [{r.missed_gated_lo:5.1%},{r.missed_gated_hi:5.1%}] "
                  f"{r.missed_spatial:6.1%} [{r.missed_spatial_lo:5.1%},{r.missed_spatial_hi:5.1%}] "
                  f"{r.difference:+6.1%} [{r.difference_lo:+5.1%},{r.difference_hi:+5.1%}]")
        nh = max(1, (~df.diseased).sum())
        print(f"  wasted spray on healthy frames, all conditions: gated "
              f"{df.wasted.sum() / nh:.1%}, spatial-only {df.wasted_spatial.sum() / nh:.1%}")
        pc.to_csv(out / "policy_comparison.csv", index=False)

    t.to_csv(out / "analysis_by_condition.csv")
    print(f"\nMISSED TREATMENT is the safety criterion: a diseased leaf the system "
          f"decided not to spray. wrote {out / 'analysis_by_condition.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
