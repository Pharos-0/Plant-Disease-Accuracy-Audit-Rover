"""Rover demonstration — four scenarios, all from committed evidence.

Runs without a rover, without a camera and without a GPU: every scenario replays
cached activations from real datasets. Nothing is simulated and no field trial is
implied.

    A  NORMAL MODE     classifier -> downstream autonomous action
    B  RESEARCH MODE   session gate -> OPERATE / REVIEW / WITHHOLD
    C  SEVERE SHIFT    what actually happens in the field
    D  SMALL-LABEL     the deployable alternative (RQ5)

The point of C is negative and is the honest headline: under the tested field
shift the classifier is ~0.50 accurate, so autonomous spraying should not be
enabled. The gate is instrumentation for that finding, **not** a proven safety
mechanism.

    python scripts/run_rover_demo.py [--arm tomatovillage] [--session-size 100]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths
from estimators.base import SourceStats
from estimators.methods import REGISTRY
from inference.cache import CacheKey, load_features, load_labels
from rover.session_gate import GateConfig, GateDecision, PerImageQueryError, SessionGate

RULE = "=" * 78


def head(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def pick_model(backbone: str = "resnet50", seed: int = 0) -> dict:
    for mp in sorted(paths.RAW_RESULTS.glob("*/manifest_*.json")):
        c = json.loads(mp.read_text(encoding="utf-8"))["config"]
        if (c["version"] in ("V2", "V9") and c["label_set"] == "SHARED_3"
                and c["backbone"] == backbone and c["seed"] == seed):
            return c
    raise SystemExit("no matching frozen model found")


def load_arm(cfg: dict, arm: str):
    k = CacheKey(cfg["version"], cfg["backbone"], cfg["seed"], arm, cfg["label_set"])
    return load_features(k), load_labels(k)


def demo_a(cfg, tgt, y, rng) -> None:
    head("DEMONSTRATION A — NORMAL MODE (research module disabled)")
    print("camera -> adaptive vision -> leaf proposal -> frozen classifier -> action")
    print("The rover is fully functional with the research module switched off.\n")
    idx = rng.choice(len(tgt["logits"]), 5, replace=False)
    classes = ["healthy", "late_blight", "leaf_mold"]
    for i in idx:
        lg = tgt["logits"][i]
        p = np.exp(lg - lg.max()) / np.exp(lg - lg.max()).sum()
        pred = int(lg.argmax())
        print(f"  crop {i:>5}  prediction={classes[pred]:<12} "
              f"confidence={p.max():.3f}  -> downstream autonomous action")
    print("\n  NOTE: 'downstream autonomous action' is deliberately abstract. No")
    print("  validated disease->chemical mapping exists in this project, so none")
    print("  is claimed.")


def demo_b(cfg, src, tgt, n: int, rng) -> None:
    head("DEMONSTRATION B — RESEARCH MODE (session gate)")
    est = REGISTRY["entropy"]["factory"]().fit(src)
    gate = SessionGate(est, GateConfig(min_session_size=n))
    print(f"estimator: {est.name} | min_session_size: {n}\n")

    v = gate.verdict()
    print(f"  after 0 observations      -> {v.decision.value}")
    idx = rng.choice(len(tgt["logits"]), n, replace=False)
    for j, i in enumerate(idx, 1):
        gate.observe(tgt["logits"][i], tgt["features"][i])
        if j in (n // 4, n // 2, n):
            v = gate.verdict()
            acc = "n/a" if v.estimated_accuracy is None else f"{v.estimated_accuracy:.3f}"
            print(f"  after {j:>4} observations   -> {v.decision.value:<26} "
                  f"estimated accuracy {acc}")

    print("\n  The gate refuses per-image questions, by construction:")
    try:
        gate.should_spray_this_leaf(0)
    except PerImageQueryError as exc:
        print(f"    PerImageQueryError: {str(exc)[:96]}...")
    print("\n  This is a RESEARCH DEMONSTRATION, not a proven safety gate.")


def demo_c(cfg, rng, n: int) -> None:
    head("DEMONSTRATION C — SEVERE SHIFT: what actually happens in the field")
    rows = []
    for arm in ("source_test", "plantdoc", "tomatovillage"):
        try:
            t, y = load_arm(cfg, arm)
        except FileNotFoundError:
            continue
        groups = t["group_ids"]
        uniq = np.unique(groups)
        take = min(n, len(uniq))
        pick = rng.choice(uniq, take, replace=False)
        idx = np.flatnonzero(np.isin(groups, pick))
        acc = float((t["logits"][idx].argmax(1) == y[idx]).mean())
        rows.append((arm, take, len(idx), acc))
    print(f"  {'arm':<15}{'photos':>7}{'crops':>7}{'TRUE accuracy':>15}")
    for arm, ph, cr, acc in rows:
        flag = "  <- lab" if acc > 0.8 else "  <- field: classifier is near chance-plus"
        print(f"  {arm:<15}{ph:>7}{cr:>7}{acc:>15.3f}{flag}")
    print("\n  Measured over the full experiment (20 models, 40 draws per cell):")
    print("    at tau_safe = 0.80, sessions genuinely safe to act on:")
    print("      source_test 100%   |   PlantDoc 0.2%   |   Tomato-Village 0.1%")
    print("    within-model ranking of field sessions: best Spearman 0.22")
    print("\n  CONCLUSION: autonomous spraying should NOT be enabled under this shift.")
    print("  The limiting factor is the classifier (~0.50 in the field), not the gate.")


def demo_d() -> None:
    head("DEMONSTRATION D — the deployable alternative: a small labelled sample")
    import pandas as pd

    lab = paths.RESULTS_DIR / "rq5" / "rq5_settingA_labelled.csv"
    lf = paths.RESULTS_DIR / "rq5" / "rq5_labelfree_full_pool.csv"
    print("  RQ5, pre-registered, Tomato-Village, 20 models, 100 repetitions,")
    print("  photograph-level sampling (391 photographs, never crop-level):\n")
    if lab.exists():
        d = pd.read_csv(lab)
        curve = d.groupby("n_label_photographs").abs_error.mean()
        best_lf = None
        if lf.exists():
            best_lf = pd.read_csv(lf).groupby("estimator").abs_error.mean().min()
        for n, mae in curve.items():
            mark = ""
            if best_lf is not None and mae < best_lf:
                mark = "  <- already better than the best label-free estimator"
            print(f"    {int(n):>4} labelled photographs -> MAE {mae:.4f}{mark}")
        if best_lf is not None:
            print(f"\n    best label-free estimator on the full 1,616-crop pool: "
                  f"MAE {best_lf:.4f}")
    else:
        print("    results/rq5/rq5_settingA_labelled.csv not found")
    print(f"\n  tables: {(paths.RESULTS_DIR / 'rq5').relative_to(ROOT)}")
    print("  This is practical evidence FOR THIS SETTING, not a universal budget.")
    print("  It is consistent with prediction-powered-inference theory: the")
    print("  labelling saving scales with the R^2 between predictions and truth,")
    print("  and our measured R^2 is 0.026.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="tomatovillage")
    ap.add_argument("--session-size", type=int, default=100)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(20260911)
    cfg = pick_model(args.backbone, args.seed)
    src_cache, src_y = load_arm(cfg, "source_val")
    src = SourceStats(logits=src_cache["logits"], labels=src_y,
                      features=src_cache["features"])
    tgt, y = load_arm(cfg, args.arm)

    print(RULE)
    print("ROVER DEMONSTRATION — replayed from cached real data")
    print(f"model: {cfg['version']}/{cfg['backbone']}/seed{cfg['seed']} "
          f"({cfg['label_set']}) | target arm: {args.arm}")
    print("No camera, no GPU, no field trial. Nothing here is simulated or invented.")
    print(RULE)

    demo_a(cfg, tgt, y, rng)
    demo_b(cfg, src, tgt, args.session_size, rng)
    demo_c(cfg, rng, args.session_size)
    demo_d()

    head("SUMMARY")
    print("  A  the rover works with the research module disabled")
    print("  B  the gate produces a session-level verdict and refuses per-image queries")
    print("  C  under field shift the classifier is ~0.50, so autonomy is withheld")
    print("  D  ~20-50 labelled photographs are the practical route to knowing accuracy")
    print("\n  Never claimed: per-image correctness, validated chemical treatment,")
    print("  or demonstrated safe autonomous spraying.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
