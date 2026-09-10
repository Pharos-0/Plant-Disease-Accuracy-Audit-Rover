"""Fast validation of the whole project (Part 29).

Checks that the repository is in the state the thesis claims, without re-running
any expensive experiment. Nothing here recomputes a result and then presents it
as freshly reproduced — it verifies that committed results exist, are internally
consistent, and match the numbers quoted in the documentation.

    FAST VALIDATION   this script                        ~1 minute
    FULL REPRODUCTION see README "Full reproduction"     hours, GPU required

    python scripts/run_all_validations.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths

R = paths.RESULTS_DIR
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def q(name: str, value: float, expected: float, tol: float = 5e-4) -> None:
    ok = abs(value - expected) <= tol
    check(name, ok, f"{value:.4f} (documented {expected:.4f})")


def main() -> int:
    print("FAST VALIDATION — verifying committed results, not recomputing them\n")

    print("1. Result directories")
    for d in ("rq4", "rq5", "rq7_gdscore", "rq7b_batchnorm", "ti_audit",
              "ti_sample_size", "ti_candidate", "gate_evaluation", "figures"):
        p = R / d
        n = len(list(p.glob("*"))) if p.exists() else 0
        check(f"results/{d}", p.exists() and n > 0, f"{n} files")

    print("\n2. Headline numbers match the documentation")
    f = R / "rq7_gdscore" / "rq7_comparison_full_pool.csv"
    if f.exists():
        m = pd.read_csv(f).groupby("estimator").abs_error.mean()
        q("entropy full-pool MAE", m["entropy"], 0.1337)
        q("cot full-pool MAE", m["cot"], 0.1339)
        q("gdscore full-pool MAE", m["gdscore"], 0.1599)

    f = R / "ti_sample_size" / "table3_rho_by_arm_N.csv"
    if f.exists():
        d = pd.read_csv(f)
        for arm, N, exp in (("source_test", 25, 0.3997), ("source_test", 100, 0.4530),
                            ("tomatovillage", 25, -0.4726),
                            ("tomatovillage", 200, -0.5047),
                            ("plantdoc", 200, -0.4121)):
            row = d[(d.arm == arm) & (d.N == N)]
            if len(row):
                q(f"reversal {arm} N={N}", float(row["mean"].iloc[0]), exp)

    print("\n3. The reversal holds at every feasible N")
    f = R / "ti_sample_size" / "table4_reversal_by_N.csv"
    if f.exists():
        c = pd.read_csv(f)
        check("all zero-shift vs shifted contrasts positive",
              bool((c.ci_lo > 0).all()), f"{len(c)} comparisons, all CIs above zero")

    print("\n4. Missing cells are preserved, never fabricated")
    f = R / "ti_sample_size" / "feasibility.csv"
    if f.exists():
        d = pd.read_csv(f)
        n400 = d[d.requested_N == 400]
        check("N=400 unavailable on every arm", bool((~n400.feasible).all()),
              f"{len(n400)} arms, none feasible")
        st = d[(d.arm == "source_test") & (d.requested_N == 200)]
        check("source_test N=200 unavailable",
              bool(len(st) and not st.feasible.iloc[0]))
    f = R / "ti_sample_size" / "sample_size_errors.csv"
    if f.exists():
        e = pd.read_csv(f)
        gd = e[e.estimator == "gdscore"]
        missing = {(a, n) for a in e.arm.unique() for n in e.N.unique()} - \
                  set(map(tuple, gd[["arm", "N"]].drop_duplicates().values))
        check("GdScore missing cells preserved", len(missing) > 0,
              f"{len(missing)} (arm, N) cells absent — GdScore needs >=128 crops")

    print("\n5. The failed candidate is recorded, not hidden")
    f = R / "ti_candidate" / "candidate_summary.csv"
    if f.exists():
        d = pd.read_csv(f, index_col=0)
        improved = int((d["ti"] < d["base"]).sum())
        check("shrinkage candidate improved 0 of 9 bases", improved == 0,
              f"{improved} improved")

    print("\n6. Manifests carry provenance")
    n_man = 0
    for m in R.rglob("manifest*.json"):
        try:
            j = json.loads(m.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "git_commit" in j:
            n_man += 1
    check("experiment manifests with git commit", n_man >= 8, f"{n_man} manifests")

    print("\n7. Figures and tables")
    nf = len(list((R / "figures").glob("*.png")))
    check("figures present", nf >= 30, f"{nf} figures")
    nt = len(list((R / "tables" / "thesis").glob("*.csv")))
    check("thesis tables present", nt >= 10, f"{nt} tables")

    print("\n8. Test suite")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                       capture_output=True, text=True)
    tail = [ln for ln in r.stdout.strip().splitlines() if "passed" in ln or "failed" in ln]
    check("pytest", r.returncode == 0, tail[-1] if tail else "no summary line")

    print("\n9. Leakage firewall")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                        "tests/test_leakage_firewall.py"], cwd=ROOT,
                       capture_output=True, text=True)
    check("estimators cannot import a label source", r.returncode == 0)

    n_ok = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"\n{'=' * 70}\n{n_ok}/{len(CHECKS)} checks passed")
    failed = [n for n, ok, _ in CHECKS if not ok]
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
        return 1
    print("FAST VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
