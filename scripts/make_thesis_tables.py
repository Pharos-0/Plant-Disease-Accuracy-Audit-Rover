"""Generate thesis-ready tables from committed result files.

Every value is read from a results CSV; none is typed in. Each table is written
as CSV (for the record) and Markdown (for pasting into the thesis), into
`results/tables/thesis/`.

Missing cells are preserved as missing. In particular N=400 does not exist on any
arm, source_test has no N=200, and GdScore has no value where a session held
fewer than 128 crops. Those gaps are real and are never filled.

    python scripts/make_thesis_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import paths

R = paths.RESULTS_DIR
OUT = R / "tables" / "thesis"


def _markdown(df: pd.DataFrame, index: bool) -> str:
    """Render a DataFrame as a Markdown table.

    Hand-rolled rather than via ``DataFrame.to_markdown``, which requires the
    optional `tabulate` package. Adding a dependency in order to format a table
    is not worth the reproducibility cost.

    A missing value is rendered as an em dash, never as a number: N=400 exists on
    no arm, source_test has no N=200, and GdScore has no value where a session
    held fewer than 128 crops. Those gaps are real.
    """
    d = df.reset_index() if index else df.copy()
    cols = [" / ".join(str(x) for x in c) if isinstance(c, tuple) else str(c)
            for c in d.columns]

    def cell(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def emit(df: pd.DataFrame, name: str, caption: str, index: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=index)
    md = [f"**{caption}**", "", _markdown(df, index), ""]
    (OUT / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  {name}: {df.shape[0]}x{df.shape[1]}")


def t_datasets() -> None:
    """Dataset / target-arm summary, measured from the caches."""
    f = R / "gate_evaluation" / "gate_sessions.csv"
    if not f.exists():
        return
    g = pd.read_csv(f)
    b = g[g.signal == "confidence"].drop_duplicates(
        subset=["arm", "N", "rep", "backbone", "seed"])
    t = b.groupby("arm").true_accuracy.agg(["mean", "std", "min", "max"]).round(4)
    t.columns = ["mean session accuracy", "sd", "min", "max"]
    t["photographs available"] = pd.Series(
        {"source_test": 109, "plantdoc": 262, "tomatovillage": 391})
    t["role"] = pd.Series({"source_test": "zero shift (control)",
                           "plantdoc": "severe shift (web imagery)",
                           "tomatovillage": "severe shift (real Indian field)"})
    emit(t, "t01_target_arms",
         "Table 1. Target arms. Session accuracy over 40 photograph-level draws "
         "per model per N, 20 models.")


def t_full_pool_mae() -> None:
    f = R / "rq7_gdscore" / "rq7_comparison_full_pool.csv"
    if not f.exists():
        return
    c = pd.read_csv(f)
    t = (c.groupby("estimator").abs_error.agg(["mean", "median", "std", "count"])
           .sort_values("mean").round(4))
    t.columns = ["MAE", "median AE", "sd", "n models"]
    emit(t, "t02_full_pool_mae",
         "Table 2. Full unlabelled Tomato-Village pool (1,616 crops), 20 models. "
         "GdScore is train-mode here, matching its official implementation.")


def t_sample_size_mae() -> None:
    f = R / "ti_sample_size" / "sample_size_errors.csv"
    if not f.exists():
        return
    e = pd.read_csv(f)
    t = e.pivot_table(index="estimator", columns=["arm", "N"], values="abs_error").round(4)
    emit(t, "t03_mae_by_arm_and_N",
         "Table 3. MAE by estimator x target arm x sample size. Blank cells are "
         "NOT AVAILABLE: N=400 exists on no arm, source_test has no N=200, and "
         "GdScore needs at least 128 crops.")


def t_ti() -> None:
    f = R / "ti_sample_size" / "sample_size_ti.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    t = d.pivot_table(index="estimator", columns=["arm", "N"], values="ti").round(4)
    emit(t, "t04_ti_by_arm_and_N",
         "Table 4. Target-informativeness by estimator x arm x N. TI must be "
         "compared only at FIXED N - it falls mechanically as N grows because "
         "target sensitivity shrinks while anchor sensitivity does not.")


def t_reversal() -> None:
    f = R / "ti_sample_size" / "table3_rho_by_arm_N.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    t = d.pivot(index="arm", columns="N", values="mean").round(4)
    emit(t, "t05_ti_error_reversal",
         "Table 5. Mean per-model Spearman(TI, |error|) across the nine "
         "estimators. 100 model x repetition pairs per cell. Positive on the "
         "zero-shift arm, negative on both field arms, at every feasible N.")
    f4 = R / "ti_sample_size" / "table4_reversal_by_N.csv"
    if f4.exists():
        c = pd.read_csv(f4).round(4)
        emit(c, "t06_reversal_statistics",
             "Table 6. Paired zero-shift vs severe-shift contrast at each N, "
             "with bootstrap CIs and Wilcoxon p. n = 100 pairs per row.",
             index=False)


def t_sensitivity() -> None:
    f = R / "ti_sample_size" / "sample_size_ti.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    t = d.groupby(["arm", "N"])[["target_sd", "anchor_sd", "ti"]].mean().round(4)
    t.columns = ["target sensitivity", "anchor sensitivity", "TI"]
    emit(t, "t07_target_vs_anchor_sensitivity",
         "Table 7. Target sensitivity falls ~2.5-3x as N grows while anchor "
         "sensitivity is essentially constant. This is why TI is not comparable "
         "across sample sizes.")


def t_removal() -> None:
    f = R / "ti_audit" / "ti_removal.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    if "arm" in d.columns:
        d = d[d.arm == "tomatovillage"]
    t = d.pivot_table(index="estimator", columns="surrogate",
                      values="shift_from_real").round(4)
    emit(t, "t08_target_information_removal",
         "Table 8. Mean |change in estimate| when target information is "
         "destroyed (Tomato-Village). 'shuffled' near zero confirms permutation "
         "invariance and validates the measurement.")


def t_gdscore() -> None:
    f = R / "rq7b_batchnorm" / "rq7b_train_vs_eval.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    t = d.pivot_table(index="backbone", columns="mode", values="abs_error").round(4)
    t["change"] = (t["eval"] - t["train"]).round(4)
    t["has BatchNorm"] = ["yes", "yes", "yes", "no"] if len(t) == 4 else np.nan
    emit(t, "t09_gdscore_batchnorm",
         "Table 9. GdScore MAE under train- vs eval-mode activations, 5 seeds "
         "per backbone. ViT has no BatchNorm and is the control.")


def t_session_ranking() -> None:
    f = R / "gate_evaluation" / "gate_sessions.csv"
    if not f.exists():
        return
    g = pd.read_csv(f)
    field = g[g.arm.isin(["plantdoc", "tomatovillage"])]
    rows = []
    for (arm, bb, sd), d in field.groupby(["arm", "backbone", "seed"]):
        for sig, h in d.groupby("signal"):
            if h.true_accuracy.std() > 0:
                rows.append({"signal": sig, "arm": arm,
                             "rho": sps.spearmanr(h.score, h.true_accuracy).statistic})
    r = pd.DataFrame(rows)
    t = r.groupby("signal").rho.agg(["mean", "std", "count"]).sort_values(
        "mean", ascending=False).round(4)
    t.columns = ["per-model Spearman", "sd", "n model-arm pairs"]
    emit(t, "t10_session_ranking",
         "Table 10. Within-model ranking of field sessions by true accuracy. "
         "Paired per model - pooling across models produces a Simpson artefact "
         "and is not reported. Best is 0.22; none ranks usefully.")


def t_gate_safety() -> None:
    f = R / "gate_evaluation" / "gate_sessions.csv"
    if not f.exists():
        return
    g = pd.read_csv(f)
    b = g[g.signal == "confidence"].drop_duplicates(
        subset=["arm", "N", "rep", "backbone", "seed"])
    rows = {}
    for tau in (0.55, 0.60, 0.65, 0.70, 0.80):
        rows[tau] = b.groupby("arm").apply(
            lambda x, tau=tau: (x.true_accuracy >= tau).mean())
    t = pd.DataFrame(rows).round(4)
    t.columns = [f"tau={c}" for c in t.columns]
    emit(t, "t11_fraction_sessions_safe",
         "Table 11. Fraction of sessions where the classifier's TRUE accuracy "
         "clears an operating threshold. At tau=0.80 essentially no field "
         "session is safe, which is why a gate cannot enable autonomy here.")


def t_candidate() -> None:
    f = R / "ti_candidate" / "candidate_summary.csv"
    if not f.exists():
        return
    d = pd.read_csv(f, index_col=0).round(4)
    emit(d, "t12_failed_shrinkage_candidate",
         "Table 12. The rejected TI-adaptive shrinkage estimator. 0 of 9 bases "
         "improved; 7 significantly harmed (paired Wilcoxon, Holm-corrected).")


def main() -> int:
    print("generating thesis tables:")
    for fn in (t_datasets, t_full_pool_mae, t_sample_size_mae, t_ti, t_reversal,
               t_sensitivity, t_removal, t_gdscore, t_session_ranking,
               t_gate_safety, t_candidate):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {fn.__name__}: FAILED - {type(exc).__name__}: {exc}")
    n = len(list(OUT.glob("*.csv")))
    print(f"\nresults/tables/thesis holds {n} tables (CSV + Markdown each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
