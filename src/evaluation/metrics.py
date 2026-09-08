"""Scoring estimates against the truth.

This module MAY read target labels. It runs strictly after estimates are written
to ``results/raw/estimates/`` (MASTER_PROMPT section 5). Nothing here may be
imported by ``src/estimators/`` - the firewall test enforces that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def session_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """The ground truth an estimator was trying to guess."""
    return float((logits.argmax(axis=1) == labels).mean())


def error_metrics(estimated: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Bias, spread and agreement between estimate and truth."""
    est, act = np.asarray(estimated, float), np.asarray(actual, float)
    err = est - act
    out = {
        "n_sessions": int(len(est)),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "std_error": float(err.std(ddof=1)) if len(err) > 1 else 0.0,
        "std_estimate": float(est.std(ddof=1)) if len(est) > 1 else 0.0,
        "std_actual": float(act.std(ddof=1)) if len(act) > 1 else 0.0,
        "mean_estimate": float(est.mean()),
        "mean_actual": float(act.mean()),
        "max_abs_error": float(np.abs(err).max()),
        "q95_abs_error": float(np.quantile(np.abs(err), 0.95)),
    }
    if len(est) > 2 and est.std() > 1e-12 and act.std() > 1e-12:
        out["pearson_r"] = float(np.corrcoef(est, act)[0, 1])
    else:
        out["pearson_r"] = float("nan")
    return out


def bootstrap_ci(
    values: np.ndarray, statistic=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of `values`."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    if len(values) < 2:
        v = float(statistic(values)) if len(values) else float("nan")
        return v, v
    stats = np.array(
        [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    )
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def gate_operating_characteristic(
    estimated: np.ndarray, actual: np.ndarray, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """The deployment gate as a decision rule.

    For threshold tau the rover operates when the *estimate* clears tau. The
    costly error is an **unsafe operate**: the estimate says go, the true
    session accuracy does not clear tau. The other error is a wasted withhold.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.50, 0.96, 0.01), 2)
    est, act = np.asarray(estimated, float), np.asarray(actual, float)

    rows = []
    for tau in thresholds:
        operate, truly_ok = est >= tau, act >= tau
        unsafe = int((operate & ~truly_ok).sum())
        wasted = int((~operate & truly_ok).sum())
        correct_op = int((operate & truly_ok).sum())
        correct_wh = int((~operate & ~truly_ok).sum())
        n = len(est)
        n_good = int(truly_ok.sum())
        n_op = int(operate.sum())
        rows.append(
            {
                "tau": float(tau),
                "n": n,
                "correct_operate": correct_op,
                "unsafe_operate": unsafe,
                "wasted_withhold": wasted,
                "correct_withhold": correct_wh,
                "unsafe_operate_rate": unsafe / n if n else float("nan"),
                "wasted_withhold_rate": wasted / n if n else float("nan"),
                "operate_rate": n_op / n if n else float("nan"),
                "accuracy_of_gate": (correct_op + correct_wh) / n if n else float("nan"),
                # --- the three below are what stop the accuracy paradox ---
                # base rate: when almost no session is truly good, a gate that
                # always withholds scores high "accuracy" while being useless.
                "base_rate_good": n_good / n if n else float("nan"),
                # of the sessions that really were good, how many did we use?
                "recall_of_good": correct_op / n_good if n_good else float("nan"),
                # of the sessions we operated on, how many really were good?
                "precision_of_operate": correct_op / n_op if n_op else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def minimum_session_size(
    per_n: pd.DataFrame, tolerance: float, *, n_col: str = "n_requested", mae_col: str = "mae"
) -> int | None:
    """Smallest N whose MAE is at or below `tolerance`, or None if none is.

    This is N* - the headline quantity of H3.
    """
    ok = per_n[per_n[mae_col] <= tolerance].sort_values(n_col)
    return int(ok.iloc[0][n_col]) if len(ok) else None
