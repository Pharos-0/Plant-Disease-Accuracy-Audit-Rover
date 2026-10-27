"""The leakage firewall, enforced in code rather than by discipline.

Study protocol (leakage control):

    src/estimators/  MAY see: source val labels, source+target logits/features.
                     NEVER target labels.
    src/evaluation/  MAY see: target labels. Runs strictly AFTER estimates are
                     written to disk.

These tests fail the build if an estimator gains a route to a target label.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

ESTIMATOR_DIR = SRC / "estimators"

# Modules an estimator may import. Anything able to reach a label store - the
# dataset index, the split tables, the evaluation code, image loading - is out.
ALLOWED_TOP_LEVEL = {
    "__future__",
    "numpy",
    "scipy",
    "typing",
    "dataclasses",
    "abc",
    "math",
    "collections",
    "functools",
    "itertools",
    "warnings",
    "json",
}

FORBIDDEN_SUBSTRINGS = (
    "data.index",
    "data.splits",
    "data.verify",
    "evaluation",
    "sessions",
    "pandas",
    "pyarrow",
    "PIL",
    "torchvision",
    "training",
)


def estimator_modules() -> list[Path]:
    return sorted(p for p in ESTIMATOR_DIR.glob("*.py") if p.name != "__init__.py")


def imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import stays inside src/estimators/, and every module
            # in that package is itself checked by this test, so it is safe.
            if node.level:
                continue
            names.append(node.module or "")
    return names


def test_estimator_dir_is_not_empty() -> None:
    assert estimator_modules(), "no estimator modules found - the firewall test would be vacuous"


@pytest.mark.parametrize("path", estimator_modules(), ids=lambda p: p.name)
def test_no_forbidden_imports(path: Path) -> None:
    for name in imported_names(path):
        if not name:
            continue
        for bad in FORBIDDEN_SUBSTRINGS:
            assert bad not in name, (
                f"{path.name} imports {name!r}, which can reach a target label. "
                "Estimators must never have that route."
            )
        top = name.split(".")[0]
        assert top in ALLOWED_TOP_LEVEL, (
            f"{path.name} imports {name!r}; only {sorted(ALLOWED_TOP_LEVEL)} are "
            "permitted inside src/estimators/."
        )


def test_estimate_signature_has_no_label_parameter() -> None:
    """No estimator may expose a parameter through which labels could arrive."""
    from estimators.methods import ALL_ESTIMATORS, build

    banned = {"y", "labels", "target_labels", "y_true", "gt", "ground_truth", "targets"}
    for name in ALL_ESTIMATORS:
        est = build(name)
        params = set(inspect.signature(est.estimate).parameters)
        overlap = params & banned
        assert not overlap, f"{name}.estimate exposes label-shaped parameters: {overlap}"
        assert params <= {"target_logits", "target_features"}, (
            f"{name}.estimate accepts unexpected parameters: {params}"
        )


def test_estimates_are_probabilities() -> None:
    """Section 14: estimator output outside [0, 1] or NaN must fail loudly."""
    from estimators.base import SourceStats
    from estimators.methods import ALL_ESTIMATORS, build

    rng = np.random.default_rng(0)
    n, k = 500, 3
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 2.5
    source = SourceStats(logits=logits, labels=y)

    target = rng.normal(0, 1, (200, k))
    for name in ALL_ESTIMATORS:
        est = build(name)
        est.fit(source)
        value = est.estimate(target)
        assert 0.0 <= value <= 1.0, f"{name} returned {value}, outside [0, 1]"
        assert np.isfinite(value), f"{name} returned a non-finite estimate"


def test_estimator_rejects_empty_target() -> None:
    from estimators.base import SourceStats
    from estimators.methods import build

    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 200)
    logits = rng.normal(0, 1, (200, 3))
    est = build("atc_mc").fit(SourceStats(logits=logits, labels=y))
    with pytest.raises(ValueError):
        est.estimate(np.empty((0, 3)))


def test_unfitted_estimator_refuses_to_estimate() -> None:
    from estimators.methods import build

    est = build("atc_mc")
    with pytest.raises(RuntimeError):
        est.estimate(np.zeros((10, 3)))
