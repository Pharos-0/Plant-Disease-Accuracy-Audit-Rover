"""Session construction and the N-sweep.

A session is what a deployed machine actually accumulates: N unlabeled images
from one outing. Sessions are sampled at the level of the **grouping unit**
(a Tomato-Village photograph, a PlantVillage leaf), never at the level of the
individual image - sibling crops of one photograph are not independent
observations, and treating them as such would make a session look easier than
it is and bias every estimate optimistically.

Sessions at large N necessarily overlap: Tomato-Village's SHARED_3 subset holds
1,616 images but only 391 distinct photographs. That overlap is measured and
reported here rather than hidden (MASTER_PROMPT section 6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SESSION_SIZES = [25, 50, 100, 200, 400]
N_BOOTSTRAP = 200


@dataclass
class SessionPlan:
    """Index arrays into a cached arm, one row per session."""

    indices: list[np.ndarray]
    n_requested: int
    arm: str
    seed: int

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def actual_sizes(self) -> np.ndarray:
        return np.array([len(i) for i in self.indices])


def sample_session(
    group_ids: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw one session of exactly `n` image indices, sampling whole groups.

    Groups are drawn without replacement until the accumulated image count
    reaches `n`; the final group is then trimmed at random so the session has
    exactly `n` images. If the pool cannot supply `n` images the whole pool is
    returned and the shortfall is visible in ``SessionPlan.actual_sizes``.
    """
    order = rng.permutation(np.unique(group_ids))
    by_group = {g: np.flatnonzero(group_ids == g) for g in order}

    chosen: list[np.ndarray] = []
    total = 0
    for g in order:
        idx = by_group[g]
        if total + len(idx) <= n:
            chosen.append(idx)
            total += len(idx)
        else:
            need = n - total
            if need > 0:
                chosen.append(rng.choice(idx, size=need, replace=False))
                total = n
            break
        if total == n:
            break

    if not chosen:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(chosen))


def build_sessions(
    group_ids: np.ndarray,
    n: int,
    *,
    arm: str,
    seed: int,
    n_sessions: int = N_BOOTSTRAP,
) -> SessionPlan:
    rng = np.random.default_rng(seed)
    return SessionPlan(
        indices=[sample_session(group_ids, n, rng) for _ in range(n_sessions)],
        n_requested=n,
        arm=arm,
        seed=seed,
    )


def overlap_stats(plan: SessionPlan, pool_size: int, n_groups: int) -> dict[str, float]:
    """How much sessions of this size must share, given the pool.

    ``mean_pairwise_jaccard`` is measured on a capped number of pairs;
    ``expected_coverage`` is the fraction of the pool one session consumes.
    """
    sizes = plan.actual_sizes
    sets = [set(i.tolist()) for i in plan.indices]
    pairs, jac = 0, 0.0
    max_pairs = 2000
    rng = np.random.default_rng(0)
    m = len(sets)
    while pairs < max_pairs and m > 1:
        a, b = rng.integers(0, m, 2)
        if a == b:
            continue
        inter = len(sets[a] & sets[b])
        union = len(sets[a] | sets[b])
        jac += inter / union if union else 0.0
        pairs += 1

    return {
        "n_requested": float(plan.n_requested),
        "mean_actual_size": float(sizes.mean()),
        "min_actual_size": float(sizes.min()),
        "short_sessions": float(int((sizes < plan.n_requested).sum())),
        "pool_size": float(pool_size),
        "pool_groups": float(n_groups),
        "expected_coverage": float(sizes.mean() / max(pool_size, 1)),
        "mean_pairwise_jaccard": float(jac / pairs) if pairs else 0.0,
    }


def validate_plan(plan: SessionPlan) -> list[str]:
    """Section 14: sessions smaller than requested must fail loudly, not silently."""
    problems = []
    sizes = plan.actual_sizes
    if len(plan) == 0:
        problems.append(f"{plan.arm} N={plan.n_requested}: no sessions built")
        return problems
    short = int((sizes < plan.n_requested).sum())
    if short:
        problems.append(
            f"{plan.arm} N={plan.n_requested}: {short}/{len(plan)} sessions are "
            f"smaller than requested (min {int(sizes.min())}) - the pool cannot supply N"
        )
    return problems
