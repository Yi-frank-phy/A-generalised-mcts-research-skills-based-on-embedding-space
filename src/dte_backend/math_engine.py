"""Small deterministic math backend for DTE allocation.

This file deliberately avoids hidden agent logic. It only transforms structured
node scores/features into UCB scores and expansion budgets.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .models import AllocationResult, SearchNode


def calculate_ucb(
    score: float,
    uncertainty: float,
    tau: float = 1.0,
    c_explore: float = 1.0,
) -> float:
    """Default DTE UCB.

    The base objective is not cost-aware:

        U = V + c * tau * uncertainty

    Cost is controlled outside this formula by hard run budgets.
    """

    return float(score + c_explore * tau * uncertainty)


def boltzmann_allocation(
    scores: Sequence[float],
    total_budget: int,
    temperature: float = 1.0,
) -> list[int]:
    """Allocate integer expansion budgets with a Boltzmann rule.

    Args:
        scores: value scores in [0, 1].
        total_budget: total number of children to allocate.
        temperature: higher means more uniform; lower means greedier.

    Returns:
        A list of nonnegative integer expansion budgets.
    """

    if total_budget <= 0:
        return [0 for _ in scores]
    if not scores:
        return []
    if len(scores) == 1:
        return [total_budget]

    values = np.asarray(scores, dtype=float)
    safe_t = max(float(temperature), 1e-10)

    # log-sum-exp for numerical stability
    log_weights = values / safe_t
    max_log_weight = float(np.max(log_weights))
    weights = np.exp(log_weights - max_log_weight)
    probs = weights / np.sum(weights)

    raw = probs * total_budget
    allocation: list[int] = []
    for quota in raw:
        if quota < 1.0:
            allocation.append(int(round(float(quota))))
        else:
            allocation.append(int(math.ceil(float(quota))))

    # If rounding zeroed everything, guarantee at least one expansion for best node.
    if sum(allocation) == 0:
        allocation[int(np.argmax(values))] = 1

    return allocation


def allocate_frontier(
    nodes: list[SearchNode],
    total_budget: int,
    tau: float = 1.0,
    c_explore: float = 1.0,
    temperature: float = 1.0,
    allocation_metric: str = "ucb",
) -> list[AllocationResult]:
    """Compute UCB scores and Boltzmann expansion budgets for frontier nodes.

    This assumes Judge has already filled `score`. If `uncertainty` is missing,
    a neutral value of 0.0 is used; real systems should use density/novelty.

    `allocation_metric="ucb"` is the prototype default because the user
    specifically wants entropy/uncertainty-guided exploration to affect actual
    expansion, not merely display ranking. Use `"score"` to recover the older
    pure-value allocation behavior.
    """

    frontier = [n for n in nodes if n.status == "frontier"]
    scores = [float(n.score if n.score is not None else n.confidence) for n in frontier]
    uncertainties = [float(n.uncertainty if n.uncertainty is not None else 0.0) for n in frontier]
    ucb_scores = [calculate_ucb(v, u, tau=tau, c_explore=c_explore) for v, u in zip(scores, uncertainties)]

    if allocation_metric == "ucb":
        allocation_values = ucb_scores
    elif allocation_metric == "score":
        allocation_values = scores
    else:
        raise ValueError("allocation_metric must be 'ucb' or 'score'")

    budgets = boltzmann_allocation(allocation_values, total_budget=total_budget, temperature=temperature)

    return [
        AllocationResult(
            node_id=node.node_id,
            score=score,
            uncertainty=uncertainty,
            ucb_score=ucb,
            expansion_budget=budget,
        )
        for node, score, uncertainty, ucb, budget in zip(frontier, scores, uncertainties, ucb_scores, budgets)
    ]
