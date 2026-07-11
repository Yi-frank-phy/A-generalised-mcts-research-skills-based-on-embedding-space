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
    allocation_mass_per_iteration: int,
    max_children_per_iteration: int,
    node_ids: Sequence[str],
    temperature: float = 1.0,
) -> list[int]:
    """Allocate integer expansion budgets with a Boltzmann rule.

    Args:
        scores: value scores in [0, 1].
        allocation_mass_per_iteration: continuous Boltzmann allocation mass.
        max_children_per_iteration: hard cap on committed children.
        node_ids: stable identifiers used for deterministic trimming.
        temperature: higher means more uniform; lower means greedier.

    Returns:
        A list of nonnegative integer expansion budgets.
    """

    if allocation_mass_per_iteration <= 0 or max_children_per_iteration <= 0:
        return [0 for _ in scores]
    if not scores:
        return []
    if len(scores) != len(node_ids):
        raise ValueError("scores and node_ids must have equal length")

    values = np.asarray(scores, dtype=float)
    safe_t = max(float(temperature), 1e-10)

    # log-sum-exp for numerical stability
    log_weights = values / safe_t
    max_log_weight = float(np.max(log_weights))
    weights = np.exp(log_weights - max_log_weight)
    probs = weights / np.sum(weights)

    quotas = probs * allocation_mass_per_iteration
    return discretize_allocation(
        quotas,
        allocation_values=values,
        node_ids=node_ids,
        max_children_per_iteration=max_children_per_iteration,
    )


def discretize_allocation(
    quotas: Sequence[float],
    allocation_values: Sequence[float],
    node_ids: Sequence[str],
    max_children_per_iteration: int,
) -> list[int]:
    """Discretize soft quotas, then enforce the deterministic hard child cap."""

    if not (len(quotas) == len(allocation_values) == len(node_ids)):
        raise ValueError("quotas, allocation_values, and node_ids must have equal length")
    if max_children_per_iteration <= 0:
        return [0 for _ in quotas]

    tentative = [
        int(math.floor(float(quota) + 0.5))
        if float(quota) < 1.0
        else int(math.ceil(float(quota)))
        for quota in quotas
    ]
    if sum(tentative) <= max_children_per_iteration:
        return tentative

    slots: list[tuple[float, float, str, int, int]] = []
    for index, (quota, value, node_id, count) in enumerate(
        zip(quotas, allocation_values, node_ids, tentative)
    ):
        for child_index in range(1, count + 1):
            marginal_support = float(quota) - (child_index - 1)
            slots.append((marginal_support, float(value), str(node_id), child_index, index))

    slots.sort(key=lambda slot: (-slot[0], -slot[1], slot[2], slot[3]))
    allocation = [0 for _ in tentative]
    for _, _, _, _, index in slots[:max_children_per_iteration]:
        allocation[index] += 1
    return allocation


def allocate_frontier(
    nodes: list[SearchNode],
    allocation_mass_per_iteration: int,
    max_children_per_iteration: int,
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

    budgets = boltzmann_allocation(
        allocation_values,
        allocation_mass_per_iteration=allocation_mass_per_iteration,
        max_children_per_iteration=max_children_per_iteration,
        node_ids=[node.node_id for node in frontier],
        temperature=temperature,
    )

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
