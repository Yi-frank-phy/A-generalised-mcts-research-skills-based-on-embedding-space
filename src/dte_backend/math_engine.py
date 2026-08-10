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
    """Return the canonical DTE upper-confidence score.

    The restored theoretical primitive is

        U = V + SD

    where ``uncertainty`` is the current standard-deviation/standard-error-like
    estimator. ``tau`` and ``c_explore`` remain accepted temporarily for call-site
    compatibility but do not alter the canonical score.
    """

    del tau, c_explore
    return float(score + uncertainty)


def boltzmann_allocation(
    scores: Sequence[float],
    allocation_mass_per_iteration: int,
    max_children_per_iteration: int,
    node_ids: Sequence[str],
    temperature: float = 1.0,
) -> list[int]:
    """Allocate integer expansion budgets with a Boltzmann rule.

    Args:
        scores: allocation values, normally canonical UCB scores.
        allocation_mass_per_iteration: continuous Boltzmann allocation mass.
        max_children_per_iteration: hard cap on committed children.
        node_ids: stable identifiers used for deterministic trimming.
        temperature: higher means more uniform; lower means greedier.

    Returns:
        A list of nonnegative integer child counts.
    """

    if allocation_mass_per_iteration <= 0 or max_children_per_iteration <= 0:
        return [0 for _ in scores]
    if not scores:
        return []
    if len(scores) != len(node_ids):
        raise ValueError("scores and node_ids must have equal length")

    values = np.asarray(scores, dtype=float)
    if float(temperature) <= 0.0:
        # Exact T -> 0 limit: only true maximizers have support. Do not use an
        # epsilon temperature here; a merely near-maximal branch must receive
        # zero probability in the canonical zero-temperature state.
        max_value = float(np.max(values))
        winners = values == max_value
        probs = winners.astype(float) / float(np.sum(winners))
    else:
        # log-sum-exp for numerical stability at finite positive temperature
        log_weights = values / float(temperature)
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
    """Compute canonical UCB scores and Boltzmann expansion budgets.

    Judge fills ``score`` and the geometry layer fills ``uncertainty``. The local
    UCB score is always ``score + uncertainty``. Legacy ``tau`` and
    ``c_explore`` arguments are accepted for compatibility but do not affect UCB.

    ``allocation_metric=\"ucb\"`` remains the default so local uncertainty affects
    actual expansion rather than display only. ``\"score\"`` retains the older
    pure-value allocation behavior as an explicit compatibility option.
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
