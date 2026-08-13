"""Production frontier scoring boundary for the `new` release line.

The historical function names remain during schema migration, but no RBF/KDE
physics is executed here. Current-frontier state is the proper-volume controller
state defined by docs/PHYSICS.md.
"""

from __future__ import annotations
from typing import TypeAlias
import numpy as np

from .cache import DTECache
from .embedding import EmbeddingProvider, HashEmbeddingProvider
from .models import SearchNode
from .new_controller import FrontierControllerState, freeze_reference_atlas, score_frontier
from .transition_state import require_completed_transition

KDEState: TypeAlias = FrontierControllerState


def _provider(provider: EmbeddingProvider | None, expected_dimension: int | None) -> EmbeddingProvider:
    if provider is None:
        return HashEmbeddingProvider(dim=expected_dimension or 3072)
    if expected_dimension is not None and provider.dim != expected_dimension:
        raise ValueError(
            "embedding provider dimension does not match expected dimension: "
            f"provider={provider.dim}, expected={expected_dimension}"
        )
    return provider


def _reference_roots(nodes: list[SearchNode]) -> list[SearchNode]:
    roots = [node for node in nodes if not node.parent_ids]
    if len(roots) < 2:
        raise ValueError("new controller requires at least two frozen root transitions")
    for node in roots:
        require_completed_transition(node)
    return roots


def estimate_frontier_kde_state(
    nodes: list[SearchNode],
    cache: DTECache | None = None,
    provider: EmbeddingProvider | None = None,
    *,
    expected_dimension: int | None = None,
    graph_k: int = 2,
    volume_bandwidth: float = 1.0,
) -> tuple[list[SearchNode], FrontierControllerState]:
    """Score the live completed-transition frontier on one frozen root atlas."""

    del cache  # transition embeddings are canonical controller state, not claim-cache state
    frontier = [node for node in nodes if node.status == "frontier"]
    if not frontier:
        raise ValueError("new controller requires a non-empty active frontier")
    resolved_provider = _provider(provider, expected_dimension)
    roots = _reference_roots(nodes)
    atlas = freeze_reference_atlas(roots, provider=resolved_provider, graph_k=graph_k)
    state = score_frontier(
        graph_nodes=nodes,
        live_nodes=frontier,
        atlas=atlas,
        provider=resolved_provider,
        volume_bandwidth=volume_bandwidth,
    )
    for node, value, rho, sd, ucb in zip(
        frontier,
        state.values,
        state.occupancy_fractions,
        state.standard_deviations,
        state.ucb_scores,
    ):
        node.score = float(value)
        node.density = float(rho)
        node.uncertainty = float(sd)
        node.ucb_score = float(ucb)
    return frontier, state


def estimate_uncertainty_from_density(
    nodes: list[SearchNode],
    cache: DTECache | None = None,
    provider: EmbeddingProvider | None = None,
    *,
    expected_dimension: int | None = None,
) -> dict[str, float]:
    """Compatibility wrapper returning proper-volume reward uncertainty."""

    frontier, state = estimate_frontier_kde_state(
        nodes,
        cache=cache,
        provider=provider,
        expected_dimension=expected_dimension,
    )
    return {
        node.node_id: float(value)
        for node, value in zip(frontier, state.standard_deviations)
    }
