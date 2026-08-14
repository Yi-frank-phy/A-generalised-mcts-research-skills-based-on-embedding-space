"""Production frontier scoring boundary for the `new` release line.

The historical function names remain during schema migration, but no RBF/KDE
physics is executed here. Current-frontier state is the proper-volume controller
state defined by docs/PHYSICS.md.
"""

from __future__ import annotations
from typing import TypeAlias

from .cache import DTECache
from .embedding import EmbeddingProvider, HashEmbeddingProvider
from .models import SearchNode
from .new_controller import FrozenReferenceAtlas, FrontierControllerState, freeze_reference_atlas, score_frontier
from .reference_atlas import combined_reference_nodes
from .transition_state import canonical_transition_text, require_completed_transition

KDEState: TypeAlias = FrontierControllerState
_ATLAS_CACHE: dict[tuple[object, ...], FrozenReferenceAtlas] = {}


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
    if not roots:
        raise ValueError("new controller requires at least one initial completed transition")
    for node in roots:
        require_completed_transition(node)
    return roots


def _frozen_atlas(
    roots: list[SearchNode],
    provider: EmbeddingProvider,
    graph_k: int,
) -> FrozenReferenceAtlas:
    key = (
        provider.name,
        str(getattr(provider, "model", provider.name)),
        provider.dim,
        int(graph_k),
        tuple(canonical_transition_text(node) for node in roots),
    )
    cached = _ATLAS_CACHE.get(key)
    if cached is not None:
        return cached
    atlas_nodes = combined_reference_nodes(roots)
    atlas = freeze_reference_atlas(atlas_nodes, provider=provider, graph_k=graph_k)
    _ATLAS_CACHE[key] = atlas
    return atlas


def estimate_frontier_kde_state(
    nodes: list[SearchNode],
    cache: DTECache | None = None,
    provider: EmbeddingProvider | None = None,
    *,
    expected_dimension: int | None = None,
    graph_k: int = 2,
    volume_bandwidth: float = 1.0,
) -> tuple[list[SearchNode], FrontierControllerState]:
    """Score live completed transitions on a frozen method-space atlas."""

    del cache
    frontier = [node for node in nodes if node.status == "frontier"]
    if not frontier:
        raise ValueError("new controller requires a non-empty active frontier")
    resolved_provider = _provider(provider, expected_dimension)
    atlas = _frozen_atlas(_reference_roots(nodes), resolved_provider, graph_k)
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
