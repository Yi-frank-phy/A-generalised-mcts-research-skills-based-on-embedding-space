"""Historical proper-volume return and local value regression."""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from .controller_atlas import FrozenReferenceAtlas
from .models import SearchNode
from .space_geometry import (
    query_geodesic_distance_matrix,
    query_reference_weights,
)
from .space_measure import intrinsic_cell_volumes, intrinsic_proper_volume_at_radii
from .transition_state import embed_transition_nodes, require_completed_transition


def proper_volume_values_for_queries(
    source_embeddings: np.ndarray,
    query_radii: np.ndarray,
    atlas: FrozenReferenceAtlas,
) -> np.ndarray:
    """Continuously extend reference proper-volume profiles to arbitrary sources.

    Reference vertex a owns the finite cumulative ball-measure profile D_a(r)
    computed on the frozen atlas. An off-atlas source x receives the same
    partition-of-unity interpolation used by the distance-profile extension:

        D_x(r) = sum_a lambda_a(x) D_a(r).

    This is continuous in x and r, monotone in r, and exactly recovers D_a at
    every frozen reference vertex.
    """
    radii = np.asarray(query_radii, dtype=float)
    if radii.ndim != 2 or radii.shape[0] != len(source_embeddings):
        raise ValueError("query_radii must be a 2D array with one row per source")
    if not np.isfinite(radii).all() or np.any(radii < 0.0):
        raise ValueError("query_radii must be finite and non-negative")

    weights = query_reference_weights(
        source_embeddings,
        atlas.embeddings,
    )
    volumes = intrinsic_cell_volumes(atlas.reference_density)
    result = np.zeros_like(radii, dtype=float)
    for source_index in range(len(source_embeddings)):
        for reference_index, weight in enumerate(weights[source_index]):
            if weight <= 0.0:
                continue
            reference_radii = atlas.geodesic_distances[reference_index]
            values = intrinsic_proper_volume_at_radii(
                reference_radii,
                volumes,
                radii[source_index],
            )
            result[source_index] += float(weight) * values
    return result


def proper_volume_distance_matrix(
    sources: np.ndarray,
    targets: np.ndarray,
    atlas: FrozenReferenceAtlas,
) -> np.ndarray:
    """Source-centred proper-volume displacement for arbitrary off-atlas queries."""
    move_radii = query_geodesic_distance_matrix(
        sources,
        targets,
        atlas.embeddings,
        atlas.geodesic_distances,
    )
    return proper_volume_values_for_queries(sources, move_radii, atlas)


def historical_parent_returns(
    graph_nodes: Sequence[SearchNode],
    atlas: FrozenReferenceAtlas,
    provider: object,
) -> tuple[np.ndarray, np.ndarray]:
    by_id = {node.node_id: node for node in graph_nodes}
    parents: list[SearchNode] = []
    returns: list[float] = []
    for child in graph_nodes:
        if not child.parent_ids:
            continue
        try:
            require_completed_transition(child)
        except ValueError:
            continue
        child_embedding = embed_transition_nodes([child], provider)[0]
        for parent_id in child.parent_ids:
            parent = by_id.get(parent_id)
            if parent is None:
                continue
            try:
                require_completed_transition(parent)
            except ValueError:
                continue
            parent_embedding = embed_transition_nodes([parent], provider)[0]
            parents.append(parent)
            returns.append(
                float(
                    proper_volume_distance_matrix(
                        parent_embedding[None, :],
                        child_embedding[None, :],
                        atlas,
                    )[0, 0]
                )
            )
    if not parents:
        return np.empty((0, atlas.embeddings.shape[1])), np.asarray([], dtype=float)
    return embed_transition_nodes(parents, provider), np.asarray(returns, dtype=float)


def regress_values(
    live_embeddings: np.ndarray,
    history_embeddings: np.ndarray,
    realized_returns: np.ndarray,
    atlas: FrozenReferenceAtlas,
    volume_bandwidth: float,
) -> np.ndarray:
    if len(realized_returns) == 0:
        return np.zeros(len(live_embeddings), dtype=float)
    distance = proper_volume_distance_matrix(
        live_embeddings,
        history_embeddings,
        atlas,
    )
    weights = np.exp(-distance / float(volume_bandwidth))
    mass = np.sum(weights, axis=1)
    return np.divide(
        weights @ realized_returns,
        mass,
        out=np.zeros(len(live_embeddings), dtype=float),
        where=mass > np.finfo(float).tiny,
    )
