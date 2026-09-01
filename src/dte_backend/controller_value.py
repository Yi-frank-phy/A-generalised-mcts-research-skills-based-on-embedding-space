"""Historical proper-volume return and local value regression."""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from .controller_atlas import FrozenReferenceAtlas
from .models import SearchNode
from .space_geometry import (
    query_geodesic_distance,
    query_geodesic_distance_matrix,
    reference_radii_for_queries,
)
from .space_measure import intrinsic_cell_volumes, intrinsic_proper_volume_at_radius
from .transition_state import embed_transition_nodes, require_completed_transition


def proper_volume_distance_matrix(sources: np.ndarray, targets: np.ndarray, atlas: FrozenReferenceAtlas) -> np.ndarray:
    """Source-centred proper-volume displacement for arbitrary off-atlas queries."""
    source_radii = reference_radii_for_queries(
        sources, atlas.embeddings, atlas.geodesic_distances
    )
    move_radii = query_geodesic_distance_matrix(
        sources, targets, atlas.embeddings, atlas.geodesic_distances
    )
    volumes = intrinsic_cell_volumes(atlas.reference_density)
    result = np.zeros((len(sources), len(targets)), dtype=float)
    for i in range(len(sources)):
        for j in range(len(targets)):
            result[i, j] = intrinsic_proper_volume_at_radius(
                source_radii[i],
                volumes,
                float(move_radii[i, j]),
            )
    return result


def historical_parent_returns(
    graph_nodes: Sequence[SearchNode],
    atlas: FrozenReferenceAtlas,
    provider: object,
) -> tuple[np.ndarray, np.ndarray]:
    by_id = {node.node_id: node for node in graph_nodes}
    parents: list[SearchNode] = []
    returns: list[float] = []
    volumes = intrinsic_cell_volumes(atlas.reference_density)
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
            radii = reference_radii_for_queries(
                parent_embedding[None, :],
                atlas.embeddings,
                atlas.geodesic_distances,
            )[0]
            radius = query_geodesic_distance(
                parent_embedding,
                child_embedding,
                atlas.embeddings,
                atlas.geodesic_distances,
            )
            parents.append(parent)
            returns.append(
                intrinsic_proper_volume_at_radius(radii, volumes, radius)
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
        live_embeddings, history_embeddings, atlas
    )
    weights = np.exp(-distance / float(volume_bandwidth))
    mass = np.sum(weights, axis=1)
    return np.divide(
        weights @ realized_returns,
        mass,
        out=np.zeros(len(live_embeddings), dtype=float),
        where=mass > np.finfo(float).tiny,
    )
