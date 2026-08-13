"""Production proper-volume controller for the `new` release line."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np

from dte_nextgen.thought_space.geometry import (
    all_pairs_geodesic_distances,
    nearest_reference_indices,
    query_geodesic_distance,
    reference_radii_for_queries,
)
from dte_nextgen.thought_space.metric_measure_controller import score_metric_measure_frontier
from dte_nextgen.thought_space.occupancy import estimate_live_occupancy
from dte_nextgen.thought_space.volume_measure import (
    intrinsic_cell_volumes,
    intrinsic_proper_volume_at_radius,
)

from .models import SearchNode
from .transition_state import embed_transition_nodes, require_completed_transition


@dataclass(frozen=True)
class FrozenReferenceAtlas:
    node_ids: tuple[str, ...]
    embeddings: np.ndarray
    geodesic_distances: np.ndarray
    reference_density: np.ndarray
    graph_k: int
    identity: str


@dataclass(frozen=True)
class FrontierControllerState:
    node_ids: tuple[str, ...]
    values: np.ndarray
    standard_deviations: np.ndarray
    ucb_scores: np.ndarray
    occupancy_fractions: np.ndarray
    target_entropies: np.ndarray
    spatial_entropy: float
    volume_bandwidth: float
    realized_returns: np.ndarray
    value_source: str = "proper_volume_history"
    sd_source: str = "proper_volume_boltzmann_reward"

    @property
    def log_density(self) -> np.ndarray:
        return np.log(self.occupancy_fractions)

    @property
    def uncertainty(self) -> list[float]:
        return self.standard_deviations.tolist()

    @property
    def bandwidth(self) -> float:
        return self.volume_bandwidth


def _resolved_graph_k(count: int, requested: int) -> int:
    if count < 2:
        raise ValueError("frozen reference atlas requires at least two completed transitions")
    value = int(requested)
    if value < 1:
        raise ValueError("graph_k must be positive")
    return min(value, count - 1)


def freeze_reference_atlas(
    nodes: Sequence[SearchNode],
    *,
    provider: object,
    graph_k: int = 2,
    reference_density: np.ndarray | None = None,
) -> FrozenReferenceAtlas:
    if len(nodes) < 2:
        raise ValueError("frozen reference atlas requires at least two completed transitions")
    for node in nodes:
        require_completed_transition(node)
    embeddings = embed_transition_nodes(nodes, provider)
    resolved_k = _resolved_graph_k(len(nodes), graph_k)
    geodesic = all_pairs_geodesic_distances(embeddings, k=resolved_k)
    density = np.ones(len(nodes), dtype=float) if reference_density is None else np.asarray(reference_density, dtype=float)
    if density.shape != (len(nodes),) or not np.isfinite(density).all() or np.any(density <= 0.0):
        raise ValueError("reference_density must be positive with one value per atlas cell")
    digest = hashlib.sha256()
    for node in nodes:
        digest.update(node.node_id.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.asarray(embeddings, dtype=np.float64).tobytes())
    digest.update(np.asarray(density, dtype=np.float64).tobytes())
    digest.update(str(resolved_k).encode("ascii"))
    return FrozenReferenceAtlas(
        node_ids=tuple(node.node_id for node in nodes),
        embeddings=np.asarray(embeddings, dtype=float),
        geodesic_distances=np.asarray(geodesic, dtype=float),
        reference_density=density.copy(),
        graph_k=resolved_k,
        identity=digest.hexdigest(),
    )


def _proper_volume_distance_matrix(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    atlas: FrozenReferenceAtlas,
) -> np.ndarray:
    source_anchors = nearest_reference_indices(source_embeddings, atlas.embeddings)
    target_anchors = nearest_reference_indices(target_embeddings, atlas.embeddings)
    volumes = intrinsic_cell_volumes(atlas.reference_density)
    distance = np.zeros((len(source_embeddings), len(target_embeddings)), dtype=float)
    for i, raw_source in enumerate(source_anchors):
        source = int(raw_source)
        radii = atlas.geodesic_distances[source]
        for j, raw_target in enumerate(target_anchors):
            move_radius = float(atlas.geodesic_distances[source, int(raw_target)])
            distance[i, j] = intrinsic_proper_volume_at_radius(radii, volumes, move_radius)
    return distance


def _historical_edges(
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
            parent_radii = reference_radii_for_queries(parent_embedding[None, :], atlas.embeddings, atlas.geodesic_distances)[0]
            move_radius = query_geodesic_distance(parent_embedding, child_embedding, atlas.embeddings, atlas.geodesic_distances)
            parents.append(parent)
            returns.append(intrinsic_proper_volume_at_radius(parent_radii, volumes, move_radius))
    if not parents:
        return np.empty((0, atlas.embeddings.shape[1])), np.asarray([], dtype=float)
    return embed_transition_nodes(parents, provider), np.asarray(returns, dtype=float)


def score_frontier(
    *,
    graph_nodes: Sequence[SearchNode],
    live_nodes: Sequence[SearchNode],
    atlas: FrozenReferenceAtlas,
    provider: object,
    volume_bandwidth: float = 1.0,
) -> FrontierControllerState:
    if not live_nodes:
        raise ValueError("live frontier must be non-empty")
    scale = float(volume_bandwidth)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("volume_bandwidth must be finite and positive")
    for node in live_nodes:
        require_completed_transition(node)
    live_embeddings = embed_transition_nodes(live_nodes, provider)
    history_embeddings, realized_returns = _historical_edges(graph_nodes, atlas, provider)
    if len(realized_returns) == 0:
        values = np.zeros(len(live_nodes), dtype=float)
    else:
        history_distance = _proper_volume_distance_matrix(live_embeddings, history_embeddings, atlas)
        weights = np.exp(-history_distance / scale)
        mass = np.sum(weights, axis=1)
        values = np.divide(weights @ realized_returns, mass, out=np.zeros(len(live_nodes), dtype=float), where=mass > np.finfo(float).tiny)
    radii = reference_radii_for_queries(live_embeddings, atlas.embeddings, atlas.geodesic_distances)
    occupancy = estimate_live_occupancy(
        live_embeddings=live_embeddings,
        reference_embeddings=atlas.embeddings,
        geodesic_distances=atlas.geodesic_distances,
        reference_density=atlas.reference_density,
        volume_bandwidth=scale,
    )
    scored = score_metric_measure_frontier(
        propulsion_values=values,
        node_radii=radii,
        reference_density=atlas.reference_density,
        occupancy_fractions=np.asarray(occupancy["occupancy_fractions"], dtype=float),
    )
    entropies = np.asarray(scored["target_entropies"], dtype=float)
    return FrontierControllerState(
        node_ids=tuple(node.node_id for node in live_nodes),
        values=np.asarray(scored["values"], dtype=float),
        standard_deviations=np.asarray(scored["standard_deviations"], dtype=float),
        ucb_scores=np.asarray(scored["ucb_scores"], dtype=float),
        occupancy_fractions=np.asarray(scored["occupancy_fractions"], dtype=float),
        target_entropies=entropies,
        spatial_entropy=float(np.mean(entropies)),
        volume_bandwidth=scale,
        realized_returns=realized_returns,
    )
