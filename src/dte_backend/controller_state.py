"""Frontier scoring for the new proper-volume controller."""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
from .controller_atlas import FrozenReferenceAtlas
from .controller_value import historical_parent_returns, proper_volume_distance_matrix, regress_values
from .models import SearchNode
from .space_distribution import node_reward_sd_from_occupancy
from .space_geometry import reference_radii_for_queries
from .transition_state import embed_transition_nodes, require_completed_transition


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
    transition_embeddings: np.ndarray
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

    @property
    def bandwidth2(self) -> float:
        return self.volume_bandwidth * self.volume_bandwidth


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
    history_embeddings, realized = historical_parent_returns(graph_nodes, atlas, provider)
    values = regress_values(live_embeddings, history_embeddings, realized, atlas, scale)
    live_distance = proper_volume_distance_matrix(live_embeddings, live_embeddings, atlas)
    np.fill_diagonal(live_distance, 0.0)
    occupancy = np.clip(np.mean(np.exp(-live_distance / scale), axis=1), np.finfo(float).tiny, 1.0)
    radii = reference_radii_for_queries(live_embeddings, atlas.embeddings, atlas.geodesic_distances)
    stats = [node_reward_sd_from_occupancy(radii[i], atlas.reference_density, float(occupancy[i])) for i in range(len(live_nodes))]
    sd = np.asarray([float(item["volume_reward_sd"]) for item in stats])
    entropies = np.asarray([float(item["target_entropy"]) for item in stats])
    return FrontierControllerState(
        node_ids=tuple(node.node_id for node in live_nodes),
        values=values,
        standard_deviations=sd,
        ucb_scores=values + sd,
        occupancy_fractions=occupancy,
        target_entropies=entropies,
        spatial_entropy=float(np.mean(entropies)),
        volume_bandwidth=scale,
        realized_returns=realized,
        transition_embeddings=live_embeddings,
    )
