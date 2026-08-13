"""Stateful frozen-atlas execution loop for next-generation DTE."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .allocation import boltzmann_probabilities, select_next_index, temperature_for_target_entropy
from .geometry import (
    all_pairs_geodesic_distances,
    nearest_reference_indices,
    query_geodesic_distance,
    reference_radii_for_queries,
)
from .history import TransitionHistory
from .metric_measure_controller import score_metric_measure_frontier
from .occupancy import estimate_live_occupancy
from .transition import MethodEpistemicTransition, embed_method_epistemic_transitions
from .volume_measure import intrinsic_cell_volumes, intrinsic_proper_volume_at_radius


class ProperVolumeTransitionSession:
    """One run-local controller with a frozen reference atlas and volume gauge."""

    def __init__(
        self,
        *,
        node_ids: Sequence[str],
        frontier: Sequence[MethodEpistemicTransition],
        reference_transitions: Sequence[MethodEpistemicTransition],
        embed_fn: Callable[[str], Sequence[float]],
        graph_k: int,
        volume_bandwidth: float = 1.0,
        reference_density: np.ndarray | None = None,
        history: TransitionHistory | None = None,
    ) -> None:
        if len(node_ids) != len(frontier) or len(frontier) == 0:
            raise ValueError("node_ids and frontier must have the same non-zero length")
        if any(not node_id for node_id in node_ids) or len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be non-empty and unique")
        if not reference_transitions:
            raise ValueError("reference_transitions must be non-empty")
        scale = float(volume_bandwidth)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("volume_bandwidth must be finite and positive")
        if history is not None and len(history) > 0:
            raise ValueError("pre-populated numeric history has no frozen atlas identity")

        self._node_ids = list(node_ids)
        self._frontier = list(frontier)
        self._reference_transitions = tuple(reference_transitions)
        self._embed_fn = embed_fn
        self._graph_k = int(graph_k)
        self._volume_bandwidth = scale
        self.history = history if history is not None else TransitionHistory()

        self._reference_embeddings = embed_method_epistemic_transitions(
            self._reference_transitions, self._embed_fn
        )
        if len(self._reference_embeddings) < len(self._frontier):
            raise ValueError("reference atlas must contain at least as many cells as the live frontier")
        self._geodesic_distances = all_pairs_geodesic_distances(
            self._reference_embeddings, k=self._graph_k
        )
        if reference_density is None:
            self._reference_density = np.ones(len(self._reference_embeddings), dtype=float)
            self._reference_density_source = "uniform_frozen_reference_measure"
        else:
            density = np.asarray(reference_density, dtype=float)
            if density.shape != (len(self._reference_embeddings),) or np.any(density <= 0.0):
                raise ValueError("reference_density must be positive with one value per atlas cell")
            self._reference_density = density.copy()
            self._reference_density_source = "supplied_experimental_correction"
        self._cell_volumes = intrinsic_cell_volumes(self._reference_density)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(self._node_ids)

    @property
    def frontier(self) -> tuple[MethodEpistemicTransition, ...]:
        return tuple(self._frontier)

    @property
    def reference_transitions(self) -> tuple[MethodEpistemicTransition, ...]:
        return self._reference_transitions

    def _frontier_embeddings(self) -> np.ndarray:
        return embed_method_epistemic_transitions(self._frontier, self._embed_fn)

    def _history_value_regression(
        self, query_embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, str]:
        history_embeddings, history_returns, history_counts = self.history.as_arrays()
        if len(history_returns) == 0:
            return (
                np.zeros(len(query_embeddings), dtype=float),
                np.empty((len(query_embeddings), 0), dtype=float),
                "zero_before_history",
            )
        if history_embeddings.shape[1] != query_embeddings.shape[1]:
            raise ValueError("history and frontier embedding dimensions must match")

        query_anchors = nearest_reference_indices(query_embeddings, self._reference_embeddings)
        history_anchors = nearest_reference_indices(history_embeddings, self._reference_embeddings)
        distance = np.zeros((len(query_embeddings), len(history_embeddings)), dtype=float)
        for i, raw_source in enumerate(query_anchors):
            source = int(raw_source)
            radii = self._geodesic_distances[source]
            for j, raw_target in enumerate(history_anchors):
                move_radius = float(self._geodesic_distances[source, int(raw_target)])
                distance[i, j] = intrinsic_proper_volume_at_radius(
                    radii, self._cell_volumes, move_radius
                )

        weights = np.exp(-distance / self._volume_bandwidth)
        weights *= np.asarray(history_counts, dtype=float)[None, :]
        mass = np.sum(weights, axis=1)
        values = np.zeros(len(query_embeddings), dtype=float)
        supported = mass > np.finfo(float).tiny
        values[supported] = (
            weights[supported] @ np.asarray(history_returns, dtype=float)
        ) / mass[supported]
        return values, distance, "proper_volume_history_kernel"

    def score(self) -> dict[str, object]:
        live = self._frontier_embeddings()
        values, history_distance, value_source = self._history_value_regression(live)
        radii = reference_radii_for_queries(
            live, self._reference_embeddings, self._geodesic_distances
        )
        occupancy = estimate_live_occupancy(
            live_embeddings=live,
            reference_embeddings=self._reference_embeddings,
            geodesic_distances=self._geodesic_distances,
            reference_density=self._reference_density,
            volume_bandwidth=self._volume_bandwidth,
        )
        scored = score_metric_measure_frontier(
            propulsion_values=values,
            node_radii=radii,
            reference_density=self._reference_density,
            occupancy_fractions=occupancy["occupancy_fractions"],
        )
        return {
            **scored,
            "live_embeddings": live,
            "reference_embeddings": self._reference_embeddings.copy(),
            "reference_density": self._reference_density.copy(),
            "reference_density_source": self._reference_density_source,
            "geodesic_distances": self._geodesic_distances.copy(),
            "node_radii": radii,
            "proper_volume_displacements": occupancy["proper_volume_displacements"],
            "volume_bandwidth": self._volume_bandwidth,
            "value_source": value_source,
            "history_value_distances": history_distance,
            "node_ids": self.node_ids,
            "transitions": self.frontier,
        }

    def select(self, *, selection_quantile: float) -> dict[str, object]:
        scored = self.score()
        ucb = np.asarray(scored["ucb_scores"], dtype=float)
        target = float(scored["allocation_target_entropy"])
        maximum = float(np.log(len(ucb)))
        if target < -1e-12 or target > maximum + 1e-10:
            raise ValueError("frontier occupancy entropy must lie in [0, log(N)]")
        target = float(np.clip(target, 0.0, maximum))
        temperature = temperature_for_target_entropy(ucb, target)
        probabilities = boltzmann_probabilities(ucb, temperature)
        selected_index = select_next_index(probabilities, selection_quantile)
        return {
            **scored,
            "temperature": float(temperature),
            "probabilities": probabilities,
            "selected_index": selected_index,
            "selected_node_id": self._node_ids[selected_index],
            "selected_transition": self._frontier[selected_index],
        }

    def complete(
        self,
        *,
        parent_index: int,
        child_node_id: str,
        child_transition: MethodEpistemicTransition,
    ) -> dict[str, object]:
        if parent_index < 0 or parent_index >= len(self._frontier):
            raise IndexError("parent_index must identify an active transition")
        if not child_node_id:
            raise ValueError("child_node_id must be non-empty")
        if child_node_id in self._node_ids and child_node_id != self._node_ids[parent_index]:
            raise ValueError("child_node_id must be unique within the active frontier")

        parent_node_id = self._node_ids[parent_index]
        parent_transition = self._frontier[parent_index]
        parent_embedding = np.asarray(self._embed_fn(parent_transition.canonical_text()), dtype=float)
        child_embedding = np.asarray(self._embed_fn(child_transition.canonical_text()), dtype=float)
        parent_radii = reference_radii_for_queries(
            parent_embedding[None, :],
            self._reference_embeddings,
            self._geodesic_distances,
        )[0]
        move_radius = query_geodesic_distance(
            parent_embedding,
            child_embedding,
            self._reference_embeddings,
            self._geodesic_distances,
        )
        observed_return = intrinsic_proper_volume_at_radius(
            parent_radii, self._cell_volumes, move_radius
        )
        self.history.record(parent_node_id, parent_embedding, observed_return)
        self._node_ids[parent_index] = child_node_id
        self._frontier[parent_index] = child_transition
        return {
            "parent_index": int(parent_index),
            "parent_node_id": parent_node_id,
            "parent_transition": parent_transition,
            "child_node_id": child_node_id,
            "child_transition": child_transition,
            "observed_return": float(observed_return),
            "return_source": "frozen_atlas_proper_volume",
        }
