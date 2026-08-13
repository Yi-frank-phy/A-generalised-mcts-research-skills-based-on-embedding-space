"""Frozen reference atlas for the new proper-volume controller."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
from collections.abc import Sequence
import numpy as np
from .models import SearchNode
from .space_geometry import all_pairs_geodesic_distances
from .transition_state import embed_transition_nodes, require_completed_transition


@dataclass(frozen=True)
class FrozenReferenceAtlas:
    node_ids: tuple[str, ...]
    embeddings: np.ndarray
    geodesic_distances: np.ndarray
    reference_density: np.ndarray
    graph_k: int
    identity: str


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
    k = min(max(1, int(graph_k)), len(nodes) - 1)
    geodesic = all_pairs_geodesic_distances(embeddings, k=k)
    density = np.ones(len(nodes), dtype=float) if reference_density is None else np.asarray(reference_density, dtype=float)
    if density.shape != (len(nodes),) or not np.isfinite(density).all() or np.any(density <= 0.0):
        raise ValueError("reference_density must be positive with one value per atlas cell")
    digest = hashlib.sha256()
    for node in nodes:
        digest.update(node.node_id.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.asarray(embeddings, dtype=np.float64).tobytes())
    digest.update(np.asarray(density, dtype=np.float64).tobytes())
    digest.update(str(k).encode("ascii"))
    return FrozenReferenceAtlas(
        node_ids=tuple(node.node_id for node in nodes),
        embeddings=np.asarray(embeddings, dtype=float),
        geodesic_distances=np.asarray(geodesic, dtype=float),
        reference_density=density.copy(),
        graph_k=k,
        identity=digest.hexdigest(),
    )
