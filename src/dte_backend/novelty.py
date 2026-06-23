"""Novelty / entropy proxy for frontier nodes."""

from __future__ import annotations

import numpy as np

from .cache import DTECache
from .models import SearchNode
from .text_features import cosine_distance_matrix, hashed_embedding, node_text_parts


def ensure_embeddings(nodes: list[SearchNode], dim: int = 64, cache: DTECache | None = None) -> None:
    """Fill missing local embeddings in-place.

    The optional cache is keyed by stable node content, not by DTE metrics.
    """

    for node in nodes:
        if node.local_embedding:
            if cache is not None:
                cache.set_embedding(node, node.local_embedding)
            continue
        cached = cache.get_embedding(node) if cache is not None else None
        if cached is not None:
            node.local_embedding = cached
            continue
        text = node_text_parts(node.claim, node.rationale, node.assumptions, node.evidence, node.risks)
        node.local_embedding = hashed_embedding(text, dim=dim)
        if cache is not None:
            cache.set_embedding(node, node.local_embedding)


def estimate_uncertainty_from_density(nodes: list[SearchNode], cache: DTECache | None = None) -> dict[str, float]:
    """Estimate novelty-style uncertainty from local density.

    This is a cheap proxy: average cosine distance to other frontier nodes.
    Sparse/outlying nodes receive larger uncertainty. The value is normalized to
    [0, 1]. A single frontier node receives uncertainty 1.0.
    """

    frontier = [n for n in nodes if n.status == "frontier"]
    if not frontier:
        return {}
    ensure_embeddings(frontier, cache=cache)
    if len(frontier) == 1:
        return {frontier[0].node_id: 1.0}

    dist = cosine_distance_matrix([n.local_embedding or [] for n in frontier])
    # Exclude diagonal by using sum/(n-1). Larger mean distance = sparser region.
    mean_dist = (dist.sum(axis=1) - np.diag(dist)) / max(1, len(frontier) - 1)
    min_v = float(np.min(mean_dist))
    max_v = float(np.max(mean_dist))
    if max_v - min_v < 1e-12:
        norm = np.full_like(mean_dist, 0.5, dtype=float)
    else:
        norm = (mean_dist - min_v) / (max_v - min_v)
    return {node.node_id: float(value) for node, value in zip(frontier, norm)}
