"""Novelty and entropy helpers for frontier nodes."""

from __future__ import annotations

from .cache import DTECache
from .context_envelope import semantic_embedding_text
from .embedding import EmbeddingProvider, HashEmbeddingProvider
from .kde import KDEState, compute_kde_state
from .models import SearchNode


def ensure_embeddings(
    nodes: list[SearchNode],
    dim: int = 3072,
    cache: DTECache | None = None,
    provider: EmbeddingProvider | None = None,
) -> None:
    """Fill missing node vectors in-place.

    Embedding input is a canonical context envelope. This improves cache hit
    rate across Codex/subagent compile variations while preserving semantic
    changes in claim/evidence/risk.
    """

    provider = provider or HashEmbeddingProvider(dim=dim)
    missing: list[tuple[SearchNode, str]] = []
    for node in nodes:
        if node.local_embedding:
            if cache is not None:
                cache.set_embedding(node, node.local_embedding)
            continue
        cached = cache.get_embedding(node) if cache is not None else None
        if cached is not None:
            node.local_embedding = cached
            continue
        missing.append((node, semantic_embedding_text(node)))

    if missing:
        vectors = provider.embed_texts([text for _, text in missing])
        for (node, _), vector in zip(missing, vectors):
            node.local_embedding = vector
            if cache is not None:
                cache.set_embedding(node, vector)


def estimate_frontier_kde_state(
    nodes: list[SearchNode],
    cache: DTECache | None = None,
    provider: EmbeddingProvider | None = None,
) -> tuple[list[SearchNode], KDEState]:
    """Return frontier nodes and their KDE observables."""

    frontier = [n for n in nodes if n.status == "frontier"]
    if not frontier:
        return [], compute_kde_state([])
    ensure_embeddings(frontier, cache=cache, provider=provider)
    embeddings = [n.local_embedding or [] for n in frontier]
    return frontier, compute_kde_state(embeddings)


def estimate_uncertainty_from_density(
    nodes: list[SearchNode],
    cache: DTECache | None = None,
    provider: EmbeddingProvider | None = None,
) -> dict[str, float]:
    """Estimate uncertainty from low-density frontier regions."""

    frontier, state = estimate_frontier_kde_state(nodes, cache=cache, provider=provider)
    return {node.node_id: value for node, value in zip(frontier, state.uncertainty)}
