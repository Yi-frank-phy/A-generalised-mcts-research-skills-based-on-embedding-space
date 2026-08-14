"""Production frontier scoring boundary for the `new` release line.

The historical function names remain during schema migration, but no RBF/KDE
physics is executed here. Current-frontier state is the proper-volume controller
state defined by docs/PHYSICS.md.
"""

from __future__ import annotations
from copy import deepcopy
import math
from pathlib import Path
from typing import Any, TypeAlias

from .cache import DTECache, EmbeddingCacheNamespace
from .context_envelope import semantic_embedding_text
from .embedding import EmbeddingProvider, HashEmbeddingProvider
from .models import SearchNode
from .new_controller import FrozenReferenceAtlas, FrontierControllerState, freeze_reference_atlas, score_frontier
from .reference_atlas import combined_reference_nodes
from .transition_state import canonical_transition_text, require_completed_transition

KDEState: TypeAlias = FrontierControllerState

def _validated_vector(value: Any, *, expected_dimension: int, source: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_dimension:
        raise ValueError(f"{source} embedding dimension mismatch")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{source} embedding must contain only finite values")
    return vector


def _ensure_relation_embeddings(
    nodes: list[SearchNode],
    *,
    cache: DTECache | None,
    provider: EmbeddingProvider,
    expected_dimension: int,
) -> None:
    """Populate semantic embeddings used by Relation, never by the new controller."""

    namespace = EmbeddingCacheNamespace(
        provider=provider.name,
        model_snapshot=str(getattr(provider, "model", provider.name)),
        dimension=expected_dimension,
        contract_version="embedding-v1",
    )
    staged: list[tuple[SearchNode, list[float], bool]] = []
    missing: list[SearchNode] = []
    for node in nodes:
        if node.local_embedding is not None:
            staged.append((node, _validated_vector(
                node.local_embedding, expected_dimension=expected_dimension,
                source=f"existing node {node.node_id!r}",
            ), False))
            continue
        cached = cache.get_embedding(node, namespace=namespace) if cache is not None else None
        if cached is not None:
            staged.append((node, _validated_vector(
                cached, expected_dimension=expected_dimension,
                source=f"cached node {node.node_id!r}",
            ), False))
        else:
            missing.append(node)

    if missing:
        raw = provider.embed_texts([semantic_embedding_text(node) for node in missing])
        if not isinstance(raw, list) or len(raw) != len(missing):
            raise ValueError("embedding provider returned the wrong number of semantic vectors")
        for node, vector in zip(missing, raw):
            staged.append((node, _validated_vector(
                vector, expected_dimension=expected_dimension,
                source=f"provider result for node {node.node_id!r}",
            ), True))

    # Install cache entries before mutating graph state, so a cache I/O failure
    # cannot leave a partially updated App state.
    if cache is not None:
        snapshot = deepcopy(cache.__dict__)
        raw_path = getattr(cache, "path", None)
        path = Path(raw_path) if raw_path is not None else None
        existed = bool(path is not None and path.exists())
        contents = path.read_bytes() if existed and path is not None else None
        try:
            for node, vector, should_write in staged:
                if should_write:
                    cache.set_embedding(node, list(vector), namespace=namespace)
        except Exception:
            cache.__dict__.clear()
            cache.__dict__.update(deepcopy(snapshot))
            if path is not None:
                if existed:
                    assert contents is not None
                    path.write_bytes(contents)
                elif path.exists():
                    path.unlink()
            raise
    for node, vector, _ in staged:
        node.local_embedding = list(vector)


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

    frontier = [node for node in nodes if node.status == "frontier"]
    if not frontier:
        raise ValueError("new controller requires a non-empty active frontier")
    resolved_provider = _provider(provider, expected_dimension)
    _ensure_relation_embeddings(
        frontier,
        cache=cache,
        provider=resolved_provider,
        expected_dimension=resolved_provider.dim,
    )
    atlas = _frozen_atlas(_reference_roots(nodes), resolved_provider, graph_k)
    state = score_frontier(
        graph_nodes=nodes,
        live_nodes=frontier,
        atlas=atlas,
        provider=resolved_provider,
        volume_bandwidth=volume_bandwidth,
    )
    for node, embedding, value, rho, sd, ucb in zip(
        frontier,
        state.transition_embeddings,
        state.values,
        state.occupancy_fractions,
        state.standard_deviations,
        state.ucb_scores,
    ):
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
