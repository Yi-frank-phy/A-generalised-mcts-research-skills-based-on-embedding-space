"""Novelty and entropy helpers for frontier nodes."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from .cache import DTECache, EmbeddingCacheNamespace, embedding_cache_key
from .context_envelope import semantic_embedding_text
from .embedding import EmbeddingProvider, HashEmbeddingProvider
from .kde import KDEState, compute_kde_state
from .models import SearchNode


def _resolve_embedding_contract(
    *,
    dim: int | None,
    expected_dimension: int | None,
    provider: EmbeddingProvider | None,
) -> tuple[EmbeddingProvider, int]:
    if dim is not None and expected_dimension is not None and dim != expected_dimension:
        raise ValueError("dim and expected_dimension must agree when both are provided")

    if expected_dimension is not None:
        expected = expected_dimension
    elif dim is not None:
        expected = dim
    elif provider is not None:
        expected = provider.dim
    else:
        expected = 3072

    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise ValueError("expected embedding dimension must be a positive integer")

    resolved_provider = provider or HashEmbeddingProvider(dim=expected)
    provider_dim = getattr(resolved_provider, "dim", None)
    if isinstance(provider_dim, bool) or not isinstance(provider_dim, int):
        raise ValueError("embedding provider dimension must be a positive integer")
    if provider_dim != expected:
        raise ValueError(
            "embedding provider dimension does not match the expected dimension: "
            f"provider={provider_dim}, expected={expected}"
        )
    return resolved_provider, expected


def _validated_vector(value: Any, *, expected_dimension: int, source: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{source} embedding must be a list of numbers")
    if len(value) != expected_dimension:
        raise ValueError(
            f"{source} embedding dimension mismatch: "
            f"got {len(value)}, expected {expected_dimension}"
        )

    vector: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{source} embedding value {index} is not a finite number")
        try:
            numeric = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source} embedding value {index} is not a finite number"
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{source} embedding value {index} is not finite")
        vector.append(numeric)
    return vector


def _cache_snapshot(cache: DTECache | None) -> tuple[dict[str, Any] | None, Path | None, bool, bytes | None]:
    if cache is None:
        return None, None, False, None
    state = deepcopy(cache.__dict__)
    raw_path = getattr(cache, "path", None)
    path = Path(raw_path) if raw_path is not None else None
    existed = bool(path is not None and path.exists())
    contents = path.read_bytes() if existed and path is not None else None
    return state, path, existed, contents


def _restore_cache(
    cache: DTECache | None,
    snapshot: tuple[dict[str, Any] | None, Path | None, bool, bytes | None],
) -> None:
    if cache is None:
        return
    state, path, existed, contents = snapshot
    assert state is not None
    cache.__dict__.clear()
    cache.__dict__.update(deepcopy(state))
    if path is None:
        return
    if existed:
        assert contents is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".embedding-rollback.tmp")
        temporary.write_bytes(contents)
        temporary.replace(path)
    elif path.exists():
        path.unlink()


def ensure_embeddings(
    nodes: list[SearchNode],
    dim: int | None = None,
    cache: DTECache | None = None,
    provider: EmbeddingProvider | None = None,
    *,
    expected_dimension: int | None = None,
) -> None:
    """Fill vectors through one copy-validate-install transaction.

    Existing node vectors, cache hits, and provider results must all satisfy the
    same dimension and finite-value contract. No node or cache write is installed
    until the complete batch has validated.
    """

    provider, expected = _resolve_embedding_contract(
        dim=dim,
        expected_dimension=expected_dimension,
        provider=provider,
    )
    namespace = EmbeddingCacheNamespace(
        provider=provider.name,
        model_snapshot=str(getattr(provider, "model", provider.name)),
        dimension=expected,
        contract_version="embedding-v1",
    )
    node_snapshot = [(node, deepcopy(node.local_embedding)) for node in nodes]
    cache_snapshot = _cache_snapshot(cache)

    staged_by_node: list[tuple[SearchNode, list[float]]] = []
    staged_by_key: dict[str, list[float]] = {}
    cache_writes: dict[str, tuple[SearchNode, list[float]]] = {}
    missing: list[tuple[SearchNode, str, str]] = []

    def stage(
        node: SearchNode,
        vector: list[float],
        *,
        key: str,
        write_cache: bool,
    ) -> None:
        previous = staged_by_key.get(key)
        if previous is not None and previous != vector:
            raise ValueError("equivalent embedding cache identities produced conflicting vectors")
        staged_by_key.setdefault(key, vector)
        staged_by_node.append((node, vector))
        if write_cache:
            cache_writes.setdefault(key, (node, vector))

    try:
        # Existing vectors are authoritative for their node, but remain detached
        # until every other source in this batch has also validated.
        for node in nodes:
            if node.local_embedding is None:
                continue
            key = embedding_cache_key(node, namespace=namespace)
            vector = _validated_vector(
                node.local_embedding,
                expected_dimension=expected,
                source=f"existing node {node.node_id!r}",
            )
            stage(node, vector, key=key, write_cache=cache is not None)

        for node in nodes:
            if node.local_embedding is not None:
                continue
            key = embedding_cache_key(node, namespace=namespace)
            staged = staged_by_key.get(key)
            if staged is not None:
                stage(node, list(staged), key=key, write_cache=False)
                continue
            cached = cache.get_embedding(node, namespace=namespace) if cache is not None else None
            if cached is not None:
                vector = _validated_vector(
                    cached,
                    expected_dimension=expected,
                    source=f"cached node {node.node_id!r}",
                )
                stage(node, vector, key=key, write_cache=False)
                continue
            missing.append((node, semantic_embedding_text(node), key))

        if missing:
            raw_vectors = provider.embed_texts([text for _, text, _ in missing])
            if not isinstance(raw_vectors, list):
                raise ValueError("embedding provider must return a list of vectors")
            if len(raw_vectors) != len(missing):
                raise ValueError(
                    "embedding provider returned the wrong number of vectors: "
                    f"got {len(raw_vectors)}, expected {len(missing)}"
                )
            for (node, _, key), raw_vector in zip(missing, raw_vectors):
                vector = _validated_vector(
                    raw_vector,
                    expected_dimension=expected,
                    source=f"provider result for node {node.node_id!r}",
                )
                stage(node, vector, key=key, write_cache=cache is not None)

        if len(staged_by_node) != len(nodes):
            raise RuntimeError("embedding transaction did not stage every node")

        # Cache installation precedes node installation. If either phase fails,
        # both the cache (including a file-backed cache) and nodes are restored.
        if cache is not None:
            for node, vector in cache_writes.values():
                cache.set_embedding(node, list(vector), namespace=namespace)
        for node, vector in staged_by_node:
            node.local_embedding = list(vector)
    except Exception as exc:
        for node, original in node_snapshot:
            node.local_embedding = deepcopy(original)
        try:
            _restore_cache(cache, cache_snapshot)
        except Exception as restore_exc:  # pragma: no cover - catastrophic storage failure
            exc.add_note(f"embedding cache rollback also failed: {restore_exc}")
        raise


def estimate_frontier_kde_state(
    nodes: list[SearchNode],
    cache: DTECache | None = None,
    provider: EmbeddingProvider | None = None,
    *,
    expected_dimension: int | None = None,
) -> tuple[list[SearchNode], KDEState]:
    """Return frontier nodes and their KDE observables."""

    frontier = [n for n in nodes if n.status == "frontier"]
    if not frontier:
        return [], compute_kde_state([])
    ensure_embeddings(
        frontier,
        cache=cache,
        provider=provider,
        expected_dimension=expected_dimension,
    )
    embeddings = [n.local_embedding or [] for n in frontier]
    return frontier, compute_kde_state(embeddings)


def estimate_uncertainty_from_density(
    nodes: list[SearchNode],
    cache: DTECache | None = None,
    provider: EmbeddingProvider | None = None,
    *,
    expected_dimension: int | None = None,
) -> dict[str, float]:
    """Estimate uncertainty from low-density frontier regions."""

    frontier, state = estimate_frontier_kde_state(
        nodes,
        cache=cache,
        provider=provider,
        expected_dimension=expected_dimension,
    )
    return {node.node_id: value for node, value in zip(frontier, state.uncertainty)}
