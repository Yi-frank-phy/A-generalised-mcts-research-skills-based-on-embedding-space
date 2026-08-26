"""Small in-memory caches for one DTE run.

DTE uses two different cache identities:

- embedding key: stable semantic geometry; ignores parent ids, confidence, status,
  scores, and run-local logs;
- judge key: semantic content plus stated confidence; still ignores controller
  outputs and parent ids.

This improves hit rate for Codex/subagent workflows where context is compiled or
reformatted between runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json

from .context_envelope import semantic_embedding_payload
from .models import SearchNode


def _hash_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingCacheNamespace:
    """Configuration identity for one embedding contract."""

    provider: str
    model_snapshot: str
    dimension: int
    contract_version: str




DEFAULT_EMBEDDING_NAMESPACE = EmbeddingCacheNamespace("hash", "hash-v1", 3072, "embedding-v1")
def embedding_cache_key(
    node: SearchNode,
    namespace: EmbeddingCacheNamespace = DEFAULT_EMBEDDING_NAMESPACE,
) -> str:
    """Hash stable semantic geometry together with its provider contract."""

    return _hash_payload({"namespace": asdict(namespace), "payload": semantic_embedding_payload(node)})










@dataclass
class CacheStats:
    """Tiny cache telemetry for traces and tests."""

    embedding_hits: int = 0
    embedding_misses: int = 0


@dataclass
class DTECache:
    """Per-run in-memory cache."""

    embeddings: dict[str, list[float]] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)

    def get_embedding(
        self,
        node: SearchNode,
        namespace: EmbeddingCacheNamespace = DEFAULT_EMBEDDING_NAMESPACE,
    ) -> list[float] | None:
        key = embedding_cache_key(node, namespace=namespace)
        value = self.embeddings.get(key)
        if value is None:
            self.stats.embedding_misses += 1
            return None
        self.stats.embedding_hits += 1
        return list(value)

    def set_embedding(
        self,
        node: SearchNode,
        embedding: list[float],
        namespace: EmbeddingCacheNamespace = DEFAULT_EMBEDDING_NAMESPACE,
    ) -> None:
        self.embeddings[embedding_cache_key(node, namespace=namespace)] = list(embedding)


