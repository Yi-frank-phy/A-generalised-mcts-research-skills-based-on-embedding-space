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

from .context_envelope import evaluation_payload, semantic_embedding_payload
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


@dataclass(frozen=True)
class JudgeCacheNamespace:
    """Configuration identity for one observable Judge contract."""

    model_snapshot: str
    reasoning_profile: str
    rubric_version: str
    prompt_version: str
    output_schema_version: str


DEFAULT_EMBEDDING_NAMESPACE = EmbeddingCacheNamespace("hash", "hash-v1", 3072, "embedding-v1")
DEFAULT_JUDGE_NAMESPACE = JudgeCacheNamespace(
    "heuristic-score-v1",
    "deterministic",
    "heuristic-rubric-v1",
    "no-prompt",
    "judge-result-v1",
)


def embedding_cache_key(
    node: SearchNode,
    namespace: EmbeddingCacheNamespace = DEFAULT_EMBEDDING_NAMESPACE,
) -> str:
    """Hash stable semantic geometry together with its provider contract."""

    return _hash_payload({"namespace": asdict(namespace), "payload": semantic_embedding_payload(node)})


def judge_cache_key(
    node: SearchNode,
    namespace: JudgeCacheNamespace = DEFAULT_JUDGE_NAMESPACE,
) -> str:
    """Hash evaluation content together with its Judge contract."""

    return _hash_payload({"namespace": asdict(namespace), "payload": evaluation_payload(node)})


def stable_node_payload(node: SearchNode) -> dict[str, object]:
    """Backward-compatible alias for the Judge/evaluation payload."""

    return evaluation_payload(node)


def stable_node_hash(node: SearchNode) -> str:
    """Backward-compatible alias for the Judge cache key."""

    return judge_cache_key(node)


@dataclass
class JudgeCacheEntry:
    """Cached Judge result for an unchanged node."""

    score: float
    reasoning: str


@dataclass
class CacheStats:
    """Tiny cache telemetry for traces and tests."""

    embedding_hits: int = 0
    embedding_misses: int = 0
    judge_hits: int = 0
    judge_misses: int = 0


@dataclass
class DTECache:
    """Per-run in-memory cache."""

    embeddings: dict[str, list[float]] = field(default_factory=dict)
    judge_scores: dict[str, JudgeCacheEntry] = field(default_factory=dict)
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

    def get_judge(
        self,
        node: SearchNode,
        namespace: JudgeCacheNamespace = DEFAULT_JUDGE_NAMESPACE,
    ) -> JudgeCacheEntry | None:
        key = judge_cache_key(node, namespace=namespace)
        value = self.judge_scores.get(key)
        if value is None:
            self.stats.judge_misses += 1
            return None
        self.stats.judge_hits += 1
        return value

    def set_judge(
        self,
        node: SearchNode,
        score: float,
        reasoning: str,
        namespace: JudgeCacheNamespace = DEFAULT_JUDGE_NAMESPACE,
    ) -> None:
        self.judge_scores[judge_cache_key(node, namespace=namespace)] = JudgeCacheEntry(
            score=score,
            reasoning=reasoning,
        )
