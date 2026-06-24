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

from dataclasses import dataclass, field
import hashlib
import json

from .context_envelope import evaluation_payload, semantic_embedding_payload
from .models import SearchNode


def _hash_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def embedding_cache_key(node: SearchNode) -> str:
    """Hash only the stable semantic geometry payload."""

    return _hash_payload(semantic_embedding_payload(node))


def judge_cache_key(node: SearchNode) -> str:
    """Hash the evaluation payload used for Judge caching."""

    return _hash_payload(evaluation_payload(node))


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

    def get_embedding(self, node: SearchNode) -> list[float] | None:
        key = embedding_cache_key(node)
        value = self.embeddings.get(key)
        if value is None:
            self.stats.embedding_misses += 1
            return None
        self.stats.embedding_hits += 1
        return list(value)

    def set_embedding(self, node: SearchNode, embedding: list[float]) -> None:
        self.embeddings[embedding_cache_key(node)] = list(embedding)

    def get_judge(self, node: SearchNode) -> JudgeCacheEntry | None:
        key = judge_cache_key(node)
        value = self.judge_scores.get(key)
        if value is None:
            self.stats.judge_misses += 1
            return None
        self.stats.judge_hits += 1
        return value

    def set_judge(self, node: SearchNode, score: float, reasoning: str) -> None:
        self.judge_scores[judge_cache_key(node)] = JudgeCacheEntry(score=score, reasoning=reasoning)
