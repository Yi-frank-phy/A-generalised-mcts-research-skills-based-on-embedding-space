"""Small in-memory caches for one DTE run.

The cache is intentionally simple: it is not a database and it does not change
DTE semantics. It only avoids recomputing deterministic node features and judge
scores for unchanged node content during one process run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from .models import SearchNode


def stable_node_payload(node: SearchNode) -> dict[str, object]:
    """Return the content fields that define a node's current meaning.

    Judge and embedding caches must not depend on mutable DTE metrics such as
    score, uncertainty, UCB, status, or expansion budget. Those are controller
    outputs, not semantic node content.
    """

    return {
        "node_type": node.node_type,
        "claim": node.claim,
        "rationale": node.rationale,
        "assumptions": list(node.assumptions),
        "evidence": list(node.evidence),
        "risks": list(node.risks),
        "parent_ids": list(node.parent_ids),
        "confidence": node.confidence,
    }


def stable_node_hash(node: SearchNode) -> str:
    """Compute a deterministic content hash for cache keys."""

    payload = stable_node_payload(node)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    """Per-run in-memory cache.

    The keys are stable semantic hashes. If a node changes its claim/evidence/
    risks, it automatically misses the cache and is re-evaluated.
    """

    embeddings: dict[str, list[float]] = field(default_factory=dict)
    judge_scores: dict[str, JudgeCacheEntry] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)

    def get_embedding(self, node: SearchNode) -> list[float] | None:
        key = stable_node_hash(node)
        value = self.embeddings.get(key)
        if value is None:
            self.stats.embedding_misses += 1
            return None
        self.stats.embedding_hits += 1
        return list(value)

    def set_embedding(self, node: SearchNode, embedding: list[float]) -> None:
        self.embeddings[stable_node_hash(node)] = list(embedding)

    def get_judge(self, node: SearchNode) -> JudgeCacheEntry | None:
        key = stable_node_hash(node)
        value = self.judge_scores.get(key)
        if value is None:
            self.stats.judge_misses += 1
            return None
        self.stats.judge_hits += 1
        return value

    def set_judge(self, node: SearchNode, score: float, reasoning: str) -> None:
        self.judge_scores[stable_node_hash(node)] = JudgeCacheEntry(score=score, reasoning=reasoning)
