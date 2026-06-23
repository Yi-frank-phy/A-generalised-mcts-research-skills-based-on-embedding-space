"""File-backed DTE cache.

Stores vectors and scalar evaluations across runs so high-quality geometry calls
are not repeated for unchanged nodes.
"""

from __future__ import annotations

import json
from pathlib import Path

from .cache import DTECache, JudgeCacheEntry, stable_node_hash
from .models import SearchNode


class FileDTECache(DTECache):
    """Simple JSON-backed cache with the same interface as DTECache."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {"vectors": {}, "scores": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            self.data.setdefault("vectors", {})
            self.data.setdefault("scores", {})

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_embedding(self, node: SearchNode) -> list[float] | None:
        key = stable_node_hash(node)
        value = self.data["vectors"].get(key)
        if value is None:
            self.stats.embedding_misses += 1
            return None
        self.stats.embedding_hits += 1
        return [float(v) for v in value]

    def set_embedding(self, node: SearchNode, embedding: list[float]) -> None:
        self.data["vectors"][stable_node_hash(node)] = list(embedding)
        self.save()

    def get_judge(self, node: SearchNode) -> JudgeCacheEntry | None:
        key = stable_node_hash(node)
        value = self.data["scores"].get(key)
        if value is None:
            self.stats.judge_misses += 1
            return None
        self.stats.judge_hits += 1
        return JudgeCacheEntry(score=float(value["score"]), reasoning=str(value["reasoning"]))

    def set_judge(self, node: SearchNode, score: float, reasoning: str) -> None:
        self.data["scores"][stable_node_hash(node)] = {"score": float(score), "reasoning": reasoning}
        self.save()
