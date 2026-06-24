"""Deterministic candidate selection for relation-oracle calls.

This module answers: which frontier node pairs are worth sending to a Relation
Oracle? It does not classify the relation itself. Classification remains a
subagent/oracle task.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .models import SearchNode
from .text_features import cosine_distance_matrix


@dataclass(frozen=True)
class RelationCandidatePair:
    """A proposed pair for relation-oracle classification."""

    node_ids: tuple[str, str]
    reason: str
    priority: float


def _score_for_tie(node: SearchNode) -> float:
    if node.ucb_score is not None:
        return node.ucb_score
    if node.score is not None:
        return node.score
    return node.confidence


def _normalized_claim(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def select_relation_candidate_pairs(
    nodes: list[SearchNode],
    max_pairs: int = 5,
    semantic_distance_threshold: float = 0.15,
    tie_threshold: float = 0.05,
    entropy_plateau: bool = False,
) -> list[RelationCandidatePair]:
    """Select frontier pairs likely to benefit from relation-oracle judgment.

    Signals:
    - exact normalized-claim duplicate;
    - close embedding distance when both embeddings exist;
    - near-tied UCB/score branches;
    - entropy plateau, where near-tied branches deserve discriminator/merge checks.
    """

    frontier = [node for node in nodes if node.status == "frontier"]
    if len(frontier) < 2:
        return []

    candidates: dict[tuple[str, str], RelationCandidatePair] = {}

    def add(a: SearchNode, b: SearchNode, reason: str, priority: float) -> None:
        key = tuple(sorted((a.node_id, b.node_id)))
        current = candidates.get(key)
        pair = RelationCandidatePair(node_ids=key, reason=reason, priority=priority)
        if current is None or pair.priority > current.priority:
            candidates[key] = pair

    # Exact duplicate fallback.
    for a, b in combinations(frontier, 2):
        if _normalized_claim(a.claim) == _normalized_claim(b.claim):
            add(a, b, "exact normalized-claim duplicate", 1.0)

    # Near-tied branches.
    ranked = sorted(frontier, key=_score_for_tie, reverse=True)
    for a, b in combinations(ranked[: min(len(ranked), 4)], 2):
        gap = abs(_score_for_tie(a) - _score_for_tie(b))
        if gap <= tie_threshold:
            add(a, b, f"near-tied branch scores: gap={gap:.4f}", 0.75 if not entropy_plateau else 0.9)

    # Embedding proximity if vectors are already present.
    embedded = [node for node in frontier if node.local_embedding]
    if len(embedded) >= 2:
        distances = cosine_distance_matrix([node.local_embedding or [] for node in embedded])
        for i, a in enumerate(embedded):
            for j, b in enumerate(embedded):
                if j <= i:
                    continue
                distance = float(distances[i, j])
                if distance <= semantic_distance_threshold:
                    add(a, b, f"embedding-close branches: cosine_distance={distance:.4f}", 0.85)

    ordered = sorted(candidates.values(), key=lambda pair: pair.priority, reverse=True)
    return ordered[:max_pairs]
