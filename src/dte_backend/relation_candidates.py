"""Deterministic candidate selection for relation-oracle calls.

This module answers: which frontier node pairs are worth sending to a Relation
Oracle? It does not classify the relation itself. Classification remains a
subagent/oracle task.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from itertools import combinations

from .models import SearchNode
from .relation_models import RelationCandidate, RelationCandidateReason, stable_relation_id
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
    max_considered_nodes: int = 12,
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

    # Exact duplicates are grouped in linear time before bounded pair expansion.
    duplicate_groups: dict[str, list[SearchNode]] = defaultdict(list)
    for node in frontier:
        duplicate_groups[_normalized_claim(node.claim)].append(node)
    for group in duplicate_groups.values():
        ordered_group = sorted(group, key=lambda node: node.node_id)
        for a, b in zip(ordered_group, ordered_group[1:]):
            add(a, b, "exact normalized-claim duplicate", 1.0)
            if len(candidates) >= max_pairs:
                break
        if len(candidates) >= max_pairs:
            break

    # Near-tied branches.
    ranked = sorted(frontier, key=lambda node: (-_score_for_tie(node), node.node_id))
    for a, b in combinations(ranked[: min(len(ranked), 4)], 2):
        gap = abs(_score_for_tie(a) - _score_for_tie(b))
        if gap <= tie_threshold:
            add(a, b, f"near-tied branch scores: gap={gap:.4f}", 0.75 if not entropy_plateau else 0.9)

    # Embedding proximity if vectors are already present.
    # Proximity is intentionally bounded to the most relevant frontier pool;
    # this is not a global all-pairs synchronization pass.
    embedded = [node for node in ranked[:max_considered_nodes] if node.local_embedding]
    if len(embedded) >= 2:
        distances = cosine_distance_matrix([node.local_embedding or [] for node in embedded])
        for i, a in enumerate(embedded):
            for j, b in enumerate(embedded):
                if j <= i:
                    continue
                distance = float(distances[i, j])
                if distance <= semantic_distance_threshold:
                    add(a, b, f"embedding-close branches: cosine_distance={distance:.4f}", 0.85)

    ordered = sorted(candidates.values(), key=lambda pair: (-pair.priority, pair.node_ids))
    return ordered[:max_pairs]


_PRIORITY_ORDER = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _normalized_evidence(node: SearchNode) -> set[str]:
    return {_normalized_claim(item) for item in node.evidence if _normalized_claim(item)}


def _candidate(
    left: SearchNode,
    right: SearchNode,
    *,
    node_revisions: dict[str, int],
    graph_revision: int,
    reason: RelationCandidateReason,
    priority: str,
    material_to_synthesis: bool,
) -> RelationCandidate:
    left_id, right_id = sorted((left.node_id, right.node_id))
    left_revision = node_revisions[left_id]
    right_revision = node_revisions[right_id]
    return RelationCandidate(
        candidate_id=stable_relation_id(
            "relcand",
            left_id,
            right_id,
            left_revision,
            right_revision,
            graph_revision,
            reason,
        ),
        left_node_id=left_id,
        right_node_id=right_id,
        left_node_revision=left_revision,
        right_node_revision=right_revision,
        candidate_reason=reason,
        priority=priority,
        material_to_synthesis=material_to_synthesis,
        created_from_graph_revision=graph_revision,
    )


def generate_relation_candidates(
    nodes: list[SearchNode],
    *,
    node_revisions: dict[str, int],
    graph_revision: int,
    provisional_synthesis_node_ids: list[str],
    entropy_plateau: bool = False,
    max_candidates: int = 16,
) -> list[RelationCandidate]:
    """Derive a bounded, deterministic Relation queue from committed state.

    Exact duplicates are found by grouping.  Medium-priority geometric/tie
    signals use the already-bounded legacy selector.  Only the provisional
    synthesis set is considered for shared-evidence material conflicts.
    """

    eligible = [
        node
        for node in nodes
        if node.status in {"frontier", "closed"} and node.node_type != "synthesis"
    ]
    by_id = {node.node_id: node for node in eligible}
    selected = [by_id[node_id] for node_id in provisional_synthesis_node_ids if node_id in by_id]
    selected_ids = set(provisional_synthesis_node_ids)
    candidates: dict[tuple[str, str], RelationCandidate] = {}

    def add(left: SearchNode, right: SearchNode, reason: RelationCandidateReason, priority: str) -> None:
        pair = tuple(sorted((left.node_id, right.node_id)))
        material = pair[0] in selected_ids and pair[1] in selected_ids
        value = _candidate(
            left,
            right,
            node_revisions=node_revisions,
            graph_revision=graph_revision,
            reason=reason,
            priority=priority,
            material_to_synthesis=material,
        )
        current = candidates.get(pair)
        if current is None or _PRIORITY_ORDER[value.priority] > _PRIORITY_ORDER[current.priority]:
            candidates[pair] = value

    duplicate_groups: dict[str, list[SearchNode]] = defaultdict(list)
    for node in eligible:
        duplicate_groups[_normalized_claim(node.claim)].append(node)
    ordered_duplicate_groups = [
        sorted(group, key=lambda node: node.node_id)
        for _, group in sorted(duplicate_groups.items())
    ]
    # First cover duplicate components inside the provisional Synthesis set.
    # A full-group chain alone can miss a selected-selected obligation when a
    # non-selected alias sorts between the two selected nodes.
    for group in ordered_duplicate_groups:
        selected_group = [node for node in group if node.node_id in selected_ids]
        for left, right in zip(selected_group, selected_group[1:]):
            add(left, right, "exact_duplicate", "critical")
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    # Then add a bounded chain for the remainder of each duplicate component.
    if len(candidates) < max_candidates:
        for group in ordered_duplicate_groups:
            for left, right in zip(group, group[1:]):
                add(left, right, "exact_duplicate", "critical")
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

    # Shared source material plus divergent conclusions is a conservative,
    # backend-observable signal for a potential material conflict.
    for left, right in combinations(selected, 2):
        if _normalized_claim(left.claim) == _normalized_claim(right.claim):
            continue
        if _normalized_evidence(left).intersection(_normalized_evidence(right)):
            add(left, right, "potential_material_conflict", "critical")

    legacy_pairs = select_relation_candidate_pairs(
        eligible,
        max_pairs=max_candidates,
        entropy_plateau=entropy_plateau,
    )
    for pair in legacy_pairs:
        left, right = by_id[pair.node_ids[0]], by_id[pair.node_ids[1]]
        if "embedding-close" in pair.reason:
            add(left, right, "embedding_close", "high")
        elif "near-tied" in pair.reason:
            add(left, right, "high_score_near_tie", "high" if entropy_plateau else "medium")

    if entropy_plateau and len(selected) >= 2:
        ranked = sorted(selected, key=lambda node: (-_score_for_tie(node), node.node_id))
        add(ranked[0], ranked[1], "entropy_plateau", "high")

    return sorted(
        candidates.values(),
        key=lambda item: (
            -_PRIORITY_ORDER[item.priority],
            item.left_node_id,
            item.right_node_id,
            item.candidate_reason,
        ),
    )[:max_candidates]


def refresh_relation_candidates(
    existing: list[RelationCandidate],
    generated: list[RelationCandidate],
    *,
    nodes: list[SearchNode],
    node_revisions: dict[str, int],
) -> list[RelationCandidate]:
    """Invalidate stale grants and add current candidates without pair duplication."""

    by_id = {node.node_id: node for node in nodes}
    refreshed = [candidate.model_copy(deep=True) for candidate in existing]
    for candidate in refreshed:
        if candidate.status in {"resolved", "superseded", "invalidated"}:
            continue
        left = by_id.get(candidate.left_node_id)
        right = by_id.get(candidate.right_node_id)
        current = (
            left is not None
            and right is not None
            and left.status != "merged"
            and right.status != "merged"
            and node_revisions.get(candidate.left_node_id) == candidate.left_node_revision
            and node_revisions.get(candidate.right_node_id) == candidate.right_node_revision
        )
        if not current:
            candidate.status = "invalidated"

    current_pairs = {
        (candidate.left_node_id, candidate.right_node_id)
        for candidate in refreshed
        if candidate.status in {"pending", "granted", "resolved"}
    }
    for candidate in generated:
        pair = (candidate.left_node_id, candidate.right_node_id)
        if pair not in current_pairs:
            refreshed.append(candidate)
            current_pairs.add(pair)
    return refreshed
