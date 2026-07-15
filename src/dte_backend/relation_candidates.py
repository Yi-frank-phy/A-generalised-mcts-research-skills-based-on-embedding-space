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
from .relation_models import (
    RelationCandidate,
    RelationCandidateReason,
    RelationRecord,
    RelationSchedulingClass,
    stable_relation_id,
)
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


def _normalized_evidence(node: SearchNode) -> set[str]:
    return {_normalized_claim(item) for item in node.evidence if _normalized_claim(item)}


def _candidate(
    left: SearchNode,
    right: SearchNode,
    *,
    node_revisions: dict[str, int],
    graph_revision: int,
    reason: RelationCandidateReason,
    scheduling_class: RelationSchedulingClass,
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
            scheduling_class,
            reason,
        ),
        left_node_id=left_id,
        right_node_id=right_id,
        left_node_revision=left_revision,
        right_node_revision=right_revision,
        candidate_reason=reason,
        scheduling_class=scheduling_class,
        priority=priority,
        material_to_synthesis=material_to_synthesis,
        created_from_graph_revision=graph_revision,
    )


def generate_blocking_relation_obligations(
    nodes: list[SearchNode],
    *,
    node_revisions: dict[str, int],
    graph_revision: int,
    provisional_synthesis_node_ids: list[str],
) -> list[RelationCandidate]:
    """Completely enumerate blockers over the bounded provisional set.

    The provisional selector is capped at eight nodes, so this is at most 28
    pairs.  No enrichment window is allowed to truncate this inventory.
    """

    eligible = [
        node
        for node in nodes
        if node.status in {"frontier", "closed"} and node.node_type != "synthesis"
    ]
    by_id = {node.node_id: node for node in eligible}
    selected = [by_id[node_id] for node_id in provisional_synthesis_node_ids if node_id in by_id]
    candidates: list[RelationCandidate] = []
    for left, right in combinations(selected, 2):
        reason: RelationCandidateReason | None = None
        if _normalized_claim(left.claim) == _normalized_claim(right.claim):
            reason = "exact_duplicate"
        elif _normalized_evidence(left).intersection(_normalized_evidence(right)):
            reason = "potential_material_conflict"
        if reason is not None:
            candidates.append(
                _candidate(
                    left,
                    right,
                    node_revisions=node_revisions,
                    graph_revision=graph_revision,
                    reason=reason,
                    scheduling_class="blocking",
                    priority="critical",
                    material_to_synthesis=True,
                )
            )

    return sorted(
        candidates,
        key=lambda item: (
            item.left_node_id,
            item.right_node_id,
            item.candidate_reason,
        ),
    )


def _candidate_pair_revision(candidate: RelationCandidate) -> tuple[str, str, int, int]:
    return (
        candidate.left_node_id,
        candidate.right_node_id,
        candidate.left_node_revision,
        candidate.right_node_revision,
    )


def _record_pair_revision(record: RelationRecord) -> tuple[str, str, int, int] | None:
    left_revision = record.selected_node_revisions.get(record.left_node_id)
    right_revision = record.selected_node_revisions.get(record.right_node_id)
    if left_revision is None or right_revision is None:
        return None
    return (record.left_node_id, record.right_node_id, left_revision, right_revision)


def _directly_related(left: SearchNode, right: SearchNode) -> bool:
    return bool(
        left.node_id in right.parent_ids
        or right.node_id in left.parent_ids
        or set(left.parent_ids).intersection(right.parent_ids)
    )


def generate_relation_enrichment_candidates(
    nodes: list[SearchNode],
    *,
    node_revisions: dict[str, int],
    graph_revision: int,
    provisional_synthesis_node_ids: list[str],
    existing: list[RelationCandidate],
    relation_ledger: list[RelationRecord],
    entropy_plateau: bool = False,
    max_candidates: int = 16,
) -> list[RelationCandidate]:
    """Generate high-value nonblocking candidates, filtering ledger state first."""

    eligible = [
        node
        for node in nodes
        if node.status in {"frontier", "closed"} and node.node_type != "synthesis"
    ]
    by_id = {node.node_id: node for node in eligible}
    selected_ids = set(provisional_synthesis_node_ids)
    selected = [by_id[node_id] for node_id in provisional_synthesis_node_ids if node_id in by_id]
    if len(selected) < 1:
        return []

    known_pair_revisions = {
        _candidate_pair_revision(candidate)
        for candidate in existing
        if candidate.status in {"pending", "granted", "resolved"}
    }
    known_pair_revisions.update(
        pair for record in relation_ledger if (pair := _record_pair_revision(record)) is not None
    )

    # Selected-selected is the primary enrichment pool.  A non-selected node
    # enters only when it has a direct parent/sibling relation to a selected
    # node.  This keeps the pass bounded without scanning whole-graph pairs.
    selected_related = [
        node
        for node in eligible
        if node.node_id in selected_ids or any(_directly_related(node, item) for item in selected)
    ]
    ranked = sorted(
        selected_related,
        key=lambda node: (node.node_id not in selected_ids, -_score_for_tie(node), node.node_id),
    )[:12]
    distances: dict[tuple[str, str], float] = {}
    embedded = [node for node in ranked if node.local_embedding]
    if len(embedded) >= 2:
        matrix = cosine_distance_matrix([node.local_embedding or [] for node in embedded])
        for i, left in enumerate(embedded):
            for j in range(i + 1, len(embedded)):
                right = embedded[j]
                distances[tuple(sorted((left.node_id, right.node_id)))] = float(matrix[i, j])

    reason_order = {"embedding_close": 0, "high_score_near_tie": 1, "entropy_plateau": 2}
    generated: dict[tuple[str, str], RelationCandidate] = {}
    for left, right in combinations(ranked, 2):
        pair = tuple(sorted((left.node_id, right.node_id)))
        both_selected = pair[0] in selected_ids and pair[1] in selected_ids
        if not both_selected and not _directly_related(left, right):
            continue
        pair_revision = (pair[0], pair[1], node_revisions[pair[0]], node_revisions[pair[1]])
        if pair_revision in known_pair_revisions:
            continue
        reasons: list[RelationCandidateReason] = []
        if distances.get(pair, 1.0) <= 0.15:
            reasons.append("embedding_close")
        if both_selected and abs(_score_for_tie(left) - _score_for_tie(right)) <= 0.05:
            reasons.append("high_score_near_tie")
        if both_selected and entropy_plateau:
            reasons.append("entropy_plateau")
        if not reasons:
            continue
        reason = min(reasons, key=lambda item: reason_order[item])
        generated[pair] = _candidate(
            left,
            right,
            node_revisions=node_revisions,
            graph_revision=graph_revision,
            reason=reason,
            scheduling_class="enrichment",
            priority="high",
            material_to_synthesis=both_selected,
        )

    # Truncation happens only after current candidates and committed records
    # have been removed, so known pairs can never hide unseen pairs.
    return sorted(
        generated.values(),
        key=lambda item: (
            reason_order[item.candidate_reason],
            item.left_node_id,
            item.right_node_id,
        ),
    )[:max_candidates]


def generate_relation_candidates(
    nodes: list[SearchNode],
    *,
    node_revisions: dict[str, int],
    graph_revision: int,
    provisional_synthesis_node_ids: list[str],
    entropy_plateau: bool = False,
    max_candidates: int = 16,
) -> list[RelationCandidate]:
    """Compatibility helper: complete blockers plus a bounded enrichment window."""

    blockers = generate_blocking_relation_obligations(
        nodes,
        node_revisions=node_revisions,
        graph_revision=graph_revision,
        provisional_synthesis_node_ids=provisional_synthesis_node_ids,
    )
    enrichment = generate_relation_enrichment_candidates(
        nodes,
        node_revisions=node_revisions,
        graph_revision=graph_revision,
        provisional_synthesis_node_ids=provisional_synthesis_node_ids,
        existing=blockers,
        relation_ledger=[],
        entropy_plateau=entropy_plateau,
        max_candidates=max_candidates,
    )
    return blockers + enrichment


def refresh_relation_candidates(
    existing: list[RelationCandidate],
    generated: list[RelationCandidate],
    *,
    nodes: list[SearchNode],
    node_revisions: dict[str, int],
    relation_ledger: list[RelationRecord] | None = None,
) -> list[RelationCandidate]:
    """Invalidate stale work and add generated candidates with current coverage."""

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

    generated_blocking_pairs = {
        _candidate_pair_revision(candidate)
        for candidate in generated
        if candidate.scheduling_class == "blocking"
    }
    for candidate in refreshed:
        if (
            candidate.scheduling_class == "enrichment"
            and candidate.status in {"pending", "granted"}
            and _candidate_pair_revision(candidate) in generated_blocking_pairs
        ):
            candidate.status = "invalidated"

    existing_ids = {candidate.candidate_id for candidate in refreshed}
    records_by_pair = {
        pair: record
        for record in relation_ledger or []
        if (pair := _record_pair_revision(record)) is not None
    }
    for candidate in generated:
        if candidate.candidate_id in existing_ids:
            continue
        covering_record = records_by_pair.get(_candidate_pair_revision(candidate))
        if covering_record is not None:
            candidate = candidate.model_copy(
                update={
                    "status": "resolved",
                    "resolved_relation_record_id": covering_record.relation_record_id,
                }
            )
        refreshed.append(candidate)
        existing_ids.add(candidate.candidate_id)
    return refreshed
