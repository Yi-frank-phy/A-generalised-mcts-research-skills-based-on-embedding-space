"""Deterministic Synthesis-readiness evaluation over committed Relation state."""

from __future__ import annotations

from collections import defaultdict

from .relation_models import (
    MergeApplicationRecord,
    RelationCandidate,
    RelationRecord,
    SynthesisReadinessRecord,
)


def _duplicate_groups(candidates: list[RelationCandidate], selected: set[str]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        if candidate.candidate_reason != "exact_duplicate":
            continue
        if candidate.left_node_id not in selected or candidate.right_node_id not in selected:
            continue
        adjacency[candidate.left_node_id].add(candidate.right_node_id)
        adjacency[candidate.right_node_id].add(candidate.left_node_id)
    groups: list[list[str]] = []
    seen: set[str] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        group: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in group:
                continue
            group.add(node_id)
            stack.extend(adjacency[node_id] - group)
        seen.update(group)
        groups.append(sorted(group))
    return groups


def evaluate_synthesis_readiness(
    *,
    graph_revision: int,
    provisional_selected_node_ids: list[str],
    candidates: list[RelationCandidate],
    relation_ledger: list[RelationRecord],
    merge_applications: list[MergeApplicationRecord],
    evaluated_at: str,
) -> SynthesisReadinessRecord:
    """Evaluate only obligations that can materially change final Synthesis."""

    selected = set(provisional_selected_node_ids)
    unresolved = [candidate for candidate in candidates if candidate.status in {"pending", "granted"}]
    blocking = [
        candidate.candidate_id
        for candidate in unresolved
        if candidate.left_node_id in selected
        and candidate.right_node_id in selected
        and candidate.candidate_reason in {"exact_duplicate", "potential_material_conflict"}
    ]
    nonblocking = [candidate.candidate_id for candidate in unresolved if candidate.candidate_id not in blocking]

    applied_relation_ids = {application.relation_record_id for application in merge_applications}
    unapplied_merges = [
        record.candidate_id
        for record in relation_ledger
        if record.relation_type == "equivalent"
        and record.left_node_id in selected
        and record.right_node_id in selected
        and record.relation_record_id not in applied_relation_ids
    ]
    unresolved_conflicts = [
        record.relation_record_id
        for record in relation_ledger
        if record.relation_type == "conflict"
        and record.material_to_synthesis
        and record.left_node_id in selected
        and record.right_node_id in selected
        and not record.disclosure_required
    ]
    disclosure_conflicts = [
        record.relation_record_id
        for record in relation_ledger
        if record.relation_type == "conflict"
        and record.material_to_synthesis
        and record.left_node_id in selected
        and record.right_node_id in selected
        and record.disclosure_required
    ]

    blockers = sorted(set(blocking + unapplied_merges))
    unresolved_conflicts = sorted(set(unresolved_conflicts))
    ready = not blockers and not unresolved_conflicts
    if blockers:
        reason = "blocking Relation candidates or confirmed-but-unapplied equivalent merges remain"
    elif unresolved_conflicts:
        reason = "material conflicts require resolution or explicit disclosure"
    elif disclosure_conflicts:
        reason = "ready with explicit material-conflict disclosure obligations"
    else:
        reason = "no blocking Relation obligation affects the provisional synthesis set"
    return SynthesisReadinessRecord(
        graph_revision=graph_revision,
        provisional_selected_node_ids=list(provisional_selected_node_ids),
        blocking_candidate_ids=blockers,
        unresolved_material_conflicts=unresolved_conflicts,
        disclosure_required_conflicts=sorted(set(disclosure_conflicts)),
        unresolved_nonblocking_candidates=sorted(set(nonblocking)),
        duplicate_groups=_duplicate_groups(candidates, selected),
        ready=ready,
        reason=reason,
        evaluated_at=evaluated_at,
    )
