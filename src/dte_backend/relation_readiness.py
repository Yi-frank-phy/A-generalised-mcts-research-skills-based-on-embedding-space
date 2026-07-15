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
    blocking_inventory_candidate_ids: list[str] | None = None,
    blocking_inventory_complete: bool = True,
    enrichment_budget_limit: int = 0,
    enrichment_pairs_committed: int = 0,
    eligible_enrichment_candidate_ids: list[str] | None = None,
) -> SynthesisReadinessRecord:
    """Evaluate a complete current blocker inventory and separate enrichment."""

    selected = set(provisional_selected_node_ids)
    expected_ids = set(
        blocking_inventory_candidate_ids
        if blocking_inventory_candidate_ids is not None
        else [
            candidate.candidate_id
            for candidate in candidates
            if candidate.scheduling_class == "blocking"
            and candidate.left_node_id in selected
            and candidate.right_node_id in selected
        ]
    )
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    inventory_complete = blocking_inventory_complete and all(
        candidate_id in candidate_by_id
        and candidate_by_id[candidate_id].scheduling_class == "blocking"
        for candidate_id in expected_ids
    )
    nonblocking = [
        candidate.candidate_id
        for candidate in candidates
        if candidate.scheduling_class == "enrichment" and candidate.status in {"pending", "granted"}
    ]

    applied_relation_ids = {application.relation_record_id for application in merge_applications}
    records_by_id = {record.relation_record_id: record for record in relation_ledger}
    unapplied_merges = [
        record
        for record in relation_ledger
        if record.relation_type == "equivalent"
        and record.left_node_id in selected
        and record.right_node_id in selected
        and record.relation_record_id not in applied_relation_ids
    ]
    unresolved_conflicts = [
        record
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

    unresolved_candidate_ids: set[str] = set()
    for candidate_id in expected_ids:
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None or candidate.status in {"pending", "granted", "invalidated", "superseded"}:
            unresolved_candidate_ids.add(candidate_id)
            continue
        if candidate.status != "resolved" or not candidate.resolved_relation_record_id:
            unresolved_candidate_ids.add(candidate_id)
            continue
        record = records_by_id.get(candidate.resolved_relation_record_id)
        if record is None:
            unresolved_candidate_ids.add(candidate_id)
        elif record.relation_type == "equivalent" and record.relation_record_id not in applied_relation_ids:
            unresolved_candidate_ids.add(candidate_id)
        elif record.relation_type == "conflict" and record.material_to_synthesis and not record.disclosure_required:
            unresolved_candidate_ids.add(candidate_id)

    unresolved_candidate_ids.update(record.candidate_id for record in unapplied_merges)
    unresolved_candidate_ids.update(record.candidate_id for record in unresolved_conflicts)
    blocking_obligation_ids = set(expected_ids)
    blocking_obligation_ids.update(record.candidate_id for record in unapplied_merges)
    blocking_obligation_ids.update(record.candidate_id for record in unresolved_conflicts)
    blocking_pair_count = len(blocking_obligation_ids)
    unresolved_blocking_pair_count = len(unresolved_candidate_ids)
    resolved_blocking_pair_count = blocking_pair_count - unresolved_blocking_pair_count
    blockers = sorted(unresolved_candidate_ids)
    unresolved_conflict_ids = sorted({record.relation_record_id for record in unresolved_conflicts})
    remaining = max(0, enrichment_budget_limit - enrichment_pairs_committed)
    eligible_enrichment_ids = sorted(set(eligible_enrichment_candidate_ids or []))
    ready = inventory_complete and unresolved_blocking_pair_count == 0
    enrichment_pending = ready and remaining > 0 and bool(eligible_enrichment_ids)
    if not inventory_complete:
        reason = "blocking Relation inventory is incomplete"
    elif blockers:
        reason = "blocking Relation candidates or confirmed-but-unapplied equivalent merges remain"
    elif unresolved_conflict_ids:
        reason = "material conflicts require resolution or explicit disclosure"
    elif disclosure_conflicts:
        reason = "ready with explicit material-conflict disclosure obligations"
    else:
        reason = "no blocking Relation obligation affects the provisional synthesis set"
    return SynthesisReadinessRecord(
        graph_revision=graph_revision,
        provisional_selected_node_ids=list(provisional_selected_node_ids),
        blocking_inventory_complete=inventory_complete,
        blocking_pair_count=blocking_pair_count,
        resolved_blocking_pair_count=resolved_blocking_pair_count,
        unresolved_blocking_pair_count=unresolved_blocking_pair_count,
        blocking_candidate_ids=blockers,
        unresolved_material_conflicts=unresolved_conflict_ids,
        disclosure_required_conflicts=sorted(set(disclosure_conflicts)),
        unresolved_nonblocking_candidates=sorted(set(nonblocking)),
        duplicate_groups=_duplicate_groups(
            [candidate_by_id[item] for item in expected_ids if item in candidate_by_id], selected
        ),
        enrichment_budget_limit=enrichment_budget_limit,
        enrichment_pairs_committed=enrichment_pairs_committed,
        enrichment_pairs_remaining=remaining,
        eligible_enrichment_candidate_ids=eligible_enrichment_ids,
        enrichment_pending=enrichment_pending,
        ready=ready,
        reason=reason,
        evaluated_at=evaluated_at,
    )
