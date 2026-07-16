"""Deterministic Synthesis-readiness evaluation over committed Relation state."""

from __future__ import annotations

from collections import defaultdict

from .merge import resolve_merge_aliases
from .relation_candidates import relation_record_covers_candidate
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
    alias_map = resolve_merge_aliases(merge_applications)

    def canonical_record_endpoints(record: RelationRecord) -> tuple[str, str]:
        left_node_id = alias_map.get(record.left_node_id, record.left_node_id)
        right_node_id = alias_map.get(record.right_node_id, record.right_node_id)
        return tuple(sorted((left_node_id, right_node_id)))

    def selected_record_endpoints(record: RelationRecord) -> bool:
        left_node_id, right_node_id = canonical_record_endpoints(record)
        return left_node_id in selected and right_node_id in selected

    def derived_disclosure_required(record: RelationRecord) -> bool:
        """Carry a formerly nonmaterial conflict forward once both ends matter."""

        return bool(
            record.relation_type == "conflict"
            and selected_record_endpoints(record)
            and (record.disclosure_required or not record.material_to_synthesis)
        )

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
        and selected_record_endpoints(record)
        and record.relation_record_id not in applied_relation_ids
    ]
    # A later blocking reclassification of the same canonical pair supersedes
    # an older nonmaterial enrichment record for readiness disclosure. Keep one
    # strongest current obligation per pair rather than duplicating it.
    conflict_by_pair: dict[tuple[str, str], RelationRecord] = {}
    for record in relation_ledger:
        if record.relation_type != "conflict" or not selected_record_endpoints(record):
            continue
        pair = canonical_record_endpoints(record)
        current = conflict_by_pair.get(pair)
        if current is None or (
            record.material_to_synthesis,
            record.disclosure_required,
            record.committed_at,
            record.relation_record_id,
        ) > (
            current.material_to_synthesis,
            current.disclosure_required,
            current.committed_at,
            current.relation_record_id,
        ):
            conflict_by_pair[pair] = record
    current_conflicts = list(conflict_by_pair.values())
    unresolved_conflicts = [
        record
        for record in current_conflicts
        if record.material_to_synthesis and not derived_disclosure_required(record)
    ]
    disclosure_conflicts = [
        record.relation_record_id
        for record in current_conflicts
        if derived_disclosure_required(record)
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
        elif not relation_record_covers_candidate(record, candidate):
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
