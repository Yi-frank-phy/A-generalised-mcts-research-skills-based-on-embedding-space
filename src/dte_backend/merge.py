"""Merge skeleton for turning beam tree search into graph search.

This is intentionally conservative. It only implements deterministic equivalent
claim merging now, while leaving complementary/conflict merge types in the data
model for later stronger model-backed merge decisions.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .models import MergeProposal, SearchNode
from .relation_models import (
    MergeApplicationRecord,
    RelationRecord,
    stable_relation_id,
)


def resolve_merge_aliases(
    applications: list[MergeApplicationRecord],
    *,
    committed_node_ids: set[str] | None = None,
) -> dict[str, str]:
    """Validate merge provenance and resolve every absorbed alias transitively."""

    canonical_by_absorbed: dict[str, str] = {}
    for application in applications:
        source_ids = set(application.source_node_ids)
        absorbed_ids = set(application.absorbed_node_ids)
        if len(source_ids) != len(application.source_node_ids):
            raise ValueError("merge provenance contains duplicate source node IDs")
        if len(absorbed_ids) != len(application.absorbed_node_ids):
            raise ValueError("merge provenance contains duplicate absorbed node IDs")
        if application.canonical_node_id in absorbed_ids:
            raise ValueError("merge provenance cannot absorb its canonical node")
        if not absorbed_ids:
            raise ValueError("merge provenance must absorb at least one source node")
        expected_sources = absorbed_ids | {application.canonical_node_id}
        if source_ids != expected_sources:
            raise ValueError(
                "merge provenance sources must equal the canonical plus absorbed nodes"
            )
        if set(application.source_node_revisions) != source_ids:
            raise ValueError("merge provenance source revisions must match source node IDs")
        if committed_node_ids is not None and not source_ids.issubset(committed_node_ids):
            missing = sorted(source_ids - committed_node_ids)
            raise ValueError(f"merge provenance references missing committed node {missing[0]}")
        for absorbed_node_id in application.absorbed_node_ids:
            existing = canonical_by_absorbed.get(absorbed_node_id)
            if existing is not None and existing != application.canonical_node_id:
                raise ValueError(
                    "merge provenance conflict: absorbed node "
                    f"{absorbed_node_id} already maps to canonical {existing}"
                )
            canonical_by_absorbed[absorbed_node_id] = application.canonical_node_id

    resolved: dict[str, str] = {}
    for absorbed_node_id in canonical_by_absorbed:
        current = absorbed_node_id
        path: set[str] = set()
        while current in canonical_by_absorbed:
            if current in path:
                raise ValueError(f"merge provenance alias cycle contains node {current}")
            path.add(current)
            current = canonical_by_absorbed[current]
        if committed_node_ids is not None and current not in committed_node_ids:
            raise ValueError(f"merge provenance resolves to missing canonical node {current}")
        resolved[absorbed_node_id] = current
    return resolved


def validate_merge_application_consistency(
    applications: list[MergeApplicationRecord],
) -> None:
    """Require one acyclic canonical provenance target for every absorbed node."""

    resolve_merge_aliases(applications)


def validate_merge_application_relation_provenance(
    application: MergeApplicationRecord,
    record: RelationRecord,
) -> None:
    """Bind one merge application to the exact equivalent observation revision."""

    if application.relation_record_id != record.relation_record_id:
        raise ValueError("merge application relation-record provenance mismatch")
    if record.relation_type != "equivalent":
        raise ValueError("merge application requires an equivalent Relation record")
    source_ids = set(application.source_node_ids)
    if source_ids != {record.left_node_id, record.right_node_id}:
        raise ValueError("merge application sources disagree with Relation provenance")
    try:
        expected_revisions = {
            node_id: record.selected_node_revisions[node_id]
            for node_id in application.source_node_ids
        }
    except KeyError as exc:
        raise ValueError(
            "merge application Relation provenance lacks a selected-node revision"
        ) from exc
    if application.source_node_revisions != expected_revisions:
        raise ValueError(
            "merge application source revisions disagree with Relation provenance"
        )


def validate_alias_projected_node_ancestry(
    nodes: list[SearchNode],
    applications: list[MergeApplicationRecord],
) -> None:
    """Reject self ancestry or cycles that appear after transitive alias projection."""

    node_ids = {node.node_id for node in nodes}
    aliases = resolve_merge_aliases(applications, committed_node_ids=node_ids)
    active_ids = {node.node_id for node in nodes if node.status != "merged"}
    projected_parents: dict[str, list[str]] = {}
    for node in nodes:
        if node.status == "merged":
            continue
        parents: list[str] = []
        for parent_id in node.parent_ids:
            if parent_id not in node_ids:
                raise ValueError(
                    f"merge-projected ancestry references missing parent {parent_id}"
                )
            projected = aliases.get(parent_id, parent_id)
            if projected == node.node_id:
                raise ValueError(
                    f"merge-projected ancestry makes node {node.node_id} its own parent"
                )
            if projected in active_ids:
                parents.append(projected)
        projected_parents[node.node_id] = parents

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("merge-projected ancestry contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for parent_id in projected_parents.get(node_id, []):
            visit(parent_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in projected_parents:
        visit(node_id)


def normalize_claim(text: str) -> str:
    """Normalize claim text for exact-equivalence merge proposals."""

    lowered = text.casefold().strip()
    return re.sub(r"\s+", " ", lowered)


def propose_equivalent_merges(nodes: list[SearchNode]) -> list[MergeProposal]:
    """Propose equivalent merges for frontier nodes with identical normalized claims."""

    groups: dict[str, list[SearchNode]] = defaultdict(list)
    for node in nodes:
        if node.status == "frontier" and node.node_type != "synthesis":
            groups[normalize_claim(node.claim)].append(node)

    proposals: list[MergeProposal] = []
    for claim_key, group in groups.items():
        if len(group) < 2:
            continue
        keep = select_canonical_node(group)
        absorbed = [node for node in group if node.node_id != keep.node_id]
        proposals.append(
            MergeProposal(
                merge_type="equivalent_merge",
                source_node_ids=[n.node_id for n in group],
                target_node_id=keep.node_id,
                rationale=f"Equivalent normalized claim: {claim_key!r}; use deterministic canonical selection.",
                merged_node=None,
                absorbed_node_ids=[n.node_id for n in absorbed],
            )
        )
    return proposals


def _canonical_information_content(node: SearchNode) -> int:
    return (
        len(set(node.assumptions))
        + len(set(node.evidence))
        + len(set(node.risks))
        + (1 if node.rationale.strip() else 0)
    )


def select_canonical_node(nodes: list[SearchNode]) -> SearchNode:
    """Choose an equivalent-merge representative by deterministic backend facts."""

    if not nodes:
        raise ValueError("canonical selection requires at least one node")
    status_rank = {"frontier": 4, "closed": 3, "archived": 2, "merged": 1, "synthesis": 0}
    return sorted(
        nodes,
        key=lambda node: (
            -status_rank[node.status],
            -_canonical_information_content(node),
            -len(set(node.evidence)),
            -(node.score if node.score is not None else node.confidence),
            -(1 if node.judge_result_provenance else 0),
            node.node_id,
        ),
    )[0]


def apply_relation_equivalent_merge(
    nodes: list[SearchNode],
    node_revisions: dict[str, int],
    *,
    source_node_ids: list[str],
    relation_record_id: str,
    applied_graph_revision: int,
    applied_at: str,
    existing_merge_applications: list[MergeApplicationRecord] | None = None,
) -> MergeApplicationRecord:
    """Apply one validated equivalent merge while preserving all source nodes."""

    unique_ids = sorted(set(source_node_ids))
    by_id = {node.node_id: node for node in nodes}
    source_nodes = [by_id[node_id] for node_id in unique_ids if node_id in by_id]
    if len(source_nodes) != len(unique_ids) or len(source_nodes) < 2:
        raise ValueError("equivalent merge requires at least two committed source nodes")

    already_merged = [node for node in source_nodes if node.status == "merged"]
    active = [node for node in source_nodes if node.status != "merged"]
    if len(active) < 2:
        canonical = next((node for node in source_nodes if node.status != "merged"), None)
        if canonical is None:
            raise ValueError("equivalent merge sources are already fully absorbed")
        absorbed_ids = sorted(node.node_id for node in source_nodes if node.node_id != canonical.node_id)
        return MergeApplicationRecord(
            merge_application_id=stable_relation_id("merge", relation_record_id, canonical.node_id, *absorbed_ids),
            relation_record_id=relation_record_id,
            canonical_node_id=canonical.node_id,
            absorbed_node_ids=absorbed_ids,
            source_node_ids=unique_ids,
            source_node_revisions={node_id: node_revisions[node_id] for node_id in unique_ids},
            applied_graph_revision=applied_graph_revision,
            applied_at=applied_at,
        )

    canonical = select_canonical_node(active)
    absorbed = [node for node in active if node.node_id != canonical.node_id]
    source_revisions = {node_id: node_revisions[node_id] for node_id in unique_ids}

    canonical.assumptions = sorted({item for node in active for item in node.assumptions})
    canonical.evidence = sorted({item for node in active for item in node.evidence})
    canonical.risks = sorted({item for node in active for item in node.risks})
    # Parent links among equivalent sources become internal to the collapsed
    # alias set. Resolve older aliases first: a parent such as A is also
    # internal when an earlier merge already mapped A -> B and B is one of the
    # current sources.
    existing_aliases = resolve_merge_aliases(
        existing_merge_applications or [],
        committed_node_ids=set(by_id),
    )
    canonicalized_parents = {
        existing_aliases.get(item, item)
        for node in active
        for item in node.parent_ids
    }
    canonical.parent_ids = sorted(
        canonicalized_parents - set(unique_ids)
    )
    if canonical.node_id in canonical.parent_ids:
        raise ValueError("equivalent merge would create a canonical self-parent")
    canonical.confidence = max(node.confidence for node in active)
    node_revisions[canonical.node_id] += 1
    for node in absorbed:
        node.status = "merged"
        node.expansion_budget = 0
        node_revisions[node.node_id] += 1

    absorbed_ids = sorted(node.node_id for node in absorbed + already_merged)
    return MergeApplicationRecord(
        merge_application_id=stable_relation_id("merge", relation_record_id, canonical.node_id, *absorbed_ids),
        relation_record_id=relation_record_id,
        canonical_node_id=canonical.node_id,
        absorbed_node_ids=absorbed_ids,
        source_node_ids=unique_ids,
        source_node_revisions=source_revisions,
        applied_graph_revision=applied_graph_revision,
        applied_at=applied_at,
    )


def apply_equivalent_merges(nodes: list[SearchNode]) -> list[MergeProposal]:
    """Apply conservative equivalent merges in-place and return proposals."""

    proposals = propose_equivalent_merges(nodes)
    by_id = {node.node_id: node for node in nodes}
    for proposal in proposals:
        for absorbed_id in proposal.absorbed_node_ids:
            node = by_id.get(absorbed_id)
            if node is None:
                continue
            node.status = "merged"
            node.expansion_budget = 0
            node.risks.append(f"merged into {proposal.target_node_id} by equivalent_merge")
    return proposals
