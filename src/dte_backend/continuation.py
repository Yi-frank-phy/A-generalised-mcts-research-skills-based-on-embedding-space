"""Deterministic node-budget and bounded continuation policy.

The gate consumes only committed graph/controller facts.  Epistemic records are
treated as bounded reasons to continue, never as verified scientific truth and
never as authority to enlarge a run budget.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .epistemic_models import EpistemicLedgerV1
from .models import DTEBaseModel, SearchNode


ContinuationDecision = Literal["continue", "prepare_synthesis"]


class ContinuationGateRecord(DTEBaseModel):
    """Durable, replayable explanation of one continuation decision."""

    schema_version: Literal["dte-continuation-gate.v1"] = (
        "dte-continuation-gate.v1"
    )
    iteration: int = Field(ge=1)
    graph_revision: int = Field(ge=0)
    committed_search_node_count: int = Field(ge=0)
    max_committed_search_nodes: int = Field(ge=1)
    remaining_search_node_slots: int = Field(ge=0)
    canonical_frontier_count: int = Field(ge=0)
    entropy_delta: float | None = Field(default=None, ge=0.0)
    consecutive_plateau_count: int = Field(ge=0)
    plateau_confirmed: bool
    trigger_signals: list[str] = Field(default_factory=list)
    material_yield_signals: list[str] = Field(default_factory=list)
    considered_epistemic_record_ids: list[str] = Field(default_factory=list)
    material_epistemic_record_ids: list[str] = Field(default_factory=list)
    continuation_target_node_ids: list[str] = Field(default_factory=list)
    provisional_synthesis_node_ids: list[str] = Field(default_factory=list)
    positive_allocation_node_ids: list[str] = Field(default_factory=list)
    decision: ContinuationDecision
    reason: str = Field(min_length=1)


def count_committed_search_nodes(nodes: list[SearchNode]) -> int:
    """Count irreversible search-node cost; merge never refunds this count."""

    return sum(node.node_type != "synthesis" for node in nodes)


def remaining_search_node_slots(
    nodes: list[SearchNode],
    max_committed_search_nodes: int,
) -> int:
    return max(
        0,
        max_committed_search_nodes - count_committed_search_nodes(nodes),
    )


def canonical_frontier_node_ids(nodes: list[SearchNode]) -> list[str]:
    return sorted(
        node.node_id
        for node in nodes
        if node.status == "frontier" and node.node_type != "synthesis"
    )


def _epistemic_record_id(record: object) -> str:
    for field_name in ("statement_id", "edge_id", "disposition_id"):
        value = getattr(record, field_name, None)
        if isinstance(value, str):
            return value
    raise TypeError("unknown epistemic record identity")


def epistemic_record_ids(ledger: EpistemicLedgerV1) -> set[str]:
    """Return all durable record identities used by continuation replay."""

    return {
        _epistemic_record_id(record)
        for record in [
            *ledger.statements,
            *ledger.edges,
            *ledger.path_dispositions,
        ]
    }


def expected_continuation_decision(record: ContinuationGateRecord) -> ContinuationDecision:
    """Recompute the gate verdict from the record's durable decision inputs."""

    if not record.trigger_signals:
        return "continue"
    if (
        record.material_yield_signals
        and record.continuation_target_node_ids
        and record.positive_allocation_node_ids
        and record.remaining_search_node_slots > 0
    ):
        return "continue"
    return "prepare_synthesis"


def _node_ids_from_ref(ref: str) -> set[str]:
    prefix = "node-claim:"
    return {ref.removeprefix(prefix)} if ref.startswith(prefix) else set()


def _new_material_epistemic_signals(
    ledger: EpistemicLedgerV1,
    *,
    previously_considered_ids: set[str],
) -> tuple[list[str], list[str], list[str], set[str]]:
    all_records = [*ledger.statements, *ledger.edges, *ledger.path_dispositions]
    new_records = [
        record
        for record in all_records
        if _epistemic_record_id(record) not in previously_considered_ids
    ]
    considered_ids = sorted(_epistemic_record_id(record) for record in new_records)
    material_ids: list[str] = []
    signals: list[str] = []
    target_node_ids: set[str] = set()

    for record in new_records:
        record_id = _epistemic_record_id(record)
        basis_refs = list(getattr(record, "basis_refs", []))
        if not basis_refs:
            continue
        material = False
        if hasattr(record, "epistemic_disposition"):
            disposition = record.epistemic_disposition
            material = disposition in {
                "counterexample_found",
                "contradicted",
                "blocked_by_assumption",
            }
            if material:
                signals.append(f"epistemic_disposition:{disposition}:{record_id}")
                target_node_ids.add(record.target_node_id)
        elif hasattr(record, "relation_type"):
            material = record.relation_type == "contradicts"
            if material:
                signals.append(f"epistemic_contradiction:{record_id}")
                target_node_ids.update(_node_ids_from_ref(record.source_ref))
                target_node_ids.update(_node_ids_from_ref(record.target_ref))
        elif hasattr(record, "statement_type"):
            material = (
                record.statement_type == "evidence"
                and record.source_type == "external_artifact_backed"
            )
            if material:
                signals.append(f"external_evidence:{record_id}")
                target_node_ids.add(record.target_node_id)
        if material:
            material_ids.append(record_id)

    return considered_ids, sorted(material_ids), signals, target_node_ids


def evaluate_continuation_gate(
    *,
    iteration: int,
    graph_revision: int,
    nodes: list[SearchNode],
    max_committed_search_nodes: int,
    entropy_delta: float | None,
    consecutive_plateau_count: int,
    plateau_confirmed: bool,
    allocations: dict[str, int],
    previous_frontier_node_ids: set[str],
    previous_positive_allocation_node_ids: set[str],
    previous_provisional_synthesis_node_ids: set[str],
    provisional_synthesis_node_ids: list[str],
    ledger: EpistemicLedgerV1,
    previously_considered_epistemic_ids: set[str],
) -> ContinuationGateRecord:
    """Evaluate one bounded continuation decision from committed facts."""

    committed_count = count_committed_search_nodes(nodes)
    remaining_slots = max(0, max_committed_search_nodes - committed_count)
    frontier_ids = canonical_frontier_node_ids(nodes)
    frontier_set = set(frontier_ids)
    positive_ids = sorted(node_id for node_id, value in allocations.items() if value > 0)
    positive_set = set(positive_ids)
    provisional_set = set(provisional_synthesis_node_ids)

    trigger_signals: list[str] = []
    if plateau_confirmed:
        trigger_signals.append("entropy_plateau_confirmed")
    if len(frontier_ids) == 1:
        trigger_signals.append("single_canonical_frontier")

    material_signals: list[str] = []
    targets: set[str] = set()
    new_allocated = sorted((positive_set - previous_frontier_node_ids) & frontier_set)
    if new_allocated:
        material_signals.append("new_judge_surviving_allocated_node")
        targets.update(new_allocated)
    if positive_set != previous_positive_allocation_node_ids:
        material_signals.append("positive_allocation_support_changed")
        targets.update(positive_set)
    if provisional_set != previous_provisional_synthesis_node_ids:
        material_signals.append("provisional_synthesis_membership_changed")
        targets.update(provisional_set & positive_set)

    considered_ids, material_ids, epistemic_signals, epistemic_targets = (
        _new_material_epistemic_signals(
            ledger,
            previously_considered_ids=previously_considered_epistemic_ids,
        )
    )
    material_signals.extend(epistemic_signals)
    targets.update(epistemic_targets & positive_set)
    targets &= frontier_set

    if not trigger_signals:
        decision: ContinuationDecision = "continue"
        reason = "continuation gate not triggered"
    elif material_signals and targets and positive_set and remaining_slots > 0:
        decision = "continue"
        reason = "bounded material yield justifies another allocated expansion"
    else:
        decision = "prepare_synthesis"
        reason = "continuation trigger has no bounded material-yield target"

    return ContinuationGateRecord(
        iteration=iteration,
        graph_revision=graph_revision,
        committed_search_node_count=committed_count,
        max_committed_search_nodes=max_committed_search_nodes,
        remaining_search_node_slots=remaining_slots,
        canonical_frontier_count=len(frontier_ids),
        entropy_delta=entropy_delta,
        consecutive_plateau_count=consecutive_plateau_count,
        plateau_confirmed=plateau_confirmed,
        trigger_signals=trigger_signals,
        material_yield_signals=material_signals,
        considered_epistemic_record_ids=considered_ids,
        material_epistemic_record_ids=material_ids,
        continuation_target_node_ids=sorted(targets),
        provisional_synthesis_node_ids=sorted(provisional_set),
        positive_allocation_node_ids=positive_ids,
        decision=decision,
        reason=reason,
    )
