"""Persistent backend protocol driven by the current Codex App main agent.

This module never launches Codex. It grants bounded logical episodes, persists
attempt lifecycle, accepts a complete structured result, and commits only
through :func:`commit_episode_result`.
"""

from __future__ import annotations

import json
import hashlib
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from .control import authorize_synthesis_control
from .continuation import (
    ContinuationGateRecord,
    count_committed_search_nodes,
    epistemic_record_ids,
    evaluate_continuation_gate,
    expected_continuation_decision,
    remaining_search_node_slots,
)
from .embedding import EmbeddingProvider, get_embedding_provider
from .entropy import evaluate_entropy_state
from .epistemic_commit import EpistemicReferenceContext
from .epistemic_models import (
    EpistemicLedgerV1,
    ResearcherLearningRecordV1,
    stable_epistemic_id,
)
from .episode_adapter import (
    build_executor_episode_request,
    build_judge_episode_request,
    build_relation_episode_request,
)
from .episode_commit import EpisodeGraph, commit_episode_result
from .episode_models import (
    CommitOutcome,
    EpisodeRequest,
    EpisodeResult,
    ExecutorEpisodeOutput,
    JudgeEpisodeOutput,
    RuntimeLimits,
    canonical_json_bytes,
    compute_output_hash,
)
from .file_cache import FileDTECache
from .math_engine import allocate_frontier
from .merge import (
    resolve_merge_aliases,
    validate_merge_application_relation_provenance,
)
from .models import DTEBaseModel, DTERunSpec, SearchNode, SynthesisControlRequest
from .novelty import estimate_frontier_kde_state
from .relation_candidates import (
    expected_relation_candidate_id,
    generate_blocking_relation_obligations,
    generate_relation_enrichment_candidates,
    refresh_relation_candidates,
    relation_record_covers_candidate,
    select_node_disjoint_relation_batch,
)
from .relation_models import (
    MergeApplicationRecord,
    ProvisionalSynthesisSelection,
    RelationCandidate,
    RelationEpisodeOutput,
    RelationRecord,
    SynthesisReadinessRecord,
    stable_relation_id,
)
from .relation_readiness import evaluate_synthesis_readiness
from .synthesis import select_provisional_synthesis_nodes
from .telemetry import EpisodeEventLog


AttemptStatus = Literal[
    "granted",
    "in_progress",
    "completed_uncommitted",
    "committed",
    "rejected",
    "failed",
    "cancelled",
    "expired",
    "superseded",
]
ControllerAction = Literal[
    "episode_required",
    "continue_controller",
    "await_operator_decision",
    "ready_for_synthesis",
    "run_complete",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid persisted timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"persisted timestamp must include a UTC offset: {value!r}")
    return parsed


def _validate_artifact_component(value: str, label: str) -> None:
    """Reject persisted identities which could be interpreted as filesystem paths."""

    if (
        not value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or Path(value).is_absolute()
        or Path(value).name != value
    ):
        raise ValueError(f"unsafe {label} for attempt artifact path")


def _episode_request_hash(request: EpisodeRequest) -> str:
    return hashlib.sha256(canonical_json_bytes(request)).hexdigest()


def _run_spec_hash(spec: DTERunSpec) -> str:
    return hashlib.sha256(canonical_json_bytes(spec)).hexdigest()


def _initial_nodes_hash(nodes: list[SearchNode]) -> str:
    payload = [node.model_dump(mode="json") for node in nodes]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class EpisodeAttemptRecord(DTEBaseModel):
    attempt_id: str
    attempt_number: int = Field(ge=1)
    status: AttemptStatus
    request: EpisodeRequest
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    granted_at: str
    deadline_at: str | None = None
    submitted_at: str | None = None
    result_hash: str | None = None
    committed_result: EpisodeResult | None = None
    commit_outcome: CommitOutcome | None = None
    failure_reason: str | None = None
    superseded_from_status: AttemptStatus | None = None
    retry_exhaustion_released: bool = False


class EpisodeLifecycleRecord(DTEBaseModel):
    episode_id: str
    run_id: str
    role: Literal["executor", "seed", "judge", "relation", "synthesis"]
    attempts: list[EpisodeAttemptRecord]
    committed_attempt_id: str | None = None


class PendingTelemetryEvent(DTEBaseModel):
    """A committed fact waiting to be idempotently appended to telemetry."""

    event_id: str
    event_type: str
    fields: dict[str, Any]


class TerminalRecord(DTEBaseModel):
    """Immutable audit record for the first controller terminal transition."""

    action: Literal["ready_for_synthesis", "run_complete"]
    source: Literal[
        "authorized_synthesis",
        "max_iterations",
        "max_search_nodes",
        "continuation_gate",
        "controller_stop",
    ]
    reason: str
    graph_revision: int = Field(ge=0)
    controller_iteration: int = Field(ge=0)
    committed_at: str


class ControllerIterationRecord(DTEBaseModel):
    """Durable proof that one deterministic allocation transition occurred."""

    iteration: int = Field(ge=1)
    input_graph_revision: int = Field(ge=0)
    output_graph_revision: int = Field(ge=1)
    frontier_node_ids: list[str]
    node_revisions_before: dict[str, int]
    allocations: dict[str, int]
    ucb_scores: dict[str, float]
    local_embeddings: dict[str, list[float]]
    densities: dict[str, float]
    uncertainties: dict[str, float]
    spatial_entropy: float
    entropy_delta: float | None = Field(default=None, ge=0.0)
    normalized_temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    plateau_signal: bool = False
    consecutive_plateau_count: int = Field(default=0, ge=0)
    effective_child_cap: int | None = Field(default=None, ge=0)


class AppRunState(DTEBaseModel):
    state_schema_version: Literal["app-run-state.v2"] = "app-run-state.v2"
    run_id: str
    spec: DTERunSpec
    spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_nodes: list[SearchNode]
    initial_nodes_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    nodes: list[SearchNode]
    graph_revision: int = Field(ge=0)
    node_revisions: dict[str, int]
    episodes: list[EpisodeLifecycleRecord] = Field(default_factory=list)
    active_episode_id: str | None = None
    active_attempt_id: str | None = None
    controller_action: ControllerAction = "continue_controller"
    synthesis_request: SynthesisControlRequest | None = None
    controller_iteration: int = Field(default=0, ge=0)
    controller_iteration_records: list[ControllerIterationRecord] = Field(
        default_factory=list
    )
    continuation_gate_records: list[ContinuationGateRecord] = Field(
        default_factory=list
    )
    previous_spatial_entropy: float | None = None
    relation_candidates: list[RelationCandidate] = Field(default_factory=list)
    relation_ledger: list[RelationRecord] = Field(default_factory=list)
    merge_applications: list[MergeApplicationRecord] = Field(default_factory=list)
    epistemic_ledger: EpistemicLedgerV1 = Field(default_factory=EpistemicLedgerV1)
    provisional_synthesis_selection: ProvisionalSynthesisSelection | None = None
    synthesis_readiness: SynthesisReadinessRecord | None = None
    relation_readiness_status: Literal["not_evaluated", "evaluated", "legacy_unchecked"] = "not_evaluated"
    pending_terminal_action: Literal["ready_for_synthesis", "run_complete"] | None = None
    pending_terminal_reason: str | None = None
    pending_terminal_source: Literal[
        "authorized_synthesis",
        "max_iterations",
        "max_search_nodes",
        "continuation_gate",
        "controller_stop",
    ] | None = None
    pending_terminal_gate_evaluated: bool = False
    terminal_record: TerminalRecord | None = None
    pending_telemetry_events: list[PendingTelemetryEvent] = Field(default_factory=list)
    created_at: str
    updated_at: str

    def graph(self) -> EpisodeGraph:
        return EpisodeGraph(
            nodes=[node.model_copy(deep=True) for node in self.nodes],
            revision=self.graph_revision,
            node_revisions=dict(self.node_revisions),
            relation_candidates=[item.model_copy(deep=True) for item in self.relation_candidates],
            relation_ledger=[item.model_copy(deep=True) for item in self.relation_ledger],
            merge_applications=[item.model_copy(deep=True) for item in self.merge_applications],
            epistemic_ledger=self.epistemic_ledger.model_copy(deep=True),
        )

    def replace_graph(self, graph: EpisodeGraph) -> None:
        self.nodes = [node.model_copy(deep=True) for node in graph.nodes]
        self.graph_revision = graph.revision
        self.node_revisions = dict(graph.node_revisions)
        self.relation_candidates = [item.model_copy(deep=True) for item in graph.relation_candidates]
        self.relation_ledger = [item.model_copy(deep=True) for item in graph.relation_ledger]
        self.merge_applications = [item.model_copy(deep=True) for item in graph.merge_applications]
        self.epistemic_ledger = graph.epistemic_ledger.model_copy(deep=True)


def _validate_terminal_intent_provenance(
    state: AppRunState,
    *,
    action: Literal["ready_for_synthesis", "run_complete"],
    reason: str,
    source: Literal[
        "authorized_synthesis",
        "max_iterations",
        "max_search_nodes",
        "continuation_gate",
        "controller_stop",
    ],
) -> None:
    """Rebuild the controller-owned terminal decision from durable facts."""

    policy_action: Literal["ready_for_synthesis", "run_complete"] = (
        "ready_for_synthesis" if state.spec.require_final_synthesis else "run_complete"
    )
    if source == "authorized_synthesis":
        if (
            state.synthesis_request is None
            or action != "ready_for_synthesis"
            or reason != "authorized synthesis request is pending"
        ):
            raise ValueError("terminal intent disagrees with its authorized synthesis command")
        return
    if state.synthesis_request is not None:
        raise ValueError("algorithmic terminal intent cannot override an authorized synthesis command")
    if action != policy_action:
        raise ValueError("terminal action disagrees with the run synthesis policy")
    if source == "max_iterations":
        if (
            state.controller_iteration < state.spec.budget.max_iterations
            or reason != "maximum controller iterations reached"
        ):
            raise ValueError("max-iteration terminal intent lacks its controller boundary")
        return
    if source == "max_search_nodes":
        if (
            count_committed_search_nodes(state.nodes)
            < state.spec.budget.max_committed_search_nodes
            or reason != "maximum committed search nodes reached"
        ):
            raise ValueError("search-node terminal intent lacks its hard boundary")
        return
    if source == "continuation_gate":
        if not state.continuation_gate_records:
            raise ValueError("continuation terminal intent lacks a durable gate record")
        gate = state.continuation_gate_records[-1]
        if gate.decision != "prepare_synthesis" or reason != gate.reason:
            raise ValueError("continuation terminal intent disagrees with its gate record")
        return

    if not state.controller_iteration_records:
        raise ValueError("controller-stop terminal intent predates any controller transition")
    latest = state.controller_iteration_records[-1]
    previous_entropy = (
        state.controller_iteration_records[-2].spatial_entropy
        if len(state.controller_iteration_records) > 1
        else None
    )
    entropy_state = evaluate_entropy_state(
        spatial_entropy=latest.spatial_entropy,
        previous_entropy=previous_entropy,
        iteration=latest.iteration,
        min_iterations=state.spec.budget.min_iterations_before_synthesis,
        entropy_change_threshold=state.spec.budget.entropy_change_threshold,
        previous_plateau_count=(
            state.controller_iteration_records[-2].consecutive_plateau_count
            if len(state.controller_iteration_records) > 1
            else 0
        ),
        plateau_confirmations=state.spec.budget.entropy_plateau_confirmations,
        t_max=state.spec.budget.t_max,
    )
    if (
        state.spec.budget.continuation_policy == "legacy_entropy_v1"
        and entropy_state.plateau_signal
    ):
        expected_reason = entropy_state.stop_reason or "entropy stopping policy reached"
    elif not any(latest.allocations.values()):
        expected_reason = "controller produced no positive expandable frontier allocation"
    elif not any(node.status == "frontier" for node in state.nodes):
        expected_reason = "no expandable frontier remains"
    else:
        raise ValueError("controller-stop terminal intent lacks a reconstructable stop condition")
    if reason != expected_reason:
        raise ValueError("controller-stop terminal reason disagrees with durable controller facts")


class NextEpisodeOutcome(DTEBaseModel):
    run_id: str
    controller_action: ControllerAction
    request: EpisodeRequest | None = None
    resumed_existing_attempt: bool = False
    reason: str | None = None


class SubmitEpisodeOutcome(DTEBaseModel):
    run_id: str
    episode_id: str
    attempt_id: str
    commit_outcome: CommitOutcome
    next_controller_action: ControllerAction


class TransitionOutcome(DTEBaseModel):
    run_id: str
    episode_id: str
    attempt_id: str
    status: AttemptStatus
    controller_action: ControllerAction
    request: EpisodeRequest | None = None
    reason: str | None = None


def _state_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "app_run_state.json"


def _event_log(run_dir: str | Path) -> EpisodeEventLog:
    return EpisodeEventLog(Path(run_dir) / "episode_events.jsonl")


def _app_cache_path(run_dir: str | Path) -> Path:
    """Stable run-scoped cache artifact; it is not part of graph state."""

    return Path(run_dir) / "dte_cache.json"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


class _BufferedEpisodeEvents:
    """Collect commit-boundary events until authoritative state is durable."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event_type: str, **fields: Any) -> None:
        self.events.append((event_type, dict(fields)))


def _queue_event(state: AppRunState, event_type: str, **fields: Any) -> None:
    state.pending_telemetry_events.append(
        PendingTelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            fields=dict(fields),
        )
    )


def _queue_buffered_events(state: AppRunState, buffered: _BufferedEpisodeEvents) -> None:
    for event_type, fields in buffered.events:
        _queue_event(state, event_type, **fields)


def _emit_best_effort(log: EpisodeEventLog, event_type: str, **fields: Any) -> None:
    try:
        log.emit(event_type, **fields)
    except Exception:
        return


def _flush_pending_events(run_dir: str | Path, state: AppRunState) -> None:
    """Best-effort outbox flush with idempotent event IDs.

    The state containing the outbox has already been atomically installed. A
    telemetry or second state-write failure therefore cannot roll back or make
    the controller commit ambiguous; the durable outbox is retried on load.
    """

    if not state.pending_telemetry_events:
        return
    original = list(state.pending_telemetry_events)
    published_ids: set[str] = set()
    log = _event_log(run_dir)
    for event in original:
        try:
            log.emit(event.event_type, event_id=event.event_id, **event.fields)
        except Exception:
            break
        published_ids.add(event.event_id)
    if not published_ids:
        return
    state.pending_telemetry_events = [
        event for event in original if event.event_id not in published_ids
    ]
    try:
        _write_json_atomic(_state_path(run_dir), state.model_dump(mode="json"))
    except Exception:
        # Disk still contains the original outbox. Event IDs make replay safe.
        state.pending_telemetry_events = original


def _save_state(run_dir: str | Path, state: AppRunState) -> None:
    state.updated_at = _iso()
    # Treat every persistence call as another machine-facing boundary.  This
    # catches invalid values introduced through assignment on nested Pydantic
    # instances before they can brick the next restart.
    canonical = AppRunState.model_validate(state.model_dump(mode="json"))
    _validate_loaded_state(canonical)
    _write_json_atomic(_state_path(run_dir), canonical.model_dump(mode="json"))
    try:
        _write_relation_artifacts(run_dir, canonical)
        _write_epistemic_artifact(run_dir, canonical)
    except Exception:
        # Relation and epistemic mirrors are derived from AppRunState and are
        # retried on the next save/load. They are never authoritative facts.
        pass
    _repair_attempt_artifacts(run_dir, canonical)
    _flush_pending_events(run_dir, canonical)
    # Keep the caller-visible object aligned with the durable outbox after a
    # successful or partially successful best-effort flush.
    state.pending_telemetry_events = [
        event.model_copy(deep=True) for event in canonical.pending_telemetry_events
    ]


def _write_relation_artifacts(run_dir: str | Path, state: AppRunState) -> None:
    relation_dir = Path(run_dir) / "relations"
    enrichment_committed = _relation_enrichment_pairs_committed(state)
    enrichment_remaining = max(
        0, state.spec.budget.max_relation_enrichment_pairs - enrichment_committed
    )
    _write_json_atomic(
        relation_dir / "candidates.json",
        {
            "schema_version": "relation-candidates.v2",
            "run_id": state.run_id,
            "blocking_candidate_count": sum(
                item.scheduling_class == "blocking" for item in state.relation_candidates
            ),
            "enrichment_candidate_count": sum(
                item.scheduling_class == "enrichment" for item in state.relation_candidates
            ),
            "candidates": [item.model_dump(mode="json") for item in state.relation_candidates],
        },
    )
    _write_json_atomic(
        relation_dir / "relation_ledger.json",
        {
            "schema_version": "relation-ledger.v2",
            "run_id": state.run_id,
            "enrichment_budget_limit": state.spec.budget.max_relation_enrichment_pairs,
            "enrichment_pairs_committed": enrichment_committed,
            "enrichment_pairs_remaining": enrichment_remaining,
            "records": [item.model_dump(mode="json") for item in state.relation_ledger],
            "merge_applications": [item.model_dump(mode="json") for item in state.merge_applications],
        },
    )
    _write_json_atomic(
        relation_dir / "synthesis_readiness.json",
        {
            "schema_version": "synthesis-readiness-artifact.v2",
            "run_id": state.run_id,
            "status": state.relation_readiness_status,
            "selection": (
                None
                if state.provisional_synthesis_selection is None
                else state.provisional_synthesis_selection.model_dump(mode="json")
            ),
            "readiness": (
                None if state.synthesis_readiness is None else state.synthesis_readiness.model_dump(mode="json")
            ),
        },
    )


def _write_epistemic_artifact(run_dir: str | Path, state: AppRunState) -> None:
    """Refresh the non-authoritative mirror from AppRunState."""

    _write_json_atomic(
        Path(run_dir) / "epistemic" / "ledger.json",
        {
            "schema_version": "dte-epistemic-ledger-mirror.v1",
            "run_id": state.run_id,
            "authoritative_source": "app_run_state.json#epistemic_ledger",
            "ledger": state.epistemic_ledger.model_dump(mode="json"),
        },
    )


def _relation_enrichment_pairs_committed(state: AppRunState) -> int:
    """Rebuild the run-level successful enrichment spend from durable ledger facts."""

    return len(
        {
            record.candidate_id
            for record in state.relation_ledger
            if record.scheduling_class == "enrichment"
        }
    )


def _validate_persisted_epistemic_ledger(
    state: AppRunState,
    attempts_by_identity: dict[
        tuple[str, str], tuple[EpisodeLifecycleRecord, EpisodeAttemptRecord]
    ],
    *,
    created_at: datetime,
    updated_at: datetime,
) -> None:
    """Rebuild ledger identities and payloads from committed episode outputs."""

    ledger = state.epistemic_ledger
    statements = {item.statement_id: item for item in ledger.statements}
    edges = {item.edge_id: item for item in ledger.edges}
    dispositions = {
        item.disposition_id: item for item in ledger.path_dispositions
    }
    all_ids = {*statements, *edges, *dispositions}
    expected_ids: set[str] = set()

    for (episode_id, attempt_id), (episode, attempt) in attempts_by_identity.items():
        if attempt.status != "committed" or episode.role not in {"executor", "judge"}:
            continue
        result = attempt.committed_result
        assert result is not None
        output = result.structured_output
        bundle = (
            output.epistemic_contributions
            if isinstance(output, (ExecutorEpisodeOutput, JudgeEpisodeOutput))
            else None
        )
        if bundle is None:
            continue
        authorized = set(attempt.request.selected_node_revisions)
        if isinstance(output, ExecutorEpisodeOutput):
            authorized.update(item.node_id for item in output.nodes)
        local_statement_ids = {
            item.local_id: stable_epistemic_id(
                "epistmt",
                run_id=state.run_id,
                episode_id=episode_id,
                attempt_id=attempt_id,
                output_hash=result.output_hash,
                local_id=item.local_id,
                record_type="statement",
            )
            for item in bundle.statements
        }

        def resolved(ref: str) -> str:
            if not ref.startswith("local-statement:"):
                return ref
            local_id = ref.removeprefix("local-statement:")
            if local_id not in local_statement_ids:
                raise ValueError("persisted epistemic output has an unknown local reference")
            return f"epistemic:{local_statement_ids[local_id]}"

        def validate_episode_authority_and_source(
            source_type: str,
            refs: list[str],
        ) -> None:
            for ref in refs:
                if ref.startswith("node-claim:") and (
                    ref.removeprefix("node-claim:") not in authorized
                ):
                    raise ValueError(
                        "persisted epistemic output exceeds its node authority"
                    )
            if source_type not in {
                "agent_reported",
                "external_artifact_backed",
            }:
                raise ValueError(
                    "persisted episode output forges epistemic source authority"
                )
            if source_type == "external_artifact_backed" and not any(
                ref.startswith(("artifact:", "external:")) for ref in refs
            ):
                raise ValueError(
                    "persisted external-artifact-backed record lacks its basis"
                )

        def ref_target_node(ref: str) -> str | None:
            if ref.startswith("node-claim:"):
                return ref.removeprefix("node-claim:")
            if ref.startswith("epistemic:"):
                statement = statements.get(ref.removeprefix("epistemic:"))
                return None if statement is None else statement.target_node_id
            return None

        for contribution in bundle.statements:
            stable_id = local_statement_ids[contribution.local_id]
            expected_ids.add(stable_id)
            record = statements.get(stable_id)
            resolved_basis = [resolved(ref) for ref in contribution.basis_refs]
            validate_episode_authority_and_source(
                contribution.source_type, resolved_basis
            )
            if record is None or (
                record.local_id != contribution.local_id
                or record.statement_type != contribution.statement_type
                or record.text != contribution.text
                or record.target_node_id != contribution.target_node_id
                or record.source_type != contribution.source_type
                or record.basis_refs != resolved_basis
                or record.target_node_id not in authorized
            ):
                raise ValueError(
                    "persisted epistemic statement disagrees with its committed output"
                )
        for contribution in bundle.edges:
            stable_id = stable_epistemic_id(
                "epiedge",
                run_id=state.run_id,
                episode_id=episode_id,
                attempt_id=attempt_id,
                output_hash=result.output_hash,
                local_id=contribution.local_id,
                record_type="edge",
            )
            expected_ids.add(stable_id)
            record = edges.get(stable_id)
            resolved_refs = [
                resolved(contribution.source_ref),
                resolved(contribution.target_ref),
                *(resolved(ref) for ref in contribution.basis_refs),
            ]
            validate_episode_authority_and_source(
                contribution.source_type, resolved_refs
            )
            target_node = ref_target_node(resolved_refs[1])
            source_node = ref_target_node(resolved_refs[0])
            if (target_node is not None and target_node not in authorized) or (
                target_node is None and source_node not in authorized
            ):
                raise ValueError(
                    "persisted epistemic edge lacks an authorized node anchor"
                )
            if record is None or (
                record.local_id != contribution.local_id
                or record.source_ref != resolved(contribution.source_ref)
                or record.target_ref != resolved(contribution.target_ref)
                or record.relation_type != contribution.relation_type
                or record.source_type != contribution.source_type
                or record.basis_refs
                != [resolved(ref) for ref in contribution.basis_refs]
                or record.explanation != contribution.explanation
            ):
                raise ValueError(
                    "persisted epistemic edge disagrees with its committed output"
                )
        for contribution in bundle.path_dispositions:
            stable_id = stable_epistemic_id(
                "epidisp",
                run_id=state.run_id,
                episode_id=episode_id,
                attempt_id=attempt_id,
                output_hash=result.output_hash,
                local_id=contribution.local_id,
                record_type="path_disposition",
            )
            expected_ids.add(stable_id)
            record = dispositions.get(stable_id)
            resolved_basis = [resolved(ref) for ref in contribution.basis_refs]
            validate_episode_authority_and_source(
                contribution.source_type, resolved_basis
            )
            if record is None or (
                record.local_id != contribution.local_id
                or record.target_node_id != contribution.target_node_id
                or record.epistemic_disposition
                != contribution.epistemic_disposition
                or record.source_type != contribution.source_type
                or record.basis_refs != resolved_basis
                or record.explanation != contribution.explanation
                or record.target_node_id not in authorized
            ):
                raise ValueError(
                    "persisted epistemic disposition disagrees with its committed output"
                )

    if expected_ids != all_ids:
        raise ValueError(
            "persisted epistemic ledger is incomplete or contains non-committed records"
        )

    committed_attempts = {
        identity
        for identity, (episode, attempt) in attempts_by_identity.items()
        if attempt.status == "committed"
        and episode.committed_attempt_id == attempt.attempt_id
    }
    node_ids = {node.node_id for node in state.nodes}
    relation_ids = {item.relation_record_id for item in state.relation_ledger}
    merge_ids = {item.merge_application_id for item in state.merge_applications}

    def validate_ref(ref: str) -> None:
        if ref.startswith("node-claim:"):
            valid = ref.removeprefix("node-claim:") in node_ids
        elif ref.startswith("epistemic:"):
            valid = ref.removeprefix("epistemic:") in all_ids
        elif ref.startswith("relation:"):
            valid = ref.removeprefix("relation:") in relation_ids
        elif ref.startswith("merge:"):
            valid = ref.removeprefix("merge:") in merge_ids
        elif ref.startswith("episode-result:"):
            parts = ref.removeprefix("episode-result:").split(":")
            valid = len(parts) == 2 and tuple(parts) in committed_attempts
        elif ref.startswith("artifact:"):
            value = ref.removeprefix("artifact:")
            path = PurePosixPath(value)
            valid = bool(value) and not path.is_absolute() and not any(
                part in {"", ".", ".."} for part in path.parts
            ) and "\\" not in value and path.as_posix() == value
        elif ref.startswith("external:"):
            valid = bool(ref.removeprefix("external:").strip())
        elif ref.startswith("learning:"):
            valid = bool(ref.removeprefix("learning:").strip())
        else:
            valid = ref == f"run:{state.run_id}"
        if not valid:
            raise ValueError(f"persisted epistemic ledger has an invalid reference: {ref}")

    for record in [*ledger.statements, *ledger.edges, *ledger.path_dispositions]:
        owner = attempts_by_identity.get((record.episode_id, record.attempt_id))
        committed_at = _parse_time(record.committed_at)
        if owner is None:
            raise ValueError("persisted epistemic record references a missing attempt")
        owner_episode, owner_attempt = owner
        if (
            record.run_id != state.run_id
            or owner_episode.role != record.role
            or owner_episode.committed_attempt_id != owner_attempt.attempt_id
            or owner_attempt.status != "committed"
            or owner_attempt.result_hash != record.output_hash
            or record.source_type
            not in {"agent_reported", "external_artifact_backed"}
            or committed_at < created_at
            or committed_at > updated_at
        ):
            raise ValueError(
                "persisted epistemic record disagrees with committed attempt provenance"
            )
        for ref in record.basis_refs:
            validate_ref(ref)
        if hasattr(record, "source_ref"):
            validate_ref(record.source_ref)
            validate_ref(record.target_ref)


def _validate_loaded_state(state: AppRunState) -> None:
    """Fail closed on persisted cross-object graph/ledger inconsistencies."""

    graph = state.graph()
    if not graph.nodes:
        raise ValueError("persisted App run has no committed initial nodes")
    node_ids = {node.node_id for node in graph.nodes}
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    if state.spec_hash != _run_spec_hash(state.spec):
        raise ValueError("persisted RunSpec disagrees with its immutable run hash")
    if state.initial_nodes_hash != _initial_nodes_hash(state.initial_nodes):
        raise ValueError("persisted initial node snapshot disagrees with its immutable hash")
    _validate_initial_nodes(state.initial_nodes)
    created_at = _parse_time(state.created_at)
    updated_at = _parse_time(state.updated_at)
    if updated_at < created_at:
        raise ValueError("persisted App run was updated before it was created")
    if state.controller_iteration > state.spec.budget.max_iterations:
        raise ValueError("persisted controller iteration exceeds the run budget")
    if (
        state.spec.budget.continuation_policy == "bounded_node_yield_v1"
        and
        count_committed_search_nodes(state.nodes)
        > state.spec.budget.max_committed_search_nodes
    ):
        raise ValueError("persisted search graph exceeds the committed-node budget")
    if len(state.controller_iteration_records) != state.controller_iteration:
        raise ValueError("persisted controller iteration lacks durable transition records")
    previous_output_revision = -1
    for expected_iteration, record in enumerate(
        state.controller_iteration_records,
        start=1,
    ):
        record_ids = set(record.frontier_node_ids)
        if (
            record.iteration != expected_iteration
            or record.output_graph_revision != record.input_graph_revision + 1
            or record.input_graph_revision < previous_output_revision
            or record.output_graph_revision > state.graph_revision
            or len(record.frontier_node_ids) != len(record_ids)
            or record_ids != set(record.node_revisions_before)
            or record_ids != set(record.allocations)
            or record_ids != set(record.ucb_scores)
            or record_ids != set(record.local_embeddings)
            or record_ids != set(record.densities)
            or record_ids != set(record.uncertainties)
            or not record_ids.issubset(node_ids)
            or any(
                budget < 0
                or budget > state.spec.budget.max_children_per_iteration
                for budget in record.allocations.values()
            )
            or sum(record.allocations.values())
            > (
                state.spec.budget.max_children_per_iteration
                if record.effective_child_cap is None
                else record.effective_child_cap
            )
        ):
            raise ValueError("persisted controller allocation transition is inconsistent")
        previous_output_revision = record.output_graph_revision
    if state.controller_iteration_records:
        if state.previous_spatial_entropy != state.controller_iteration_records[-1].spatial_entropy:
            raise ValueError("persisted controller entropy disagrees with its latest transition")
    elif state.previous_spatial_entropy is not None:
        raise ValueError("persisted controller entropy predates any allocation transition")
    if state.spec.budget.continuation_policy == "bounded_node_yield_v1":
        if len(state.continuation_gate_records) != state.controller_iteration:
            raise ValueError("persisted controller iteration lacks a continuation-gate record")
        durable_epistemic_ids = epistemic_record_ids(state.epistemic_ledger)
        initial_node_ids = {node.node_id for node in state.initial_nodes}
        committed_child_revisions: dict[str, int] = {}
        attempt_commit_revisions: dict[tuple[str, str], int] = {}
        for episode in state.episodes:
            for attempt in episode.attempts:
                outcome = attempt.commit_outcome
                if (
                    attempt.status != "committed"
                    or outcome is None
                    or not outcome.accepted
                ):
                    continue
                attempt_commit_revisions[(episode.episode_id, attempt.attempt_id)] = (
                    outcome.graph_revision_after
                )
                if episode.role == "executor":
                    for node_id in outcome.accepted_node_ids:
                        committed_child_revisions[node_id] = outcome.graph_revision_after

        def ledger_at_revision(graph_revision: int) -> EpistemicLedgerV1:
            def visible(record: object) -> bool:
                identity = (record.episode_id, record.attempt_id)  # type: ignore[attr-defined]
                revision = attempt_commit_revisions.get(identity)
                return revision is not None and revision <= graph_revision

            return EpistemicLedgerV1(
                statements=[
                    record
                    for record in state.epistemic_ledger.statements
                    if visible(record)
                ],
                edges=[
                    record
                    for record in state.epistemic_ledger.edges
                    if visible(record)
                ],
                path_dispositions=[
                    record
                    for record in state.epistemic_ledger.path_dispositions
                    if visible(record)
                ],
            )

        previously_considered_ids: set[str] = set()
        previous_committed_count = 0
        for expected_iteration, gate in enumerate(
            state.continuation_gate_records,
            start=1,
        ):
            controller_record = state.controller_iteration_records[
                expected_iteration - 1
            ]
            expected_triggers = []
            if gate.plateau_confirmed:
                expected_triggers.append("entropy_plateau_confirmed")
            if gate.canonical_frontier_count == 1:
                expected_triggers.append("single_canonical_frontier")
            positive_allocation_ids = sorted(
                node_id
                for node_id, budget in controller_record.allocations.items()
                if budget > 0
            )
            node_ids_at_revision = initial_node_ids | {
                node_id
                for node_id, revision in committed_child_revisions.items()
                if revision <= gate.graph_revision
            }
            nodes_at_revision = [
                nodes_by_id[node_id].model_copy(
                    update={
                        "status": (
                            "frontier"
                            if node_id in controller_record.frontier_node_ids
                            else "closed"
                        )
                    }
                )
                for node_id in sorted(node_ids_at_revision)
            ]
            previous_controller_record = (
                state.controller_iteration_records[expected_iteration - 2]
                if expected_iteration > 1
                else None
            )
            previous_gate = (
                state.continuation_gate_records[expected_iteration - 2]
                if expected_iteration > 1
                else None
            )
            replayed_gate = evaluate_continuation_gate(
                iteration=gate.iteration,
                graph_revision=gate.graph_revision,
                nodes=nodes_at_revision,
                max_committed_search_nodes=(
                    state.spec.budget.max_committed_search_nodes
                ),
                entropy_delta=controller_record.entropy_delta,
                consecutive_plateau_count=(
                    controller_record.consecutive_plateau_count
                ),
                plateau_confirmed=(
                    gate.iteration
                    >= state.spec.budget.min_iterations_before_synthesis
                    and controller_record.consecutive_plateau_count
                    >= state.spec.budget.entropy_plateau_confirmations
                ),
                allocations=controller_record.allocations,
                previous_frontier_node_ids=(
                    set(previous_controller_record.frontier_node_ids)
                    if previous_controller_record is not None
                    else set()
                ),
                previous_positive_allocation_node_ids=(
                    {
                        node_id
                        for node_id, budget in previous_controller_record.allocations.items()
                        if budget > 0
                    }
                    if previous_controller_record is not None
                    else set()
                ),
                previous_provisional_synthesis_node_ids=(
                    set(previous_gate.provisional_synthesis_node_ids)
                    if previous_gate is not None
                    else set()
                ),
                provisional_synthesis_node_ids=gate.provisional_synthesis_node_ids,
                ledger=ledger_at_revision(gate.graph_revision),
                previously_considered_epistemic_ids=previously_considered_ids,
            )
            expected_committed_count = count_committed_search_nodes(
                nodes_at_revision
            )
            if (
                gate.iteration != expected_iteration
                or gate.graph_revision != controller_record.output_graph_revision
                or gate.max_committed_search_nodes
                != state.spec.budget.max_committed_search_nodes
                or gate.committed_search_node_count < previous_committed_count
                or gate.committed_search_node_count != expected_committed_count
                or gate.committed_search_node_count
                > count_committed_search_nodes(state.nodes)
                or gate.remaining_search_node_slots
                != max(
                    0,
                    gate.max_committed_search_nodes
                    - gate.committed_search_node_count,
                )
                or len(gate.considered_epistemic_record_ids)
                != len(set(gate.considered_epistemic_record_ids))
                or len(gate.material_epistemic_record_ids)
                != len(set(gate.material_epistemic_record_ids))
                or set(gate.considered_epistemic_record_ids)
                & previously_considered_ids
                or not set(gate.considered_epistemic_record_ids).issubset(
                    durable_epistemic_ids
                )
                or not set(gate.material_epistemic_record_ids).issubset(
                    set(gate.considered_epistemic_record_ids)
                )
                or gate.entropy_delta != controller_record.entropy_delta
                or gate.consecutive_plateau_count
                != controller_record.consecutive_plateau_count
                or gate.plateau_confirmed
                != (
                    gate.iteration
                    >= state.spec.budget.min_iterations_before_synthesis
                    and gate.consecutive_plateau_count
                    >= state.spec.budget.entropy_plateau_confirmations
                )
                or gate.trigger_signals != expected_triggers
                or gate.canonical_frontier_count
                != len(controller_record.frontier_node_ids)
                or gate.positive_allocation_node_ids != positive_allocation_ids
                or not set(gate.continuation_target_node_ids).issubset(
                    set(gate.positive_allocation_node_ids)
                )
                or not set(gate.provisional_synthesis_node_ids).issubset(node_ids)
                or gate.decision != expected_continuation_decision(gate)
                or gate.model_dump(mode="json")
                != replayed_gate.model_dump(mode="json")
            ):
                raise ValueError("persisted continuation-gate record is inconsistent")
            previously_considered_ids.update(
                gate.considered_epistemic_record_ids
            )
            previous_committed_count = gate.committed_search_node_count
    elif state.continuation_gate_records:
        raise ValueError("legacy entropy state cannot contain bounded continuation records")

    def require_unique(values: list[str], label: str) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"persisted state contains duplicate {label}")

    require_unique([episode.episode_id for episode in state.episodes], "episode IDs")
    require_unique(
        [
            attempt.attempt_id
            for episode in state.episodes
            for attempt in episode.attempts
        ],
        "attempt IDs",
    )
    require_unique(
        [candidate.candidate_id for candidate in state.relation_candidates],
        "Relation candidate IDs",
    )
    require_unique(
        [record.relation_record_id for record in state.relation_ledger],
        "Relation record IDs",
    )
    require_unique(
        [application.merge_application_id for application in state.merge_applications],
        "merge application IDs",
    )
    require_unique(
        [application.relation_record_id for application in state.merge_applications],
        "merge application Relation record IDs",
    )

    attempts_by_identity: dict[
        tuple[str, str], tuple[EpisodeLifecycleRecord, EpisodeAttemptRecord]
    ] = {}
    active_lifecycle_records: list[
        tuple[EpisodeLifecycleRecord, EpisodeAttemptRecord]
    ] = []
    active_statuses = {"granted", "in_progress", "completed_uncommitted"}
    for episode in state.episodes:
        _validate_artifact_component(episode.episode_id, "episode_id")
        if episode.run_id != state.run_id:
            raise ValueError("persisted episode run_id disagrees with App run")
        if not episode.attempts:
            raise ValueError("persisted episode has no attempts")
        attempt_numbers = [attempt.attempt_number for attempt in episode.attempts]
        if attempt_numbers != list(range(1, len(episode.attempts) + 1)):
            raise ValueError("persisted episode attempt numbers are not contiguous")
        retry_limit = episode.attempts[0].request.runtime_limits.max_retries
        if (
            len(episode.attempts) > retry_limit + 1
            or any(
                attempt.request.runtime_limits.max_retries != retry_limit
                for attempt in episode.attempts
            )
        ):
            raise ValueError("persisted episode attempts exceed or rewrite the retry grant")
        for previous in episode.attempts[:-1]:
            if (
                previous.status != "superseded"
                or previous.superseded_from_status
                not in {
                    "granted",
                    "in_progress",
                    "completed_uncommitted",
                    "rejected",
                    "failed",
                    "cancelled",
                    "expired",
                }
            ):
                raise ValueError("persisted retry chain lacks a superseded predecessor")
        if episode.attempts[-1].status == "superseded":
            raise ValueError("persisted retry chain ends in a superseded attempt")
        if any(
            attempt.status != "superseded"
            and attempt.superseded_from_status is not None
            for attempt in episode.attempts
        ):
            raise ValueError("non-superseded attempt retains retry predecessor metadata")
        committed = [attempt for attempt in episode.attempts if attempt.status == "committed"]
        if episode.committed_attempt_id is None:
            if committed:
                raise ValueError("persisted committed attempt lacks episode commit identity")
        elif (
            len(committed) != 1
            or committed[0].attempt_id != episode.committed_attempt_id
            or committed[0].result_hash is None
            or committed[0].commit_outcome is None
            or not committed[0].commit_outcome.accepted
            or committed[0].commit_outcome.episode_id != episode.episode_id
        ):
            raise ValueError("persisted episode commit identity is inconsistent")
        if committed:
            attempt = committed[0]
            result = attempt.committed_result
            outcome = attempt.commit_outcome
            assert outcome is not None
            if result is None:
                raise ValueError("persisted committed attempt lacks its authoritative result")
            if (
                result.episode_id != episode.episode_id
                or result.attempt_id != attempt.attempt_id
                or result.run_id != state.run_id
                or result.role != episode.role
                or result.status != "completed"
                or result.input_graph_revision != attempt.request.input_graph_revision
                or result.selected_node_revisions != attempt.request.selected_node_revisions
                or result.schema_version != attempt.request.output_schema_version
                or result.output_hash != attempt.result_hash
                or result.output_hash
                != compute_output_hash(result.structured_output, result.schema_version)
                or outcome.graph_revision_before != attempt.request.input_graph_revision
                or outcome.graph_revision_after > state.graph_revision
            ):
                raise ValueError("persisted committed result disagrees with its attempt envelope")
            if episode.role == "judge" and isinstance(
                result.structured_output, JudgeEpisodeOutput
            ):
                accepted_ids = [
                    observation.node_id for observation in result.structured_output.observations
                ]
                expected_after = attempt.request.input_graph_revision + 1
            elif episode.role == "executor" and isinstance(
                result.structured_output, ExecutorEpisodeOutput
            ):
                accepted_ids = [node.node_id for node in result.structured_output.nodes]
                expected_after = attempt.request.input_graph_revision + 1
            elif episode.role == "relation" and isinstance(
                result.structured_output, RelationEpisodeOutput
            ):
                accepted_ids = [
                    observation.candidate_id
                    for observation in result.structured_output.observations
                ]
                expected_after = attempt.request.input_graph_revision + 1 + int(
                    any(
                        observation.relation_type == "equivalent"
                        for observation in result.structured_output.observations
                    )
                )
            else:
                raise ValueError("persisted committed result has the wrong role output schema")
            if (
                outcome.accepted_node_ids != accepted_ids
                or outcome.accepted_node_count != len(accepted_ids)
                or outcome.graph_revision_after != expected_after
            ):
                raise ValueError("persisted committed result disagrees with its commit outcome")
        for attempt in episode.attempts:
            _validate_artifact_component(attempt.attempt_id, "attempt_id")
            if attempt.status != "committed" and (
                attempt.committed_result is not None
                or attempt.result_hash is not None
                or (
                    attempt.commit_outcome is not None
                    and attempt.commit_outcome.accepted
                )
            ):
                raise ValueError("non-committed attempt claims an accepted commit fact")
            if attempt.retry_exhaustion_released and (
                episode.role != "relation"
                or attempt is not episode.attempts[-1]
                or episode.committed_attempt_id is not None
                or attempt.status
                not in {"rejected", "failed", "cancelled", "expired"}
                or attempt.attempt_number <= attempt.request.runtime_limits.max_retries
            ):
                raise ValueError("retry exhaustion release lacks a terminal Relation attempt")
            granted_at = _parse_time(attempt.granted_at)
            deadline_at = (
                None if attempt.deadline_at is None else _parse_time(attempt.deadline_at)
            )
            submitted_at = (
                None if attempt.submitted_at is None else _parse_time(attempt.submitted_at)
            )
            if granted_at < created_at or granted_at > updated_at:
                raise ValueError("persisted attempt grant timestamp is outside the run lifetime")
            wall_clock_seconds = attempt.request.runtime_limits.wall_clock_seconds
            expected_deadline = (
                None
                if wall_clock_seconds is None
                else granted_at + timedelta(seconds=wall_clock_seconds)
            )
            if deadline_at != expected_deadline:
                raise ValueError("persisted attempt deadline disagrees with its runtime grant")
            if submitted_at is not None and (
                submitted_at < granted_at or submitted_at > updated_at
            ):
                raise ValueError("persisted attempt submission is outside its run lifetime")
            if attempt.status in {"committed", "rejected", "completed_uncommitted"} and (
                submitted_at is None
            ):
                raise ValueError("persisted submitted attempt lacks its submission timestamp")
            if (
                attempt.status == "committed"
                and deadline_at is not None
                and submitted_at is not None
                and submitted_at > deadline_at
            ):
                raise ValueError("persisted committed attempt was submitted after its deadline")
            request = attempt.request
            if attempt.request_hash != _episode_request_hash(request):
                raise ValueError("persisted episode request disagrees with its durable grant hash")
            if request.run_id != state.run_id:
                raise ValueError("persisted attempt request run_id disagrees with App run")
            if request.episode_id != episode.episode_id:
                raise ValueError("persisted attempt request episode_id disagrees with lifecycle")
            if request.attempt_id != attempt.attempt_id:
                raise ValueError("persisted attempt request attempt_id disagrees with lifecycle")
            if request.role != episode.role:
                raise ValueError("persisted attempt request role disagrees with lifecycle")
            if request.role == "relation":
                payload = request.relation_payload
                if payload is None:
                    raise ValueError("persisted Relation request lacks its candidate payload")
                granted_node_ids = [
                    node_id
                    for pair in payload.candidate_pairs
                    for node_id in (pair.left.node_id, pair.right.node_id)
                ]
                if (
                    len(payload.candidate_pairs)
                    > state.spec.budget.max_relation_pairs_per_episode
                    or len(granted_node_ids) != len(set(granted_node_ids))
                ):
                    raise ValueError("persisted Relation grant is not node-disjoint and bounded")
            identity = (episode.episode_id, attempt.attempt_id)
            attempts_by_identity[identity] = (episode, attempt)
            if attempt.status in active_statuses:
                active_lifecycle_records.append((episode, attempt))
    if len(active_lifecycle_records) > 1:
        raise ValueError("persisted App state contains multiple active attempt lifecycles")

    _validate_persisted_epistemic_ledger(
        state,
        attempts_by_identity,
        created_at=created_at,
        updated_at=updated_at,
    )

    require_unique(
        [event.event_id for event in state.pending_telemetry_events],
        "pending telemetry event IDs",
    )
    commit_event_roles = {
        "judge_observations_committed": "judge",
        "nodes_committed": "executor",
        "relation_observations_committed": "relation",
        "merge_proposed": "relation",
        "complementarity_recorded": "relation",
        "conflict_recorded": "relation",
        "merge_applied": "relation",
        "relation_enrichment_committed": "relation",
        "episode_completed": None,
    }
    for event in state.pending_telemetry_events:
        try:
            uuid.UUID(event.event_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("pending telemetry event has an invalid durable identity") from exc
        fields = event.fields
        if fields.get("run_id") != state.run_id:
            raise ValueError("pending telemetry event disagrees with the App run")
        if event.event_type in commit_event_roles:
            identity = (fields.get("episode_id"), fields.get("attempt_id"))
            owner = attempts_by_identity.get(identity)  # type: ignore[arg-type]
            if owner is None:
                raise ValueError("pending commit telemetry references a missing attempt")
            owner_episode, owner_attempt = owner
            outcome = owner_attempt.commit_outcome
            expected_role = commit_event_roles[event.event_type]
            if (
                owner_episode.committed_attempt_id != owner_attempt.attempt_id
                or owner_attempt.status != "committed"
                or outcome is None
                or not outcome.accepted
                or (expected_role is not None and owner_episode.role != expected_role)
                or fields.get("role") != owner_episode.role
                or fields.get("status") != "committed"
                or fields.get("input_graph_revision")
                != owner_attempt.request.input_graph_revision
                or (
                    fields.get("graph_revision") is not None
                    and fields.get("graph_revision") != outcome.graph_revision_after
                )
                or (
                    fields.get("accepted_node_count") is not None
                    and fields.get("accepted_node_count") != outcome.accepted_node_count
                )
            ):
                raise ValueError("pending commit telemetry disagrees with its durable commit")
        elif event.event_type == "run_completed":
            if state.controller_action != "run_complete" or state.terminal_record is None:
                raise ValueError("pending run-completed telemetry lacks a terminal run fact")
        elif event.event_type == "allocation_recorded":
            record = next(
                (
                    item
                    for item in state.controller_iteration_records
                    if item.input_graph_revision == fields.get("input_graph_revision")
                ),
                None,
            )
            if record is None or (
                fields.get("allocated_child_count") != sum(record.allocations.values())
                or fields.get("spatial_entropy") != record.spatial_entropy
            ):
                raise ValueError("pending allocation telemetry lacks a controller transition")

    judge_observations: dict[
        str, tuple[EpisodeLifecycleRecord, EpisodeAttemptRecord, Any]
    ] = {}
    for episode in state.episodes:
        if episode.role != "judge" or episode.committed_attempt_id is None:
            continue
        attempt = next(
            item for item in episode.attempts if item.attempt_id == episode.committed_attempt_id
        )
        result = attempt.committed_result
        if result is None or not isinstance(result.structured_output, JudgeEpisodeOutput):
            raise ValueError("committed Judge attempt lacks its authoritative observations")
        output_ids = [item.node_id for item in result.structured_output.observations]
        payload = attempt.request.judge_payload
        payload_ids = (
            [] if payload is None else [item.node_id for item in payload.selected_frontier_nodes]
        )
        if (
            len(output_ids) != len(set(output_ids))
            or set(output_ids) != set(attempt.request.selected_node_revisions)
            or set(payload_ids) != set(output_ids)
        ):
            raise ValueError("committed Judge attempt does not cover its exact grant")
        for observation in result.structured_output.observations:
            if observation.node_id in judge_observations:
                raise ValueError("persisted node was committed by multiple Judge attempts")
            judge_observations[observation.node_id] = (episode, attempt, observation)

    allocation_record_by_node = {
        node_id: record
        for record in state.controller_iteration_records
        for node_id in record.frontier_node_ids
    }
    committed_executor_by_parent: dict[str, EpisodeAttemptRecord] = {}
    producer_origins: dict[str, dict[str, Any]] = {
        node.node_id: node.model_dump(mode="json") for node in state.initial_nodes
    }
    for episode in state.episodes:
        if episode.role != "executor" or episode.committed_attempt_id is None:
            continue
        attempt = next(
            item for item in episode.attempts if item.attempt_id == episode.committed_attempt_id
        )
        result = attempt.committed_result
        if result is None or not isinstance(result.structured_output, ExecutorEpisodeOutput):
            raise ValueError("committed Executor attempt lacks its authoritative child set")
        parent_id = attempt.request.parent_node_id
        if parent_id is None:
            raise ValueError("committed Executor attempt lacks its parent grant")
        if parent_id in committed_executor_by_parent:
            raise ValueError("persisted allocation parent has multiple committed Executor episodes")
        committed_executor_by_parent[parent_id] = attempt
        for candidate in result.structured_output.nodes:
            if candidate.node_id in producer_origins:
                raise ValueError("persisted graph node has multiple producer origins")
            producer_origins[candidate.node_id] = candidate.model_dump(mode="json")

    if set(producer_origins) != node_ids:
        raise ValueError("persisted graph nodes disagree with authoritative producer outputs")
    producer_fields = {
        "node_id",
        "node_type",
        "claim",
        "rationale",
        "assumptions",
        "evidence",
        "risks",
        "parent_ids",
        "confidence",
    }
    expected_semantics = {
        node_id: {field: origin[field] for field in producer_fields}
        for node_id, origin in producer_origins.items()
    }
    absorbed_so_far: set[str] = set()
    applied_so_far: list[MergeApplicationRecord] = []
    for application in state.merge_applications:
        source_ids = set(application.source_node_ids)
        active_source_ids = sorted(source_ids - absorbed_so_far)
        if (
            not active_source_ids
            or application.canonical_node_id not in active_source_ids
        ):
            raise ValueError("merge application canonical was not active when committed")
        if len(active_source_ids) >= 2:
            canonical = expected_semantics[application.canonical_node_id]
            canonical["assumptions"] = sorted(
                {
                    item
                    for node_id in active_source_ids
                    for item in expected_semantics[node_id]["assumptions"]
                }
            )
            canonical["evidence"] = sorted(
                {
                    item
                    for node_id in active_source_ids
                    for item in expected_semantics[node_id]["evidence"]
                }
            )
            canonical["risks"] = sorted(
                {
                    item
                    for node_id in active_source_ids
                    for item in expected_semantics[node_id]["risks"]
                }
            )
            prior_aliases = resolve_merge_aliases(
                applied_so_far,
                committed_node_ids=node_ids,
            )
            canonical["parent_ids"] = sorted(
                {
                    prior_aliases.get(parent_id, parent_id)
                    for node_id in active_source_ids
                    for parent_id in expected_semantics[node_id]["parent_ids"]
                }
                - source_ids
            )
            if application.canonical_node_id in canonical["parent_ids"]:
                raise ValueError("replayed equivalent merge creates a canonical self-parent")
            canonical["confidence"] = max(
                expected_semantics[node_id]["confidence"]
                for node_id in active_source_ids
            )
            absorbed_so_far.update(
                node_id
                for node_id in active_source_ids
                if node_id != application.canonical_node_id
            )
        applied_so_far.append(application)

    for node in state.nodes:
        current = node.model_dump(mode="json", include=producer_fields)
        expected = expected_semantics[node.node_id]
        if current != expected:
            raise ValueError("persisted graph semantics disagree with authoritative producer output")
        expected_status = (
            "merged"
            if node.node_id in absorbed_so_far
            else "closed"
            if node.node_id in committed_executor_by_parent
            else "frontier"
        )
        if node.status != expected_status:
            raise ValueError("persisted node status disagrees with committed lifecycle transitions")

    for node in state.nodes:
        judge_fields_present = bool(
            node.judge_reasoning is not None
            or node.judge_risks
            or node.judge_uncertainty_evidence
            or node.judge_result_provenance is not None
        )
        geometry = (
            node.local_embedding,
            node.density,
            node.uncertainty,
            node.ucb_score,
        )
        allocation_record = allocation_record_by_node.get(node.node_id)
        if any(value is not None for value in geometry) and allocation_record is None:
            raise ValueError("persisted controller geometry lacks an allocation transition")
        if allocation_record is not None:
            committed_executor = committed_executor_by_parent.get(node.node_id)
            expected_remaining = (
                0
                if committed_executor is not None
                else allocation_record.allocations[node.node_id]
            )
            if committed_executor is not None:
                result = committed_executor.committed_result
                assert result is not None
                assert isinstance(result.structured_output, ExecutorEpisodeOutput)
                if (
                    committed_executor.request.input_graph_revision
                    < allocation_record.output_graph_revision
                    or len(result.structured_output.nodes)
                    > allocation_record.allocations[node.node_id]
                ):
                    raise ValueError("committed Executor exceeded its controller allocation")
            if (
                state.node_revisions.get(node.node_id, -1)
                <= allocation_record.node_revisions_before[node.node_id]
                or node.ucb_score != allocation_record.ucb_scores[node.node_id]
                or node.local_embedding
                != allocation_record.local_embeddings[node.node_id]
                or node.density != allocation_record.densities[node.node_id]
                or node.uncertainty
                != allocation_record.uncertainties[node.node_id]
                or (
                    node.status != "merged"
                    and node.expansion_budget != expected_remaining
                )
                or (node.status == "merged" and node.expansion_budget != 0)
            ):
                raise ValueError("persisted node allocation/child accounting is inconsistent")
        if node.score is None:
            if judge_fields_present or any(value is not None for value in geometry) or node.expansion_budget:
                raise ValueError("unscored node contains persisted controller-owned state")
            continue
        owner = judge_observations.get(node.node_id)
        if owner is None:
            raise ValueError("scored node lacks an authoritative committed Judge observation")
        episode, attempt, observation = owner
        expected_provenance = {
            "run_id": state.run_id,
            "episode_id": episode.episode_id,
            "attempt_id": attempt.attempt_id,
            "schema_version": attempt.request.output_schema_version,
            "output_hash": attempt.result_hash or "",
        }
        if (
            node.score != observation.score
            or node.judge_reasoning != observation.reasoning
            or node.judge_risks != observation.risks
            or node.judge_uncertainty_evidence != observation.uncertainty_evidence
            or node.judge_result_provenance != expected_provenance
            or state.node_revisions.get(node.node_id, -1)
            <= attempt.request.selected_node_revisions[node.node_id]
        ):
            raise ValueError("persisted Judge-owned node state disagrees with committed output")

        if any(value is not None for value in geometry) and not all(
            value is not None for value in geometry
        ):
            raise ValueError("persisted controller geometry/allocation state is partial")
        if any(value is not None for value in geometry) and state.controller_iteration == 0:
            raise ValueError("persisted controller geometry predates any controller iteration")
        if node.expansion_budget > 0 and (
            node.status != "frontier"
            or not all(value is not None for value in geometry)
            or state.controller_iteration == 0
        ):
            raise ValueError("persisted positive expansion budget lacks controller allocation facts")

    aliases = resolve_merge_aliases(
        state.merge_applications,
        committed_node_ids=node_ids,
    )
    records_by_id = {
        record.relation_record_id: record for record in state.relation_ledger
    }
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in state.relation_candidates
    }
    for episode in state.episodes:
        for attempt in episode.attempts:
            if not attempt.retry_exhaustion_released:
                continue
            payload = attempt.request.relation_payload
            assert payload is not None
            for pair in payload.candidate_pairs:
                candidate = candidates_by_id.get(pair.candidate_id)
                if candidate is not None and (
                    candidate.granted_episode_id == episode.episode_id
                    or candidate.granted_attempt_id == attempt.attempt_id
                ):
                    raise ValueError("retry exhaustion release retained a Relation grant")
    for candidate in state.relation_candidates:
        if candidate.candidate_id != expected_relation_candidate_id(candidate):
            raise ValueError("persisted Relation candidate ID disagrees with its identity")
        if candidate.left_node_id not in node_ids or candidate.right_node_id not in node_ids:
            raise ValueError("Relation candidate references a missing committed node")
        if candidate.status == "resolved":
            record = records_by_id.get(candidate.resolved_relation_record_id or "")
            if record is None or not relation_record_covers_candidate(record, candidate):
                raise ValueError("resolved Relation candidate has an inconsistent ledger link")

    records_by_attempt: dict[tuple[str, str], list[RelationRecord]] = {}
    for record in state.relation_ledger:
        committed_at = _parse_time(record.committed_at)
        if committed_at < created_at or committed_at > updated_at:
            raise ValueError("Relation record timestamp is outside the run lifetime")
        candidate = candidates_by_id.get(record.candidate_id)
        if (
            candidate is None
            or candidate.status != "resolved"
            or candidate.resolved_relation_record_id != record.relation_record_id
            or not relation_record_covers_candidate(record, candidate)
        ):
            raise ValueError("Relation ledger record lacks its exact resolved candidate")
        owner = attempts_by_identity.get((record.episode_id, record.attempt_id))
        if owner is None:
            raise ValueError("Relation ledger record references a missing attempt")
        owner_episode, owner_attempt = owner
        request = owner_attempt.request
        outcome = owner_attempt.commit_outcome
        if (
            owner_episode.role != "relation"
            or owner_episode.committed_attempt_id != owner_attempt.attempt_id
            or owner_attempt.status != "committed"
            or owner_attempt.result_hash != record.output_hash
            or outcome is None
            or not outcome.accepted
            or outcome.episode_id != owner_episode.episode_id
            or outcome.graph_revision_before != request.input_graph_revision
            or request.relation_payload is None
            or record.input_graph_revision != request.input_graph_revision
            or record.selected_node_revisions != request.selected_node_revisions
            or record.schema_version != request.output_schema_version
        ):
            raise ValueError("Relation ledger record disagrees with its committed attempt")
        pair = next(
            (
                item
                for item in request.relation_payload.candidate_pairs
                if item.candidate_id == record.candidate_id
            ),
            None,
        )
        if pair is None or (
            pair.left.node_id,
            pair.right.node_id,
            pair.left_node_revision,
            pair.right_node_revision,
            pair.scheduling_class,
            pair.candidate_reason,
            pair.priority,
            pair.material_to_synthesis,
        ) != (
            record.left_node_id,
            record.right_node_id,
            record.selected_node_revisions.get(record.left_node_id),
            record.selected_node_revisions.get(record.right_node_id),
            record.scheduling_class,
            candidate.candidate_reason,
            candidate.priority,
            record.material_to_synthesis,
        ):
            raise ValueError("Relation ledger record disagrees with its granted pair")
        expected_disclosure = bool(
            record.relation_type == "conflict"
            and (record.observation.disclosure_required or candidate.material_to_synthesis)
        )
        if record.disclosure_required != expected_disclosure:
            raise ValueError("Relation ledger disclosure disagrees with committed candidate facts")
        expected_record_id = stable_relation_id(
            "relrec",
            record.candidate_id,
            record.episode_id,
            record.attempt_id,
            record.output_hash,
        )
        if record.relation_record_id != expected_record_id:
            raise ValueError("Relation ledger record has a forged stable identity")
        records_by_attempt.setdefault((record.episode_id, record.attempt_id), []).append(record)

    for episode in state.episodes:
        if episode.role != "relation" or episode.committed_attempt_id is None:
            continue
        attempt = next(
            item for item in episode.attempts if item.attempt_id == episode.committed_attempt_id
        )
        payload = attempt.request.relation_payload
        if payload is None:
            raise ValueError("committed Relation attempt lacks a request payload")
        expected_candidate_ids = {item.candidate_id for item in payload.candidate_pairs}
        actual_records = records_by_attempt.get((episode.episode_id, attempt.attempt_id), [])
        if {record.candidate_id for record in actual_records} != expected_candidate_ids:
            raise ValueError("committed Relation attempt has an incomplete or extra ledger record set")
        outcome = attempt.commit_outcome
        assert outcome is not None
        actual_candidate_ids = [record.candidate_id for record in actual_records]
        reconstructed_hash = compute_output_hash(
            RelationEpisodeOutput(
                observations=[record.observation for record in actual_records]
            ),
            attempt.request.output_schema_version,
        )
        if (
            attempt.result_hash != reconstructed_hash
            or any(record.output_hash != reconstructed_hash for record in actual_records)
            or outcome.accepted_node_ids != actual_candidate_ids
            or outcome.accepted_node_count != len(actual_candidate_ids)
        ):
            raise ValueError("committed Relation attempt output hash or accepted set is inconsistent")
        expected_after = attempt.request.input_graph_revision + 1 + int(
            any(record.relation_type == "equivalent" for record in actual_records)
        )
        if outcome.graph_revision_after != expected_after:
            raise ValueError("committed Relation attempt has an inconsistent graph revision outcome")

    for application in state.merge_applications:
        applied_at = _parse_time(application.applied_at)
        if applied_at < created_at or applied_at > updated_at:
            raise ValueError("merge application timestamp is outside the run lifetime")
        record = records_by_id.get(application.relation_record_id)
        if record is None:
            raise ValueError("merge application lacks an equivalent Relation record")
        validate_merge_application_relation_provenance(application, record)
        expected_merge_id = stable_relation_id(
            "merge",
            record.relation_record_id,
            application.canonical_node_id,
            *sorted(application.absorbed_node_ids),
        )
        if (
            application.merge_application_id != expected_merge_id
            or application.source_node_ids != sorted(application.source_node_ids)
            or application.absorbed_node_ids != sorted(application.absorbed_node_ids)
            or application.applied_at != record.committed_at
            or application.applied_graph_revision != record.input_graph_revision + 2
            or application.applied_graph_revision > state.graph_revision
        ):
            raise ValueError("merge application has inconsistent stable commit provenance")
        for source_node_id, source_revision in application.source_node_revisions.items():
            if state.node_revisions.get(source_node_id, -1) <= source_revision:
                raise ValueError("merge application was not reflected in source node revisions")
    equivalent_record_ids = {
        record.relation_record_id
        for record in state.relation_ledger
        if record.relation_type == "equivalent"
    }
    applied_record_ids = {
        application.relation_record_id for application in state.merge_applications
    }
    if applied_record_ids != equivalent_record_ids:
        raise ValueError("equivalent Relation ledger and merge applications are not one-to-one")

    transition_events: list[
        tuple[int, int, str, ControllerIterationRecord | tuple[EpisodeLifecycleRecord, EpisodeAttemptRecord]]
    ] = []
    for record in state.controller_iteration_records:
        transition_events.append(
            (
                record.input_graph_revision,
                record.output_graph_revision,
                "controller",
                record,
            )
        )
    for episode in state.episodes:
        if episode.committed_attempt_id is None:
            continue
        attempt = next(
            item for item in episode.attempts if item.attempt_id == episode.committed_attempt_id
        )
        outcome = attempt.commit_outcome
        assert outcome is not None
        transition_events.append(
            (
                outcome.graph_revision_before,
                outcome.graph_revision_after,
                episode.role,
                (episode, attempt),
            )
        )
    transition_events.sort(key=lambda item: (item[0], item[1], item[2]))

    replay_graph_revision = 0
    replay_node_revisions = {node.node_id: 0 for node in state.initial_nodes}
    replay_statuses = {node.node_id: "frontier" for node in state.initial_nodes}
    applications_by_record = {
        application.relation_record_id: application
        for application in state.merge_applications
    }
    for revision_before, revision_after, transition_kind, payload in transition_events:
        if revision_before != replay_graph_revision or revision_after <= revision_before:
            raise ValueError("persisted graph revision transitions are not one continuous chain")
        if transition_kind == "controller":
            assert isinstance(payload, ControllerIterationRecord)
            frontier_ids = {
                node_id
                for node_id, status in replay_statuses.items()
                if status == "frontier"
            }
            if (
                set(payload.frontier_node_ids) != frontier_ids
                or payload.node_revisions_before
                != {
                    node_id: replay_node_revisions[node_id]
                    for node_id in payload.frontier_node_ids
                }
            ):
                raise ValueError("controller transition disagrees with replayed node revisions")
            for node_id in payload.frontier_node_ids:
                replay_node_revisions[node_id] += 1
        else:
            assert isinstance(payload, tuple)
            episode, attempt = payload
            request = attempt.request
            if request.selected_node_revisions != {
                node_id: replay_node_revisions.get(node_id, -1)
                for node_id in request.selected_node_revisions
            }:
                raise ValueError("episode grant disagrees with replayed node revisions")
            result = attempt.committed_result
            assert result is not None
            if transition_kind == "judge":
                for node_id in request.selected_node_revisions:
                    if replay_statuses.get(node_id) != "frontier":
                        raise ValueError("Judge transition targeted a non-frontier node")
                    replay_node_revisions[node_id] += 1
            elif transition_kind == "executor":
                parent_id = request.parent_node_id
                if parent_id is None or replay_statuses.get(parent_id) != "frontier":
                    raise ValueError("Executor transition targeted a non-frontier parent")
                replay_node_revisions[parent_id] += 1
                replay_statuses[parent_id] = "closed"
                assert isinstance(result.structured_output, ExecutorEpisodeOutput)
                for candidate in result.structured_output.nodes:
                    if candidate.node_id in replay_node_revisions:
                        raise ValueError("Executor transition replay found a duplicate child ID")
                    replay_node_revisions[candidate.node_id] = 0
                    replay_statuses[candidate.node_id] = "frontier"
            elif transition_kind == "relation":
                for node_id in request.selected_node_revisions:
                    if replay_statuses.get(node_id) == "merged":
                        raise ValueError("Relation transition targeted an already merged node")
                for record in records_by_attempt.get(
                    (episode.episode_id, attempt.attempt_id),
                    [],
                ):
                    if record.relation_type != "equivalent":
                        continue
                    application = applications_by_record.get(record.relation_record_id)
                    if application is None:
                        raise ValueError("equivalent Relation replay lacks its merge application")
                    if application.source_node_revisions != {
                        node_id: replay_node_revisions.get(node_id, -1)
                        for node_id in application.source_node_ids
                    }:
                        raise ValueError("merge application disagrees with replayed source revisions")
                    active_sources = [
                        node_id
                        for node_id in application.source_node_ids
                        if replay_statuses.get(node_id) != "merged"
                    ]
                    if (
                        not active_sources
                        or application.canonical_node_id not in active_sources
                    ):
                        raise ValueError("merge replay canonical was not active")
                    if len(active_sources) >= 2:
                        for node_id in active_sources:
                            replay_node_revisions[node_id] += 1
                            if node_id != application.canonical_node_id:
                                replay_statuses[node_id] = "merged"
            else:  # pragma: no cover - App-native commit roles are exhaustive above
                raise ValueError(f"unsupported committed transition role: {transition_kind}")
        replay_graph_revision = revision_after

    if replay_graph_revision != state.graph_revision:
        raise ValueError("persisted graph revision is not backed by committed transitions")
    if replay_node_revisions != state.node_revisions:
        raise ValueError("persisted node revisions disagree with committed transition replay")
    if replay_statuses != {node.node_id: node.status for node in state.nodes}:
        raise ValueError("persisted node statuses disagree with committed transition replay")

    nodes_by_id = {node.node_id: node for node in state.nodes}
    for node in state.nodes:
        if node.local_embedding is not None and (
            len(node.local_embedding) != state.spec.embedding_dimension
            or any(not math.isfinite(value) for value in node.local_embedding)
        ):
            raise ValueError(
                f"node {node.node_id!r} has an invalid persisted embedding dimension/value"
            )
    for absorbed_id, canonical_id in aliases.items():
        if nodes_by_id[absorbed_id].status != "merged":
            raise ValueError("absorbed merge alias is not marked merged")
        if nodes_by_id[canonical_id].status == "merged":
            raise ValueError("merge alias resolves to a merged canonical node")
    if state.synthesis_request is not None and state.synthesis_request.scope == "node_ids":
        for requested_id in state.synthesis_request.node_ids:
            if requested_id not in nodes_by_id:
                raise ValueError("persisted synthesis request references a missing node")
            canonical_id = aliases.get(requested_id, requested_id)
            target = nodes_by_id[canonical_id]
            if target.status not in {"frontier", "closed"} or target.node_type == "synthesis":
                raise ValueError("persisted synthesis request references an ineligible node")
    if state.synthesis_request is not None:
        authorize_synthesis_control(state.spec, state.synthesis_request)
        canonical_request = _canonicalize_synthesis_request(
            state,
            state.synthesis_request,
        )
        if (
            canonical_request.model_dump(mode="json")
            != state.synthesis_request.model_dump(mode="json")
        ):
            raise ValueError("persisted synthesis request is not canonical")

    active_ids = {
        node.node_id for node in state.nodes if node.status != "merged"
    }
    projected_parents: dict[str, list[str]] = {}
    for node in state.nodes:
        if node.status == "merged":
            continue
        if len(node.parent_ids) != len(set(node.parent_ids)):
            raise ValueError(f"node {node.node_id!r} contains duplicate parent IDs")
        missing = sorted(set(node.parent_ids) - node_ids)
        if missing:
            raise ValueError(f"node {node.node_id!r} references missing parent {missing[0]!r}")
        parents = [aliases.get(parent_id, parent_id) for parent_id in node.parent_ids]
        if node.node_id in parents:
            raise ValueError(f"node {node.node_id!r} has alias-projected self ancestry")
        projected_parents[node.node_id] = [parent for parent in parents if parent in active_ids]
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("persisted node ancestry contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for parent_id in projected_parents.get(node_id, []):
            visit(parent_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in projected_parents:
        visit(node_id)
    active_identity = (state.active_episode_id, state.active_attempt_id)
    if (active_identity[0] is None) != (active_identity[1] is None):
        raise ValueError("persisted App state has a partial active-attempt identity")
    active_record: tuple[EpisodeLifecycleRecord, EpisodeAttemptRecord] | None = None
    if active_identity[0] is not None:
        active_record = attempts_by_identity.get(
            (active_identity[0], active_identity[1])  # type: ignore[arg-type]
        )
        if active_record is None:
            raise ValueError("persisted App state references a missing active attempt")
        if active_record[1].status not in active_statuses:
            raise ValueError("persisted App state points at a non-active attempt lifecycle")
        request = active_record[1].request
        if request.input_graph_revision != state.graph_revision or any(
            state.node_revisions.get(node_id) != revision
            for node_id, revision in request.selected_node_revisions.items()
        ):
            raise ValueError("active episode grant is stale relative to committed graph state")
        judge_input_fields = {
            "node_id",
            "node_type",
            "claim",
            "rationale",
            "assumptions",
            "evidence",
            "risks",
            "confidence",
        }
        executor_parent_fields = judge_input_fields | {"parent_ids"}
        if request.role == "executor":
            parent = nodes_by_id.get(request.parent_node_id or "")
            payload = request.executor_payload
            if (
                parent is None
                or payload is None
                or parent.status != "frontier"
                or parent.expansion_budget <= 0
                or request.selected_node_revisions
                != {parent.node_id: state.node_revisions[parent.node_id]}
                or request.parent_node_revision != state.node_revisions[parent.node_id]
                or request.max_returned_children
                != min(
                    parent.expansion_budget,
                    state.spec.budget.max_children_per_iteration,
                    remaining_search_node_slots(
                        state.nodes,
                        state.spec.budget.max_committed_search_nodes,
                    ),
                )
                or payload.parent.model_dump(mode="json")
                != parent.model_dump(mode="json", include=executor_parent_fields)
                or payload.iteration != max(1, state.controller_iteration)
                or payload.constraints != state.spec.constraints
                or request.objective != f"{state.spec.goal}: expand {parent.claim}"
            ):
                raise ValueError("active Executor grant disagrees with controller allocation")
        elif request.role == "judge":
            expected_nodes = _select_unjudged_frontier(state)
            payload = request.judge_payload
            if (
                payload is None
                or list(request.selected_node_revisions)
                != [node.node_id for node in expected_nodes]
                or payload.problem != state.spec.problem
                or payload.goal != state.spec.goal
                or payload.constraints != state.spec.constraints
                or [item.model_dump(mode="json") for item in payload.selected_frontier_nodes]
                != [
                    node.model_dump(mode="json", include=judge_input_fields)
                    for node in expected_nodes
                ]
            ):
                raise ValueError("active Judge grant disagrees with current unjudged frontier")
        elif request.role == "relation":
            payload = request.relation_payload
            if (
                payload is None
                or len(payload.candidate_pairs)
                > state.spec.budget.max_relation_pairs_per_episode
            ):
                raise ValueError("active Relation grant exceeds the run pair budget")
    if active_lifecycle_records:
        lifecycle_identity = (
            active_lifecycle_records[0][0].episode_id,
            active_lifecycle_records[0][1].attempt_id,
        )
        if lifecycle_identity != active_identity:
            raise ValueError("persisted active lifecycle disagrees with App active identity")
    elif active_identity[0] is not None:
        raise ValueError("persisted App active identity has no active lifecycle")
    if active_record is not None and state.controller_action != "episode_required":
        raise ValueError("persisted active attempt requires controller_action='episode_required'")
    if active_record is None and state.controller_action == "episode_required":
        raise ValueError("persisted episode_required action has no active attempt")

    if state.synthesis_readiness is not None:
        readiness_evaluated_at = _parse_time(state.synthesis_readiness.evaluated_at)
        if readiness_evaluated_at < created_at or readiness_evaluated_at > updated_at:
            raise ValueError("synthesis readiness timestamp is outside the run lifetime")

    for candidate in state.relation_candidates:
        if candidate.status != "granted":
            continue
        owner_identity = (
            candidate.granted_episode_id or "",
            candidate.granted_attempt_id or "",
        )
        owner = attempts_by_identity.get(owner_identity)
        if owner is None:
            raise ValueError("granted Relation candidate references a missing attempt")
        owner_episode, owner_attempt = owner
        payload = owner_attempt.request.relation_payload
        if owner_attempt.request.role != "relation" or payload is None:
            raise ValueError("granted Relation candidate is owned by a non-Relation attempt")
        pair = next(
            (item for item in payload.candidate_pairs if item.candidate_id == candidate.candidate_id),
            None,
        )
        if pair is None or (
            pair.left.node_id,
            pair.right.node_id,
            pair.left_node_revision,
            pair.right_node_revision,
            pair.candidate_reason,
            pair.scheduling_class,
            pair.priority,
            pair.material_to_synthesis,
        ) != (
            candidate.left_node_id,
            candidate.right_node_id,
            candidate.left_node_revision,
            candidate.right_node_revision,
            candidate.candidate_reason,
            candidate.scheduling_class,
            candidate.priority,
            candidate.material_to_synthesis,
        ):
            raise ValueError("granted Relation candidate disagrees with its attempt request")
        if owner_identity != active_identity and not (
            owner_attempt.status == "rejected"
            and owner_episode.committed_attempt_id is None
            and owner_attempt is owner_episode.attempts[-1]
            and state.controller_action == "await_operator_decision"
        ):
            raise ValueError("granted Relation candidate lacks a retryable owner lifecycle")
    if active_record is not None and active_record[1].request.role == "relation":
        payload = active_record[1].request.relation_payload
        if payload is None:
            raise ValueError("active Relation attempt is missing its payload")
        for pair in payload.candidate_pairs:
            candidate = candidates_by_id.get(pair.candidate_id)
            if candidate is None or (
                candidate.status != "granted"
                or candidate.granted_episode_id != active_record[0].episode_id
                or candidate.granted_attempt_id != active_record[1].attempt_id
            ):
                raise ValueError("active Relation request lacks its committed candidate grant")

    terminal = state.controller_action in {"ready_for_synthesis", "run_complete"}
    selection_present = state.provisional_synthesis_selection is not None
    readiness_present = state.synthesis_readiness is not None
    if selection_present != readiness_present:
        raise ValueError("persisted synthesis selection/readiness must be present together")
    if state.relation_readiness_status == "evaluated":
        if not selection_present or not (
            terminal or state.pending_terminal_gate_evaluated
        ):
            raise ValueError("evaluated synthesis readiness lacks an active gate or terminal state")
    elif state.relation_readiness_status == "not_evaluated":
        if selection_present:
            raise ValueError("unevaluated synthesis state contains controller-owned readiness")
    else:
        raise ValueError("legacy unchecked synthesis readiness cannot authorize App state")
    if terminal and active_identity[0] is not None:
        raise ValueError("terminal App state cannot retain an active attempt")
    pending_values = (
        state.pending_terminal_action,
        state.pending_terminal_reason,
        state.pending_terminal_source,
    )
    if any(value is None for value in pending_values) and any(
        value is not None for value in pending_values
    ):
        raise ValueError("persisted pending terminal action/reason/source must be present together")
    if terminal and state.pending_terminal_action is not None:
        raise ValueError("terminal App state cannot retain pending terminal intent")
    if state.pending_terminal_action is None and state.pending_terminal_gate_evaluated:
        raise ValueError("Relation gate marker lacks pending terminal intent")
    if state.pending_terminal_action is not None:
        if not state.pending_terminal_reason or not state.pending_terminal_reason.strip():
            raise ValueError("persisted pending terminal reason must be non-empty")
        assert state.pending_terminal_source is not None
        _validate_terminal_intent_provenance(
            state,
            action=state.pending_terminal_action,
            reason=state.pending_terminal_reason,
            source=state.pending_terminal_source,
        )
        if (
            state.pending_terminal_gate_evaluated
            and active_record is not None
            and active_record[1].request.role != "relation"
        ):
            raise ValueError("pending terminal intent may only coexist with a Relation attempt")
        if (
            state.synthesis_request is not None
            and state.pending_terminal_action != "ready_for_synthesis"
        ):
            raise ValueError("authorized synthesis request disagrees with pending terminal action")
        if state.pending_terminal_gate_evaluated:
            if state.relation_readiness_status != "evaluated" or (
                state.provisional_synthesis_selection is None
                or state.synthesis_readiness is None
            ):
                raise ValueError("pending terminal intent lacks a completed Relation gate evaluation")
        elif state.pending_terminal_source not in {
            "controller_stop",
            "authorized_synthesis",
            "max_iterations",
            "max_search_nodes",
            "continuation_gate",
        }:
            raise ValueError("ungated pending terminal intent lacks durable stop provenance")
        if not _has_synthesis_safe_checkpoint(state):
            raise ValueError("pending terminal intent predates a synthesis-safe checkpoint")
        if state.pending_terminal_source == "authorized_synthesis":
            if (
                state.synthesis_request is None
                or state.pending_terminal_action != "ready_for_synthesis"
            ):
                raise ValueError("pending authorized synthesis lacks its controller command")
        elif state.pending_terminal_source == "max_iterations":
            if state.controller_iteration < state.spec.budget.max_iterations:
                raise ValueError("pending max-iteration terminal intent is premature")

    blocking_operator_attempts: list[tuple[EpisodeLifecycleRecord, EpisodeAttemptRecord]] = []
    for episode in state.episodes:
        if episode.committed_attempt_id is not None:
            continue
        latest = episode.attempts[-1]
        if latest.status not in {"rejected", "failed", "cancelled", "expired"}:
            continue
        if not latest.retry_exhaustion_released:
            blocking_operator_attempts.append((episode, latest))
    relation_gate_blocked = bool(
        state.pending_terminal_gate_evaluated
        and state.synthesis_readiness is not None
        and not state.synthesis_readiness.ready
    )
    if state.controller_action == "await_operator_decision":
        if not blocking_operator_attempts and not relation_gate_blocked:
            raise ValueError("await_operator_decision lacks a durable blocking fact")
    elif blocking_operator_attempts:
        raise ValueError("controller action bypasses an unresolved operator decision")
    if terminal and state.terminal_record is None:
        raise ValueError("terminal App state lacks its immutable terminal record")
    if state.terminal_record is not None:
        terminal_committed_at = _parse_time(state.terminal_record.committed_at)
        if terminal_committed_at < created_at or terminal_committed_at > updated_at:
            raise ValueError("terminal record timestamp is outside the run lifetime")
        _validate_terminal_intent_provenance(
            state,
            action=state.terminal_record.action,
            reason=state.terminal_record.reason,
            source=state.terminal_record.source,
        )
        if not terminal or state.terminal_record.action != state.controller_action:
            raise ValueError("persisted terminal record disagrees with controller action")
        if not state.terminal_record.reason.strip():
            raise ValueError("persisted terminal record has an empty reason")
        if (
            state.terminal_record.graph_revision != state.graph_revision
            or state.terminal_record.controller_iteration != state.controller_iteration
        ):
            raise ValueError("persisted terminal record disagrees with current controller state")
    if terminal:
        selection = state.provisional_synthesis_selection
        readiness = state.synthesis_readiness
        if (
            selection is None
            or readiness is None
            or state.relation_readiness_status != "evaluated"
        ):
            raise ValueError("terminal App state lacks evaluated synthesis readiness")
        if any(
            node.expansion_budget > 0
            or (node.status == "frontier" and node.score is None)
            for node in state.nodes
        ):
            raise ValueError("terminal App state retains authorized or unjudged work")
        if (
            not readiness.ready
            or readiness.graph_revision != state.graph_revision
            or selection.selection_revision != state.graph_revision
            or readiness.provisional_selected_node_ids != selection.selected_node_ids
        ):
            raise ValueError("terminal synthesis readiness is stale or internally inconsistent")
        eligible_ids = {
            node.node_id
            for node in state.nodes
            if node.status in {"frontier", "closed"} and node.node_type != "synthesis"
        }
        if not set(selection.selected_node_ids).issubset(eligible_ids):
            raise ValueError("terminal synthesis selection contains an ineligible node")
        synthesis_request = state.synthesis_request
        if synthesis_request is not None:
            synthesis_request = _canonicalize_synthesis_request(state, synthesis_request)
        expected_selection = select_provisional_synthesis_nodes(
            state.nodes,
            graph_revision=state.graph_revision,
            synthesis_request=synthesis_request,
        )
        if selection.model_dump(mode="json") != expected_selection.model_dump(mode="json"):
            raise ValueError("terminal synthesis selection disagrees with committed graph ranking")
        expected_blocking = generate_blocking_relation_obligations(
            state.nodes,
            node_revisions=state.node_revisions,
            graph_revision=state.graph_revision,
            provisional_synthesis_node_ids=expected_selection.selected_node_ids,
        )
        enrichment_committed = _relation_enrichment_pairs_committed(state)
        expected_readiness = evaluate_synthesis_readiness(
            graph_revision=state.graph_revision,
            provisional_selected_node_ids=expected_selection.selected_node_ids,
            candidates=state.relation_candidates,
            relation_ledger=state.relation_ledger,
            merge_applications=state.merge_applications,
            evaluated_at=readiness.evaluated_at,
            blocking_inventory_candidate_ids=[
                candidate.candidate_id for candidate in expected_blocking
            ],
            blocking_inventory_complete=True,
            enrichment_budget_limit=state.spec.budget.max_relation_enrichment_pairs,
            enrichment_pairs_committed=enrichment_committed,
            eligible_enrichment_candidate_ids=[
                candidate.candidate_id
                for candidate in state.relation_candidates
                if candidate.scheduling_class == "enrichment"
                and candidate.priority == "high"
                and candidate.status == "pending"
            ],
        )
        if readiness.model_dump(mode="json") != expected_readiness.model_dump(mode="json"):
            raise ValueError("terminal synthesis readiness cannot be reproduced from durable facts")
        if not expected_readiness.ready:
            raise ValueError("terminal App state has unresolved Relation obligations")


def load_app_run(run_dir: str | Path) -> AppRunState:
    raw = json.loads(_state_path(run_dir).read_text(encoding="utf-8"))
    raw_spec = raw.get("spec") if isinstance(raw, dict) else None
    raw_budget = raw_spec.get("budget") if isinstance(raw_spec, dict) else None
    if isinstance(raw_budget, dict) and "max_committed_search_nodes" not in raw_budget:
        # Validate the immutable pre-upgrade spec before inserting compatibility
        # fields. Existing runs keep the old entropy stop and receive a node cap
        # loose enough not to reduce their original expansion envelope.
        old_hash = hashlib.sha256(canonical_json_bytes(raw_spec)).hexdigest()
        if raw.get("spec_hash") != old_hash:
            raise ValueError("persisted legacy RunSpec disagrees with its immutable run hash")
        initial_count = sum(
            item.get("node_type") != "synthesis"
            for item in raw.get("initial_nodes", [])
            if isinstance(item, dict)
        )
        current_count = sum(
            item.get("node_type") != "synthesis"
            for item in raw.get("nodes", [])
            if isinstance(item, dict)
        )
        legacy_envelope = initial_count + (
            int(raw_budget.get("max_iterations", 2))
            * int(raw_budget.get("max_children_per_iteration", 5))
        )
        raw_budget["max_committed_search_nodes"] = min(
            100,
            max(1, current_count, legacy_envelope),
        )
        raw_budget["entropy_plateau_confirmations"] = 1
        raw_budget["continuation_policy"] = "legacy_entropy_v1"
        migrated_spec = DTERunSpec.model_validate(raw_spec)
        raw["spec_hash"] = _run_spec_hash(migrated_spec)
    state = AppRunState.model_validate(raw)
    _validate_loaded_state(state)
    _flush_pending_events(run_dir, state)
    try:
        _write_relation_artifacts(run_dir, state)
        _write_epistemic_artifact(run_dir, state)
    except Exception:
        pass
    _repair_attempt_artifacts(run_dir, state)
    return state


def _validate_initial_nodes(initial_nodes: list[SearchNode]) -> list[SearchNode]:
    """Re-parse and enforce the producer-safe new-run node boundary."""

    nodes = [
        SearchNode.model_validate(node.model_dump(mode="json"))
        for node in initial_nodes
    ]
    if not nodes:
        raise ValueError("App-native create-run requires at least one initial node")
    ids = [node.node_id for node in nodes]
    if any(not node_id.strip() for node_id in ids):
        raise ValueError("initial node IDs must be non-empty")
    if len(ids) != len(set(ids)):
        raise ValueError("initial nodes contain duplicate node IDs")
    known_ids = set(ids)
    for node in nodes:
        polluted: list[str] = []
        if node.status != "frontier":
            polluted.append("status")
        if node.node_type == "synthesis":
            polluted.append("node_type")
        for field_name in (
            "local_embedding",
            "judge_reasoning",
            "judge_result_provenance",
            "score",
            "density",
            "uncertainty",
            "ucb_score",
        ):
            if getattr(node, field_name) is not None:
                polluted.append(field_name)
        if node.judge_risks:
            polluted.append("judge_risks")
        if node.judge_uncertainty_evidence:
            polluted.append("judge_uncertainty_evidence")
        if node.expansion_budget != 0:
            polluted.append("expansion_budget")
        if polluted:
            raise ValueError(
                f"initial node {node.node_id!r} sets controller-owned fields: "
                + ", ".join(sorted(set(polluted)))
            )
        if len(node.parent_ids) != len(set(node.parent_ids)):
            raise ValueError(f"initial node {node.node_id!r} contains duplicate parent IDs")
        invalid_parents = sorted(set(node.parent_ids) - known_ids)
        if invalid_parents:
            raise ValueError(
                f"initial node {node.node_id!r} references unknown parent: {invalid_parents[0]}"
            )
        if node.node_id in node.parent_ids:
            raise ValueError(f"initial node {node.node_id!r} cannot parent itself")

    parents_by_id = {node.node_id: list(node.parent_ids) for node in nodes}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("initial node ancestry contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for parent_id in parents_by_id[node_id]:
            visit(parent_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in ids:
        visit(node_id)
    return nodes


def create_app_run(
    run_dir: str | Path,
    spec: DTERunSpec,
    initial_nodes: list[SearchNode],
    *,
    run_id: str | None = None,
) -> AppRunState:
    """Create committed state for an App-driven run without starting a runtime."""

    path = _state_path(run_dir)
    if path.exists():
        raise FileExistsError(f"App run already exists: {path}")
    validated_spec = DTERunSpec.model_validate(spec.model_dump(mode="json"))
    validated_initial_nodes = _validate_initial_nodes(initial_nodes)
    initial_search_node_count = count_committed_search_nodes(validated_initial_nodes)
    if initial_search_node_count > validated_spec.budget.max_committed_search_nodes:
        raise ValueError(
            "initial committed search nodes exceed max_committed_search_nodes"
        )
    graph = EpisodeGraph(nodes=validated_initial_nodes)
    created = _iso()
    state = AppRunState(
        run_id=run_id or str(uuid.uuid4()),
        spec=validated_spec,
        spec_hash=_run_spec_hash(validated_spec),
        initial_nodes=[node.model_copy(deep=True) for node in validated_initial_nodes],
        initial_nodes_hash=_initial_nodes_hash(validated_initial_nodes),
        nodes=graph.nodes,
        graph_revision=graph.revision,
        node_revisions=graph.node_revisions,
        controller_action="continue_controller",
        created_at=created,
        updated_at=created,
    )
    _queue_event(
        state,
        "run_created",
        run_id=state.run_id,
        status="created",
        input_graph_revision=state.graph_revision,
        usage_source="unavailable",
    )
    _save_state(run_dir, state)
    return state


def _find_episode(state: AppRunState, episode_id: str) -> EpisodeLifecycleRecord:
    episode = next((item for item in state.episodes if item.episode_id == episode_id), None)
    if episode is None:
        raise KeyError(f"unknown episode_id: {episode_id}")
    return episode


def _find_attempt(episode: EpisodeLifecycleRecord, attempt_id: str) -> EpisodeAttemptRecord:
    attempt = next((item for item in episode.attempts if item.attempt_id == attempt_id), None)
    if attempt is None:
        raise KeyError(f"unknown attempt_id: {attempt_id}")
    return attempt


def _active_attempt(state: AppRunState) -> tuple[EpisodeLifecycleRecord, EpisodeAttemptRecord] | None:
    if state.active_episode_id is None or state.active_attempt_id is None:
        return None
    episode = _find_episode(state, state.active_episode_id)
    return episode, _find_attempt(episode, state.active_attempt_id)


def _attempt_expired(
    attempt: EpisodeAttemptRecord,
    *,
    at: datetime | None = None,
) -> bool:
    observed_at = at or _now()
    return (
        attempt.deadline_at is not None
        and observed_at >= _parse_time(attempt.deadline_at)
    )


def _release_relation_grants(state: AppRunState, request: EpisodeRequest) -> None:
    """Return an uncommitted Relation grant to schedulable durable state."""

    if request.role != "relation" or request.relation_payload is None:
        return
    candidate_ids = {pair.candidate_id for pair in request.relation_payload.candidate_pairs}
    for candidate in state.relation_candidates:
        if (
            candidate.candidate_id in candidate_ids
            and candidate.status == "granted"
            and candidate.granted_episode_id == request.episode_id
            and candidate.granted_attempt_id == request.attempt_id
        ):
            candidate.status = "pending"
            candidate.granted_episode_id = None
            candidate.granted_attempt_id = None
    state.pending_terminal_gate_evaluated = False
    state.provisional_synthesis_selection = None
    state.synthesis_readiness = None
    state.relation_readiness_status = "not_evaluated"


def _mark_expired(run_dir: str | Path, state: AppRunState, attempt: EpisodeAttemptRecord) -> None:
    attempt.status = "expired"
    attempt.failure_reason = "attempt deadline elapsed before submission"
    state.active_episode_id = None
    state.active_attempt_id = None
    state.controller_action = "await_operator_decision"
    _release_relation_grants(state, attempt.request)
    _queue_event(
        state,
        "episode_expired",
        run_id=state.run_id,
        episode_id=attempt.request.episode_id,
        attempt_id=attempt.attempt_id,
        role=attempt.request.role,
        status="expired",
        input_graph_revision=attempt.request.input_graph_revision,
        rejection_reason=attempt.failure_reason,
        usage_source="unavailable",
    )


def _select_executor_parent(state: AppRunState) -> SearchNode | None:
    eligible = [
        node
        for node in state.nodes
        if node.status == "frontier" and node.expansion_budget > 0
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda node: (
            -(node.ucb_score if node.ucb_score is not None else float("-inf")),
            node.node_id,
        ),
    )[0]


def _select_unjudged_frontier(state: AppRunState) -> list[SearchNode]:
    """Select one deterministic bounded Judge batch.

    A positive expansion budget is treated as an already committed controller
    grant for backward compatibility with the Executor vertical slice.
    """

    candidates = sorted(
        (
            node
            for node in state.nodes
            if node.status == "frontier" and node.score is None and node.expansion_budget == 0
        ),
        key=lambda node: node.node_id,
    )
    return candidates[: state.spec.budget.max_children_per_iteration]


def _ensure_nonterminal(state: AppRunState, operation: str) -> None:
    if state.controller_action in {"ready_for_synthesis", "run_complete"}:
        raise ValueError(
            f"cannot {operation}: controller terminal action {state.controller_action!r} is sticky"
        )


def _has_synthesis_safe_checkpoint(state: AppRunState) -> bool:
    if state.controller_iteration > 0:
        return True
    return any(
        episode.role == "executor" and episode.committed_attempt_id is not None
        for episode in state.episodes
    )


def _canonicalize_synthesis_request(
    state: AppRunState,
    request: SynthesisControlRequest,
) -> SynthesisControlRequest:
    request = SynthesisControlRequest.model_validate(request.model_dump(mode="json"))
    if request.scope == "all":
        return request
    nodes_by_id = {node.node_id: node for node in state.nodes}
    missing = [node_id for node_id in request.node_ids if node_id not in nodes_by_id]
    if missing:
        raise ValueError(
            "synthesis request references unknown node IDs: " + ", ".join(missing)
        )
    aliases = resolve_merge_aliases(
        state.merge_applications,
        committed_node_ids=set(nodes_by_id),
    )
    canonical_ids: list[str] = []
    for requested_id in request.node_ids:
        canonical_id = aliases.get(requested_id, requested_id)
        node = nodes_by_id[canonical_id]
        if node.status not in {"frontier", "closed"} or node.node_type == "synthesis":
            raise ValueError(
                f"synthesis request target is not an eligible committed node: {requested_id}"
            )
        if canonical_id not in canonical_ids:
            canonical_ids.append(canonical_id)
    if not canonical_ids:
        raise ValueError("targeted synthesis requires a non-empty effective node selection")
    return request.model_copy(update={"node_ids": canonical_ids})


def _progress_controller(
    run_dir: str | Path,
    state: AppRunState,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> tuple[ControllerAction, str]:
    """Run one complete deterministic geometry/entropy/UCB/allocation transition."""

    terminal_action: ControllerAction = (
        "ready_for_synthesis" if state.spec.require_final_synthesis else "run_complete"
    )

    frontier = [node for node in state.nodes if node.status == "frontier"]
    if not frontier:
        return terminal_action, "no expandable frontier remains"
    if any(node.score is None for node in frontier):
        raise RuntimeError("controller progression requires every frontier node to be judged")
    if (
        state.spec.budget.continuation_policy == "bounded_node_yield_v1"
        and count_committed_search_nodes(state.nodes)
        >= state.spec.budget.max_committed_search_nodes
    ):
        return terminal_action, "maximum committed search nodes reached"
    if state.controller_iteration >= state.spec.budget.max_iterations:
        return terminal_action, "maximum controller iterations reached"

    next_nodes = [node.model_copy(deep=True) for node in state.nodes]
    provider = embedding_provider or get_embedding_provider(
        state.spec.embedding_provider,
        dim=state.spec.embedding_dimension,
    )
    next_frontier, kde_state = estimate_frontier_kde_state(
        next_nodes,
        cache=FileDTECache(_app_cache_path(run_dir)),
        provider=provider,
        expected_dimension=state.spec.embedding_dimension,
    )
    for node, log_density, uncertainty in zip(
        next_frontier,
        kde_state.log_density,
        kde_state.uncertainty,
    ):
        node.density = math.exp(log_density)
        node.uncertainty = uncertainty

    iteration = state.controller_iteration + 1
    previous_plateau_count = (
        state.controller_iteration_records[-1].consecutive_plateau_count
        if state.controller_iteration_records
        else 0
    )
    entropy_state = evaluate_entropy_state(
        spatial_entropy=kde_state.spatial_entropy,
        previous_entropy=state.previous_spatial_entropy,
        iteration=iteration,
        min_iterations=state.spec.budget.min_iterations_before_synthesis,
        entropy_change_threshold=state.spec.budget.entropy_change_threshold,
        previous_plateau_count=previous_plateau_count,
        plateau_confirmations=state.spec.budget.entropy_plateau_confirmations,
        t_max=state.spec.budget.t_max,
    )
    remaining_slots = remaining_search_node_slots(
        next_nodes,
        state.spec.budget.max_committed_search_nodes,
    )
    effective_child_cap = min(
        state.spec.budget.max_children_per_iteration,
        remaining_slots,
    )
    allocations = allocate_frontier(
        next_nodes,
        allocation_mass_per_iteration=state.spec.budget.allocation_mass_per_iteration,
        max_children_per_iteration=effective_child_cap,
        tau=max(entropy_state.normalized_temperature, 0.05),
        c_explore=1.0,
        temperature=max(entropy_state.effective_temperature, 0.05),
    )
    allocation_by_id = {allocation.node_id: allocation for allocation in allocations}
    next_revisions = dict(state.node_revisions)
    revisions_before = {
        node.node_id: state.node_revisions[node.node_id] for node in next_frontier
    }
    for node in next_frontier:
        allocation = allocation_by_id[node.node_id]
        node.ucb_score = allocation.ucb_score
        node.expansion_budget = allocation.expansion_budget
        next_revisions[node.node_id] += 1

    input_graph_revision = state.graph_revision
    state.controller_iteration_records.append(
        ControllerIterationRecord(
            iteration=iteration,
            input_graph_revision=input_graph_revision,
            output_graph_revision=input_graph_revision + 1,
            frontier_node_ids=[node.node_id for node in next_frontier],
            node_revisions_before=revisions_before,
            allocations={
                node.node_id: allocation_by_id[node.node_id].expansion_budget
                for node in next_frontier
            },
            ucb_scores={
                node.node_id: allocation_by_id[node.node_id].ucb_score
                for node in next_frontier
            },
            local_embeddings={
                node.node_id: list(node.local_embedding or []) for node in next_frontier
            },
            densities={node.node_id: node.density or 0.0 for node in next_frontier},
            uncertainties={
                node.node_id: node.uncertainty or 0.0 for node in next_frontier
            },
            spatial_entropy=kde_state.spatial_entropy,
            entropy_delta=entropy_state.entropy_delta,
            normalized_temperature=entropy_state.normalized_temperature,
            plateau_signal=entropy_state.plateau_signal,
            consecutive_plateau_count=entropy_state.consecutive_plateau_count,
            effective_child_cap=effective_child_cap,
        )
    )
    state.nodes = next_nodes
    state.node_revisions = next_revisions
    state.graph_revision += 1
    state.controller_iteration = iteration
    state.previous_spatial_entropy = kde_state.spatial_entropy
    _queue_event(
        state,
        "allocation_recorded",
        run_id=state.run_id,
        status="committed",
        input_graph_revision=state.graph_revision - 1,
        selected_node_count=len(next_frontier),
        allocated_child_count=sum(allocation.expansion_budget for allocation in allocations),
        spatial_entropy=kde_state.spatial_entropy,
        usage_source="unavailable",
    )
    if state.spec.budget.continuation_policy == "bounded_node_yield_v1":
        synthesis_request = state.synthesis_request
        if synthesis_request is not None:
            synthesis_request = _canonicalize_synthesis_request(state, synthesis_request)
        provisional = select_provisional_synthesis_nodes(
            state.nodes,
            graph_revision=state.graph_revision,
            synthesis_request=synthesis_request,
        )
        previous_record = (
            state.controller_iteration_records[-2]
            if len(state.controller_iteration_records) > 1
            else None
        )
        previous_gate = (
            state.continuation_gate_records[-1]
            if state.continuation_gate_records
            else None
        )
        considered_epistemic_ids = {
            record_id
            for record in state.continuation_gate_records
            for record_id in record.considered_epistemic_record_ids
        }
        gate = evaluate_continuation_gate(
            iteration=iteration,
            graph_revision=state.graph_revision,
            nodes=state.nodes,
            max_committed_search_nodes=state.spec.budget.max_committed_search_nodes,
            entropy_delta=entropy_state.entropy_delta,
            consecutive_plateau_count=entropy_state.consecutive_plateau_count,
            plateau_confirmed=entropy_state.plateau_signal,
            allocations={
                node.node_id: allocation_by_id[node.node_id].expansion_budget
                for node in next_frontier
            },
            previous_frontier_node_ids=(
                set(previous_record.frontier_node_ids)
                if previous_record is not None
                else set()
            ),
            previous_positive_allocation_node_ids=(
                {
                    node_id
                    for node_id, budget in previous_record.allocations.items()
                    if budget > 0
                }
                if previous_record is not None
                else set()
            ),
            previous_provisional_synthesis_node_ids=(
                set(previous_gate.provisional_synthesis_node_ids)
                if previous_gate is not None
                else set()
            ),
            provisional_synthesis_node_ids=provisional.selected_node_ids,
            ledger=state.epistemic_ledger,
            previously_considered_epistemic_ids=considered_epistemic_ids,
        )
        state.continuation_gate_records.append(gate)
        _queue_event(
            state,
            "continuation_gate_evaluated",
            run_id=state.run_id,
            status="committed",
            input_graph_revision=input_graph_revision,
            graph_revision=state.graph_revision,
            controller_iteration=iteration,
            committed_search_node_count=gate.committed_search_node_count,
            remaining_search_node_slots=gate.remaining_search_node_slots,
            canonical_frontier_count=gate.canonical_frontier_count,
            entropy_delta=gate.entropy_delta,
            consecutive_plateau_count=gate.consecutive_plateau_count,
            material_yield_signals=gate.material_yield_signals,
            continuation_target_node_ids=gate.continuation_target_node_ids,
            decision=gate.decision,
            reason=gate.reason,
            usage_source="unavailable",
        )
        if gate.trigger_signals:
            _queue_event(
                state,
                (
                    "continuation_granted"
                    if gate.decision == "continue"
                    else "early_synthesis_selected"
                ),
                run_id=state.run_id,
                status=gate.decision,
                input_graph_revision=input_graph_revision,
                graph_revision=state.graph_revision,
                controller_iteration=iteration,
                trigger_signals=gate.trigger_signals,
                material_yield_signals=gate.material_yield_signals,
                continuation_target_node_ids=gate.continuation_target_node_ids,
                usage_source="unavailable",
            )
        if gate.decision == "prepare_synthesis":
            return terminal_action, gate.reason
    elif entropy_state.plateau_signal:
        return terminal_action, entropy_state.stop_reason or "entropy stopping policy reached"
    if any(allocation.expansion_budget > 0 for allocation in allocations):
        return "continue_controller", "deterministic controller progression assigned expansion budget"
    return terminal_action, "controller produced no positive expandable frontier allocation"


def _evaluate_relation_gate(
    run_dir: str | Path,
    state: AppRunState,
    *,
    entropy_plateau: bool,
) -> SynthesisReadinessRecord:
    if state.synthesis_request is not None:
        state.synthesis_request = _canonicalize_synthesis_request(
            state,
            state.synthesis_request,
        )
    selection = select_provisional_synthesis_nodes(
        state.nodes,
        graph_revision=state.graph_revision,
        synthesis_request=state.synthesis_request,
    )
    blocking_inventory = generate_blocking_relation_obligations(
        state.nodes,
        node_revisions=state.node_revisions,
        graph_revision=state.graph_revision,
        provisional_synthesis_node_ids=selection.selected_node_ids,
    )
    previous_ids = {candidate.candidate_id for candidate in state.relation_candidates}
    state.relation_candidates = refresh_relation_candidates(
        state.relation_candidates,
        blocking_inventory,
        nodes=state.nodes,
        node_revisions=state.node_revisions,
        relation_ledger=state.relation_ledger,
    )
    enrichment = generate_relation_enrichment_candidates(
        state.nodes,
        node_revisions=state.node_revisions,
        graph_revision=state.graph_revision,
        provisional_synthesis_node_ids=selection.selected_node_ids,
        existing=state.relation_candidates,
        relation_ledger=state.relation_ledger,
        entropy_plateau=entropy_plateau,
        max_candidates=max(16, state.spec.budget.max_relation_pairs_per_episode * 4),
    )
    state.relation_candidates = refresh_relation_candidates(
        state.relation_candidates,
        enrichment,
        nodes=state.nodes,
        node_revisions=state.node_revisions,
        relation_ledger=state.relation_ledger,
    )
    added_count = sum(candidate.candidate_id not in previous_ids for candidate in state.relation_candidates)
    enrichment_committed = _relation_enrichment_pairs_committed(state)
    eligible_enrichment_ids = [
        candidate.candidate_id
        for candidate in state.relation_candidates
        if candidate.scheduling_class == "enrichment"
        and candidate.priority == "high"
        and candidate.status == "pending"
    ]
    readiness = evaluate_synthesis_readiness(
        graph_revision=state.graph_revision,
        provisional_selected_node_ids=selection.selected_node_ids,
        candidates=state.relation_candidates,
        relation_ledger=state.relation_ledger,
        merge_applications=state.merge_applications,
        evaluated_at=_iso(),
        blocking_inventory_candidate_ids=[item.candidate_id for item in blocking_inventory],
        blocking_inventory_complete=True,
        enrichment_budget_limit=state.spec.budget.max_relation_enrichment_pairs,
        enrichment_pairs_committed=enrichment_committed,
        eligible_enrichment_candidate_ids=eligible_enrichment_ids,
    )
    state.provisional_synthesis_selection = selection
    state.synthesis_readiness = readiness
    state.relation_readiness_status = "evaluated"
    if added_count:
        _queue_event(
            state,
            "relation_candidates_generated",
            run_id=state.run_id,
            status="committed",
            input_graph_revision=state.graph_revision,
            selected_pair_count=added_count,
            blocking_candidate_count=len(readiness.blocking_candidate_ids),
            enrichment_candidate_count=len(readiness.eligible_enrichment_candidate_ids),
            usage_source="unavailable",
        )
    inventory_fields = dict(
        run_id=state.run_id,
        role="relation",
        status="complete" if readiness.blocking_inventory_complete else "incomplete",
        input_graph_revision=state.graph_revision,
        graph_revision=state.graph_revision,
        selected_node_count=len(selection.selected_node_ids),
        provisional_selected_node_count=len(selection.selected_node_ids),
        blocking_pair_count=readiness.blocking_pair_count,
        resolved_blocking_pair_count=readiness.resolved_blocking_pair_count,
        unresolved_blocking_pair_count=readiness.unresolved_blocking_pair_count,
        blocking_inventory_complete=readiness.blocking_inventory_complete,
        enrichment_candidate_count=len(readiness.eligible_enrichment_candidate_ids),
        enrichment_pairs_committed=readiness.enrichment_pairs_committed,
        enrichment_pairs_remaining=readiness.enrichment_pairs_remaining,
        selected_pair_count=readiness.blocking_pair_count,
        usage_source="unavailable",
    )
    _queue_event(state, "relation_blocking_inventory_evaluated", **inventory_fields)
    if readiness.blocking_inventory_complete:
        _queue_event(state, "relation_blocking_inventory_completed", **inventory_fields)
    _queue_event(
        state,
        "synthesis_readiness_evaluated",
        run_id=state.run_id,
        status="ready" if readiness.ready else "blocked",
        input_graph_revision=state.graph_revision,
        selected_node_count=len(selection.selected_node_ids),
        blocking_candidate_count=len(readiness.blocking_candidate_ids),
        material_conflict_count=len(readiness.unresolved_material_conflicts),
        graph_revision=state.graph_revision,
        provisional_selected_node_count=len(selection.selected_node_ids),
        blocking_pair_count=readiness.blocking_pair_count,
        resolved_blocking_pair_count=readiness.resolved_blocking_pair_count,
        unresolved_blocking_pair_count=readiness.unresolved_blocking_pair_count,
        blocking_inventory_complete=readiness.blocking_inventory_complete,
        enrichment_candidate_count=len(readiness.eligible_enrichment_candidate_ids),
        enrichment_pairs_committed=readiness.enrichment_pairs_committed,
        enrichment_pairs_remaining=readiness.enrichment_pairs_remaining,
        role="relation",
        usage_source="unavailable",
    )
    if not readiness.ready:
        _queue_event(
            state,
            "synthesis_blocked_by_relation",
            run_id=state.run_id,
            status="blocked",
            input_graph_revision=state.graph_revision,
            blocking_candidate_count=(
                len(readiness.blocking_candidate_ids)
                + len(readiness.unresolved_material_conflicts)
            ),
            usage_source="unavailable",
        )
    return readiness


def _prepare_terminal_or_relation(
    run_dir: str | Path,
    state: AppRunState,
    *,
    terminal_action: Literal["ready_for_synthesis", "run_complete"],
    terminal_reason: str,
    terminal_source: Literal[
        "authorized_synthesis",
        "max_iterations",
        "max_search_nodes",
        "continuation_gate",
        "controller_stop",
    ],
    runtime_limits: RuntimeLimits,
    profile: Literal["legacy-explicit", "native-guided", "native-autonomous"],
    entropy_plateau: bool = False,
) -> NextEpisodeOutcome:
    """Run the Relation gate before committing a new sticky terminal action."""

    state.pending_terminal_action = terminal_action
    state.pending_terminal_reason = terminal_reason
    state.pending_terminal_source = terminal_source
    readiness = _evaluate_relation_gate(run_dir, state, entropy_plateau=entropy_plateau)
    state.pending_terminal_gate_evaluated = True
    blocking_ids = set(readiness.blocking_candidate_ids)
    pending = [
        candidate
        for candidate in state.relation_candidates
        if candidate.candidate_id in blocking_ids and candidate.status == "pending"
    ]
    ordered_pending = sorted(
        pending,
        key=lambda candidate: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}[candidate.priority],
            candidate.left_node_id,
            candidate.right_node_id,
        ),
    )
    pending = select_node_disjoint_relation_batch(
        ordered_pending,
        max_pairs=state.spec.budget.max_relation_pairs_per_episode,
    )
    if pending:
        assert state.provisional_synthesis_selection is not None
        request = build_relation_episode_request(
            state.graph(),
            pending,
            run_id=state.run_id,
            problem=state.spec.problem,
            goal=state.spec.goal,
            constraints=list(state.spec.constraints),
            provisional_synthesis_node_ids=state.provisional_synthesis_selection.selected_node_ids,
            max_relation_pairs_per_episode=state.spec.budget.max_relation_pairs_per_episode,
            native_orchestration_allowed=True,
            runtime_limits=runtime_limits,
            transport_hints={"profile": profile, "runtime": "current-codex-app"},
        )
        for candidate in pending:
            candidate.status = "granted"
            candidate.granted_episode_id = request.episode_id
            candidate.granted_attempt_id = request.attempt_id
        return _grant_new_episode(run_dir, state, request, profile=profile)

    if not readiness.ready:
        state.controller_action = "await_operator_decision"
        _save_state(run_dir, state)
        return NextEpisodeOutcome(
            run_id=state.run_id,
            controller_action="await_operator_decision",
            reason=readiness.reason,
        )

    enrichment_ids = set(readiness.eligible_enrichment_candidate_ids)
    enrichment_pending = [
        candidate
        for candidate in state.relation_candidates
        if candidate.candidate_id in enrichment_ids and candidate.status == "pending"
    ]
    enrichment_pending = sorted(
        enrichment_pending,
        key=lambda candidate: (
            candidate.left_node_id,
            candidate.right_node_id,
            candidate.candidate_reason,
        ),
    )
    enrichment_pending = select_node_disjoint_relation_batch(
        enrichment_pending,
        max_pairs=min(
            state.spec.budget.max_relation_pairs_per_episode,
            readiness.enrichment_pairs_remaining,
        ),
    )
    if readiness.enrichment_pending and enrichment_pending:
        assert state.provisional_synthesis_selection is not None
        request = build_relation_episode_request(
            state.graph(),
            enrichment_pending,
            run_id=state.run_id,
            problem=state.spec.problem,
            goal=state.spec.goal,
            constraints=list(state.spec.constraints),
            provisional_synthesis_node_ids=state.provisional_synthesis_selection.selected_node_ids,
            max_relation_pairs_per_episode=state.spec.budget.max_relation_pairs_per_episode,
            native_orchestration_allowed=True,
            runtime_limits=runtime_limits,
            transport_hints={"profile": profile, "runtime": "current-codex-app"},
        )
        for candidate in enrichment_pending:
            candidate.status = "granted"
            candidate.granted_episode_id = request.episode_id
            candidate.granted_attempt_id = request.attempt_id
        return _grant_new_episode(run_dir, state, request, profile=profile)

    if readiness.enrichment_pairs_remaining == 0:
        _queue_event(
            state,
            "relation_enrichment_budget_exhausted",
            run_id=state.run_id,
            role="relation",
            status="exhausted",
            input_graph_revision=state.graph_revision,
            graph_revision=state.graph_revision,
            provisional_selected_node_count=len(readiness.provisional_selected_node_ids),
            blocking_pair_count=readiness.blocking_pair_count,
            resolved_blocking_pair_count=readiness.resolved_blocking_pair_count,
            unresolved_blocking_pair_count=readiness.unresolved_blocking_pair_count,
            blocking_inventory_complete=readiness.blocking_inventory_complete,
            enrichment_candidate_count=len(readiness.eligible_enrichment_candidate_ids),
            enrichment_pairs_committed=readiness.enrichment_pairs_committed,
            enrichment_pairs_remaining=0,
            selected_pair_count=0,
            usage_source="unavailable",
        )

    committed_terminal_source = terminal_source
    state.pending_terminal_action = None
    state.pending_terminal_reason = None
    state.pending_terminal_source = None
    state.pending_terminal_gate_evaluated = False
    if state.terminal_record is None:
        state.terminal_record = TerminalRecord(
            action=terminal_action,
            source=committed_terminal_source,
            reason=terminal_reason,
            graph_revision=state.graph_revision,
            controller_iteration=state.controller_iteration,
            committed_at=_iso(),
        )
        if terminal_action == "run_complete":
            _queue_event(
                state,
                "run_completed",
                run_id=state.run_id,
                status="completed",
                input_graph_revision=state.graph_revision,
                graph_revision=state.graph_revision,
                usage_source="unavailable",
            )
    state.controller_action = terminal_action
    _save_state(run_dir, state)
    return NextEpisodeOutcome(
        run_id=state.run_id,
        controller_action=terminal_action,
        reason=terminal_reason,
    )


def _request_artifact_path(run_dir: str | Path, request: EpisodeRequest) -> Path:
    return _attempt_artifact_dir(run_dir, request) / "request.json"


def _status_artifact_path(run_dir: str | Path, request: EpisodeRequest) -> Path:
    return _attempt_artifact_dir(run_dir, request) / "status.json"


def _attempt_artifact_dir(run_dir: str | Path, request: EpisodeRequest) -> Path:
    """Resolve one attempt mirror below the run without trusting persisted IDs as paths."""

    for label, value in (
        ("episode_id", request.episode_id),
        ("attempt_id", request.attempt_id),
    ):
        _validate_artifact_component(value, label)
    root = (Path(run_dir) / "episodes").resolve()
    target = (root / request.episode_id / request.attempt_id).resolve()
    if root not in target.parents:
        raise ValueError("attempt artifact path escapes the App run directory")
    return target


def _epistemic_reference_context(
    run_dir: str | Path,
    state: AppRunState,
) -> EpistemicReferenceContext:
    """Snapshot safe pre-commit identities without creating any artifact."""

    root = Path(run_dir).resolve()
    reserved_files = {
        "app_run_state.json",
        "episode_events.jsonl",
        "dte_cache.json",
        "epistemic/ledger.json",
        "epistemic/researcher_learning.jsonl",
        "observability/feedback.jsonl",
    }
    artifact_paths: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if (
            relative in reserved_files
            or relative.startswith("episodes/")
            or relative.startswith("relations/")
            or relative.startswith("epistemic/")
            or relative.startswith("observability/")
        ):
            continue
        artifact_paths.add(relative)

    confirmed_learning_ids: set[str] = set()
    duplicate_learning_ids: set[str] = set()
    seen_learning_ids: set[str] = set()
    learning_path = root / "epistemic" / "researcher_learning.jsonl"
    if learning_path.exists():
        for line in learning_path.read_bytes().splitlines():
            try:
                raw = json.loads(line.decode("utf-8"))
                learning = ResearcherLearningRecordV1.model_validate(raw)
            except Exception:
                continue
            if learning.learning_id in seen_learning_ids:
                duplicate_learning_ids.add(learning.learning_id)
                confirmed_learning_ids.discard(learning.learning_id)
                continue
            seen_learning_ids.add(learning.learning_id)
            if (
                learning.run_id == state.run_id
                and learning.source == "user"
                and learning.user_confirmed
            ):
                confirmed_learning_ids.add(learning.learning_id)
    confirmed_learning_ids.difference_update(duplicate_learning_ids)

    return EpistemicReferenceContext(
        committed_episode_attempts={
            (episode.episode_id, attempt.attempt_id)
            for episode in state.episodes
            for attempt in episode.attempts
            if attempt.status == "committed"
            and episode.committed_attempt_id == attempt.attempt_id
        },
        artifact_paths=artifact_paths,
        user_confirmed_learning_ids=confirmed_learning_ids,
    )


def _write_attempt_artifacts(run_dir: str | Path, attempt: EpisodeAttemptRecord) -> None:
    _write_json_atomic(_request_artifact_path(run_dir, attempt.request), attempt.request.model_dump(mode="json"))
    _write_json_atomic(_status_artifact_path(run_dir, attempt.request), attempt.model_dump(mode="json"))


def _repair_attempt_artifacts(run_dir: str | Path, state: AppRunState) -> None:
    """Best-effort reconstruction of non-authoritative request/status mirrors."""

    for episode in state.episodes:
        for attempt in episode.attempts:
            try:
                _write_attempt_artifacts(run_dir, attempt)
            except Exception:
                # AppRunState is authoritative. A later save or restart retries
                # the mirror without changing any controller lifecycle fact.
                continue


def _grant_new_episode(
    run_dir: str | Path,
    state: AppRunState,
    request: EpisodeRequest,
    *,
    profile: str,
) -> NextEpisodeOutcome:
    # Re-parse even builder-produced Pydantic instances.  Pydantic does not
    # revalidate nested model instances by default, and assignment validation
    # is intentionally not relied on at this machine-facing boundary.
    request = EpisodeRequest.model_validate(request.model_dump(mode="json"))
    if request.run_id != state.run_id:
        raise ValueError("episode grant run_id does not match App run state")
    if _active_attempt(state) is not None:
        raise ValueError("cannot grant a second episode while an attempt is active")
    if any(episode.episode_id == request.episode_id for episode in state.episodes):
        raise ValueError("episode grant reuses an existing episode_id")
    if any(
        attempt.attempt_id == request.attempt_id
        for episode in state.episodes
        for attempt in episode.attempts
    ):
        raise ValueError("episode grant reuses an existing attempt_id")
    granted_at = _now()
    deadline = None
    if request.runtime_limits.wall_clock_seconds is not None:
        deadline = granted_at + timedelta(seconds=request.runtime_limits.wall_clock_seconds)
    attempt = EpisodeAttemptRecord(
        attempt_id=request.attempt_id,
        attempt_number=1,
        status="in_progress",
        request=request,
        request_hash=_episode_request_hash(request),
        granted_at=_iso(granted_at),
        deadline_at=None if deadline is None else _iso(deadline),
    )
    state.episodes.append(
        EpisodeLifecycleRecord(
            episode_id=request.episode_id,
            run_id=state.run_id,
            role=request.role,
            attempts=[attempt],
        )
    )
    state.active_episode_id = request.episode_id
    state.active_attempt_id = request.attempt_id
    state.controller_action = "episode_required"
    common = dict(
        run_id=state.run_id,
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        role=request.role,
        adapter_name="codex-app-main-agent",
        transport_name="current-app-runtime",
        profile=profile,
        input_graph_revision=request.input_graph_revision,
        selected_node_count=len(request.selected_node_revisions),
        usage_source="unavailable",
    )
    _queue_event(state, "episode_granted", status="granted", **common)
    _queue_event(state, "episode_started", status="in_progress", **common)
    if request.role == "relation":
        scheduling_classes = {
            pair.scheduling_class
            for pair in (request.relation_payload.candidate_pairs if request.relation_payload else [])
        }
        _queue_event(
            state,
            "relation_episode_granted",
            status="granted",
            selected_pair_count=(
                0 if request.relation_payload is None else len(request.relation_payload.candidate_pairs)
            ),
            **common,
        )
        if scheduling_classes == {"enrichment"}:
            committed = _relation_enrichment_pairs_committed(state)
            _queue_event(
                state,
                "relation_enrichment_granted",
                status="granted",
                selected_pair_count=len(request.relation_payload.candidate_pairs),
                graph_revision=state.graph_revision,
                provisional_selected_node_count=len(
                    request.relation_payload.provisional_synthesis_node_ids
                ),
                blocking_pair_count=(
                    None if state.synthesis_readiness is None else state.synthesis_readiness.blocking_pair_count
                ),
                resolved_blocking_pair_count=(
                    None
                    if state.synthesis_readiness is None
                    else state.synthesis_readiness.resolved_blocking_pair_count
                ),
                unresolved_blocking_pair_count=(
                    None
                    if state.synthesis_readiness is None
                    else state.synthesis_readiness.unresolved_blocking_pair_count
                ),
                blocking_inventory_complete=(
                    None
                    if state.synthesis_readiness is None
                    else state.synthesis_readiness.blocking_inventory_complete
                ),
                enrichment_candidate_count=len(state.synthesis_readiness.eligible_enrichment_candidate_ids),
                enrichment_pairs_committed=committed,
                enrichment_pairs_remaining=max(
                    0, state.spec.budget.max_relation_enrichment_pairs - committed
                ),
                **common,
            )
    _save_state(run_dir, state)
    try:
        _write_attempt_artifacts(run_dir, attempt)
    except Exception:
        pass
    return NextEpisodeOutcome(
        run_id=state.run_id,
        controller_action="episode_required",
        request=request,
    )


def next_app_episode(
    run_dir: str | Path,
    *,
    runtime_limits: RuntimeLimits | None = None,
    profile: Literal["legacy-explicit", "native-guided", "native-autonomous"] = "native-autonomous",
    embedding_provider: EmbeddingProvider | None = None,
) -> NextEpisodeOutcome:
    """Grant or resume one episode; never invoke a model runtime or subprocess."""

    state = load_app_run(run_dir)
    active = _active_attempt(state)
    if active is not None:
        _, attempt = active
        if _attempt_expired(attempt):
            _mark_expired(run_dir, state, attempt)
            _save_state(run_dir, state)
            try:
                _write_attempt_artifacts(run_dir, attempt)
            except Exception:
                pass
            return NextEpisodeOutcome(
                run_id=state.run_id,
                controller_action="await_operator_decision",
                reason=attempt.failure_reason,
            )
        return NextEpisodeOutcome(
            run_id=state.run_id,
            controller_action="episode_required",
            request=attempt.request,
            resumed_existing_attempt=True,
        )

    source_limits = runtime_limits or RuntimeLimits(max_retries=1)
    limits = RuntimeLimits.model_validate(source_limits.model_dump(mode="json"))
    if state.controller_action in {"ready_for_synthesis", "run_complete"}:
        return NextEpisodeOutcome(
            run_id=state.run_id,
            controller_action=state.controller_action,
            reason=(
                state.terminal_record.reason
                if state.terminal_record is not None
                else "controller terminal action is sticky"
            ),
        )
    if state.controller_action == "await_operator_decision":
        return NextEpisodeOutcome(
            run_id=state.run_id,
            controller_action="await_operator_decision",
            reason="the previous attempt requires an explicit retry or operator decision",
        )
    # The loop consumes backend-only transitions. It never asks the App main
    # agent to interpret `continue_controller` or run controller mathematics.
    while True:
        parent = _select_executor_parent(state)
        if parent is not None:
            remaining_slots = remaining_search_node_slots(
                state.nodes,
                state.spec.budget.max_committed_search_nodes,
            )
            request = build_executor_episode_request(
                state.graph(),
                parent,
                run_id=state.run_id,
                iteration=max(1, state.controller_iteration),
                max_returned_children=min(
                    parent.expansion_budget,
                    state.spec.budget.max_children_per_iteration,
                    remaining_slots,
                ),
                objective=f"{state.spec.goal}: expand {parent.claim}",
                constraints=list(state.spec.constraints),
                native_orchestration_allowed=state.spec.allow_self_organized_executor,
                runtime_limits=limits,
                transport_hints={"profile": profile, "runtime": "current-codex-app"},
            )
            return _grant_new_episode(run_dir, state, request, profile=profile)

        unjudged = _select_unjudged_frontier(state)
        if unjudged:
            request = build_judge_episode_request(
                state.graph(),
                unjudged,
                run_id=state.run_id,
                problem=state.spec.problem,
                goal=state.spec.goal,
                constraints=list(state.spec.constraints),
                native_orchestration_allowed=True,
                runtime_limits=limits,
                transport_hints={"profile": profile, "runtime": "current-codex-app"},
            )
            return _grant_new_episode(run_dir, state, request, profile=profile)

        if state.pending_terminal_action is not None:
            return _prepare_terminal_or_relation(
                run_dir,
                state,
                terminal_action=state.pending_terminal_action,
                terminal_reason=state.pending_terminal_reason or "controller intends to terminate",
                terminal_source=state.pending_terminal_source or "controller_stop",
                runtime_limits=limits,
                profile=profile,
            )

        if state.synthesis_request is not None and _has_synthesis_safe_checkpoint(state):
            return _prepare_terminal_or_relation(
                run_dir,
                state,
                terminal_action="ready_for_synthesis",
                terminal_reason="authorized synthesis request is pending",
                terminal_source="authorized_synthesis",
                runtime_limits=limits,
                profile=profile,
            )

        if (
            state.spec.budget.continuation_policy == "bounded_node_yield_v1"
            and count_committed_search_nodes(state.nodes)
            >= state.spec.budget.max_committed_search_nodes
        ):
            terminal_action = (
                "ready_for_synthesis" if state.spec.require_final_synthesis else "run_complete"
            )
            _queue_event(
                state,
                "search_node_budget_exhausted",
                run_id=state.run_id,
                status="exhausted",
                input_graph_revision=state.graph_revision,
                graph_revision=state.graph_revision,
                controller_iteration=state.controller_iteration,
                committed_search_node_count=count_committed_search_nodes(state.nodes),
                remaining_search_node_slots=0,
                usage_source="unavailable",
            )
            return _prepare_terminal_or_relation(
                run_dir,
                state,
                terminal_action=terminal_action,
                terminal_reason="maximum committed search nodes reached",
                terminal_source="max_search_nodes",
                runtime_limits=limits,
                profile=profile,
            )

        # The iteration cap prevents another controller allocation. It does
        # not revoke already-authorized Executor output: every committed child
        # must still pass a bounded Judge episode before Relation/readiness.
        if state.controller_iteration >= state.spec.budget.max_iterations:
            terminal_action: ControllerAction = (
                "ready_for_synthesis" if state.spec.require_final_synthesis else "run_complete"
            )
            return _prepare_terminal_or_relation(
                run_dir,
                state,
                terminal_action=terminal_action,
                terminal_reason="maximum controller iterations reached",
                terminal_source="max_iterations",
                runtime_limits=limits,
                profile=profile,
            )

        action, reason = _progress_controller(
            run_dir,
            state,
            embedding_provider=embedding_provider,
        )
        terminal_intent: tuple[
            Literal["ready_for_synthesis", "run_complete"],
            str,
            Literal[
                "authorized_synthesis",
                "max_iterations",
                "max_search_nodes",
                "continuation_gate",
                "controller_stop",
            ],
        ] | None = None
        if state.synthesis_request is not None and _has_synthesis_safe_checkpoint(state):
            terminal_intent = (
                "ready_for_synthesis",
                "authorized synthesis request is pending",
                "authorized_synthesis",
            )
        elif action != "continue_controller":
            if reason == "maximum committed search nodes reached":
                source = "max_search_nodes"
            elif reason == "maximum controller iterations reached":
                source = "max_iterations"
            elif (
                state.continuation_gate_records
                and state.continuation_gate_records[-1].decision == "prepare_synthesis"
                and reason == state.continuation_gate_records[-1].reason
            ):
                source = "continuation_gate"
            else:
                source = "controller_stop"
            terminal_intent = (action, reason, source)  # type: ignore[assignment]
        if terminal_intent is not None:
            terminal_action, terminal_reason, terminal_source = terminal_intent
            if _select_executor_parent(state) is not None or _select_unjudged_frontier(state):
                state.pending_terminal_action = terminal_action
                state.pending_terminal_reason = terminal_reason
                state.pending_terminal_source = terminal_source
                state.pending_terminal_gate_evaluated = False
                state.controller_action = "continue_controller"
                _save_state(run_dir, state)
                continue
            return _prepare_terminal_or_relation(
                run_dir,
                state,
                terminal_action=terminal_action,
                terminal_reason=terminal_reason,
                terminal_source=terminal_source,
                runtime_limits=limits,
                profile=profile,
                entropy_plateau="entropy" in reason.casefold(),
            )
        state.controller_action = action
        _save_state(run_dir, state)


def _result_payload(raw_result: Any) -> dict[str, Any]:
    """Materialize one JSON-safe result snapshot for the whole transaction.

    A caller-provided ``model_dump`` implementation is untrusted and may be
    stateful.  Calling it again after lifecycle validation would make the
    graph commit, result artifact, and recorded hash observe different values.
    """

    if isinstance(raw_result, EpisodeResult):
        dumped: Any = raw_result.model_dump(mode="json")
    elif isinstance(raw_result, Mapping):
        dumped = dict(raw_result)
    else:
        model_dump = getattr(raw_result, "model_dump", None)
        if not callable(model_dump):
            raise TypeError("episode result must be a mapping or a model with model_dump()")
        dumped = model_dump(mode="json")
    if not isinstance(dumped, Mapping):
        raise TypeError("episode result model_dump() must return a mapping")
    detached = json.loads(
        json.dumps(
            dict(dumped),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    if not isinstance(detached, dict):
        raise TypeError("episode result JSON payload must be an object")
    return detached


def _result_identity_hint(raw_result: Any) -> dict[str, Any]:
    """Extract only audit identity without invoking an untrusted dumper twice."""

    hint: dict[str, Any] = {}
    for field_name in ("episode_id", "attempt_id"):
        try:
            value = (
                raw_result.get(field_name)
                if isinstance(raw_result, Mapping)
                else getattr(raw_result, field_name, None)
            )
        except Exception:
            value = None
        if isinstance(value, str):
            hint[field_name] = value
    return hint


def _rejection_outcome(state: AppRunState, request: EpisodeRequest, reason: str) -> CommitOutcome:
    return CommitOutcome(
        accepted=False,
        episode_id=request.episode_id,
        graph_revision_before=state.graph_revision,
        graph_revision_after=state.graph_revision,
        rejection_reason=reason,
    )


def _identity_rejection(
    run_dir: str | Path,
    state: AppRunState,
    payload: Mapping[str, Any],
    reason: str,
) -> SubmitEpisodeOutcome:
    episode_value = payload.get("episode_id")
    attempt_value = payload.get("attempt_id")
    episode_id = episode_value if isinstance(episode_value, str) else ""
    attempt_id = attempt_value if isinstance(attempt_value, str) else ""
    outcome = CommitOutcome(
        accepted=False,
        episode_id=episode_id,
        graph_revision_before=state.graph_revision,
        graph_revision_after=state.graph_revision,
        rejection_reason=reason,
    )
    _emit_best_effort(
        _event_log(run_dir),
        "output_rejected",
        run_id=state.run_id,
        episode_id=episode_id or None,
        attempt_id=attempt_id or None,
        status="rejected",
        input_graph_revision=state.graph_revision,
        accepted_node_count=0,
        rejection_reason=reason,
        schema_valid=False,
        usage_source="unavailable",
    )
    return SubmitEpisodeOutcome(
        run_id=state.run_id,
        episode_id=episode_id,
        attempt_id=attempt_id,
        commit_outcome=outcome,
        next_controller_action=state.controller_action,
    )


def submit_app_episode_result(
    run_dir: str | Path,
    raw_result: Any,
) -> SubmitEpisodeOutcome:
    """Validate lifecycle plus result, then commit through the sole graph boundary."""

    state = load_app_run(run_dir)
    try:
        payload = _result_payload(raw_result)
    except Exception as exc:
        return _identity_rejection(
            run_dir,
            state,
            _result_identity_hint(raw_result),
            f"episode result JSON detachment failed: {exc}",
        )
    if state.controller_action in {"ready_for_synthesis", "run_complete"}:
        episode_value = payload.get("episode_id")
        attempt_value = payload.get("attempt_id")
        episode_id = episode_value if isinstance(episode_value, str) else ""
        attempt_id = attempt_value if isinstance(attempt_value, str) else ""
        outcome = CommitOutcome(
            accepted=False,
            episode_id=episode_id,
            graph_revision_before=state.graph_revision,
            graph_revision_after=state.graph_revision,
            rejection_reason="controller terminal action is sticky",
        )
        return SubmitEpisodeOutcome(
            run_id=state.run_id,
            episode_id=episode_id,
            attempt_id=attempt_id,
            commit_outcome=outcome,
            next_controller_action=state.controller_action,
        )
    for field_name in ("episode_id", "attempt_id"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return _identity_rejection(
                run_dir,
                state,
                payload,
                f"episode result is missing a non-empty {field_name}",
            )
    episode_id = payload["episode_id"]
    attempt_id = payload["attempt_id"]
    try:
        episode = _find_episode(state, episode_id)
        attempt = _find_attempt(episode, attempt_id)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else str(exc)
        return _identity_rejection(
            run_dir,
            state,
            payload,
            f"episode identity lookup failed: {detail}",
        )
    result_path = _attempt_artifact_dir(run_dir, attempt.request) / "result.json"
    submitted_at = _now()
    wall_clock_ms = max(0, round((submitted_at - _parse_time(attempt.granted_at)).total_seconds() * 1000))
    raw_output = payload.get("structured_output")
    returned_node_count = None
    returned_observation_count = None
    if isinstance(raw_output, Mapping):
        if isinstance(raw_output.get("nodes"), list):
            returned_node_count = len(raw_output["nodes"])
        if isinstance(raw_output.get("observations"), list):
            returned_observation_count = len(raw_output["observations"])
    _queue_event(
        state,
        "episode_submitted",
        run_id=state.run_id,
        episode_id=episode_id,
        attempt_id=attempt_id,
        role=episode.role,
        adapter_name="codex-app-main-agent",
        transport_name="current-app-runtime",
        wall_clock_ms=wall_clock_ms,
        retry_count=attempt.attempt_number - 1,
        status="submitted",
        input_graph_revision=attempt.request.input_graph_revision,
        selected_node_count=len(attempt.request.selected_node_revisions),
        returned_node_count=returned_node_count,
        returned_observation_count=returned_observation_count,
        usage_source="unavailable",
    )

    invalid_lifecycle = attempt.status not in {"granted", "in_progress", "completed_uncommitted"}
    not_active = state.active_episode_id != episode_id or state.active_attempt_id != attempt_id
    already_committed = episode.committed_attempt_id is not None
    if _attempt_expired(attempt, at=submitted_at) and not invalid_lifecycle:
        _mark_expired(run_dir, state, attempt)
        reason = "expired attempt cannot commit"
    elif invalid_lifecycle or not_active or already_committed:
        reason = f"attempt lifecycle forbids commit: status={attempt.status}"
    else:
        reason = ""

    if reason:
        outcome = _rejection_outcome(state, attempt.request, reason)
        if attempt.commit_outcome is None:
            attempt.commit_outcome = outcome
        rejected_path = result_path.with_name(f"rejected_result_{uuid.uuid4().hex}.json")
        _write_json_atomic(rejected_path, payload)
        _queue_event(
            state,
            "output_rejected",
            run_id=state.run_id,
            episode_id=episode_id,
            attempt_id=attempt_id,
            role=episode.role,
            status="rejected",
            input_graph_revision=attempt.request.input_graph_revision,
            accepted_node_count=0,
            selected_node_count=len(attempt.request.selected_node_revisions),
            returned_observation_count=returned_observation_count,
            accepted_observation_count=0 if episode.role in {"judge", "relation"} else None,
            rejection_reason=reason,
            schema_valid=None,
            usage_source="unavailable",
        )
        if episode.role == "relation":
            _queue_event(
                state,
                "relation_result_rejected",
                run_id=state.run_id,
                episode_id=episode_id,
                attempt_id=attempt_id,
                role=episode.role,
                status="rejected",
                input_graph_revision=attempt.request.input_graph_revision,
                selected_pair_count=(
                    0
                    if attempt.request.relation_payload is None
                    else len(attempt.request.relation_payload.candidate_pairs)
                ),
                returned_observation_count=returned_observation_count,
                accepted_observation_count=0,
                rejection_reason=reason,
                schema_valid=None,
                usage_source="unavailable",
            )
        _save_state(run_dir, state)
        try:
            _write_attempt_artifacts(run_dir, attempt)
        except Exception:
            pass
        return SubmitEpisodeOutcome(
            run_id=state.run_id,
            episode_id=episode_id,
            attempt_id=attempt_id,
            commit_outcome=outcome,
            next_controller_action=state.controller_action,
        )

    epistemic_context = _epistemic_reference_context(run_dir, state)
    attempt.submitted_at = _iso(submitted_at)
    attempt.status = "completed_uncommitted"
    _write_json_atomic(result_path, payload)
    graph = state.graph()
    buffered_events = _BufferedEpisodeEvents()
    outcome = commit_episode_result(
        graph,
        attempt.request,
        payload,
        telemetry=buffered_events,  # type: ignore[arg-type]
        epistemic_context=epistemic_context,
    )
    attempt.commit_outcome = outcome
    if outcome.accepted:
        state.replace_graph(graph)
        if state.synthesis_request is not None:
            state.synthesis_request = _canonicalize_synthesis_request(
                state,
                state.synthesis_request,
            )
        attempt.status = "committed"
        episode.committed_attempt_id = attempt_id
        parsed = EpisodeResult.model_validate(payload)
        attempt.result_hash = parsed.output_hash
        attempt.committed_result = parsed.model_copy(deep=True)
        state.active_episode_id = None
        state.active_attempt_id = None
        state.controller_action = "continue_controller"
        _queue_buffered_events(state, buffered_events)
        _queue_event(
            state,
            "episode_completed",
            run_id=state.run_id,
            episode_id=episode_id,
            attempt_id=attempt_id,
            role=episode.role,
            adapter_name="codex-app-main-agent",
            transport_name="current-app-runtime",
            wall_clock_ms=wall_clock_ms,
            retry_count=attempt.attempt_number - 1,
            status="committed",
            input_graph_revision=attempt.request.input_graph_revision,
            selected_node_count=len(attempt.request.selected_node_revisions),
            returned_node_count=outcome.accepted_node_count if episode.role == "executor" else None,
            accepted_node_count=outcome.accepted_node_count,
            returned_observation_count=(
                outcome.accepted_node_count if episode.role in {"judge", "relation"} else None
            ),
            accepted_observation_count=(
                outcome.accepted_node_count if episode.role in {"judge", "relation"} else None
            ),
            usage_source="unavailable",
            schema_valid=True,
        )
        if episode.role == "relation" and attempt.request.relation_payload is not None:
            enrichment_count = sum(
                pair.scheduling_class == "enrichment"
                for pair in attempt.request.relation_payload.candidate_pairs
            )
            if enrichment_count:
                committed = _relation_enrichment_pairs_committed(state)
                readiness = state.synthesis_readiness
                _queue_event(
                    state,
                    "relation_enrichment_committed",
                    run_id=state.run_id,
                    episode_id=episode_id,
                    attempt_id=attempt_id,
                    role="relation",
                    status="committed",
                    input_graph_revision=attempt.request.input_graph_revision,
                    graph_revision=state.graph_revision,
                    provisional_selected_node_count=len(
                        attempt.request.relation_payload.provisional_synthesis_node_ids
                    ),
                    blocking_pair_count=(None if readiness is None else readiness.blocking_pair_count),
                    resolved_blocking_pair_count=(
                        None if readiness is None else readiness.resolved_blocking_pair_count
                    ),
                    unresolved_blocking_pair_count=(
                        None if readiness is None else readiness.unresolved_blocking_pair_count
                    ),
                    blocking_inventory_complete=(
                        None if readiness is None else readiness.blocking_inventory_complete
                    ),
                    enrichment_candidate_count=enrichment_count,
                    enrichment_pairs_committed=committed,
                    enrichment_pairs_remaining=max(
                        0, state.spec.budget.max_relation_enrichment_pairs - committed
                    ),
                    selected_pair_count=enrichment_count,
                    usage_source="unavailable",
                )
        if episode.role == "relation":
            state.pending_terminal_gate_evaluated = False
            state.provisional_synthesis_selection = None
            state.synthesis_readiness = None
            state.relation_readiness_status = "not_evaluated"
    else:
        attempt.status = "rejected"
        state.active_episode_id = None
        state.active_attempt_id = None
        state.controller_action = "await_operator_decision"
        _queue_buffered_events(state, buffered_events)
    _save_state(run_dir, state)
    try:
        _write_attempt_artifacts(run_dir, attempt)
    except Exception:
        pass
    return SubmitEpisodeOutcome(
        run_id=state.run_id,
        episode_id=episode_id,
        attempt_id=attempt_id,
        commit_outcome=outcome,
        next_controller_action=state.controller_action,
    )


def _transition_attempt(
    run_dir: str | Path,
    episode_id: str,
    attempt_id: str,
    *,
    status: Literal["failed", "cancelled"],
    reason: str,
) -> TransitionOutcome:
    state = load_app_run(run_dir)
    _ensure_nonterminal(state, f"mark an episode {status}")
    episode = _find_episode(state, episode_id)
    attempt = _find_attempt(episode, attempt_id)
    if attempt.status not in {"granted", "in_progress", "completed_uncommitted"}:
        raise ValueError(f"cannot mark attempt {status} from status={attempt.status}")
    if state.active_episode_id != episode_id or state.active_attempt_id != attempt_id:
        raise ValueError("only the active attempt may transition")
    attempt.status = status
    attempt.failure_reason = reason
    state.active_episode_id = None
    state.active_attempt_id = None
    state.controller_action = "await_operator_decision"
    _release_relation_grants(state, attempt.request)
    _queue_event(
        state,
        "episode_failed" if status == "failed" else "episode_cancelled",
        run_id=state.run_id,
        episode_id=episode_id,
        attempt_id=attempt_id,
        role=episode.role,
        status=status,
        input_graph_revision=attempt.request.input_graph_revision,
        rejection_reason=reason,
        usage_source="unavailable",
    )
    _save_state(run_dir, state)
    try:
        _write_attempt_artifacts(run_dir, attempt)
    except Exception:
        pass
    return TransitionOutcome(
        run_id=state.run_id,
        episode_id=episode_id,
        attempt_id=attempt_id,
        status=status,
        controller_action=state.controller_action,
    )


def fail_app_episode(run_dir: str | Path, episode_id: str, attempt_id: str, reason: str) -> TransitionOutcome:
    return _transition_attempt(
        run_dir,
        episode_id,
        attempt_id,
        status="failed",
        reason=reason,
    )


def cancel_app_episode(run_dir: str | Path, episode_id: str, attempt_id: str, reason: str) -> TransitionOutcome:
    return _transition_attempt(
        run_dir,
        episode_id,
        attempt_id,
        status="cancelled",
        reason=reason,
    )


def retry_app_episode(
    run_dir: str | Path,
    episode_id: str,
    *,
    selected_by: Literal["user", "main_agent", "run_default"] = "main_agent",
    wall_clock_seconds: int | None = None,
) -> TransitionOutcome:
    """Supersede the latest non-committed attempt and issue a fresh attempt ID."""

    state = load_app_run(run_dir)
    _ensure_nonterminal(state, "retry an episode")
    active = _active_attempt(state)
    if active is not None:
        active_episode, active_attempt = active
        if active_episode.episode_id != episode_id:
            raise ValueError("another episode has an active attempt")
    episode = _find_episode(state, episode_id)
    if episode.committed_attempt_id is not None:
        raise ValueError("a committed episode cannot be retried")
    previous = episode.attempts[-1]
    previous_status = previous.status
    if previous_status == "committed":
        raise ValueError("a committed attempt cannot be retried")
    if previous.attempt_number > previous.request.runtime_limits.max_retries:
        # A failed retry request must not invalidate work which is still
        # legitimately active and submittable.  A non-active rejected
        # Relation attempt, on the other hand, has no future commit path once
        # its retry budget is exhausted, so release that durable grant.
        if active is None and previous.request.role == "relation":
            _release_relation_grants(state, previous.request)
            previous.retry_exhaustion_released = True
            # Relation is a mandatory readiness obligation. Once this logical
            # episode has no remaining retry, return its candidates to the
            # controller so a fresh bounded Relation episode can be granted;
            # leaving await_operator_decision here has no general continue API
            # and can deadlock policies that disallow synthesis commands.
            state.controller_action = "continue_controller"
            _save_state(run_dir, state)
            return TransitionOutcome(
                run_id=state.run_id,
                episode_id=episode_id,
                attempt_id=previous.attempt_id,
                status=previous.status,
                controller_action="continue_controller",
                reason="episode retry limit exhausted; Relation grants released",
            )
        raise ValueError("episode retry limit exhausted")
    previous.status = "superseded"
    previous.superseded_from_status = previous_status
    state.active_episode_id = None
    state.active_attempt_id = None

    graph = state.graph()
    limits_payload = previous.request.runtime_limits.model_dump(mode="json")
    limits_payload["selected_by"] = selected_by
    if wall_clock_seconds is not None:
        limits_payload["wall_clock_seconds"] = wall_clock_seconds
    limits = RuntimeLimits.model_validate(limits_payload)
    if previous.request.role == "executor":
        parent_id = previous.request.parent_node_id
        if parent_id is None:
            raise ValueError("Executor retry is missing parent_node_id")
        parent = graph.node_by_id(parent_id)
        if parent is None or parent.status != "frontier":
            raise ValueError("retry parent is no longer an expandable committed node")
        request = build_executor_episode_request(
            graph,
            parent,
            run_id=state.run_id,
            iteration=previous.request.executor_payload.iteration if previous.request.executor_payload else 1,
            max_returned_children=previous.request.max_returned_children or 0,
            objective=previous.request.objective,
            constraints=(previous.request.executor_payload.constraints if previous.request.executor_payload else []),
            coverage_requirements=list(previous.request.coverage_requirements),
            allowed_output_types=list(previous.request.allowed_output_types),
            native_orchestration_allowed=previous.request.native_orchestration_allowed,
            runtime_limits=limits,
            tool_policy=previous.request.tool_policy,
            transport_hints=previous.request.transport_hints,
        )
    elif previous.request.role == "judge":
        nodes = [graph.node_by_id(node_id) for node_id in previous.request.selected_node_revisions]
        if any(node is None or node.status != "frontier" for node in nodes):
            raise ValueError("Judge retry targets are no longer committed frontier nodes")
        request = build_judge_episode_request(
            graph,
            [node for node in nodes if node is not None],
            run_id=state.run_id,
            problem=state.spec.problem,
            goal=state.spec.goal,
            constraints=list(state.spec.constraints),
            rubric_version=(
                previous.request.judge_payload.rubric_version
                if previous.request.judge_payload is not None
                else "research-potential.v1"
            ),
            native_orchestration_allowed=previous.request.native_orchestration_allowed,
            runtime_limits=limits,
            tool_policy=previous.request.tool_policy,
            transport_hints=previous.request.transport_hints,
        )
    elif previous.request.role == "relation":
        if previous.request.relation_payload is None:
            raise ValueError("Relation retry is missing relation_payload")
        candidate_ids = {
            pair.candidate_id for pair in previous.request.relation_payload.candidate_pairs
        }
        candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in state.relation_candidates
            if candidate.candidate_id in candidate_ids
            and candidate.status in {"pending", "granted"}
        }
        candidates = [
            candidates_by_id[pair.candidate_id]
            for pair in previous.request.relation_payload.candidate_pairs
            if pair.candidate_id in candidates_by_id
        ]
        if len(candidates) != len(candidate_ids):
            raise ValueError("Relation retry candidates are no longer current grants")
        retry_cap = state.spec.budget.max_relation_pairs_per_episode
        if all(candidate.scheduling_class == "enrichment" for candidate in candidates):
            retry_cap = min(
                retry_cap,
                max(
                    0,
                    state.spec.budget.max_relation_enrichment_pairs
                    - _relation_enrichment_pairs_committed(state),
                ),
            )
        retry_candidates = select_node_disjoint_relation_batch(
            candidates,
            max_pairs=retry_cap,
        )
        if not retry_candidates:
            raise ValueError("Relation retry has no node-disjoint candidates within remaining budget")
        request = build_relation_episode_request(
            graph,
            retry_candidates,
            run_id=state.run_id,
            problem=state.spec.problem,
            goal=state.spec.goal,
            constraints=list(state.spec.constraints),
            provisional_synthesis_node_ids=(
                previous.request.relation_payload.provisional_synthesis_node_ids
            ),
            max_relation_pairs_per_episode=state.spec.budget.max_relation_pairs_per_episode,
            rubric_version=previous.request.relation_payload.rubric_version,
            native_orchestration_allowed=previous.request.native_orchestration_allowed,
            runtime_limits=limits,
            tool_policy=previous.request.tool_policy,
            transport_hints=previous.request.transport_hints,
        )
        retry_candidate_ids = {candidate.candidate_id for candidate in retry_candidates}
        for candidate in candidates:
            if candidate.candidate_id not in retry_candidate_ids:
                candidate.status = "pending"
                candidate.granted_episode_id = None
                candidate.granted_attempt_id = None
    else:
        raise ValueError(f"retry is not implemented for role={previous.request.role}")
    # A retry is another attempt of the same logical episode.
    request.episode_id = episode_id
    if request.role == "relation":
        assert request.relation_payload is not None
        retry_candidate_ids = {
            pair.candidate_id for pair in request.relation_payload.candidate_pairs
        }
        for candidate in state.relation_candidates:
            if candidate.candidate_id in retry_candidate_ids:
                candidate.status = "granted"
                candidate.granted_episode_id = episode_id
                candidate.granted_attempt_id = request.attempt_id
    granted_at = _now()
    deadline = None
    if limits.wall_clock_seconds is not None:
        deadline = granted_at + timedelta(seconds=limits.wall_clock_seconds)
    attempt = EpisodeAttemptRecord(
        attempt_id=request.attempt_id,
        attempt_number=previous.attempt_number + 1,
        status="in_progress",
        request=request,
        request_hash=_episode_request_hash(request),
        granted_at=_iso(granted_at),
        deadline_at=None if deadline is None else _iso(deadline),
    )
    episode.attempts.append(attempt)
    state.active_episode_id = episode_id
    state.active_attempt_id = attempt.attempt_id
    state.controller_action = "episode_required"
    _queue_event(
        state,
        "episode_superseded",
        run_id=state.run_id,
        episode_id=episode_id,
        attempt_id=previous.attempt_id,
        role=episode.role,
        status="superseded",
        input_graph_revision=previous.request.input_graph_revision,
        usage_source="unavailable",
    )
    for event_type, status in (("episode_granted", "granted"), ("episode_started", "in_progress")):
        _queue_event(
            state,
            event_type,
            run_id=state.run_id,
            episode_id=episode_id,
            attempt_id=attempt.attempt_id,
            role=episode.role,
            adapter_name="codex-app-main-agent",
            transport_name="current-app-runtime",
            retry_count=attempt.attempt_number - 1,
            status=status,
            input_graph_revision=request.input_graph_revision,
            usage_source="unavailable",
        )
    _save_state(run_dir, state)
    try:
        _write_attempt_artifacts(run_dir, previous)
        _write_attempt_artifacts(run_dir, attempt)
    except Exception:
        pass
    return TransitionOutcome(
        run_id=state.run_id,
        episode_id=episode_id,
        attempt_id=attempt.attempt_id,
        status="in_progress",
        controller_action="episode_required",
        request=request,
    )


def request_app_synthesis(
    run_dir: str | Path,
    request: SynthesisControlRequest,
) -> AppRunState:
    """Apply the existing OperatorPolicy authorization without committing synthesis."""

    state = load_app_run(run_dir)
    _ensure_nonterminal(state, "request synthesis")
    if _active_attempt(state) is not None:
        raise ValueError("synthesis cannot be requested while an episode attempt is active")
    for episode in state.episodes:
        if episode.committed_attempt_id is not None:
            continue
        latest = episode.attempts[-1]
        if (
            latest.status in {"rejected", "failed", "cancelled", "expired"}
            and not latest.retry_exhaustion_released
        ):
            role_label = "Relation " if episode.role == "relation" else ""
            raise ValueError(
                f"unresolved {role_label}attempt must be retried before synthesis can be requested"
            )
    request = SynthesisControlRequest.model_validate(request.model_dump(mode="json"))
    authorize_synthesis_control(state.spec, request)
    request = _canonicalize_synthesis_request(state, request)
    state.synthesis_request = request
    if state.pending_terminal_action is not None:
        # A previously pending algorithmic ``run_complete`` may be waiting on
        # Relation recovery.  A newly authorized synthesis command changes the
        # intended terminal action, but only once a pending safe-boundary gate
        # already exists; a brand-new run must still reach its first Judge and
        # controller checkpoint.
        state.pending_terminal_action = "ready_for_synthesis"
        state.pending_terminal_reason = "authorized synthesis request is pending"
        state.pending_terminal_source = "authorized_synthesis"
    # The authorization records stop intent.  `next-episode` performs the
    # Relation readiness gate before a new sticky terminal action is committed.
    state.controller_action = "continue_controller"
    _save_state(run_dir, state)
    return state


def app_run_status(run_dir: str | Path) -> AppRunState:
    state = load_app_run(run_dir)
    active = _active_attempt(state)
    if active is not None and _attempt_expired(active[1]):
        _mark_expired(run_dir, state, active[1])
        _save_state(run_dir, state)
        try:
            _write_attempt_artifacts(run_dir, active[1])
        except Exception:
            pass
    if (
        state.controller_action in {"ready_for_synthesis", "run_complete"}
        and state.synthesis_readiness is None
    ):
        # Persisted pre-Relation terminal runs remain sticky and are never
        # migrated or reopened.  The status response labels the missing gate.
        state.relation_readiness_status = "legacy_unchecked"
    return state
