"""Persistent backend protocol driven by the current Codex App main agent.

This module never launches Codex. It grants bounded logical episodes, persists
attempt lifecycle, accepts a complete structured result, and commits only
through :func:`commit_episode_result`.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from .control import authorize_synthesis_control
from .embedding import EmbeddingProvider, get_embedding_provider
from .entropy import evaluate_entropy_state
from .episode_adapter import (
    build_executor_episode_request,
    build_judge_episode_request,
    build_relation_episode_request,
)
from .episode_commit import EpisodeGraph, commit_episode_result
from .episode_models import CommitOutcome, EpisodeRequest, EpisodeResult, RuntimeLimits
from .file_cache import FileDTECache
from .math_engine import allocate_frontier
from .models import DTEBaseModel, DTERunSpec, SearchNode, SynthesisControlRequest
from .novelty import estimate_frontier_kde_state
from .relation_candidates import (
    generate_blocking_relation_obligations,
    generate_relation_enrichment_candidates,
    refresh_relation_candidates,
    select_node_disjoint_relation_batch,
)
from .relation_models import (
    MergeApplicationRecord,
    ProvisionalSynthesisSelection,
    RelationCandidate,
    RelationRecord,
    SynthesisReadinessRecord,
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
    return datetime.fromisoformat(value)


class EpisodeAttemptRecord(DTEBaseModel):
    attempt_id: str
    attempt_number: int = Field(ge=1)
    status: AttemptStatus
    request: EpisodeRequest
    granted_at: str
    deadline_at: str | None = None
    submitted_at: str | None = None
    result_hash: str | None = None
    commit_outcome: CommitOutcome | None = None
    failure_reason: str | None = None
    superseded_from_status: AttemptStatus | None = None


class EpisodeLifecycleRecord(DTEBaseModel):
    episode_id: str
    run_id: str
    role: Literal["executor", "seed", "judge", "relation", "synthesis"]
    attempts: list[EpisodeAttemptRecord]
    committed_attempt_id: str | None = None


class AppRunState(DTEBaseModel):
    run_id: str
    spec: DTERunSpec
    nodes: list[SearchNode]
    graph_revision: int = Field(ge=0)
    node_revisions: dict[str, int]
    episodes: list[EpisodeLifecycleRecord] = Field(default_factory=list)
    active_episode_id: str | None = None
    active_attempt_id: str | None = None
    controller_action: ControllerAction = "continue_controller"
    synthesis_request: SynthesisControlRequest | None = None
    controller_iteration: int = Field(default=0, ge=0)
    previous_spatial_entropy: float | None = None
    relation_candidates: list[RelationCandidate] = Field(default_factory=list)
    relation_ledger: list[RelationRecord] = Field(default_factory=list)
    merge_applications: list[MergeApplicationRecord] = Field(default_factory=list)
    provisional_synthesis_selection: ProvisionalSynthesisSelection | None = None
    synthesis_readiness: SynthesisReadinessRecord | None = None
    relation_readiness_status: Literal["not_evaluated", "evaluated", "legacy_unchecked"] = "not_evaluated"
    pending_terminal_action: Literal["ready_for_synthesis", "run_complete"] | None = None
    pending_terminal_reason: str | None = None
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
        )

    def replace_graph(self, graph: EpisodeGraph) -> None:
        self.nodes = [node.model_copy(deep=True) for node in graph.nodes]
        self.graph_revision = graph.revision
        self.node_revisions = dict(graph.node_revisions)
        self.relation_candidates = [item.model_copy(deep=True) for item in graph.relation_candidates]
        self.relation_ledger = [item.model_copy(deep=True) for item in graph.relation_ledger]
        self.merge_applications = [item.model_copy(deep=True) for item in graph.merge_applications]


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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_state(run_dir: str | Path, state: AppRunState) -> None:
    state.updated_at = _iso()
    _write_json_atomic(_state_path(run_dir), state.model_dump(mode="json"))
    _write_relation_artifacts(run_dir, state)


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


def _relation_enrichment_pairs_committed(state: AppRunState) -> int:
    """Rebuild the run-level successful enrichment spend from durable ledger facts."""

    return len(
        {
            record.candidate_id
            for record in state.relation_ledger
            if record.scheduling_class == "enrichment"
        }
    )


def load_app_run(run_dir: str | Path) -> AppRunState:
    return AppRunState.model_validate_json(_state_path(run_dir).read_text(encoding="utf-8"))


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
    graph = EpisodeGraph(nodes=[node.model_copy(deep=True) for node in initial_nodes])
    created = _iso()
    state = AppRunState(
        run_id=run_id or str(uuid.uuid4()),
        spec=spec,
        nodes=graph.nodes,
        graph_revision=graph.revision,
        node_revisions=graph.node_revisions,
        controller_action="continue_controller",
        created_at=created,
        updated_at=created,
    )
    _save_state(run_dir, state)
    _event_log(run_dir).emit(
        "run_created",
        run_id=state.run_id,
        status="created",
        input_graph_revision=state.graph_revision,
        usage_source="unavailable",
    )
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


def _attempt_expired(attempt: EpisodeAttemptRecord) -> bool:
    return attempt.deadline_at is not None and _now() >= _parse_time(attempt.deadline_at)


def _mark_expired(run_dir: str | Path, state: AppRunState, attempt: EpisodeAttemptRecord) -> None:
    attempt.status = "expired"
    attempt.failure_reason = "attempt deadline elapsed before submission"
    state.active_episode_id = None
    state.active_attempt_id = None
    state.controller_action = "await_operator_decision"
    _event_log(run_dir).emit(
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
    )
    for node, log_density, uncertainty in zip(
        next_frontier,
        kde_state.log_density,
        kde_state.uncertainty,
    ):
        node.density = math.exp(log_density)
        node.uncertainty = uncertainty

    iteration = state.controller_iteration + 1
    entropy_state = evaluate_entropy_state(
        spatial_entropy=kde_state.spatial_entropy,
        previous_entropy=state.previous_spatial_entropy,
        iteration=iteration,
        min_iterations=state.spec.budget.min_iterations_before_synthesis,
        entropy_change_threshold=state.spec.budget.entropy_change_threshold,
        t_max=state.spec.budget.t_max,
    )
    allocations = allocate_frontier(
        next_nodes,
        allocation_mass_per_iteration=state.spec.budget.allocation_mass_per_iteration,
        max_children_per_iteration=state.spec.budget.max_children_per_iteration,
        tau=max(entropy_state.normalized_temperature, 0.05),
        c_explore=1.0,
        temperature=max(entropy_state.effective_temperature, 0.05),
    )
    allocation_by_id = {allocation.node_id: allocation for allocation in allocations}
    next_revisions = dict(state.node_revisions)
    for node in next_frontier:
        allocation = allocation_by_id[node.node_id]
        node.ucb_score = allocation.ucb_score
        node.expansion_budget = allocation.expansion_budget
        next_revisions[node.node_id] += 1

    state.nodes = next_nodes
    state.node_revisions = next_revisions
    state.graph_revision += 1
    state.controller_iteration = iteration
    state.previous_spatial_entropy = kde_state.spatial_entropy
    _event_log(run_dir).emit(
        "allocation_recorded",
        run_id=state.run_id,
        status="committed",
        input_graph_revision=state.graph_revision - 1,
        selected_node_count=len(next_frontier),
        allocated_child_count=sum(allocation.expansion_budget for allocation in allocations),
        spatial_entropy=kde_state.spatial_entropy,
        usage_source="unavailable",
    )
    if entropy_state.should_synthesize:
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
    log = _event_log(run_dir)
    if added_count:
        log.emit(
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
    log.emit("relation_blocking_inventory_evaluated", **inventory_fields)
    if readiness.blocking_inventory_complete:
        log.emit("relation_blocking_inventory_completed", **inventory_fields)
    log.emit(
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
        log.emit(
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
    runtime_limits: RuntimeLimits,
    profile: Literal["legacy-explicit", "native-guided", "native-autonomous"],
    entropy_plateau: bool = False,
) -> NextEpisodeOutcome:
    """Run the Relation gate before committing a new sticky terminal action."""

    state.pending_terminal_action = terminal_action
    state.pending_terminal_reason = terminal_reason
    readiness = _evaluate_relation_gate(run_dir, state, entropy_plateau=entropy_plateau)
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
        _event_log(run_dir).emit(
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

    state.pending_terminal_action = None
    state.pending_terminal_reason = None
    state.controller_action = terminal_action
    _save_state(run_dir, state)
    return NextEpisodeOutcome(
        run_id=state.run_id,
        controller_action=terminal_action,
        reason=terminal_reason,
    )


def _request_artifact_path(run_dir: str | Path, request: EpisodeRequest) -> Path:
    return Path(run_dir) / "episodes" / request.episode_id / request.attempt_id / "request.json"


def _status_artifact_path(run_dir: str | Path, request: EpisodeRequest) -> Path:
    return Path(run_dir) / "episodes" / request.episode_id / request.attempt_id / "status.json"


def _write_attempt_artifacts(run_dir: str | Path, attempt: EpisodeAttemptRecord) -> None:
    _write_json_atomic(_request_artifact_path(run_dir, attempt.request), attempt.request.model_dump(mode="json"))
    _write_json_atomic(_status_artifact_path(run_dir, attempt.request), attempt.model_dump(mode="json"))


def _grant_new_episode(
    run_dir: str | Path,
    state: AppRunState,
    request: EpisodeRequest,
    *,
    profile: str,
) -> NextEpisodeOutcome:
    granted_at = _now()
    deadline = None
    if request.runtime_limits.wall_clock_seconds is not None:
        deadline = granted_at + timedelta(seconds=request.runtime_limits.wall_clock_seconds)
    attempt = EpisodeAttemptRecord(
        attempt_id=request.attempt_id,
        attempt_number=1,
        status="in_progress",
        request=request,
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
    _write_attempt_artifacts(run_dir, attempt)
    _save_state(run_dir, state)
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
    log = _event_log(run_dir)
    log.emit("episode_granted", status="granted", **common)
    log.emit("episode_started", status="in_progress", **common)
    if request.role == "relation":
        scheduling_classes = {
            pair.scheduling_class
            for pair in (request.relation_payload.candidate_pairs if request.relation_payload else [])
        }
        log.emit(
            "relation_episode_granted",
            status="granted",
            selected_pair_count=(
                0 if request.relation_payload is None else len(request.relation_payload.candidate_pairs)
            ),
            **common,
        )
        if scheduling_classes == {"enrichment"}:
            committed = _relation_enrichment_pairs_committed(state)
            log.emit(
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
            _write_attempt_artifacts(run_dir, attempt)
            _save_state(run_dir, state)
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

    limits = runtime_limits or RuntimeLimits(max_retries=1)
    if state.controller_action in {"ready_for_synthesis", "run_complete"}:
        return NextEpisodeOutcome(
            run_id=state.run_id,
            controller_action=state.controller_action,
            reason="controller terminal action is sticky",
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
        if state.pending_terminal_action is not None:
            return _prepare_terminal_or_relation(
                run_dir,
                state,
                terminal_action=state.pending_terminal_action,
                terminal_reason=state.pending_terminal_reason or "controller intends to terminate",
                runtime_limits=limits,
                profile=profile,
            )

        parent = _select_executor_parent(state)
        if parent is not None:
            request = build_executor_episode_request(
                state.graph(),
                parent,
                run_id=state.run_id,
                iteration=max(1, state.controller_iteration),
                max_returned_children=min(
                    parent.expansion_budget,
                    state.spec.budget.max_children_per_iteration,
                ),
                objective=f"{state.spec.goal}: expand {parent.claim}",
                constraints=list(state.spec.constraints),
                native_orchestration_allowed=state.spec.allow_self_organized_executor,
                runtime_limits=limits,
                transport_hints={"profile": profile, "runtime": "current-codex-app"},
            )
            return _grant_new_episode(run_dir, state, request, profile=profile)

        if state.synthesis_request is not None:
            return _prepare_terminal_or_relation(
                run_dir,
                state,
                terminal_action="ready_for_synthesis",
                terminal_reason="authorized synthesis request is pending",
                runtime_limits=limits,
                profile=profile,
            )

        if state.controller_iteration >= state.spec.budget.max_iterations:
            terminal_action: ControllerAction = (
                "ready_for_synthesis" if state.spec.require_final_synthesis else "run_complete"
            )
            return _prepare_terminal_or_relation(
                run_dir,
                state,
                terminal_action=terminal_action,
                terminal_reason="maximum controller iterations reached",
                runtime_limits=limits,
                profile=profile,
            )

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

        action, reason = _progress_controller(
            run_dir,
            state,
            embedding_provider=embedding_provider,
        )
        if action != "continue_controller":
            return _prepare_terminal_or_relation(
                run_dir,
                state,
                terminal_action=action,
                terminal_reason=reason,
                runtime_limits=limits,
                profile=profile,
                entropy_plateau="entropy" in reason.casefold(),
            )
        state.controller_action = action
        _save_state(run_dir, state)


def _result_payload(raw_result: Any) -> dict[str, Any] | None:
    if isinstance(raw_result, EpisodeResult):
        return raw_result.model_dump(mode="json")
    if isinstance(raw_result, Mapping):
        return dict(raw_result)
    model_dump = getattr(raw_result, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return None


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
    _event_log(run_dir).emit(
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
    payload = _result_payload(raw_result)
    if payload is None:
        return _identity_rejection(
            run_dir,
            state,
            {},
            "episode result must be a mapping or a model with model_dump()",
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
    result_path = Path(run_dir) / "episodes" / episode_id / attempt_id / "result.json"
    submitted_at = _now()
    attempt.submitted_at = _iso(submitted_at)
    wall_clock_ms = max(0, round((submitted_at - _parse_time(attempt.granted_at)).total_seconds() * 1000))
    log = _event_log(run_dir)
    raw_output = payload.get("structured_output")
    returned_node_count = None
    returned_observation_count = None
    if isinstance(raw_output, Mapping):
        if isinstance(raw_output.get("nodes"), list):
            returned_node_count = len(raw_output["nodes"])
        if isinstance(raw_output.get("observations"), list):
            returned_observation_count = len(raw_output["observations"])
    log.emit(
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
    if _attempt_expired(attempt) and not invalid_lifecycle:
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
        log.emit(
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
            log.emit(
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
        _write_attempt_artifacts(run_dir, attempt)
        _save_state(run_dir, state)
        return SubmitEpisodeOutcome(
            run_id=state.run_id,
            episode_id=episode_id,
            attempt_id=attempt_id,
            commit_outcome=outcome,
            next_controller_action=state.controller_action,
        )

    attempt.status = "completed_uncommitted"
    _write_json_atomic(result_path, payload)
    graph = state.graph()
    outcome = commit_episode_result(graph, attempt.request, raw_result, telemetry=log)
    attempt.commit_outcome = outcome
    if outcome.accepted:
        state.replace_graph(graph)
        attempt.status = "committed"
        episode.committed_attempt_id = attempt_id
        parsed = raw_result if isinstance(raw_result, EpisodeResult) else EpisodeResult.model_validate(raw_result)
        attempt.result_hash = parsed.output_hash
        state.active_episode_id = None
        state.active_attempt_id = None
        state.controller_action = "continue_controller"
        log.emit(
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
                log.emit(
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
    else:
        attempt.status = "rejected"
        state.active_episode_id = None
        state.active_attempt_id = None
        state.controller_action = "await_operator_decision"
    _write_attempt_artifacts(run_dir, attempt)
    _save_state(run_dir, state)
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
    _write_attempt_artifacts(run_dir, attempt)
    _save_state(run_dir, state)
    _event_log(run_dir).emit(
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
    if _active_attempt(state) is not None:
        active_episode, active_attempt = _active_attempt(state)  # type: ignore[misc]
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
        raise ValueError("episode retry limit exhausted")
    previous.status = "superseded"
    previous.superseded_from_status = previous_status
    state.active_episode_id = None
    state.active_attempt_id = None

    graph = state.graph()
    limits = previous.request.runtime_limits.model_copy(deep=True)
    limits.selected_by = selected_by
    if wall_clock_seconds is not None:
        limits.wall_clock_seconds = wall_clock_seconds
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
            if candidate.candidate_id in candidate_ids and candidate.status == "granted"
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
        granted_at=_iso(granted_at),
        deadline_at=None if deadline is None else _iso(deadline),
    )
    episode.attempts.append(attempt)
    state.active_episode_id = episode_id
    state.active_attempt_id = attempt.attempt_id
    state.controller_action = "episode_required"
    _write_attempt_artifacts(run_dir, previous)
    _write_attempt_artifacts(run_dir, attempt)
    _save_state(run_dir, state)
    log = _event_log(run_dir)
    log.emit(
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
        log.emit(
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
    if _active_attempt(state) is not None:
        raise ValueError("synthesis cannot be requested while an episode attempt is active")
    authorize_synthesis_control(state.spec, request)
    state.synthesis_request = request
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
        _write_attempt_artifacts(run_dir, active[1])
        _save_state(run_dir, state)
    if (
        state.controller_action in {"ready_for_synthesis", "run_complete"}
        and state.synthesis_readiness is None
    ):
        # Persisted pre-Relation terminal runs remain sticky and are never
        # migrated or reopened.  The status response labels the missing gate.
        state.relation_readiness_status = "legacy_unchecked"
    return state
