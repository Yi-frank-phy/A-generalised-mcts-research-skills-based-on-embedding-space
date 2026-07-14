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
from .episode_adapter import build_executor_episode_request, build_judge_episode_request
from .episode_commit import EpisodeGraph, commit_episode_result
from .episode_models import CommitOutcome, EpisodeRequest, EpisodeResult, RuntimeLimits
from .file_cache import FileDTECache
from .math_engine import allocate_frontier
from .models import DTEBaseModel, DTERunSpec, SearchNode, SynthesisControlRequest
from .novelty import estimate_frontier_kde_state
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
    created_at: str
    updated_at: str

    def graph(self) -> EpisodeGraph:
        return EpisodeGraph(
            nodes=[node.model_copy(deep=True) for node in self.nodes],
            revision=self.graph_revision,
            node_revisions=dict(self.node_revisions),
        )

    def replace_graph(self, graph: EpisodeGraph) -> None:
        self.nodes = [node.model_copy(deep=True) for node in graph.nodes]
        self.graph_revision = graph.revision
        self.node_revisions = dict(graph.node_revisions)


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
    if state.synthesis_request is not None:
        state.controller_action = "ready_for_synthesis"
        _save_state(run_dir, state)
        return NextEpisodeOutcome(
            run_id=state.run_id,
            controller_action="ready_for_synthesis",
            reason="authorized synthesis request is pending",
        )

    # The loop consumes backend-only transitions. It never asks the App main
    # agent to interpret `continue_controller` or run controller mathematics.
    while True:
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

        if state.controller_iteration >= state.spec.budget.max_iterations:
            terminal_action: ControllerAction = (
                "ready_for_synthesis" if state.spec.require_final_synthesis else "run_complete"
            )
            state.controller_action = terminal_action
            _save_state(run_dir, state)
            return NextEpisodeOutcome(
                run_id=state.run_id,
                controller_action=terminal_action,
                reason="maximum controller iterations reached",
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
        state.controller_action = action
        _save_state(run_dir, state)
        if action != "continue_controller":
            return NextEpisodeOutcome(
                run_id=state.run_id,
                controller_action=action,
                reason=reason,
            )


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
            accepted_observation_count=0 if episode.role == "judge" else None,
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
            returned_observation_count=outcome.accepted_node_count if episode.role == "judge" else None,
            accepted_observation_count=outcome.accepted_node_count if episode.role == "judge" else None,
            usage_source="unavailable",
            schema_valid=True,
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
    else:
        raise ValueError(f"retry is not implemented for role={previous.request.role}")
    # A retry is another attempt of the same logical episode.
    request.episode_id = episode_id
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
    state.controller_action = "ready_for_synthesis"
    _save_state(run_dir, state)
    return state


def app_run_status(run_dir: str | Path) -> AppRunState:
    state = load_app_run(run_dir)
    active = _active_attempt(state)
    if active is not None and _attempt_expired(active[1]):
        _mark_expired(run_dir, state, active[1])
        _write_attempt_artifacts(run_dir, active[1])
        _save_state(run_dir, state)
    return state
