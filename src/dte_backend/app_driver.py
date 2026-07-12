"""Persistent backend protocol driven by the current Codex App main agent.

This module never launches Codex. It grants bounded logical episodes, persists
attempt lifecycle, accepts a complete structured result, and commits only
through :func:`commit_episode_result`.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import Field

from .control import authorize_synthesis_control
from .episode_adapter import build_executor_episode_request
from .episode_commit import EpisodeGraph, commit_episode_result
from .episode_models import CommitOutcome, EpisodeRequest, EpisodeResult, RuntimeLimits
from .models import DTEBaseModel, DTERunSpec, SearchNode, SynthesisControlRequest
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


def _request_artifact_path(run_dir: str | Path, request: EpisodeRequest) -> Path:
    return Path(run_dir) / "episodes" / request.episode_id / request.attempt_id / "request.json"


def _status_artifact_path(run_dir: str | Path, request: EpisodeRequest) -> Path:
    return Path(run_dir) / "episodes" / request.episode_id / request.attempt_id / "status.json"


def _write_attempt_artifacts(run_dir: str | Path, attempt: EpisodeAttemptRecord) -> None:
    _write_json_atomic(_request_artifact_path(run_dir, attempt.request), attempt.request.model_dump(mode="json"))
    _write_json_atomic(_status_artifact_path(run_dir, attempt.request), attempt.model_dump(mode="json"))


def next_app_episode(
    run_dir: str | Path,
    *,
    runtime_limits: RuntimeLimits | None = None,
    profile: Literal["legacy-explicit", "native-guided", "native-autonomous"] = "native-autonomous",
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

    parent = _select_executor_parent(state)
    if parent is None:
        if any(node.status == "frontier" for node in state.nodes):
            state.controller_action = "continue_controller"
            reason = "no frontier node currently has a controller-assigned expansion budget"
        else:
            state.controller_action = "ready_for_synthesis"
            reason = "no expandable frontier remains"
        _save_state(run_dir, state)
        return NextEpisodeOutcome(
            run_id=state.run_id,
            controller_action=state.controller_action,
            reason=reason,
        )

    graph = state.graph()
    limits = runtime_limits or RuntimeLimits(max_retries=1)
    request = build_executor_episode_request(
        graph,
        parent,
        run_id=state.run_id,
        iteration=len(state.episodes) + 1,
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
    granted_at = _now()
    deadline = None
    if limits.wall_clock_seconds is not None:
        deadline = granted_at + timedelta(seconds=limits.wall_clock_seconds)
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
    log = _event_log(run_dir)
    common = dict(
        run_id=state.run_id,
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        role=request.role,
        adapter_name="codex-app-main-agent",
        transport_name="current-app-runtime",
        profile=profile,
        status="granted",
        input_graph_revision=request.input_graph_revision,
        usage_source="unavailable",
    )
    log.emit("episode_granted", **common)
    common["status"] = "in_progress"
    log.emit("episode_started", **common)
    return NextEpisodeOutcome(
        run_id=state.run_id,
        controller_action="episode_required",
        request=request,
    )


def _result_payload(raw_result: EpisodeResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw_result, EpisodeResult):
        return raw_result.model_dump(mode="json")
    return dict(raw_result)


def _rejection_outcome(state: AppRunState, request: EpisodeRequest, reason: str) -> CommitOutcome:
    return CommitOutcome(
        accepted=False,
        episode_id=request.episode_id,
        graph_revision_before=state.graph_revision,
        graph_revision_after=state.graph_revision,
        rejection_reason=reason,
    )


def submit_app_episode_result(
    run_dir: str | Path,
    raw_result: EpisodeResult | Mapping[str, Any],
) -> SubmitEpisodeOutcome:
    """Validate lifecycle plus result, then commit through the sole graph boundary."""

    state = load_app_run(run_dir)
    payload = _result_payload(raw_result)
    episode_id = str(payload.get("episode_id", ""))
    attempt_id = str(payload.get("attempt_id", ""))
    episode = _find_episode(state, episode_id)
    attempt = _find_attempt(episode, attempt_id)
    result_path = Path(run_dir) / "episodes" / episode_id / attempt_id / "result.json"
    submitted_at = _now()
    attempt.submitted_at = _iso(submitted_at)
    wall_clock_ms = max(0, round((submitted_at - _parse_time(attempt.granted_at)).total_seconds() * 1000))
    log = _event_log(run_dir)
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
        returned_node_count=len(payload.get("structured_output", {}).get("nodes", []))
        if isinstance(payload.get("structured_output"), Mapping)
        else None,
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
            returned_node_count=outcome.accepted_node_count,
            accepted_node_count=outcome.accepted_node_count,
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
    parent_id = previous.request.parent_node_id
    if parent_id is None:
        raise ValueError("Executor retry is missing parent_node_id")
    parent = graph.node_by_id(parent_id)
    if parent is None or parent.status != "frontier":
        raise ValueError("retry parent is no longer an expandable committed node")
    limits = previous.request.runtime_limits.model_copy(deep=True)
    limits.selected_by = selected_by
    if wall_clock_seconds is not None:
        limits.wall_clock_seconds = wall_clock_seconds
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
