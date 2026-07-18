"""Deterministic read model and append-only feedback for persistent DTE runs.

The functions in this module deliberately avoid :func:`load_app_run` and
``EpisodeEventLog.read_events`` because both may repair mirrors or JSONL tails.
Summary construction is a pure projection over bytes already on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import uuid
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, get_args

from .app_driver import AppRunState, _validate_loaded_state
from .episode_models import (
    ExecutorEpisodeOutput,
    JudgeEpisodeOutput,
    RelationEpisodeOutput,
    canonical_json_bytes,
)
from .merge import resolve_merge_aliases
from .observability_models import (
    AllocationOutcomeRecordV1,
    AttemptObservabilityRecordV1,
    ControllerTrajectoryRecordV1,
    DescriptiveStatsV1,
    EpisodeFunnelV1,
    EpisodeObservabilityRecordV1,
    FeedbackLedgerDiagnosticsV1,
    FeedbackRecordV1,
    FeedbackSource,
    FeedbackTargetType,
    JudgeNodePosteriorRecordV1,
    JudgeOutcomeSummaryV1,
    NodeAllocationHistoryRecordV1,
    NodeFunnelV1,
    NodeLineageRecordV1,
    NodeRelationOutcomeRecordV1,
    ObservabilityDataQualityV1,
    ObservabilityExportResultV1,
    ObservabilityExportSkippedRunV1,
    RejectionCategory,
    RejectionCategoryCountV1,
    RejectionSummaryV1,
    RelationOutcomeSummaryV1,
    RelationReasonYieldV1,
    RoleEpisodeFunnelV1,
    RunBudgetSnapshotV1,
    RunIdentityObservabilityV1,
    RunObservabilitySummaryV1,
    RuntimeAggregateDiagnosticsV1,
)
from .relation_models import RelationCandidateReason


EXPORT_SCHEMA_VERSION = "dte-observability-export.v1"
FEEDBACK_LEDGER_PATH = Path("observability") / "feedback.jsonl"


class ObservabilityReadError(ValueError):
    """The minimum persistent facts needed for a run summary are unreadable."""


class DuplicateFeedbackError(ValueError):
    """A feedback ID already exists in the append-only ledger."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _wall_clock_ms(granted_at: str | None, submitted_at: str | None) -> int | None:
    granted = _parse_time(granted_at)
    submitted = _parse_time(submitted_at)
    if granted is None or submitted is None or submitted < granted:
        return None
    return max(0, round((submitted - granted).total_seconds() * 1000))


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _descriptive(values: Iterable[float]) -> DescriptiveStatsV1:
    samples = list(values)
    if not samples:
        return DescriptiveStatsV1(count=0)
    return DescriptiveStatsV1(
        count=len(samples),
        minimum=min(samples),
        maximum=max(samples),
        mean=statistics.fmean(samples),
        median=statistics.median(samples),
    )


def allocation_decision_id(run_id: str, controller_iteration: int, node_id: str) -> str:
    """Return the stable public identity of one parent allocation decision."""

    payload = f"{run_id}\x1f{controller_iteration}\x1f{node_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"allocation-{digest}"


def _load_state_read_only(
    run_dir: str | Path,
) -> tuple[
    AppRunState,
    str | None,
    bool,
    str | None,
    list[str],
    dict[str, Any],
]:
    """Parse current and reconstructable legacy App state without side effects."""

    state_path = Path(run_dir) / "app_run_state.json"
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ObservabilityReadError(f"missing App run state: {state_path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservabilityReadError(f"invalid App run state JSON: {state_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ObservabilityReadError("App run state JSON must be an object")

    payload = deepcopy(raw)
    original_schema = payload.get("state_schema_version")
    reconstructed: list[str] = []
    if original_schema != "app-run-state.v2":
        payload["state_schema_version"] = "app-run-state.v2"
        reconstructed.append("state_schema_version")
    if "spec_hash" not in payload and isinstance(payload.get("spec"), dict):
        payload["spec_hash"] = hashlib.sha256(
            canonical_json_bytes(payload["spec"])
        ).hexdigest()
        reconstructed.append("spec_hash")
    if "initial_nodes_hash" not in payload and isinstance(
        payload.get("initial_nodes"), list
    ):
        payload["initial_nodes_hash"] = hashlib.sha256(
            canonical_json_bytes(payload["initial_nodes"])
        ).hexdigest()
        reconstructed.append("initial_nodes_hash")

    try:
        state = AppRunState.model_validate(payload)
    except Exception as exc:
        raise ObservabilityReadError(
            f"App run state is not reconstructable by the v1 read model: {exc}"
        ) from exc

    state_validation_error = None
    try:
        _validate_loaded_state(state)
    except Exception as exc:
        # Observability is allowed to expose partial legacy state. It never
        # treats this as controller-valid or writes the reconstructed payload.
        state_validation_error = str(exc)
    partial = bool(reconstructed or state_validation_error)
    return (
        state,
        original_schema if isinstance(original_schema, str) else None,
        partial,
        state_validation_error,
        reconstructed,
        raw,
    )


def _read_jsonl_objects_read_only(
    path: Path,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Read valid JSON objects without repairing or quarantining any bytes."""

    if not path.exists():
        return [], 0, False
    raw = path.read_bytes()
    physical = raw.split(b"\n")
    nonempty_indices = [index for index, line in enumerate(physical) if line.rstrip(b"\r")]
    final_nonempty = nonempty_indices[-1] if nonempty_indices else None
    records: list[dict[str, Any]] = []
    malformed_interior = 0
    corrupt_tail = False
    for index, line in enumerate(physical):
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line:
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index == final_nonempty:
                corrupt_tail = True
            else:
                malformed_interior += 1
            continue
        if not isinstance(value, dict):
            if index == final_nonempty:
                corrupt_tail = True
            else:
                malformed_interior += 1
            continue
        records.append(value)
    return records, malformed_interior, corrupt_tail


def _read_telemetry_read_only(
    run_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[str], list[str], bool, bool]:
    path = Path(run_dir) / "episode_events.jsonl"
    records, malformed_interior, corrupt_tail = _read_jsonl_objects_read_only(path)
    recoverable: list[str] = []
    if malformed_interior:
        recoverable.append(
            f"episode_events.jsonl contains {malformed_interior} malformed interior line(s)"
        )
    required = {
        "timestamp",
        "event_type",
        "run_id",
        "episode_id",
        "attempt_id",
        "role",
        "status",
        "input_graph_revision",
        "usage_source",
    }
    missing_fields = sorted(
        {
            field
            for record in records
            for field in required
            if field not in record
        }
    )
    deduplicated: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for record in records:
        event_id = record.get("event_id")
        if isinstance(event_id, str) and event_id:
            if event_id in seen_event_ids:
                recoverable.append(f"duplicate telemetry event_id ignored: {event_id}")
                continue
            seen_event_ids.add(event_id)
        deduplicated.append(record)
    repaired = path.with_suffix(path.suffix + ".corrupt").exists()
    return deduplicated, missing_fields, sorted(set(recoverable)), corrupt_tail, repaired


def read_feedback_ledger(
    run_dir: str | Path,
) -> tuple[list[FeedbackRecordV1], FeedbackLedgerDiagnosticsV1]:
    """Read valid feedback records without changing the ledger or its tail."""

    path = Path(run_dir) / FEEDBACK_LEDGER_PATH
    raw_records, malformed_interior, corrupt_tail = _read_jsonl_objects_read_only(path)
    records: list[FeedbackRecordV1] = []
    schema_invalid = 0
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    for raw in raw_records:
        try:
            record = FeedbackRecordV1.model_validate(raw)
        except Exception:
            schema_invalid += 1
            continue
        if record.feedback_id in seen:
            duplicate_ids.append(record.feedback_id)
            continue
        seen.add(record.feedback_id)
        records.append(record)
    return records, FeedbackLedgerDiagnosticsV1(
        path_present=path.exists(),
        valid_record_count=len(records),
        malformed_interior_line_count=malformed_interior + schema_invalid,
        corrupt_tail_detected=corrupt_tail,
        corrupt_tail_repaired=path.with_suffix(path.suffix + ".corrupt").exists(),
        duplicate_feedback_ids=sorted(set(duplicate_ids)),
    )


def _replace_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _repair_jsonl_tail_for_append(path: Path) -> bool:
    """Apply the telemetry-style damaged-tail policy before a feedback append."""

    if not path.exists():
        return False
    raw = path.read_bytes()
    if not raw:
        return False
    tail_end = len(raw)
    while tail_end > 0 and raw[tail_end - 1 : tail_end] == b"\n":
        tail_end -= 1
        if tail_end > 0 and raw[tail_end - 1 : tail_end] == b"\r":
            tail_end -= 1
    if tail_end == 0:
        return False
    tail_start = raw.rfind(b"\n", 0, tail_end) + 1
    tail_line = raw[tail_start:tail_end]
    try:
        decoded = json.loads(tail_line.decode("utf-8"))
        complete = isinstance(decoded, dict)
    except (UnicodeDecodeError, json.JSONDecodeError):
        complete = False
    if complete:
        if not raw.endswith(b"\n"):
            _replace_bytes_atomic(path, raw + b"\n")
        return False

    corrupt_tail = raw[tail_start:]
    quarantine = path.with_suffix(path.suffix + ".corrupt")
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    with quarantine.open("ab") as handle:
        handle.write(corrupt_tail)
        if not corrupt_tail.endswith(b"\n"):
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_bytes_atomic(path, raw[:tail_start])
    return True


def _feedback_target_ids(state: AppRunState) -> dict[FeedbackTargetType, set[str]]:
    return {
        "run": {state.run_id},
        "episode": {episode.episode_id for episode in state.episodes},
        "attempt": {
            attempt.attempt_id
            for episode in state.episodes
            for attempt in episode.attempts
        },
        "node": {node.node_id for node in state.nodes},
        "relation_record": {
            record.relation_record_id for record in state.relation_ledger
        },
        "merge_application": {
            application.merge_application_id
            for application in state.merge_applications
        },
        "allocation_decision": {
            allocation_decision_id(state.run_id, record.iteration, node_id)
            for record in state.controller_iteration_records
            for node_id, budget in record.allocations.items()
            if budget > 0
        },
    }


def record_feedback(
    run_dir: str | Path,
    *,
    target_type: FeedbackTargetType,
    metric: str,
    source: FeedbackSource,
    target_id: str | None = None,
    score: float | None = None,
    label: str | None = None,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
    feedback_id: str | None = None,
    timestamp: str | None = None,
) -> FeedbackRecordV1:
    """Validate and durably append one evaluation without touching run state."""

    state, _, _, _, _, _ = _load_state_read_only(run_dir)
    record = FeedbackRecordV1(
        feedback_id=feedback_id or str(uuid.uuid4()),
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        run_id=state.run_id,
        target_type=target_type,
        target_id=target_id,
        metric=metric,
        score=score,
        label=label,
        comment=comment,
        source=source,
        metadata=metadata,
    )
    target_ids = _feedback_target_ids(state)
    if target_type != "run" and record.target_id not in target_ids[target_type]:
        raise ValueError(
            f"feedback target does not exist: {target_type}={record.target_id!r}"
        )

    path = Path(run_dir) / FEEDBACK_LEDGER_PATH
    _repair_jsonl_tail_for_append(path)
    existing, diagnostics = read_feedback_ledger(run_dir)
    if diagnostics.duplicate_feedback_ids:
        raise ValueError("feedback ledger contains duplicate feedback IDs")
    if diagnostics.malformed_interior_line_count:
        raise ValueError(
            "feedback ledger contains invalid existing records; refusing to append"
        )
    if any(item.feedback_id == record.feedback_id for item in existing):
        raise DuplicateFeedbackError(
            f"feedback_id already exists: {record.feedback_id}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return record


def classify_rejection_reason(
    reason: str | None,
    *,
    controller_field_violation_count: int = 0,
) -> RejectionCategory:
    """Classify backend rejection strings by an explicit deterministic table."""

    value = (reason or "").casefold()
    if "node-disjoint" in value or "relation overlap" in value or "overlapping" in value:
        return "relation_overlap"
    if (
        "merge provenance" in value
        or "absorbed node" in value
        or "canonical target" in value
        or "alias cycle" in value
        or "merge-projected" in value
    ):
        return "merge_provenance_conflict"
    if "stale" in value or "revision mismatch" in value or "revisions mismatch" in value:
        return "stale_revision"
    if "exceeds grant" in value or "over-grant" in value:
        return "over_grant"
    if "duplicate" in value or "collision" in value:
        return "duplicate_output"
    if (
        "expired" in value
        or "deadline" in value
        or "timed_out" in value
        or "timed out" in value
        or "timeout" in value
    ):
        return "timeout_expire"
    if any(
        marker in value
        for marker in (
            "id mismatch",
            "identity",
            "role mismatch",
            "ungranted",
            "omitted granted",
            "pair mismatch",
            "unknown episode",
            "unknown attempt",
        )
    ):
        return "identity_mismatch"
    if any(
        marker in value
        for marker in (
            "lifecycle",
            "status=",
            "cannot commit",
            "terminal action is sticky",
            "already committed",
            "cancelled attempt",
            "failed attempt",
            "superseded attempt",
        )
    ):
        return "lifecycle_rejection"
    if controller_field_violation_count > 0 or "controller-owned" in value:
        return "controller_owned_field_violation"
    if any(
        marker in value
        for marker in (
            "schema validation",
            "json detachment",
            "must be a mapping",
            "output hash mismatch",
            "hash validation failed",
            "output schema version mismatch",
            "wrong structured output schema",
        )
    ):
        return "schema_rejection"
    return "other"


def _event_index(
    events: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        episode_id = event.get("episode_id")
        attempt_id = event.get("attempt_id")
        if isinstance(episode_id, str) and isinstance(attempt_id, str):
            indexed[(episode_id, attempt_id)].append(event)
    return indexed


def _attempt_counts_from_events(
    events: list[dict[str, Any]], event_type: str, field: str
) -> int | None:
    values = [
        event.get(field)
        for event in events
        if event.get("event_type") == event_type
        and isinstance(event.get(field), int)
    ]
    return values[-1] if values else None


def _build_episode_records(
    state: AppRunState,
    events: list[dict[str, Any]],
) -> list[EpisodeObservabilityRecordV1]:
    indexed_events = _event_index(events)
    output: list[EpisodeObservabilityRecordV1] = []
    for episode in state.episodes:
        attempts: list[AttemptObservabilityRecordV1] = []
        episode_created: list[str] = []
        selected: set[str] = set()
        relation_ids: list[str] = []
        for attempt in episode.attempts:
            request = attempt.request
            selected.update(request.selected_node_revisions)
            attempt_events = indexed_events.get((episode.episode_id, attempt.attempt_id), [])
            result = attempt.committed_result
            returned_node_count: int | None = None
            returned_observation_count: int | None = None
            diagnostics: RuntimeAggregateDiagnosticsV1 | None = None
            if result is not None:
                # Legacy envelopes allowed arbitrary opaque internal metadata.
                # The v1 read model exposes only the explicit aggregate fields,
                # never worker names, prompts, traces, or transcripts.
                diagnostics = RuntimeAggregateDiagnosticsV1.model_validate(
                    result.runtime_diagnostics.model_dump(
                        mode="json",
                        exclude={"internal_subagent_metadata"},
                    )
                )
                structured = result.structured_output
                if isinstance(structured, ExecutorEpisodeOutput):
                    returned_node_count = len(structured.nodes)
                    episode_created.extend(node.node_id for node in structured.nodes)
                elif isinstance(structured, (JudgeEpisodeOutput, RelationEpisodeOutput)):
                    returned_observation_count = len(structured.observations)
            if returned_node_count is None:
                returned_node_count = _attempt_counts_from_events(
                    attempt_events, "episode_submitted", "returned_node_count"
                )
            if returned_observation_count is None:
                returned_observation_count = _attempt_counts_from_events(
                    attempt_events, "episode_submitted", "returned_observation_count"
                )
            if request.relation_payload is not None:
                relation_ids.extend(
                    pair.candidate_id for pair in request.relation_payload.candidate_pairs
                )
            reason = (
                attempt.commit_outcome.rejection_reason
                if attempt.commit_outcome is not None
                and not attempt.commit_outcome.accepted
                else attempt.failure_reason
            )
            violation_count = max(
                [
                    event.get("controller_field_violation_count", 0)
                    for event in attempt_events
                    if isinstance(event.get("controller_field_violation_count"), int)
                ]
                or [0]
            )
            attempts.append(
                AttemptObservabilityRecordV1(
                    attempt_id=attempt.attempt_id,
                    attempt_number=attempt.attempt_number,
                    status=attempt.status,
                    superseded_from_status=attempt.superseded_from_status,
                    granted_at=attempt.granted_at,
                    deadline_at=attempt.deadline_at,
                    submitted_at=attempt.submitted_at,
                    wall_clock_ms=_wall_clock_ms(
                        attempt.granted_at, attempt.submitted_at
                    ),
                    selected_node_ids=sorted(request.selected_node_revisions),
                    returned_node_count=returned_node_count,
                    accepted_node_count=(
                        None
                        if attempt.commit_outcome is None
                        else attempt.commit_outcome.accepted_node_count
                    ),
                    returned_observation_count=returned_observation_count,
                    rejection_reason=reason,
                    rejection_category=(
                        None
                        if reason is None
                        else classify_rejection_reason(
                            reason,
                            controller_field_violation_count=violation_count,
                        )
                    ),
                    runtime_diagnostics=diagnostics,
                )
            )
        lifecycle_status = (
            "committed"
            if episode.committed_attempt_id is not None
            else episode.attempts[-1].status
        )
        output.append(
            EpisodeObservabilityRecordV1(
                episode_id=episode.episode_id,
                run_id=episode.run_id,
                role=episode.role,
                lifecycle_status=lifecycle_status,
                attempt_count=len(episode.attempts),
                retry_count=max(0, len(episode.attempts) - 1),
                committed_attempt_id=episode.committed_attempt_id,
                selected_node_ids=sorted(selected),
                created_node_ids=sorted(set(episode_created)),
                relation_candidate_ids=sorted(set(relation_ids)),
                attempts=attempts,
            )
        )
    return output


def _role_funnel(
    episodes: list[EpisodeObservabilityRecordV1], role: str
) -> RoleEpisodeFunnelV1:
    selected = [episode for episode in episodes if episode.role == role]
    attempts = [attempt for episode in selected for attempt in episode.attempts]

    def lifecycle_count(status: str) -> int:
        return sum(
            attempt.status == status or attempt.superseded_from_status == status
            for attempt in attempts
        )

    wall_samples = [
        attempt.wall_clock_ms
        for attempt in attempts
        if attempt.wall_clock_ms is not None
    ]
    count = len(attempts)
    committed = sum(attempt.status == "committed" for attempt in attempts)
    rejected = lifecycle_count("rejected")
    retried = sum(attempt.attempt_number > 1 for attempt in attempts)
    return RoleEpisodeFunnelV1(
        episode_count=len(selected),
        attempt_count=count,
        granted_attempt_count=count,
        started_attempt_count=count,
        in_progress_attempt_count=sum(
            attempt.status in {"granted", "in_progress", "completed_uncommitted"}
            for attempt in attempts
        ),
        submitted_attempt_count=sum(attempt.submitted_at is not None for attempt in attempts),
        committed_attempt_count=committed,
        rejected_attempt_count=rejected,
        failed_attempt_count=lifecycle_count("failed"),
        cancelled_attempt_count=lifecycle_count("cancelled"),
        expired_attempt_count=lifecycle_count("expired"),
        superseded_attempt_count=sum(attempt.status == "superseded" for attempt in attempts),
        retried_attempt_count=retried,
        retried_episode_count=sum(episode.retry_count > 0 for episode in selected),
        commit_rate=_ratio(committed, count),
        rejection_rate=_ratio(rejected, count),
        retry_rate=_ratio(retried, count),
        wall_clock_sample_count=len(wall_samples),
        wall_clock_total_ms=sum(wall_samples) if wall_samples else None,
        wall_clock_mean_ms=(statistics.fmean(wall_samples) if wall_samples else None),
        wall_clock_median_ms=(statistics.median(wall_samples) if wall_samples else None),
    )


def _committed_attempt(state: AppRunState, episode: Any) -> Any | None:
    if episode.committed_attempt_id is None:
        return None
    return next(
        (
            attempt
            for attempt in episode.attempts
            if attempt.attempt_id == episode.committed_attempt_id
        ),
        None,
    )


def _committed_judge_observations(
    state: AppRunState,
    recoverable: list[str],
) -> dict[str, tuple[Any, Any, Any]]:
    """Index authoritative committed Judge observations by node identity."""

    output: dict[str, tuple[Any, Any, Any]] = {}
    for episode in state.episodes:
        if episode.role != "judge":
            continue
        attempt = _committed_attempt(state, episode)
        if attempt is None or attempt.committed_result is None:
            continue
        structured = attempt.committed_result.structured_output
        if not isinstance(structured, JudgeEpisodeOutput):
            continue
        for observation in structured.observations:
            if observation.node_id in output:
                recoverable.append(
                    "multiple committed Judge observations claim node "
                    f"{observation.node_id}"
                )
                continue
            output[observation.node_id] = (episode, attempt, observation)
    return output


def _creation_and_children(
    state: AppRunState,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, Any]]:
    creation: dict[str, dict[str, Any]] = {
        node.node_id: {
            "episode_id": None,
            "attempt_id": None,
            "graph_revision": 0,
            "parent_ids": list(node.parent_ids),
        }
        for node in state.initial_nodes
    }
    children: dict[str, set[str]] = defaultdict(set)
    for node in state.initial_nodes:
        for parent_id in node.parent_ids:
            children[parent_id].add(node.node_id)
    expansions: dict[str, Any] = {}
    for episode in state.episodes:
        if episode.role != "executor":
            continue
        attempt = _committed_attempt(state, episode)
        if attempt is None or attempt.committed_result is None:
            continue
        structured = attempt.committed_result.structured_output
        if not isinstance(structured, ExecutorEpisodeOutput):
            continue
        parent_id = attempt.request.parent_node_id
        if parent_id is not None:
            expansions[parent_id] = attempt
        graph_revision = (
            None
            if attempt.commit_outcome is None
            else attempt.commit_outcome.graph_revision_after
        )
        for candidate in structured.nodes:
            creation[candidate.node_id] = {
                "episode_id": episode.episode_id,
                "attempt_id": attempt.attempt_id,
                "graph_revision": graph_revision,
                "parent_ids": list(candidate.parent_ids),
            }
            for candidate_parent in candidate.parent_ids:
                children[candidate_parent].add(candidate.node_id)
    return creation, children, expansions


def _descendants(node_id: str, children: Mapping[str, set[str]]) -> set[str]:
    result: set[str] = set()
    stack = list(children.get(node_id, set()))
    while stack:
        child = stack.pop()
        if child == node_id or child in result:
            continue
        result.add(child)
        stack.extend(children.get(child, set()) - result)
    return result


def _produced_children_by_parent(state: AppRunState) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for episode in state.episodes:
        if episode.role != "executor":
            continue
        attempt = _committed_attempt(state, episode)
        if (
            attempt is None
            or attempt.request.parent_node_id is None
            or attempt.committed_result is None
            or not isinstance(
                attempt.committed_result.structured_output, ExecutorEpisodeOutput
            )
        ):
            continue
        output[attempt.request.parent_node_id].update(
            node.node_id
            for node in attempt.committed_result.structured_output.nodes
        )
    return output


def _relation_outcomes_by_node(state: AppRunState) -> dict[str, list[NodeRelationOutcomeRecordV1]]:
    records_by_id = {
        record.relation_record_id: record for record in state.relation_ledger
    }
    output: dict[str, list[NodeRelationOutcomeRecordV1]] = defaultdict(list)
    for candidate in state.relation_candidates:
        record = records_by_id.get(candidate.resolved_relation_record_id or "")
        for node_id, other_id in (
            (candidate.left_node_id, candidate.right_node_id),
            (candidate.right_node_id, candidate.left_node_id),
        ):
            output[node_id].append(
                NodeRelationOutcomeRecordV1(
                    candidate_id=candidate.candidate_id,
                    relation_record_id=(None if record is None else record.relation_record_id),
                    other_node_id=other_id,
                    candidate_reason=candidate.candidate_reason,
                    scheduling_class=candidate.scheduling_class,
                    relation_type=(None if record is None else record.relation_type),
                    material_to_synthesis=candidate.material_to_synthesis,
                    disclosure_required=(None if record is None else record.disclosure_required),
                )
            )
    for records in output.values():
        records.sort(key=lambda item: (item.candidate_id, item.other_node_id))
    return output


def _allocation_history_by_node(
    state: AppRunState,
) -> dict[str, list[NodeAllocationHistoryRecordV1]]:
    output: dict[str, list[NodeAllocationHistoryRecordV1]] = defaultdict(list)
    for record in state.controller_iteration_records:
        for node_id, budget in record.allocations.items():
            output[node_id].append(
                NodeAllocationHistoryRecordV1(
                    allocation_decision_id=allocation_decision_id(
                        state.run_id, record.iteration, node_id
                    ),
                    controller_iteration=record.iteration,
                    input_graph_revision=record.input_graph_revision,
                    output_graph_revision=record.output_graph_revision,
                    allocated_expansion_budget=budget,
                    ucb_score=record.ucb_scores.get(node_id),
                    spatial_entropy=record.spatial_entropy,
                )
            )
    return output


def _safe_alias_map(
    state: AppRunState,
    recoverable: list[str],
) -> dict[str, str]:
    try:
        return resolve_merge_aliases(
            state.merge_applications,
            committed_node_ids={node.node_id for node in state.nodes},
        )
    except Exception as exc:
        recoverable.append(f"merge alias reconstruction failed: {exc}")
        return {}


def _build_lineage_and_allocations(
    state: AppRunState,
    recoverable: list[str],
) -> tuple[list[NodeLineageRecordV1], list[AllocationOutcomeRecordV1], NodeFunnelV1]:
    creation, children, expansions = _creation_and_children(state)
    histories = _allocation_history_by_node(state)
    relation_by_node = _relation_outcomes_by_node(state)
    aliases = _safe_alias_map(state, recoverable)
    judge_observations = _committed_judge_observations(state, recoverable)
    selected = set(
        []
        if state.provisional_synthesis_selection is None
        else state.provisional_synthesis_selection.selected_node_ids
    )
    nodes_by_id = {node.node_id: node for node in state.nodes}
    produced_children = _produced_children_by_parent(state)
    lineage: list[NodeLineageRecordV1] = []
    for node in sorted(state.nodes, key=lambda item: item.node_id):
        origin = creation.get(
            node.node_id,
            {
                "episode_id": None,
                "attempt_id": None,
                "graph_revision": None,
                "parent_ids": list(node.parent_ids),
            },
        )
        history = histories.get(node.node_id, [])
        judge_owner = judge_observations.get(node.node_id)
        if judge_owner is None:
            judge_score = None
            judge_reasoning = None
            judge_risks: list[str] = []
            judge_uncertainty_evidence: list[str] = []
            judge_episode_id = None
            judge_attempt_id = None
            if (
                node.score is not None
                or node.judge_reasoning is not None
                or node.judge_risks
                or node.judge_uncertainty_evidence
                or node.judge_result_provenance is not None
            ):
                recoverable.append(
                    f"node {node.node_id} has Judge-like graph fields without a "
                    "committed Judge observation"
                )
        else:
            judge_episode, judge_attempt, judge_observation = judge_owner
            judge_score = judge_observation.score
            judge_reasoning = judge_observation.reasoning
            judge_risks = list(judge_observation.risks)
            judge_uncertainty_evidence = list(
                judge_observation.uncertainty_evidence
            )
            judge_episode_id = judge_episode.episode_id
            judge_attempt_id = judge_attempt.attempt_id
            if (
                node.score != judge_score
                or node.judge_reasoning != judge_reasoning
                or node.judge_risks != judge_risks
                or node.judge_uncertainty_evidence
                != judge_uncertainty_evidence
            ):
                recoverable.append(
                    f"node {node.node_id} Judge graph fields disagree with its "
                    "committed Judge observation"
                )
        descendants = _descendants(node.node_id, children)
        lineage.append(
            NodeLineageRecordV1(
                node_id=node.node_id,
                node_type=node.node_type,
                parent_ids=list(origin["parent_ids"]),
                creation_episode_id=origin["episode_id"],
                creation_attempt_id=origin["attempt_id"],
                creation_graph_revision=origin["graph_revision"],
                current_node_revision=state.node_revisions.get(node.node_id),
                judge_score=judge_score,
                judge_reasoning=judge_reasoning,
                judge_risks=judge_risks,
                judge_uncertainty_evidence=judge_uncertainty_evidence,
                judge_episode_id=judge_episode_id,
                judge_attempt_id=judge_attempt_id,
                allocation_history=history,
                total_expansion_budget_granted=sum(
                    item.allocated_expansion_budget for item in history
                ),
                expansion_episode_count=int(node.node_id in expansions),
                children_committed=sorted(children.get(node.node_id, set())),
                descendant_count=len(descendants),
                selected_for_synthesis=node.node_id in selected,
                merged=node.status == "merged" or node.node_id in aliases,
                canonical_target=aliases.get(node.node_id),
                relation_outcomes=relation_by_node.get(node.node_id, []),
                current_status=node.status,
            )
        )

    conflict_records = [
        record for record in state.relation_ledger if record.relation_type == "conflict"
    ]
    lineage_by_id = {item.node_id: item for item in lineage}
    allocation_outcomes: list[AllocationOutcomeRecordV1] = []
    for record in state.controller_iteration_records:
        for parent_id, budget in sorted(record.allocations.items()):
            if budget <= 0:
                continue
            direct = set(produced_children.get(parent_id, set()))
            descendants = _descendants(parent_id, children)
            selected_descendants = descendants & selected
            conflicts = {
                relation.relation_record_id
                for relation in conflict_records
                if relation.left_node_id in descendants
                or relation.right_node_id in descendants
            }
            allocation_outcomes.append(
                AllocationOutcomeRecordV1(
                    allocation_decision_id=allocation_decision_id(
                        state.run_id, record.iteration, parent_id
                    ),
                    parent_node_id=parent_id,
                    controller_iteration=record.iteration,
                    input_graph_revision=record.input_graph_revision,
                    output_graph_revision=record.output_graph_revision,
                    allocated_expansion_budget=budget,
                    actual_committed_children=len(direct),
                    unused_granted_capacity=max(0, budget - len(direct)),
                    direct_children_later_judged=sum(
                        lineage_by_id[child].judge_score is not None
                        for child in direct
                        if child in lineage_by_id
                    ),
                    direct_children_later_selected=len(direct & selected),
                    descendant_count=len(descendants),
                    selected_descendant_count=len(selected_descendants),
                    merged_descendant_count=sum(
                        nodes_by_id[item].status == "merged"
                        for item in descendants
                        if item in nodes_by_id
                    ),
                    relation_conflicts_involving_descendants=len(conflicts),
                    committed_child_yield=_ratio(len(direct), budget),
                    selected_descendant_yield=_ratio(
                        len(selected_descendants), len(descendants)
                    ),
                )
            )

    committed_children = {
        node_id for node_id, origin in creation.items() if origin["episode_id"] is not None
    }
    positive_nodes = {
        node_id
        for record in state.controller_iteration_records
        for node_id, budget in record.allocations.items()
        if budget > 0
    }
    funnel = NodeFunnelV1(
        node_count=len(state.nodes),
        unique_node_count=len(nodes_by_id),
        initial_node_count=len(state.initial_nodes),
        all_committed_node_count=len(state.nodes),
        judged_node_count=sum(item.judge_score is not None for item in lineage),
        positive_allocation_unique_node_count=len(positive_nodes),
        parent_expansion_count=len(expansions),
        expanded_parent_unique_node_count=len(expansions),
        committed_executor_child_count=len(committed_children),
        frontier_node_count=sum(node.status == "frontier" for node in state.nodes),
        closed_node_count=sum(node.status == "closed" for node in state.nodes),
        merged_node_count=sum(node.status == "merged" for node in state.nodes),
        provisional_synthesis_selected_node_count=len(selected),
    )
    return lineage, allocation_outcomes, funnel


def _build_judge_summary(
    state: AppRunState,
    lineage: list[NodeLineageRecordV1],
) -> JudgeOutcomeSummaryV1:
    selected = {
        item.node_id for item in lineage if item.selected_for_synthesis
    }
    conflict_nodes = {
        node_id
        for record in state.relation_ledger
        if record.relation_type == "conflict"
        for node_id in (record.left_node_id, record.right_node_id)
    }
    records: list[JudgeNodePosteriorRecordV1] = []
    for lineage_record in lineage:
        if lineage_record.judge_score is None:
            continue
        records.append(
            JudgeNodePosteriorRecordV1(
                node_id=lineage_record.node_id,
                score=lineage_record.judge_score,
                later_received_positive_allocation=any(
                    item.allocated_expansion_budget > 0
                    for item in lineage_record.allocation_history
                ),
                later_expanded=lineage_record.expansion_episode_count > 0,
                later_provisional_selected=lineage_record.node_id in selected,
                later_merged=lineage_record.merged,
                later_involved_in_conflict=lineage_record.node_id in conflict_nodes,
            )
        )
    selected_scores = [record.score for record in records if record.node_id in selected]
    nonselected_scores = [record.score for record in records if record.node_id not in selected]
    threshold = 0.75
    high = [record for record in records if record.score >= threshold]
    return JudgeOutcomeSummaryV1(
        score_distribution=_descriptive(record.score for record in records),
        selected_score_distribution=_descriptive(selected_scores),
        nonselected_score_distribution=_descriptive(nonselected_scores),
        high_score_threshold=threshold,
        high_score_node_count=len(high),
        high_score_selected_count=sum(record.node_id in selected for record in high),
        posterior_records=records,
    )


def _build_relation_summary(
    state: AppRunState,
    episodes: list[EpisodeObservabilityRecordV1],
    events: list[dict[str, Any]],
    enrichment_budget_limit: int | None,
) -> RelationOutcomeSummaryV1:
    records_by_candidate: dict[str, list[Any]] = defaultdict(list)
    for record in state.relation_ledger:
        records_by_candidate[record.candidate_id].append(record)
    reason_rows: list[RelationReasonYieldV1] = []
    for reason in get_args(RelationCandidateReason):
        candidates = [
            candidate
            for candidate in state.relation_candidates
            if candidate.candidate_reason == reason
        ]
        records = [
            record
            for candidate in candidates
            for record in records_by_candidate.get(candidate.candidate_id, [])
        ]
        denominator = len(records)
        counts = {
            relation_type: sum(
                record.relation_type == relation_type for record in records
            )
            for relation_type in (
                "equivalent",
                "complementary",
                "conflict",
                "independent",
            )
        }
        reason_rows.append(
            RelationReasonYieldV1(
                candidate_reason=reason,
                candidate_count=len(candidates),
                committed_relation_count=denominator,
                equivalent_count=counts["equivalent"],
                complementary_count=counts["complementary"],
                conflict_count=counts["conflict"],
                independent_count=counts["independent"],
                equivalent_yield=_ratio(counts["equivalent"], denominator),
                complementary_yield=_ratio(counts["complementary"], denominator),
                conflict_yield=_ratio(counts["conflict"], denominator),
                independent_yield=_ratio(counts["independent"], denominator),
            )
        )
    rejected_transaction_keys = {
        (episode.episode_id, attempt.attempt_id, attempt.rejection_reason or "")
        for episode in episodes
        if episode.role == "relation"
        for attempt in episode.attempts
        if attempt.status == "rejected"
        or attempt.superseded_from_status == "rejected"
    }
    rejected_transaction_keys.update(
        (
            event.get("episode_id") if isinstance(event.get("episode_id"), str) else "",
            event.get("attempt_id") if isinstance(event.get("attempt_id"), str) else "",
            event.get("rejection_reason")
            if isinstance(event.get("rejection_reason"), str)
            else "",
        )
        for event in events
        if event.get("role") == "relation"
        and event.get("event_type") in {"output_rejected", "relation_result_rejected"}
    )
    enrichment_records = {
        record.candidate_id
        for record in state.relation_ledger
        if record.scheduling_class == "enrichment"
    }
    budget_limit = enrichment_budget_limit
    consumed = len(enrichment_records)
    return RelationOutcomeSummaryV1(
        candidate_count=len(state.relation_candidates),
        blocking_candidates_generated=sum(
            candidate.scheduling_class == "blocking"
            for candidate in state.relation_candidates
        ),
        blocking_pairs_resolved=sum(
            candidate.scheduling_class == "blocking" and candidate.status == "resolved"
            for candidate in state.relation_candidates
        ),
        enrichment_candidates_generated=sum(
            candidate.scheduling_class == "enrichment"
            for candidate in state.relation_candidates
        ),
        enrichment_pairs_committed=consumed,
        equivalent_count=sum(
            record.relation_type == "equivalent" for record in state.relation_ledger
        ),
        complementary_count=sum(
            record.relation_type == "complementary" for record in state.relation_ledger
        ),
        conflict_count=sum(
            record.relation_type == "conflict" for record in state.relation_ledger
        ),
        independent_count=sum(
            record.relation_type == "independent" for record in state.relation_ledger
        ),
        merge_count=len(state.merge_applications),
        material_conflict_count=sum(
            record.relation_type == "conflict" and record.material_to_synthesis
            for record in state.relation_ledger
        ),
        disclosure_required_count=sum(
            record.disclosure_required for record in state.relation_ledger
        ),
        rejected_relation_transactions=len(rejected_transaction_keys),
        relation_budget_limit=budget_limit,
        relation_budget_consumed=consumed,
        relation_budget_remaining=(
            None if budget_limit is None else max(0, budget_limit - consumed)
        ),
        by_candidate_reason=reason_rows,
    )


def _build_controller_trajectory(
    state: AppRunState,
    events: list[dict[str, Any]],
    lineage: list[NodeLineageRecordV1],
    allocation_mass_parameter: int | None,
) -> list[ControllerTrajectoryRecordV1]:
    lineage_by_id = {item.node_id: item for item in lineage}
    produced_children = _produced_children_by_parent(state)
    output: list[ControllerTrajectoryRecordV1] = []
    terminal_iteration = (
        None if state.terminal_record is None else state.terminal_record.controller_iteration
    )
    for record in state.controller_iteration_records:
        output.append(
            ControllerTrajectoryRecordV1(
                controller_iteration=record.iteration,
                input_graph_revision=record.input_graph_revision,
                output_graph_revision=record.output_graph_revision,
                frontier_size=len(record.frontier_node_ids),
                judged_frontier_size=sum(
                    lineage_by_id.get(node_id) is not None
                    and lineage_by_id[node_id].judge_score is not None
                    for node_id in record.frontier_node_ids
                ),
                spatial_entropy=record.spatial_entropy,
                allocation_mass_parameter=allocation_mass_parameter,
                allocated_child_count=sum(record.allocations.values()),
                positive_budget_parent_count=sum(
                    budget > 0 for budget in record.allocations.values()
                ),
                children_committed=sum(
                    len(produced_children.get(node_id, set()))
                    for node_id, budget in record.allocations.items()
                    if budget > 0
                ),
                graph_revision=record.output_graph_revision,
                readiness_transition=(
                    None
                    if record.iteration != state.controller_iteration
                    or state.synthesis_readiness is None
                    else "ready"
                    if state.synthesis_readiness.ready
                    else "blocked"
                ),
                terminal_transition=(
                    state.controller_action
                    if terminal_iteration == record.iteration
                    and state.controller_action in {"ready_for_synthesis", "run_complete"}
                    else None
                ),
            )
        )
    if output:
        return output

    allocation_events = [
        event for event in events if event.get("event_type") == "allocation_recorded"
    ]
    for index, event in enumerate(allocation_events, start=1):
        output.append(
            ControllerTrajectoryRecordV1(
                controller_iteration=index,
                input_graph_revision=(
                    event.get("input_graph_revision")
                    if isinstance(event.get("input_graph_revision"), int)
                    else None
                ),
                output_graph_revision=(
                    event.get("graph_revision")
                    if isinstance(event.get("graph_revision"), int)
                    else None
                ),
                frontier_size=(
                    event.get("selected_node_count")
                    if isinstance(event.get("selected_node_count"), int)
                    else None
                ),
                judged_frontier_size=None,
                spatial_entropy=(
                    event.get("spatial_entropy")
                    if isinstance(event.get("spatial_entropy"), (int, float))
                    else None
                ),
                allocation_mass_parameter=allocation_mass_parameter,
                allocated_child_count=(
                    event.get("allocated_child_count")
                    if isinstance(event.get("allocated_child_count"), int)
                    else None
                ),
                positive_budget_parent_count=None,
                children_committed=None,
                graph_revision=(
                    event.get("graph_revision")
                    if isinstance(event.get("graph_revision"), int)
                    else None
                ),
            )
        )
    return output


def _build_rejection_summary(
    state: AppRunState,
    events: list[dict[str, Any]],
) -> RejectionSummaryV1:
    observations: dict[tuple[str, str, str], RejectionCategory] = {}
    for episode in state.episodes:
        for attempt in episode.attempts:
            reasons: list[str] = []
            if attempt.commit_outcome is not None and not attempt.commit_outcome.accepted:
                if attempt.commit_outcome.rejection_reason:
                    reasons.append(attempt.commit_outcome.rejection_reason)
            if attempt.failure_reason:
                reasons.append(attempt.failure_reason)
            for reason in reasons:
                observations[(episode.episode_id, attempt.attempt_id, reason)] = (
                    classify_rejection_reason(reason)
                )
    for event in events:
        if event.get("event_type") not in {
            "output_rejected",
            "relation_result_rejected",
            "episode_failed",
            "episode_cancelled",
            "episode_expired",
        }:
            continue
        reason = event.get("rejection_reason")
        if not isinstance(reason, str) or not reason:
            continue
        episode_id = event.get("episode_id")
        attempt_id = event.get("attempt_id")
        violation_count = event.get("controller_field_violation_count", 0)
        observations[
            (
                episode_id if isinstance(episode_id, str) else "",
                attempt_id if isinstance(attempt_id, str) else "",
                reason,
            )
        ] = classify_rejection_reason(
            reason,
            controller_field_violation_count=(
                violation_count if isinstance(violation_count, int) else 0
            ),
        )
    categories = list(get_args(RejectionCategory))
    counts = {
        category: sum(value == category for value in observations.values())
        for category in categories
    }
    return RejectionSummaryV1(
        total_rejection_or_error_count=len(observations),
        by_category=[
            RejectionCategoryCountV1(category=category, count=counts[category])
            for category in categories
        ],
    )


def _artifact_quality(
    run_dir: Path,
    state: AppRunState,
    missing: list[str],
    recoverable: list[str],
) -> None:
    enrichment_committed = len(
        {
            record.candidate_id
            for record in state.relation_ledger
            if record.scheduling_class == "enrichment"
        }
    )
    relation_expectations: dict[Path, dict[str, Any]] = {
        Path("relations") / "candidates.json": {
            "schema_version": "relation-candidates.v2",
            "run_id": state.run_id,
            "blocking_candidate_count": sum(
                item.scheduling_class == "blocking"
                for item in state.relation_candidates
            ),
            "enrichment_candidate_count": sum(
                item.scheduling_class == "enrichment"
                for item in state.relation_candidates
            ),
            "candidates": [
                item.model_dump(mode="json") for item in state.relation_candidates
            ],
        },
        Path("relations") / "relation_ledger.json": {
            "schema_version": "relation-ledger.v2",
            "run_id": state.run_id,
            "enrichment_budget_limit": state.spec.budget.max_relation_enrichment_pairs,
            "enrichment_pairs_committed": enrichment_committed,
            "enrichment_pairs_remaining": max(
                0,
                state.spec.budget.max_relation_enrichment_pairs
                - enrichment_committed,
            ),
            "records": [
                item.model_dump(mode="json") for item in state.relation_ledger
            ],
            "merge_applications": [
                item.model_dump(mode="json") for item in state.merge_applications
            ],
        },
        Path("relations") / "synthesis_readiness.json": {
            "schema_version": "synthesis-readiness-artifact.v2",
            "run_id": state.run_id,
            "status": state.relation_readiness_status,
            "selection": (
                None
                if state.provisional_synthesis_selection is None
                else state.provisional_synthesis_selection.model_dump(mode="json")
            ),
            "readiness": (
                None
                if state.synthesis_readiness is None
                else state.synthesis_readiness.model_dump(mode="json")
            ),
        },
    }
    for relative, expected_payload in relation_expectations.items():
        path = run_dir / relative
        if not path.exists():
            missing.append(str(relative))
            continue
        try:
            actual_payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            recoverable.append(f"invalid derived Relation artifact: {relative}")
            continue
        if actual_payload != expected_payload:
            recoverable.append(
                f"derived Relation artifact disagrees with AppRunState: {relative}"
            )

    for episode in state.episodes:
        for attempt in episode.attempts:
            unsafe = any(
                not value
                or value in {".", ".."}
                or "/" in value
                or "\\" in value
                or ":" in value
                or "\x00" in value
                or Path(value).name != value
                for value in (episode.episode_id, attempt.attempt_id)
            )
            if unsafe:
                recoverable.append(
                    "unsafe persisted episode/attempt identity prevented mirror inspection"
                )
                continue
            base = Path("episodes") / episode.episode_id / attempt.attempt_id
            expected: list[tuple[Path, Any, type[Any]]] = [
                (
                    base / "request.json",
                    attempt.request.model_dump(mode="json"),
                    type(attempt.request),
                ),
                (
                    base / "status.json",
                    attempt.model_dump(mode="json"),
                    type(attempt),
                ),
            ]
            if attempt.status == "committed" and attempt.committed_result is not None:
                expected.append(
                    (
                        base / "result.json",
                        attempt.committed_result.model_dump(mode="json"),
                        type(attempt.committed_result),
                    )
                )
            for relative, expected_payload, model_type in expected:
                path = run_dir / relative
                if not path.exists():
                    missing.append(str(relative))
                    continue
                try:
                    actual_payload = json.loads(path.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    recoverable.append(f"invalid episode mirror artifact: {relative}")
                    continue
                try:
                    normalized_actual = model_type.model_validate(
                        actual_payload
                    ).model_dump(mode="json")
                except Exception:
                    recoverable.append(
                        f"invalid episode mirror artifact schema: {relative}"
                    )
                    continue
                if normalized_actual != expected_payload:
                    recoverable.append(
                        f"episode mirror disagrees with AppRunState: {relative}"
                    )


def _runtime_availability(
    episodes: list[EpisodeObservabilityRecordV1],
) -> tuple[bool, bool]:
    diagnostics = [
        attempt.runtime_diagnostics
        for episode in episodes
        for attempt in episode.attempts
        if attempt.runtime_diagnostics is not None
    ]
    usage_available = any(
        diagnostic.usage_source != "unavailable"
        and any(
            value is not None
            for value in (
                diagnostic.input_tokens,
                diagnostic.output_tokens,
                diagnostic.cached_tokens,
                diagnostic.provider_reported_cost,
                diagnostic.estimated_cost,
                diagnostic.quota_delta,
            )
        )
        for diagnostic in diagnostics
    )
    aggregate_available = any(
        diagnostic.diagnostics_source != "unavailable"
        and any(
            getattr(diagnostic, field_name) is not None
            for field_name in (
                "internal_subagent_count",
                "max_internal_parallelism",
                "internal_tool_call_count",
                "internal_round_count",
                "internal_failure_count",
                "internal_input_tokens",
                "internal_output_tokens",
            )
        )
        for diagnostic in diagnostics
    )
    return not usage_available, not aggregate_available


def build_run_observability_summary(
    run_dir: str | Path,
) -> RunObservabilitySummaryV1:
    """Build the formal v1 summary without modifying any run artifact."""

    directory = Path(run_dir)
    state, original_schema, partial, validation_error, reconstructed, raw_state = (
        _load_state_read_only(directory)
    )
    events, missing_event_fields, telemetry_issues, corrupt_tail, repaired_tail = (
        _read_telemetry_read_only(directory)
    )
    feedback, feedback_diagnostics = read_feedback_ledger(directory)
    missing_artifacts: list[str] = []
    recoverable = list(telemetry_issues)
    if not (directory / "episode_events.jsonl").exists():
        missing_artifacts.append("episode_events.jsonl")
    _artifact_quality(directory, state, missing_artifacts, recoverable)
    if feedback_diagnostics.malformed_interior_line_count:
        recoverable.append(
            "feedback ledger contains "
            f"{feedback_diagnostics.malformed_interior_line_count} invalid line(s)"
        )
    if feedback_diagnostics.duplicate_feedback_ids:
        recoverable.append(
            "feedback ledger duplicate IDs ignored: "
            + ", ".join(feedback_diagnostics.duplicate_feedback_ids)
        )
    if reconstructed:
        recoverable.append(
            "deterministically reconstructed legacy fields: "
            + ", ".join(sorted(reconstructed))
        )

    raw_spec = raw_state.get("spec")
    if not isinstance(raw_spec, dict):
        raw_spec = {}
    raw_budget = raw_spec.get("budget")
    if not isinstance(raw_budget, dict):
        raw_budget = {}
    raw_budget_presence = dict(raw_budget)
    if "total_child_budget" in raw_budget:
        raw_budget_presence.setdefault(
            "allocation_mass_per_iteration", raw_budget["total_child_budget"]
        )
    configuration_fields = {
        "spec.mode": (raw_spec, "mode"),
        "spec.embedding_provider": (raw_spec, "embedding_provider"),
        "spec.embedding_dimension": (raw_spec, "embedding_dimension"),
        "spec.budget.max_iterations": (raw_budget, "max_iterations"),
        "spec.budget.allocation_mass_per_iteration": (
            raw_budget_presence,
            "allocation_mass_per_iteration",
        ),
        "spec.budget.max_children_per_iteration": (
            raw_budget,
            "max_children_per_iteration",
        ),
        "spec.budget.max_relation_pairs_per_episode": (
            raw_budget,
            "max_relation_pairs_per_episode",
        ),
        "spec.budget.max_relation_enrichment_pairs": (
            raw_budget,
            "max_relation_enrichment_pairs",
        ),
        "spec.budget.max_research_iterations": (
            raw_budget,
            "max_research_iterations",
        ),
        "spec.budget.min_iterations_before_synthesis": (
            raw_budget,
            "min_iterations_before_synthesis",
        ),
        "spec.budget.entropy_change_threshold": (
            raw_budget,
            "entropy_change_threshold",
        ),
        "spec.budget.t_max": (raw_budget, "t_max"),
    }
    missing_configuration_fields = sorted(
        path
        for path, (container, field_name) in configuration_fields.items()
        if field_name not in container
    )
    if missing_configuration_fields:
        partial = True
        recoverable.append(
            "legacy source omits configuration field(s): "
            + ", ".join(missing_configuration_fields)
        )
    if "controller_iteration" not in raw_state:
        partial = True
        recoverable.append("legacy source omits state field: controller_iteration")

    episodes = _build_episode_records(state, events)
    lineage, allocations, node_funnel = _build_lineage_and_allocations(
        state, recoverable
    )
    episode_funnel = EpisodeFunnelV1(
        judge=_role_funnel(episodes, "judge"),
        executor=_role_funnel(episodes, "executor"),
        relation=_role_funnel(episodes, "relation"),
        unmodeled_role_episode_count=sum(
            episode.role not in {"judge", "executor", "relation"}
            for episode in episodes
        ),
    )
    runtime_profiles = sorted(
        {
            profile
            for episode in state.episodes
            for attempt in episode.attempts
            for profile in (
                (
                    attempt.request.transport_hints.get("profile")
                    if attempt.request.transport_hints
                    and isinstance(attempt.request.transport_hints.get("profile"), str)
                    else None
                ),
                (
                    attempt.committed_result.runtime_diagnostics.profile
                    if attempt.committed_result is not None
                    else None
                ),
            )
            if profile is not None
        }
    )
    usage_unavailable, runtime_diagnostics_unavailable = _runtime_availability(
        episodes
    )
    limitations: list[str] = []
    if usage_unavailable:
        limitations.append("provider/runtime usage is unavailable or unreported")
    if runtime_diagnostics_unavailable:
        limitations.append("aggregate internal runtime diagnostics are unavailable")
    if not state.controller_iteration_records and state.controller_iteration:
        limitations.append(
            "controller trajectory is partial because durable iteration records are absent"
        )
    if partial:
        limitations.append(
            "legacy or invalid-for-controller state was read only as a partial observation"
        )
    if corrupt_tail:
        limitations.append(
            "the current telemetry tail is corrupt and was ignored without repair"
        )
    if feedback_diagnostics.corrupt_tail_detected:
        limitations.append(
            "the current feedback tail is corrupt and was ignored without repair"
        )

    terminal_reason = (
        state.terminal_record.reason
        if state.terminal_record is not None
        else state.pending_terminal_reason
    )
    terminal_source = (
        state.terminal_record.source
        if state.terminal_record is not None
        else state.pending_terminal_source
    )
    summary = RunObservabilitySummaryV1(
        run=RunIdentityObservabilityV1(
            run_id=state.run_id,
            state_schema_version=original_schema,
            controller_profile=(
                state.spec.mode if "mode" in raw_spec else None
            ),
            runtime_profiles=runtime_profiles,
            problem_sha256=_sha256_text(state.spec.problem),
            goal_sha256=_sha256_text(state.spec.goal),
            budget=RunBudgetSnapshotV1(
                max_iterations=(
                    state.spec.budget.max_iterations
                    if "max_iterations" in raw_budget
                    else None
                ),
                allocation_mass_per_iteration=(
                    state.spec.budget.allocation_mass_per_iteration
                    if "allocation_mass_per_iteration" in raw_budget_presence
                    else None
                ),
                max_children_per_iteration=(
                    state.spec.budget.max_children_per_iteration
                    if "max_children_per_iteration" in raw_budget
                    else None
                ),
                max_relation_pairs_per_episode=(
                    state.spec.budget.max_relation_pairs_per_episode
                    if "max_relation_pairs_per_episode" in raw_budget
                    else None
                ),
                max_relation_enrichment_pairs=(
                    state.spec.budget.max_relation_enrichment_pairs
                    if "max_relation_enrichment_pairs" in raw_budget
                    else None
                ),
                max_research_iterations=(
                    state.spec.budget.max_research_iterations
                    if "max_research_iterations" in raw_budget
                    else None
                ),
                min_iterations_before_synthesis=(
                    state.spec.budget.min_iterations_before_synthesis
                    if "min_iterations_before_synthesis" in raw_budget
                    else None
                ),
                entropy_change_threshold=(
                    state.spec.budget.entropy_change_threshold
                    if "entropy_change_threshold" in raw_budget
                    else None
                ),
                t_max=(
                    state.spec.budget.t_max if "t_max" in raw_budget else None
                ),
            ),
            embedding_provider=(
                state.spec.embedding_provider
                if "embedding_provider" in raw_spec
                else None
            ),
            embedding_dimension=(
                state.spec.embedding_dimension
                if "embedding_dimension" in raw_spec
                else None
            ),
            terminal_action=(
                state.controller_action
                if state.controller_action in {"ready_for_synthesis", "run_complete"}
                else None
            ),
            terminal_reason=terminal_reason,
            terminal_source=terminal_source,
            controller_iterations=(
                state.controller_iteration
                if "controller_iteration" in raw_state
                else None
            ),
            graph_revision=state.graph_revision,
            code_version=None,
            observability_status="partial_legacy" if partial else "current",
            created_at=state.created_at,
            updated_at=state.updated_at,
        ),
        episode_funnel=episode_funnel,
        node_funnel=node_funnel,
        episodes=episodes,
        node_lineage=lineage,
        allocation_outcomes=allocations,
        judge_outcomes=_build_judge_summary(state, lineage),
        relation_outcomes=_build_relation_summary(
            state,
            episodes,
            events,
            (
                state.spec.budget.max_relation_enrichment_pairs
                if "max_relation_enrichment_pairs" in raw_budget
                else None
            ),
        ),
        controller_trajectory=_build_controller_trajectory(
            state,
            events,
            lineage,
            (
                state.spec.budget.allocation_mass_per_iteration
                if "allocation_mass_per_iteration" in raw_budget_presence
                else None
            ),
        ),
        rejections=_build_rejection_summary(state, events),
        feedback=feedback,
        data_quality=ObservabilityDataQualityV1(
            missing_event_fields=missing_event_fields,
            missing_artifacts=sorted(set(missing_artifacts)),
            inconsistent_but_recoverable_records=sorted(set(recoverable)),
            corrupt_telemetry_tail_detected=corrupt_tail,
            corrupt_telemetry_tail_repaired=repaired_tail,
            usage_unavailable=usage_unavailable,
            runtime_diagnostics_unavailable=runtime_diagnostics_unavailable,
            partial_legacy_reconstruction=partial,
            state_validation_error=validation_error,
            telemetry_record_count=len(events),
            feedback_record_count=len(feedback),
            limitations=sorted(set(limitations)),
        ),
    )
    # Re-parse the serialized shape so callers never receive a model carrying
    # an assignment mutation that bypassed strict round-trip validation.
    return RunObservabilitySummaryV1.model_validate(
        summary.model_dump(mode="json")
    )


def render_observability_text(summary: RunObservabilitySummaryV1) -> str:
    """Render a compact human/main-agent view while JSON remains authoritative."""

    run = summary.run
    funnel = summary.node_funnel
    role_lines = []
    for role in ("judge", "executor", "relation"):
        values = getattr(summary.episode_funnel, role)
        role_lines.append(
            f"- {role}: {values.episode_count} episodes / {values.attempt_count} attempts; "
            f"committed {values.committed_attempt_count}, rejected {values.rejected_attempt_count}, "
            f"retried {values.retried_attempt_count}"
        )
    allocations = [
        f"{item.parent_node_id} {item.actual_committed_children}/{item.allocated_expansion_budget} children"
        for item in summary.allocation_outcomes
    ]
    relation = summary.relation_outcomes
    quality_notes = list(summary.data_quality.limitations)
    if summary.data_quality.missing_artifacts:
        quality_notes.append(
            "missing artifacts: "
            + ", ".join(summary.data_quality.missing_artifacts)
        )
    if summary.data_quality.missing_event_fields:
        quality_notes.append(
            "missing telemetry fields: "
            + ", ".join(summary.data_quality.missing_event_fields)
        )
    if summary.data_quality.inconsistent_but_recoverable_records:
        quality_notes.append(
            f"{len(summary.data_quality.inconsistent_but_recoverable_records)} "
            "recoverable inconsistency(s)"
        )
    iteration_text = (
        "unavailable"
        if run.controller_iterations is None
        else str(run.controller_iterations)
    )
    return "\n".join(
        [
            f"DTE observability summary ({summary.schema_version})",
            f"run: {run.run_id}",
            f"terminal: {run.terminal_action or 'not terminal'}"
            + (f" — {run.terminal_reason}" if run.terminal_reason else ""),
            f"controller: {iteration_text} iteration(s), graph revision {run.graph_revision}",
            "episodes:",
            *role_lines,
            (
                "nodes: "
                f"{funnel.initial_node_count} initial -> {funnel.all_committed_node_count} committed "
                f"-> {funnel.provisional_synthesis_selected_node_count} provisional-selected; "
                f"{funnel.merged_node_count} merged"
            ),
            "allocations: " + ("; ".join(allocations) if allocations else "none"),
            (
                "relations: "
                f"{relation.equivalent_count} equivalent, "
                f"{relation.complementary_count} complementary, "
                f"{relation.conflict_count} conflict, "
                f"{relation.independent_count} independent; "
                f"{relation.merge_count} merge(s), "
                f"{relation.disclosure_required_count} disclosure(s)"
            ),
            (
                "rejections/retries: "
                f"{summary.rejections.total_rejection_or_error_count} classified rejection/error fact(s)"
            ),
            "data quality: "
            + ("; ".join(quality_notes) if quality_notes else "no reported limitation"),
            "proxy note: internal process metrics do not establish external research effectiveness.",
        ]
    )


def export_observability_jsonl(
    runs_root: str | Path,
    output_path: str | Path,
) -> ObservabilityExportResultV1:
    """Export distinguishable run/episode/node/allocation/feedback JSONL records."""

    root = Path(runs_root)
    output = Path(output_path)
    state_paths = sorted(root.rglob("app_run_state.json"), key=lambda item: str(item))
    records: list[dict[str, Any]] = []
    skipped: list[ObservabilityExportSkippedRunV1] = []
    processed = 0
    for state_path in state_paths:
        run_dir = state_path.parent
        try:
            summary = build_run_observability_summary(run_dir)
        except Exception as exc:
            skipped.append(
                ObservabilityExportSkippedRunV1(
                    run_dir=str(run_dir),
                    reason=str(exc),
                )
            )
            continue
        processed += 1
        records.append(
            {
                "record_type": "run",
                "schema_version": EXPORT_SCHEMA_VERSION,
                "run_id": summary.run.run_id,
                "summary": summary.model_dump(mode="json"),
            }
        )
        for episode in summary.episodes:
            records.append(
                {
                    "record_type": "episode",
                    "schema_version": EXPORT_SCHEMA_VERSION,
                    "run_id": summary.run.run_id,
                    "episode": episode.model_dump(mode="json"),
                }
            )
        for node in summary.node_lineage:
            records.append(
                {
                    "record_type": "node",
                    "schema_version": EXPORT_SCHEMA_VERSION,
                    "run_id": summary.run.run_id,
                    "node": node.model_dump(mode="json"),
                }
            )
        for allocation in summary.allocation_outcomes:
            records.append(
                {
                    "record_type": "allocation",
                    "schema_version": EXPORT_SCHEMA_VERSION,
                    "run_id": summary.run.run_id,
                    "allocation": allocation.model_dump(mode="json"),
                }
            )
        for feedback in summary.feedback:
            records.append(
                {
                    "record_type": "feedback",
                    "schema_version": EXPORT_SCHEMA_VERSION,
                    "run_id": summary.run.run_id,
                    "feedback": feedback.model_dump(mode="json"),
                }
            )

    encoded = b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )
    _replace_bytes_atomic(output, encoded)
    return ObservabilityExportResultV1(
        output_path=str(output),
        processed_run_count=processed,
        skipped_run_count=len(skipped),
        record_count=len(records),
        skipped_runs=skipped,
    )
