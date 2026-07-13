"""Atomic validation and graph commit boundary for AgentEpisode results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from pydantic import ValidationError

from .episode_models import (
    CommitOutcome,
    EpisodeRequest,
    EpisodeResult,
    ExecutorEpisodeOutput,
    JudgeEpisodeOutput,
    compute_output_hash,
)
from .models import SearchNode
from .telemetry import EpisodeEventLog


CONTROLLER_OWNED_FIELDS = frozenset(
    {
        "embedding",
        "local_embedding",
        "score",
        "density",
        "judge_reasoning",
        "judge_verdict",
        "judge_verdict_reference",
        "judge_verdict_references",
        "judge_risks",
        "judge_uncertainty_evidence",
        "uncertainty",
        "ucb_score",
        "expansion_budget",
        "allocation",
        "committed_graph_revision",
        "graph_revision",
        "controller_stop_reason",
        "stop_reason",
        "synthesis_checkpoint",
        "node_revision",
        "judge_result_provenance",
    }
)


@dataclass
class EpisodeGraph:
    """Minimum in-memory revision state needed for the vertical slice."""

    nodes: list[SearchNode]
    revision: int = 0
    node_revisions: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("graph contains duplicate node IDs")
        if not self.node_revisions:
            self.node_revisions = {node_id: 0 for node_id in ids}
        if set(self.node_revisions) != set(ids):
            raise ValueError("node_revisions must match committed graph node IDs")
        if self.revision < 0 or any(revision < 0 for revision in self.node_revisions.values()):
            raise ValueError("graph revisions must be non-negative")

    def node_by_id(self, node_id: str) -> SearchNode | None:
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "node_revisions": dict(self.node_revisions),
            "nodes": [node.model_dump(mode="json") for node in self.nodes],
        }


def _raw_structured_output(raw_result: Any) -> Any:
    if isinstance(raw_result, EpisodeResult):
        return None if raw_result.structured_output is None else raw_result.structured_output.model_dump(mode="json")
    if isinstance(raw_result, Mapping):
        return raw_result.get("structured_output")
    return None


def _raw_nodes(raw_result: Any) -> list[Any]:
    output = _raw_structured_output(raw_result)
    if isinstance(output, Mapping) and isinstance(output.get("nodes"), list):
        return list(output["nodes"])
    return []


def _raw_judge_observations(raw_result: Any) -> list[Any]:
    output = _raw_structured_output(raw_result)
    if isinstance(output, Mapping) and isinstance(output.get("observations"), list):
        return list(output["observations"])
    return []


def _quality_counts(raw_result: Any) -> tuple[int, int]:
    raw_nodes = _raw_nodes(raw_result) or _raw_judge_observations(raw_result)
    controller_violations = 0
    node_ids: list[str] = []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping):
            continue
        controller_violations += len(CONTROLLER_OWNED_FIELDS.intersection(raw_node))
        if isinstance(raw_node.get("node_id"), str):
            node_ids.append(raw_node["node_id"])
    duplicate_count = len(node_ids) - len(set(node_ids))
    return controller_violations, duplicate_count


def _reject(
    graph: EpisodeGraph,
    request: EpisodeRequest,
    reason: str,
    telemetry: EpisodeEventLog | None,
    *,
    schema_valid: bool,
    controller_violations: int,
    duplicate_count: int,
    returned_observation_count: int | None,
) -> CommitOutcome:
    if telemetry is not None:
        telemetry.emit(
            "output_rejected",
            run_id=request.run_id,
            episode_id=request.episode_id,
            attempt_id=request.attempt_id,
            role=request.role,
            status="rejected",
            input_graph_revision=request.input_graph_revision,
            returned_node_count=None,
            accepted_node_count=0,
            selected_node_count=len(request.selected_node_revisions),
            returned_observation_count=returned_observation_count,
            accepted_observation_count=0 if request.role == "judge" else None,
            rejection_reason=reason,
            schema_valid=schema_valid,
            controller_field_violation_count=controller_violations,
            duplicate_within_result_count=duplicate_count,
        )
    return CommitOutcome(
        accepted=False,
        episode_id=request.episode_id,
        graph_revision_before=graph.revision,
        graph_revision_after=graph.revision,
        rejection_reason=reason,
    )


def commit_episode_result(
    graph: EpisodeGraph,
    request: EpisodeRequest,
    raw_result: EpisodeResult | Mapping[str, Any],
    telemetry: EpisodeEventLog | None = None,
) -> CommitOutcome:
    """Validate the complete result before atomically replacing graph state."""

    controller_violations, duplicate_count = _quality_counts(raw_result)
    raw_observations = _raw_judge_observations(raw_result)
    returned_observation_count = len(raw_observations) if request.role == "judge" else None
    try:
        result = raw_result if isinstance(raw_result, EpisodeResult) else EpisodeResult.model_validate(raw_result)
    except ValidationError as exc:
        return _reject(
            graph,
            request,
            f"episode result schema validation failed: {exc}",
            telemetry,
            schema_valid=False,
            controller_violations=controller_violations,
            duplicate_count=duplicate_count,
            returned_observation_count=returned_observation_count,
        )

    def reject(reason: str) -> CommitOutcome:
        return _reject(
            graph,
            request,
            reason,
            telemetry,
            schema_valid=True,
            controller_violations=controller_violations,
            duplicate_count=duplicate_count,
            returned_observation_count=returned_observation_count,
        )

    if result.episode_id != request.episode_id:
        return reject("episode ID mismatch")
    if result.attempt_id != request.attempt_id:
        return reject("attempt ID mismatch")
    if result.run_id != request.run_id:
        return reject("run ID mismatch")
    if result.role != request.role:
        return reject("role mismatch")
    if result.status != "completed":
        return reject(f"result status is {result.status}, not completed")
    if request.input_graph_revision != graph.revision or result.input_graph_revision != graph.revision:
        return reject("stale graph revision")
    if result.selected_node_revisions != request.selected_node_revisions:
        return reject("selected node revisions mismatch")
    if result.schema_version != request.output_schema_version:
        return reject("output schema version mismatch")
    if result.output_hash != compute_output_hash(result.structured_output, result.schema_version):
        return reject("output hash mismatch")

    if request.role == "judge":
        for node_id, revision in request.selected_node_revisions.items():
            if graph.node_revisions.get(node_id) != revision:
                return reject(f"stale selected-node revision: {node_id}")
        if not isinstance(result.structured_output, JudgeEpisodeOutput):
            return reject("completed Judge result has the wrong structured output schema")
        granted_ids = set(request.selected_node_revisions)
        observations = result.structured_output.observations
        observation_ids = [observation.node_id for observation in observations]
        if len(observation_ids) != len(set(observation_ids)):
            return reject("duplicate Judge node ID inside result")
        returned_ids = set(observation_ids)
        missing = sorted(granted_ids - returned_ids)
        if missing:
            return reject(f"Judge result omitted granted node: {missing[0]}")
        extra = sorted(returned_ids - granted_ids)
        if extra:
            return reject(f"Judge result returned ungranted node: {extra[0]}")

        next_nodes = deepcopy(graph.nodes)
        next_revisions = dict(graph.node_revisions)
        by_id = {node.node_id: node for node in next_nodes}
        for observation in observations:
            node = by_id.get(observation.node_id)
            if node is None or node.status != "frontier":
                return reject(f"Judge target is not a committed frontier node: {observation.node_id}")
            node.score = observation.score
            node.judge_reasoning = observation.reasoning
            node.judge_risks = list(observation.risks)
            node.judge_uncertainty_evidence = list(observation.uncertainty_evidence)
            node.judge_result_provenance = {
                "run_id": request.run_id,
                "episode_id": request.episode_id,
                "attempt_id": request.attempt_id,
                "schema_version": result.schema_version,
                "output_hash": result.output_hash,
            }
            next_revisions[node.node_id] += 1

        revision_before = graph.revision
        graph.nodes = next_nodes
        graph.node_revisions = next_revisions
        graph.revision = revision_before + 1
        if telemetry is not None:
            telemetry.emit(
                "judge_observations_committed",
                run_id=request.run_id,
                episode_id=request.episode_id,
                attempt_id=request.attempt_id,
                role=request.role,
                status="committed",
                input_graph_revision=request.input_graph_revision,
                selected_node_count=len(granted_ids),
                returned_observation_count=len(observations),
                accepted_observation_count=len(observations),
                schema_valid=True,
                controller_field_violation_count=controller_violations,
                duplicate_within_result_count=duplicate_count,
                usage_source="unavailable",
            )
        return CommitOutcome(
            accepted=True,
            episode_id=request.episode_id,
            accepted_node_ids=observation_ids,
            accepted_node_count=len(observations),
            graph_revision_before=revision_before,
            graph_revision_after=graph.revision,
        )

    if request.role != "executor":
        return reject(f"commit path is not implemented for role={request.role}")
    if not isinstance(result.structured_output, ExecutorEpisodeOutput):
        return reject("completed Executor result has the wrong structured output schema")

    assert request.parent_node_id is not None
    assert request.parent_node_revision is not None
    assert request.max_returned_children is not None
    parent = graph.node_by_id(request.parent_node_id)
    if parent is None:
        return reject("assigned parent is not committed")
    if graph.node_revisions.get(parent.node_id) != request.parent_node_revision:
        return reject("stale parent revision")
    candidates = result.structured_output.nodes
    if len(candidates) > request.max_returned_children:
        return reject("returned child count exceeds grant")
    candidate_ids = [candidate.node_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        return reject("duplicate node ID inside result")
    committed_ids = {node.node_id for node in graph.nodes}
    collisions = sorted(committed_ids.intersection(candidate_ids))
    if collisions:
        return reject(f"node ID collision with committed graph: {collisions[0]}")

    for candidate in candidates:
        if candidate.node_type not in request.allowed_output_types:
            return reject(f"forbidden node type: {candidate.node_type}")
        if candidate.status != "frontier":
            return reject(f"forbidden node status: {candidate.status}")
        if candidate.node_type == "synthesis":
            return reject("Executor episode may not produce synthesis nodes")
        if request.required_parent_id_on_children and request.parent_node_id not in candidate.parent_ids:
            return reject("child does not reference assigned parent")

    # Copy first; only the final assignments mutate the caller-visible graph.
    next_nodes = deepcopy(graph.nodes)
    next_revisions = dict(graph.node_revisions)
    next_parent = next(node for node in next_nodes if node.node_id == request.parent_node_id)
    next_parent.status = "closed"
    next_parent.expansion_budget = 0
    next_revisions[next_parent.node_id] += 1
    for candidate in candidates:
        next_nodes.append(SearchNode.model_validate(candidate.model_dump(mode="json")))
        next_revisions[candidate.node_id] = 0

    revision_before = graph.revision
    graph.nodes = next_nodes
    graph.node_revisions = next_revisions
    graph.revision = revision_before + 1
    # Preserve the legacy caller-visible parent reference after the atomic
    # replacement succeeds. No external object is touched on rejection.
    parent.status = "closed"
    parent.expansion_budget = 0

    if telemetry is not None:
        telemetry.emit(
            "nodes_committed",
            run_id=request.run_id,
            episode_id=request.episode_id,
            attempt_id=request.attempt_id,
            role=request.role,
            status="committed",
            input_graph_revision=request.input_graph_revision,
            returned_node_count=len(candidates),
            accepted_node_count=len(candidates),
            schema_valid=True,
            controller_field_violation_count=controller_violations,
            duplicate_within_result_count=duplicate_count,
        )
    return CommitOutcome(
        accepted=True,
        episode_id=request.episode_id,
        accepted_node_ids=candidate_ids,
        accepted_node_count=len(candidate_ids),
        graph_revision_before=revision_before,
        graph_revision_after=graph.revision,
    )
