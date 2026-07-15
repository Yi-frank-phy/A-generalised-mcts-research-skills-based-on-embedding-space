"""Atomic validation and graph commit boundary for AgentEpisode results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from .merge import apply_relation_equivalent_merge, validate_merge_application_consistency
from .models import SearchNode
from .relation_models import (
    MergeApplicationRecord,
    RelationCandidate,
    RelationEpisodeOutput,
    RelationRecord,
    stable_relation_id,
)
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
        "controller_iteration",
        "controller_action",
        "canonical_node_id",
        "node_status",
        "parent_rewrites",
        "synthesis_readiness",
        "ready_for_synthesis",
        "run_complete",
    }
)
RELATION_FORBIDDEN_FIELDS = CONTROLLER_OWNED_FIELDS.union(
    {"status", "claim", "evidence", "assumptions", "risks", "parent_ids", "parent_links"}
)


@dataclass
class EpisodeGraph:
    """Minimum in-memory revision state needed for the vertical slice."""

    nodes: list[SearchNode]
    revision: int = 0
    node_revisions: dict[str, int] = field(default_factory=dict)
    relation_candidates: list[RelationCandidate] = field(default_factory=list)
    relation_ledger: list[RelationRecord] = field(default_factory=list)
    merge_applications: list[MergeApplicationRecord] = field(default_factory=list)

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
            "relation_candidates": [item.model_dump(mode="json") for item in self.relation_candidates],
            "relation_ledger": [item.model_dump(mode="json") for item in self.relation_ledger],
            "merge_applications": [item.model_dump(mode="json") for item in self.merge_applications],
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


def _raw_relation_observations(raw_result: Any) -> list[Any]:
    output = _raw_structured_output(raw_result)
    if isinstance(output, Mapping) and isinstance(output.get("observations"), list):
        return list(output["observations"])
    return []


def _controller_owned_pollution(value: Any, forbidden: frozenset[str]) -> int:
    if isinstance(value, Mapping):
        return sum(
            (1 if str(key) in forbidden else 0) + _controller_owned_pollution(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(_controller_owned_pollution(item, forbidden) for item in value)
    return 0


def _quality_counts(raw_result: Any) -> tuple[int, int]:
    raw_nodes = _raw_nodes(raw_result) or _raw_judge_observations(raw_result) or _raw_relation_observations(raw_result)
    role = raw_result.role if isinstance(raw_result, EpisodeResult) else raw_result.get("role") if isinstance(raw_result, Mapping) else None
    forbidden = RELATION_FORBIDDEN_FIELDS if role == "relation" else CONTROLLER_OWNED_FIELDS
    controller_violations = _controller_owned_pollution(_raw_structured_output(raw_result), forbidden)
    node_ids: list[str] = []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping):
            continue
        identifier = raw_node.get("candidate_id", raw_node.get("node_id"))
        if isinstance(identifier, str):
            node_ids.append(identifier)
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
        rejection_fields = dict(
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
            accepted_observation_count=0 if request.role in {"judge", "relation"} else None,
            rejection_reason=reason,
            schema_valid=schema_valid,
            controller_field_violation_count=controller_violations,
            duplicate_within_result_count=duplicate_count,
        )
        telemetry.emit("output_rejected", **rejection_fields)
        if request.role == "relation":
            telemetry.emit("relation_result_rejected", **rejection_fields)
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
    raw_observations = (
        _raw_judge_observations(raw_result)
        if request.role == "judge"
        else _raw_relation_observations(raw_result)
        if request.role == "relation"
        else []
    )
    returned_observation_count = len(raw_observations) if request.role in {"judge", "relation"} else None
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

    if request.role == "relation":
        for node_id, revision in request.selected_node_revisions.items():
            if graph.node_revisions.get(node_id) != revision:
                return reject(f"stale selected-node revision: {node_id}")
        if request.relation_payload is None:
            return reject("Relation request is missing relation_payload")
        request_nodes = [
            node_id
            for pair in request.relation_payload.candidate_pairs
            for node_id in (pair.left.node_id, pair.right.node_id)
        ]
        if len(request_nodes) != len(set(request_nodes)):
            return reject("Relation episode candidate pairs are not node-disjoint")
        if not isinstance(result.structured_output, RelationEpisodeOutput):
            return reject("completed Relation result has the wrong structured output schema")

        granted_pairs = {pair.candidate_id: pair for pair in request.relation_payload.candidate_pairs}
        observations = result.structured_output.observations
        candidate_ids = [observation.candidate_id for observation in observations]
        if len(candidate_ids) != len(set(candidate_ids)):
            return reject("duplicate Relation candidate ID inside result")
        unordered_pairs = [(observation.left_node_id, observation.right_node_id) for observation in observations]
        if len(unordered_pairs) != len(set(unordered_pairs)):
            return reject("duplicate unordered Relation pair inside result")
        returned_ids = set(candidate_ids)
        missing = sorted(set(granted_pairs) - returned_ids)
        if missing:
            return reject(f"Relation result omitted granted candidate: {missing[0]}")
        extra = sorted(returned_ids - set(granted_pairs))
        if extra:
            return reject(f"Relation result returned ungranted candidate: {extra[0]}")

        candidate_by_id = {candidate.candidate_id: candidate for candidate in graph.relation_candidates}
        for observation in observations:
            pair = granted_pairs[observation.candidate_id]
            if (observation.left_node_id, observation.right_node_id) != (
                pair.left.node_id,
                pair.right.node_id,
            ):
                return reject(f"Relation observation pair mismatch: {observation.candidate_id}")
            evidence_refs = {
                evidence.evidence_ref
                for node in (pair.left, pair.right)
                for evidence in node.evidence
            }
            invalid_refs = sorted(set(observation.evidence_refs) - evidence_refs)
            if invalid_refs:
                return reject(f"invalid Relation evidence reference: {invalid_refs[0]}")
            candidate = candidate_by_id.get(observation.candidate_id)
            if candidate is None:
                return reject(f"Relation candidate is not committed: {observation.candidate_id}")
            if candidate.status != "granted":
                return reject(f"Relation candidate is not granted: {observation.candidate_id}")
            if (
                candidate.granted_episode_id != request.episode_id
                or candidate.granted_attempt_id != request.attempt_id
            ):
                return reject(f"Relation candidate grant provenance mismatch: {observation.candidate_id}")
            if (
                candidate.left_node_revision != pair.left_node_revision
                or candidate.right_node_revision != pair.right_node_revision
            ):
                return reject(f"Relation candidate is stale: {observation.candidate_id}")

        next_nodes = deepcopy(graph.nodes)
        next_revisions = dict(graph.node_revisions)
        next_candidates = deepcopy(graph.relation_candidates)
        next_ledger = deepcopy(graph.relation_ledger)
        next_merges = deepcopy(graph.merge_applications)
        next_candidate_by_id = {candidate.candidate_id: candidate for candidate in next_candidates}
        committed_at = datetime.now(timezone.utc).isoformat()
        equivalent_records: list[RelationRecord] = []
        counts = {"equivalent": 0, "complementary": 0, "conflict": 0, "independent": 0}
        material_conflict_count = 0
        for observation in observations:
            pair = granted_pairs[observation.candidate_id]
            candidate = next_candidate_by_id[observation.candidate_id]
            # Full discriminator scheduling is deferred.  A material conflict
            # is therefore preserved as an explicit future-Synthesis disclosure
            # obligation unless the observation already requests disclosure.
            disclosure_required = bool(
                observation.relation_type == "conflict"
                and (observation.disclosure_required or candidate.material_to_synthesis)
            )
            record_id = stable_relation_id(
                "relrec",
                candidate.candidate_id,
                request.episode_id,
                request.attempt_id,
                result.output_hash,
            )
            record = RelationRecord(
                relation_record_id=record_id,
                candidate_id=candidate.candidate_id,
                left_node_id=observation.left_node_id,
                right_node_id=observation.right_node_id,
                relation_type=observation.relation_type,
                scheduling_class=candidate.scheduling_class,
                confidence=observation.confidence,
                rationale=observation.rationale,
                evidence_refs=list(observation.evidence_refs),
                material_to_synthesis=candidate.material_to_synthesis,
                materiality_assessment=observation.materiality_assessment,
                observation=observation,
                disclosure_required=disclosure_required,
                episode_id=request.episode_id,
                attempt_id=request.attempt_id,
                input_graph_revision=request.input_graph_revision,
                selected_node_revisions=dict(request.selected_node_revisions),
                output_hash=result.output_hash,
                schema_version=result.schema_version,
                committed_at=committed_at,
            )
            next_ledger.append(record)
            candidate.status = "resolved"
            candidate.resolved_relation_record_id = record_id
            counts[observation.relation_type] += 1
            if observation.relation_type == "equivalent":
                equivalent_records.append(record)
            if observation.relation_type == "conflict" and candidate.material_to_synthesis:
                material_conflict_count += 1

        revision_before = graph.revision
        relation_revision = revision_before + 1
        merge_revision = relation_revision + (1 if equivalent_records else 0)
        try:
            validate_merge_application_consistency(next_merges)
            for record in equivalent_records:
                application = apply_relation_equivalent_merge(
                    next_nodes,
                    next_revisions,
                    source_node_ids=[record.left_node_id, record.right_node_id],
                    relation_record_id=record.relation_record_id,
                    applied_graph_revision=merge_revision,
                    applied_at=committed_at,
                )
                validate_merge_application_consistency([*next_merges, application])
                next_merges.append(application)
        except ValueError as exc:
            return reject(str(exc))

        graph.nodes = next_nodes
        graph.node_revisions = next_revisions
        graph.relation_candidates = next_candidates
        graph.relation_ledger = next_ledger
        graph.merge_applications = next_merges
        graph.revision = merge_revision
        if telemetry is not None:
            telemetry.emit(
                "relation_observations_committed",
                run_id=request.run_id,
                episode_id=request.episode_id,
                attempt_id=request.attempt_id,
                role=request.role,
                status="committed",
                input_graph_revision=request.input_graph_revision,
                selected_node_count=len(request.selected_node_revisions),
                selected_pair_count=len(granted_pairs),
                returned_observation_count=len(observations),
                accepted_observation_count=len(observations),
                equivalent_count=counts["equivalent"],
                complementary_count=counts["complementary"],
                conflict_count=counts["conflict"],
                independent_count=counts["independent"],
                material_conflict_count=material_conflict_count,
                enrichment_candidate_count=sum(
                    pair.scheduling_class == "enrichment" for pair in granted_pairs.values()
                ),
                merge_count=len(equivalent_records),
                schema_valid=True,
                usage_source="unavailable",
            )
            if counts["equivalent"]:
                telemetry.emit(
                    "merge_proposed",
                    run_id=request.run_id,
                    episode_id=request.episode_id,
                    attempt_id=request.attempt_id,
                    role=request.role,
                    status="committed",
                    input_graph_revision=request.input_graph_revision,
                    merge_count=counts["equivalent"],
                    usage_source="unavailable",
                )
            if counts["complementary"]:
                telemetry.emit(
                    "complementarity_recorded",
                    run_id=request.run_id,
                    episode_id=request.episode_id,
                    attempt_id=request.attempt_id,
                    role=request.role,
                    status="committed",
                    input_graph_revision=request.input_graph_revision,
                    complementary_count=counts["complementary"],
                    usage_source="unavailable",
                )
            if counts["conflict"]:
                telemetry.emit(
                    "conflict_recorded",
                    run_id=request.run_id,
                    episode_id=request.episode_id,
                    attempt_id=request.attempt_id,
                    role=request.role,
                    status="committed",
                    input_graph_revision=request.input_graph_revision,
                    conflict_count=counts["conflict"],
                    material_conflict_count=material_conflict_count,
                    usage_source="unavailable",
                )
            for record in equivalent_records:
                telemetry.emit(
                    "merge_applied",
                    run_id=request.run_id,
                    episode_id=request.episode_id,
                    attempt_id=request.attempt_id,
                    role=request.role,
                    status="committed",
                    input_graph_revision=request.input_graph_revision,
                    merge_count=1,
                    usage_source="unavailable",
                )
        return CommitOutcome(
            accepted=True,
            episode_id=request.episode_id,
            accepted_node_ids=candidate_ids,
            accepted_node_count=len(candidate_ids),
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
