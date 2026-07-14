"""Unified adapters for transport-neutral bounded AgentEpisodes."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol
import uuid

from .adapter import ExecutorAdapter, build_subprocess_adapter, validate_search_node_output
from .episode_commit import EpisodeGraph, commit_episode_result
from .episode_models import (
    CommitOutcome,
    EpisodeRequest,
    EpisodeResult,
    ExecutorEpisodeOutput,
    ExecutorNodeCandidate,
    JudgeEpisodePayload,
    JudgeNodeInput,
    RuntimeDiagnostics,
    RuntimeLimits,
    compute_output_hash,
)
from .models import ExpansionRequest, SearchNode
from .relation_models import (
    RelationCandidate,
    RelationEpisodePayload,
    RelationEvidenceInput,
    RelationNodeInput,
    RelationPairInput,
)
from .telemetry import EpisodeEventLog


class AgentEpisodeAdapter(Protocol):
    """Stable transport-neutral interface; internal topology is deliberately absent."""

    adapter_name: str
    transport_name: str

    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        """Run one bounded episode and return one complete result envelope."""


def build_executor_episode_request(
    graph: EpisodeGraph,
    parent: SearchNode,
    *,
    run_id: str,
    iteration: int,
    max_returned_children: int,
    objective: str,
    constraints: list[str] | None = None,
    coverage_requirements: list[str] | None = None,
    allowed_output_types: list[str] | None = None,
    native_orchestration_allowed: bool = True,
    runtime_limits: RuntimeLimits | None = None,
    tool_policy: Any = None,
    transport_hints: dict[str, Any] | None = None,
) -> EpisodeRequest:
    """Create a producer-safe Executor grant from committed graph state."""

    if parent.node_id not in graph.node_revisions:
        raise ValueError("cannot grant an episode for an uncommitted parent")
    producer_parent = {
        key: value
        for key, value in parent.model_dump(mode="json").items()
        if key
        not in {
            "local_embedding",
            "judge_reasoning",
            "judge_risks",
            "judge_uncertainty_evidence",
            "judge_result_provenance",
            "score",
            "density",
            "uncertainty",
            "ucb_score",
            "expansion_budget",
            "status",
        }
    }
    parent_revision = graph.node_revisions[parent.node_id]
    return EpisodeRequest(
        episode_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        run_id=run_id,
        role="executor",
        input_graph_revision=graph.revision,
        selected_node_revisions={parent.node_id: parent_revision},
        objective=objective,
        coverage_requirements=coverage_requirements or [],
        allowed_output_types=allowed_output_types or ["candidate", "evidence", "counterexample", "merge"],
        output_schema_version="executor-output.v1",
        native_orchestration_allowed=native_orchestration_allowed,
        runtime_limits=runtime_limits or RuntimeLimits(),
        tool_policy=tool_policy,
        transport_hints=transport_hints,
        parent_node_id=parent.node_id,
        parent_node_revision=parent_revision,
        max_returned_children=max_returned_children,
        required_parent_id_on_children=True,
        executor_payload={
            "parent": producer_parent,
            "iteration": iteration,
            "constraints": constraints or [],
        },
    )


def build_judge_episode_request(
    graph: EpisodeGraph,
    nodes: list[SearchNode],
    *,
    run_id: str,
    problem: str,
    goal: str,
    constraints: list[str] | None = None,
    rubric_version: str = "research-potential.v1",
    native_orchestration_allowed: bool = True,
    runtime_limits: RuntimeLimits | None = None,
    tool_policy: Any = None,
    transport_hints: dict[str, Any] | None = None,
) -> EpisodeRequest:
    """Create one bounded observable Judge grant from committed frontier nodes."""

    if not nodes:
        raise ValueError("Judge grant requires at least one selected node")
    selected_revisions: dict[str, int] = {}
    selected_inputs: list[JudgeNodeInput] = []
    for node in nodes:
        if node.status != "frontier" or node.node_id not in graph.node_revisions:
            raise ValueError("Judge grants require committed frontier nodes")
        selected_revisions[node.node_id] = graph.node_revisions[node.node_id]
        selected_inputs.append(
            JudgeNodeInput.model_validate(
                node.model_dump(
                    mode="json",
                    include={
                        "node_id",
                        "node_type",
                        "claim",
                        "rationale",
                        "assumptions",
                        "evidence",
                        "risks",
                        "confidence",
                    },
                )
            )
        )
    return EpisodeRequest(
        episode_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        run_id=run_id,
        role="judge",
        input_graph_revision=graph.revision,
        selected_node_revisions=selected_revisions,
        objective=f"Judge research potential for: {goal}",
        coverage_requirements=[
            "score every granted node exactly once",
            "state observable reasoning and material risks",
            "do not return controller-owned geometry, allocation, revision, stopping, or synthesis fields",
        ],
        allowed_output_types=[],
        output_schema_version="judge-output.v1",
        native_orchestration_allowed=native_orchestration_allowed,
        runtime_limits=runtime_limits or RuntimeLimits(),
        tool_policy=tool_policy,
        transport_hints=transport_hints,
        required_parent_id_on_children=False,
        judge_payload=JudgeEpisodePayload(
            rubric_version=rubric_version,
            problem=problem,
            goal=goal,
            constraints=constraints or [],
            selected_frontier_nodes=selected_inputs,
            required_output_fields=["node_id", "score", "reasoning", "risks"],
        ),
    )


def _relation_node_input(node: SearchNode, revision: int) -> RelationNodeInput:
    return RelationNodeInput(
        node_id=node.node_id,
        node_revision=revision,
        node_type=node.node_type,
        claim=node.claim,
        rationale=node.rationale,
        assumptions=list(node.assumptions),
        evidence=[
            RelationEvidenceInput(
                evidence_ref=f"{node.node_id}:evidence:{index}",
                text=value,
            )
            for index, value in enumerate(node.evidence)
        ],
        risks=list(node.risks),
        confidence=node.confidence,
        judge_reasoning=node.judge_reasoning,
        judge_risks=list(node.judge_risks),
        judge_uncertainty_evidence=list(node.judge_uncertainty_evidence),
        judge_result_provenance=(
            None if node.judge_result_provenance is None else dict(node.judge_result_provenance)
        ),
        parent_ids=list(node.parent_ids),
    )


def build_relation_episode_request(
    graph: EpisodeGraph,
    candidates: list[RelationCandidate],
    *,
    run_id: str,
    problem: str,
    goal: str,
    constraints: list[str],
    provisional_synthesis_node_ids: list[str],
    max_relation_pairs_per_episode: int,
    rubric_version: str = "semantic-relation.v1",
    native_orchestration_allowed: bool = True,
    runtime_limits: RuntimeLimits | None = None,
    tool_policy: Any = None,
    transport_hints: dict[str, Any] | None = None,
) -> EpisodeRequest:
    """Build one bounded Relation grant from exact backend-selected candidates."""

    if not candidates:
        raise ValueError("Relation grant requires at least one candidate")
    if len(candidates) > max_relation_pairs_per_episode:
        raise ValueError("Relation grant exceeds max_relation_pairs_per_episode")
    selected_revisions: dict[str, int] = {}
    pair_inputs: list[RelationPairInput] = []
    for candidate in candidates:
        left = graph.node_by_id(candidate.left_node_id)
        right = graph.node_by_id(candidate.right_node_id)
        if left is None or right is None:
            raise ValueError("Relation candidate references an uncommitted node")
        left_revision = graph.node_revisions[left.node_id]
        right_revision = graph.node_revisions[right.node_id]
        if (
            candidate.left_node_revision != left_revision
            or candidate.right_node_revision != right_revision
        ):
            raise ValueError("Relation candidate revision is stale")
        selected_revisions[left.node_id] = left_revision
        selected_revisions[right.node_id] = right_revision
        pair_inputs.append(
            RelationPairInput(
                candidate_id=candidate.candidate_id,
                left=_relation_node_input(left, left_revision),
                right=_relation_node_input(right, right_revision),
                left_node_revision=left_revision,
                right_node_revision=right_revision,
                candidate_reason=candidate.candidate_reason,
                priority=candidate.priority,
                material_to_synthesis=candidate.material_to_synthesis,
            )
        )
    return EpisodeRequest(
        episode_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        run_id=run_id,
        role="relation",
        input_graph_revision=graph.revision,
        selected_node_revisions=selected_revisions,
        objective="Classify only the granted committed-node Relation candidates",
        coverage_requirements=[
            "classify every granted candidate exactly once",
            "use only granted node pairs and evidence references",
            "return observations or proposals, never graph mutations or controller fields",
        ],
        allowed_output_types=[],
        output_schema_version="relation-output.v1",
        native_orchestration_allowed=native_orchestration_allowed,
        runtime_limits=runtime_limits or RuntimeLimits(),
        tool_policy=tool_policy,
        transport_hints=transport_hints,
        required_parent_id_on_children=False,
        relation_payload=RelationEpisodePayload(
            rubric_version=rubric_version,
            problem=problem,
            goal=goal,
            constraints=list(constraints),
            candidate_pairs=pair_inputs,
            provisional_synthesis_node_ids=list(provisional_synthesis_node_ids),
        ),
    )


def _parent_as_search_node(request: EpisodeRequest) -> SearchNode:
    if request.executor_payload is None:
        raise ValueError("Executor adapter requires executor_payload")
    return SearchNode.model_validate(request.executor_payload.parent.model_dump(mode="json"))


def _completed_result(
    request: EpisodeRequest,
    output: ExecutorEpisodeOutput,
    diagnostics: RuntimeDiagnostics,
) -> EpisodeResult:
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role=request.role,
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status="completed",
        structured_output=output,
        runtime_diagnostics=diagnostics,
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


class LegacyExecutorEpisodeAdapter:
    """Wrap an existing ExpansionRequest -> SearchNodes adapter without replacing it."""

    adapter_name = "legacy-executor-bridge"
    transport_name = "legacy-callable"

    def __init__(self, executor_adapter: ExecutorAdapter, *, runtime_profile: str | None = None) -> None:
        self.executor_adapter = executor_adapter
        self.runtime_profile = runtime_profile

    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        if request.role != "executor" or request.executor_payload is None:
            raise ValueError("legacy bridge supports Executor episodes only")
        if request.tool_policy is not None:
            raise ValueError("legacy Executor adapter cannot enforce ToolPolicy; omit it or use a policy-aware adapter")
        assert request.max_returned_children is not None
        started = time.monotonic()
        expansion_request = ExpansionRequest(
            parent=_parent_as_search_node(request),
            child_count=max(1, request.max_returned_children),
            iteration=request.executor_payload.iteration,
        )
        if hasattr(self.executor_adapter, "expand"):
            children = self.executor_adapter.expand(expansion_request)  # type: ignore[union-attr]
        else:
            children = self.executor_adapter(expansion_request)
        children = validate_search_node_output(
            expansion_request.parent,
            request.max_returned_children,
            children,
        )
        output = ExecutorEpisodeOutput(
            nodes=[
                ExecutorNodeCandidate.model_validate(
                    child.model_dump(
                        mode="json",
                        exclude={
                            "local_embedding",
                            "judge_reasoning",
                            "judge_risks",
                            "judge_uncertainty_evidence",
                            "judge_result_provenance",
                            "score",
                            "density",
                            "uncertainty",
                            "ucb_score",
                            "expansion_budget",
                        },
                    )
                )
                for child in children
            ]
        )
        diagnostics = RuntimeDiagnostics(
            adapter_name=self.adapter_name,
            transport_name=self.transport_name,
            profile="legacy-explicit",
            runtime_profile=self.runtime_profile,
            wall_clock_ms=round((time.monotonic() - started) * 1000),
            usage_source="unavailable",
        )
        return _completed_result(request, output, diagnostics)


class CommandAgentEpisodeAdapter(LegacyExecutorEpisodeAdapter):
    """Command/subprocess fallback exposed through AgentEpisodeAdapter."""

    adapter_name = "command-agent-episode"
    transport_name = "subprocess"

    def __init__(self, command: Sequence[str], timeout_seconds: float = 120.0) -> None:
        super().__init__(build_subprocess_adapter(command, timeout=timeout_seconds))
        self.command = list(command)
        self.timeout_seconds = timeout_seconds

    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        requested_timeout = request.runtime_limits.wall_clock_seconds
        effective_timeout = self.timeout_seconds
        if requested_timeout is not None:
            effective_timeout = min(effective_timeout, float(requested_timeout))
        bridge = LegacyExecutorEpisodeAdapter(
            build_subprocess_adapter(self.command, timeout=effective_timeout),
            runtime_profile="command-fallback",
        )
        result = bridge.run_episode(request)
        result.runtime_diagnostics.adapter_name = self.adapter_name
        result.runtime_diagnostics.transport_name = self.transport_name
        return result


class NativeStubEpisodeAdapter:
    """Deterministic native-shaped test double; it is not a native integration."""

    adapter_name = "deterministic-native-stub"
    transport_name = "in-process-stub"

    def __init__(
        self,
        output_factory: Callable[[EpisodeRequest], ExecutorEpisodeOutput] | None = None,
        *,
        profile: str = "native-autonomous",
    ) -> None:
        self.output_factory = output_factory or (lambda request: ExecutorEpisodeOutput())
        self.profile = profile

    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        started = time.monotonic()
        output = self.output_factory(request)
        diagnostics = RuntimeDiagnostics(
            adapter_name=self.adapter_name,
            transport_name=self.transport_name,
            profile=self.profile,
            runtime_profile="deterministic-test-double",
            wall_clock_ms=round((time.monotonic() - started) * 1000),
            usage_source="unavailable",
        )
        return _completed_result(request, output, diagnostics)


def run_and_commit_episode(
    graph: EpisodeGraph,
    request: EpisodeRequest,
    adapter: AgentEpisodeAdapter,
    telemetry: EpisodeEventLog | None = None,
) -> CommitOutcome:
    """Emit lifecycle events, run one adapter call, then use the sole commit boundary."""

    adapter_name = getattr(adapter, "adapter_name", adapter.__class__.__name__)
    transport_name = getattr(adapter, "transport_name", "unknown")
    if telemetry is not None:
        telemetry.emit(
            "episode_granted",
            run_id=request.run_id,
            episode_id=request.episode_id,
            attempt_id=request.attempt_id,
            role=request.role,
            adapter_name=adapter_name,
            transport_name=transport_name,
            status="granted",
            input_graph_revision=request.input_graph_revision,
        )
        telemetry.emit(
            "episode_started",
            run_id=request.run_id,
            episode_id=request.episode_id,
            attempt_id=request.attempt_id,
            role=request.role,
            adapter_name=adapter_name,
            transport_name=transport_name,
            status="started",
            input_graph_revision=request.input_graph_revision,
        )
    try:
        result = adapter.run_episode(request)
    except Exception as exc:
        if telemetry is not None:
            telemetry.emit(
                "episode_failed",
                run_id=request.run_id,
                episode_id=request.episode_id,
                attempt_id=request.attempt_id,
                role=request.role,
                adapter_name=adapter_name,
                transport_name=transport_name,
                status="failed",
                input_graph_revision=request.input_graph_revision,
                rejection_reason=str(exc),
            )
            telemetry.emit(
                "output_rejected",
                run_id=request.run_id,
                episode_id=request.episode_id,
                attempt_id=request.attempt_id,
                role=request.role,
                adapter_name=adapter_name,
                transport_name=transport_name,
                status="rejected",
                input_graph_revision=request.input_graph_revision,
                accepted_node_count=0,
                rejection_reason=f"runtime call failed: {exc}",
                schema_valid=False,
            )
        return CommitOutcome(
            accepted=False,
            episode_id=request.episode_id,
            graph_revision_before=graph.revision,
            graph_revision_after=graph.revision,
            rejection_reason=f"runtime call failed: {exc}",
        )

    diagnostics = result.runtime_diagnostics
    if result.structured_output is None:
        returned = 0
    elif isinstance(result.structured_output, ExecutorEpisodeOutput):
        returned = len(result.structured_output.nodes)
    else:
        returned = len(result.structured_output.observations)
    if telemetry is not None:
        telemetry.emit(
            "episode_completed" if result.status == "completed" else "episode_failed",
            run_id=request.run_id,
            episode_id=request.episode_id,
            attempt_id=request.attempt_id,
            role=request.role,
            adapter_name=diagnostics.adapter_name,
            transport_name=diagnostics.transport_name,
            profile=diagnostics.profile,
            runtime_profile=diagnostics.runtime_profile,
            model=diagnostics.model,
            wall_clock_ms=diagnostics.wall_clock_ms,
            queue_or_io_ms=diagnostics.queue_or_io_ms,
            retry_count=diagnostics.retry_count,
            status=result.status,
            input_graph_revision=result.input_graph_revision,
            returned_node_count=returned,
            input_tokens=diagnostics.input_tokens,
            output_tokens=diagnostics.output_tokens,
            cached_tokens=diagnostics.cached_tokens,
            provider_reported_cost=diagnostics.provider_reported_cost,
            estimated_cost=diagnostics.estimated_cost,
            quota_delta=diagnostics.quota_delta,
            usage_source=diagnostics.usage_source,
        )
    return commit_episode_result(graph, request, result, telemetry=telemetry)
