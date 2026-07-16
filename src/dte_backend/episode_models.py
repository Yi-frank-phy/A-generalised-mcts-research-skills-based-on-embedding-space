"""Strict transport-neutral contracts for bounded DTE agent episodes."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, model_validator

from .models import DTEBaseModel
from .relation_models import RelationEpisodeOutput, RelationEpisodePayload


EpisodeRole = Literal["executor", "seed", "judge", "relation", "synthesis"]
EpisodeStatus = Literal["completed", "failed", "timed_out", "cancelled"]
ComparisonProfile = Literal["legacy-explicit", "native-guided", "native-autonomous"]
UsageSource = Literal["provider_reported", "estimated", "unavailable"]
PolicySelector = Literal["user", "main_agent", "run_default"]


class RuntimeLimits(DTEBaseModel):
    wall_clock_seconds: int | None = Field(default=None, ge=1)
    max_retries: int = Field(default=0, ge=0, le=20)
    max_parallelism_hint: int | None = Field(default=None, ge=1)
    max_tool_calls_hint: int | None = Field(default=None, ge=0)
    selected_by: PolicySelector = "run_default"


class ToolPolicy(DTEBaseModel):
    network_allowed: bool = False
    shell_allowed: bool = True
    write_allowed: bool = False
    allowed_write_roots: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_write_roots(self) -> "ToolPolicy":
        if self.allowed_write_roots and not self.write_allowed:
            raise ValueError("allowed_write_roots requires write_allowed=true")
        return self


class ExecutorParentContext(DTEBaseModel):
    """Producer-visible parent context with no controller-owned fields."""

    node_id: str = Field(min_length=1)
    node_type: Literal["candidate", "evidence", "counterexample", "merge"] = "candidate"
    claim: str = Field(min_length=1)
    rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExecutorEpisodePayload(DTEBaseModel):
    parent: ExecutorParentContext
    iteration: int = Field(ge=1)
    constraints: list[str] = Field(default_factory=list)


class ExecutorNodeCandidate(DTEBaseModel):
    """Executor-produced node fields; controller-owned fields are absent by design."""

    node_id: str = Field(min_length=1)
    node_type: Literal["candidate", "evidence", "counterexample", "merge"] = "candidate"
    claim: str = Field(min_length=1)
    rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: Literal["frontier"] = "frontier"


class ExecutorEpisodeOutput(DTEBaseModel):
    nodes: list[ExecutorNodeCandidate] = Field(default_factory=list)
    episode_summary: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)


class JudgeNodeInput(DTEBaseModel):
    """Producer-visible Judge input with no controller mutation authority."""

    node_id: str = Field(min_length=1)
    node_type: Literal["candidate", "evidence", "counterexample", "merge"] = "candidate"
    claim: str = Field(min_length=1)
    rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class JudgeEpisodePayload(DTEBaseModel):
    rubric_version: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    selected_frontier_nodes: list[JudgeNodeInput] = Field(min_length=1)
    required_output_fields: list[
        Literal["node_id", "score", "reasoning", "risks", "uncertainty_evidence"]
    ] = Field(default_factory=lambda: ["node_id", "score", "reasoning", "risks"])

    @model_validator(mode="after")
    def validate_selected_nodes(self) -> "JudgeEpisodePayload":
        node_ids = [node.node_id for node in self.selected_frontier_nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Judge payload selected node IDs must be unique")
        if len(self.required_output_fields) != len(set(self.required_output_fields)):
            raise ValueError("Judge required_output_fields must be unique")
        return self


class JudgeObservation(DTEBaseModel):
    node_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    risks: list[str]
    uncertainty_evidence: list[str] = Field(default_factory=list)


class JudgeEpisodeOutput(DTEBaseModel):
    observations: list[JudgeObservation] = Field(min_length=1)


class RuntimeDiagnostics(DTEBaseModel):
    adapter_name: str = Field(min_length=1)
    transport_name: str = Field(min_length=1)
    profile: ComparisonProfile
    runtime_profile: str | None = None
    model: str | None = None
    wall_clock_ms: int | None = Field(default=None, ge=0)
    queue_or_io_ms: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)
    runtime_reference: str | None = None
    internal_subagent_metadata: dict[str, Any] | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    provider_reported_cost: float | None = Field(default=None, ge=0.0)
    estimated_cost: float | None = Field(default=None, ge=0.0)
    quota_delta: float | None = Field(default=None, ge=0.0)
    usage_source: UsageSource = "unavailable"


class EpisodeRequest(DTEBaseModel):
    episode_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    role: EpisodeRole
    input_graph_revision: int = Field(ge=0)
    selected_node_revisions: dict[str, int]
    objective: str = Field(min_length=1)
    coverage_requirements: list[str] = Field(default_factory=list)
    allowed_output_types: list[Literal["candidate", "evidence", "counterexample", "merge"]]
    output_schema_version: str = Field(min_length=1)
    native_orchestration_allowed: bool = True
    runtime_limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    tool_policy: ToolPolicy | None = None
    transport_hints: dict[str, Any] | None = None
    parent_node_id: str | None = None
    parent_node_revision: int | None = Field(default=None, ge=0)
    max_returned_children: int | None = Field(default=None, ge=0, le=50)
    required_parent_id_on_children: bool = True
    executor_payload: ExecutorEpisodePayload | None = None
    judge_payload: JudgeEpisodePayload | None = None
    relation_payload: RelationEpisodePayload | None = None

    @model_validator(mode="after")
    def validate_role_payload(self) -> "EpisodeRequest":
        if self.role == "executor":
            required = {
                "parent_node_id": self.parent_node_id,
                "parent_node_revision": self.parent_node_revision,
                "max_returned_children": self.max_returned_children,
                "executor_payload": self.executor_payload,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"executor request is missing: {', '.join(missing)}")
            assert self.parent_node_id is not None
            assert self.parent_node_revision is not None
            assert self.executor_payload is not None
            if self.executor_payload.parent.node_id != self.parent_node_id:
                raise ValueError("executor payload parent does not match parent_node_id")
            if self.selected_node_revisions.get(self.parent_node_id) != self.parent_node_revision:
                raise ValueError("selected_node_revisions must include the assigned parent revision")
            if not self.required_parent_id_on_children:
                raise ValueError("Executor request must require the assigned parent ID on children")
            if self.judge_payload is not None:
                raise ValueError("judge_payload requires role='judge'")
            if self.relation_payload is not None:
                raise ValueError("relation_payload requires role='relation'")
        elif self.role == "judge":
            if self.judge_payload is None:
                raise ValueError("judge request is missing judge_payload")
            executor_fields = (
                self.parent_node_id,
                self.parent_node_revision,
                self.max_returned_children,
                self.executor_payload,
            )
            if any(value is not None for value in executor_fields):
                raise ValueError("executor-specific fields require role='executor'")
            if self.required_parent_id_on_children:
                raise ValueError("Judge request must not grant child-parent authority")
            selected_ids = [node.node_id for node in self.judge_payload.selected_frontier_nodes]
            if set(selected_ids) != set(self.selected_node_revisions):
                raise ValueError("Judge payload nodes must exactly match selected_node_revisions")
            if self.allowed_output_types:
                raise ValueError("Judge request must not grant graph output types")
            if self.relation_payload is not None:
                raise ValueError("relation_payload requires role='relation'")
        elif self.role == "relation":
            if self.relation_payload is None:
                raise ValueError("relation request is missing relation_payload")
            executor_fields = (
                self.parent_node_id,
                self.parent_node_revision,
                self.max_returned_children,
                self.executor_payload,
            )
            if any(value is not None for value in executor_fields):
                raise ValueError("executor-specific fields require role='executor'")
            if self.judge_payload is not None:
                raise ValueError("judge_payload requires role='judge'")
            if self.required_parent_id_on_children:
                raise ValueError("Relation request must not grant child-parent authority")
            if self.allowed_output_types:
                raise ValueError("Relation request must not grant graph output types")
            pair_revisions: dict[str, int] = {}
            for pair in self.relation_payload.candidate_pairs:
                pair_revisions[pair.left.node_id] = pair.left_node_revision
                pair_revisions[pair.right.node_id] = pair.right_node_revision
            if pair_revisions != self.selected_node_revisions:
                raise ValueError("Relation payload nodes must exactly match selected_node_revisions")
        elif any(
            value is not None
            for value in (
                self.parent_node_id,
                self.parent_node_revision,
                self.max_returned_children,
                self.executor_payload,
            )
        ):
            raise ValueError("executor-specific fields require role='executor'")
        elif self.judge_payload is not None:
            raise ValueError("judge_payload requires role='judge'")
        elif self.relation_payload is not None:
            raise ValueError("relation_payload requires role='relation'")
        if len(set(self.allowed_output_types)) != len(self.allowed_output_types):
            raise ValueError("allowed_output_types must not contain duplicates")
        return self


class EpisodeResult(DTEBaseModel):
    episode_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    role: EpisodeRole
    input_graph_revision: int = Field(ge=0)
    selected_node_revisions: dict[str, int]
    status: EpisodeStatus
    structured_output: ExecutorEpisodeOutput | JudgeEpisodeOutput | RelationEpisodeOutput | None
    runtime_diagnostics: RuntimeDiagnostics
    output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_completion_shape(self) -> "EpisodeResult":
        if self.status == "completed" and self.structured_output is None:
            raise ValueError("completed result requires structured_output")
        if self.status != "completed" and self.structured_output is not None:
            raise ValueError("non-completed result must not contain structured_output")
        return self


class CommitOutcome(DTEBaseModel):
    accepted: bool
    episode_id: str
    accepted_node_ids: list[str] = Field(default_factory=list)
    accepted_node_count: int = Field(default=0, ge=0)
    graph_revision_before: int = Field(ge=0)
    graph_revision_after: int = Field(ge=0)
    rejection_reason: str | None = None


def canonical_json_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_output_hash(
    output: ExecutorEpisodeOutput | JudgeEpisodeOutput | RelationEpisodeOutput | None,
    schema_version: str,
) -> str:
    payload = {
        "schema_version": schema_version,
        "structured_output": None if output is None else output.model_dump(mode="json"),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
