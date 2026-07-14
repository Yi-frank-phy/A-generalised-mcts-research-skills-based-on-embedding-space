"""Strict models for App-native Relation episodes and readiness state.

Relation output is an observation layer.  These models deliberately contain no
graph-mutation, controller-score, allocation, or terminal-action authority.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from .models import DTEBaseModel


RelationType = Literal["equivalent", "complementary", "conflict", "independent"]
RelationCandidateReason = Literal[
    "exact_duplicate",
    "embedding_close",
    "high_score_near_tie",
    "shared_evidence_divergence",
    "synthesis_set_overlap",
    "potential_material_conflict",
    "entropy_plateau",
    "manual_operator_request",
]
RelationCandidatePriority = Literal["critical", "high", "medium", "low"]
RelationCandidateStatus = Literal["pending", "granted", "resolved", "superseded", "invalidated"]


def canonical_pair(left_node_id: str, right_node_id: str) -> tuple[str, str]:
    if left_node_id == right_node_id:
        raise ValueError("Relation candidate requires two distinct node IDs")
    return tuple(sorted((left_node_id, right_node_id)))  # type: ignore[return-value]


def stable_relation_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


class RelationCandidate(DTEBaseModel):
    candidate_id: str = Field(min_length=1)
    left_node_id: str = Field(min_length=1)
    right_node_id: str = Field(min_length=1)
    left_node_revision: int = Field(ge=0)
    right_node_revision: int = Field(ge=0)
    candidate_reason: RelationCandidateReason
    priority: RelationCandidatePriority
    material_to_synthesis: bool
    created_from_graph_revision: int = Field(ge=0)
    status: RelationCandidateStatus = "pending"
    granted_episode_id: str | None = None
    granted_attempt_id: str | None = None
    resolved_relation_record_id: str | None = None

    @model_validator(mode="after")
    def validate_pair_and_lifecycle(self) -> "RelationCandidate":
        left, right = canonical_pair(self.left_node_id, self.right_node_id)
        if (left, right) != (self.left_node_id, self.right_node_id):
            raise ValueError("Relation candidate node pair must use canonical ordering")
        if self.status == "granted" and (not self.granted_episode_id or not self.granted_attempt_id):
            raise ValueError("granted Relation candidate requires episode and attempt provenance")
        if self.status == "resolved" and not self.resolved_relation_record_id:
            raise ValueError("resolved Relation candidate requires a relation record")
        return self


class RelationEvidenceInput(DTEBaseModel):
    evidence_ref: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RelationNodeInput(DTEBaseModel):
    node_id: str = Field(min_length=1)
    node_revision: int = Field(ge=0)
    node_type: Literal["candidate", "evidence", "counterexample", "merge"]
    claim: str = Field(min_length=1)
    rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[RelationEvidenceInput] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    judge_reasoning: str | None = None
    judge_risks: list[str] = Field(default_factory=list)
    judge_uncertainty_evidence: list[str] = Field(default_factory=list)
    judge_result_provenance: dict[str, str] | None = None
    parent_ids: list[str] = Field(default_factory=list)


class RelationPairInput(DTEBaseModel):
    candidate_id: str = Field(min_length=1)
    left: RelationNodeInput
    right: RelationNodeInput
    left_node_revision: int = Field(ge=0)
    right_node_revision: int = Field(ge=0)
    candidate_reason: RelationCandidateReason
    priority: RelationCandidatePriority
    material_to_synthesis: bool

    @model_validator(mode="after")
    def validate_pair(self) -> "RelationPairInput":
        left, right = canonical_pair(self.left.node_id, self.right.node_id)
        if (left, right) != (self.left.node_id, self.right.node_id):
            raise ValueError("Relation pair input must use canonical ordering")
        if self.left.node_revision != self.left_node_revision:
            raise ValueError("left node revision does not match pair revision")
        if self.right.node_revision != self.right_node_revision:
            raise ValueError("right node revision does not match pair revision")
        return self


class RelationEpisodePayload(DTEBaseModel):
    relation_schema_version: Literal["relation-payload.v1"] = "relation-payload.v1"
    rubric_version: str = Field(default="semantic-relation.v1", min_length=1)
    problem: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    candidate_pairs: list[RelationPairInput] = Field(min_length=1)
    provisional_synthesis_node_ids: list[str]
    required_output_fields: list[str] = Field(
        default_factory=lambda: [
            "candidate_id",
            "left_node_id",
            "right_node_id",
            "relation_type",
            "confidence",
            "rationale",
            "evidence_refs",
            "materiality_assessment",
        ]
    )

    @model_validator(mode="after")
    def validate_candidates(self) -> "RelationEpisodePayload":
        candidate_ids = [pair.candidate_id for pair in self.candidate_pairs]
        unordered_pairs = [(pair.left.node_id, pair.right.node_id) for pair in self.candidate_pairs]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Relation payload candidate IDs must be unique")
        if len(unordered_pairs) != len(set(unordered_pairs)):
            raise ValueError("Relation payload unordered pairs must be unique")
        if len(self.required_output_fields) != len(set(self.required_output_fields)):
            raise ValueError("Relation required_output_fields must be unique")
        return self


class DiscriminatorTaskProposal(DTEBaseModel):
    task_type: Literal[
        "counterexample_search",
        "formal_derivation",
        "numerical_check",
        "source_verification",
        "boundary_condition_comparison",
    ]
    objective: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    material_to_synthesis: bool


class RelationObservation(DTEBaseModel):
    candidate_id: str = Field(min_length=1)
    left_node_id: str = Field(min_length=1)
    right_node_id: str = Field(min_length=1)
    relation_type: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    materiality_assessment: Literal["material", "non_material", "uncertain"]

    merge_recommended: bool | None = None
    canonicality_factors: list[str] = Field(default_factory=list)
    information_loss_risk: str | None = None

    complementarity_summary: str | None = None
    recommended_joint_use: str | None = None
    distinct_contributions: list[str] = Field(default_factory=list)

    conflict_summary: str | None = None
    conflicting_assumptions: list[str] = Field(default_factory=list)
    conflicting_claims: list[str] = Field(default_factory=list)
    possible_resolution: str | None = None
    disclosure_required: bool = False
    discriminator_task_proposal: DiscriminatorTaskProposal | None = None

    independence_summary: str | None = None

    @model_validator(mode="after")
    def validate_relation_specific_fields(self) -> "RelationObservation":
        left, right = canonical_pair(self.left_node_id, self.right_node_id)
        if (left, right) != (self.left_node_id, self.right_node_id):
            raise ValueError("Relation observation node pair must use canonical ordering")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("Relation observation evidence_refs must be unique")
        if self.relation_type == "equivalent" and self.merge_recommended is None:
            raise ValueError("equivalent observation requires merge_recommended")
        if self.relation_type == "complementary" and not self.complementarity_summary:
            raise ValueError("complementary observation requires complementarity_summary")
        if self.relation_type == "conflict" and not self.conflict_summary:
            raise ValueError("conflict observation requires conflict_summary")
        if self.relation_type == "independent" and not self.independence_summary:
            raise ValueError("independent observation requires independence_summary")
        return self


class RelationEpisodeOutput(DTEBaseModel):
    observations: list[RelationObservation] = Field(min_length=1)


class RelationRecord(DTEBaseModel):
    relation_record_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    left_node_id: str
    right_node_id: str
    relation_type: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)
    material_to_synthesis: bool
    materiality_assessment: Literal["material", "non_material", "uncertain"]
    observation: RelationObservation
    disclosure_required: bool = False
    episode_id: str
    attempt_id: str
    input_graph_revision: int = Field(ge=0)
    selected_node_revisions: dict[str, int]
    output_hash: str
    schema_version: str
    committed_at: str


class MergeApplicationRecord(DTEBaseModel):
    merge_application_id: str
    relation_record_id: str
    canonical_node_id: str
    absorbed_node_ids: list[str]
    source_node_ids: list[str]
    source_node_revisions: dict[str, int]
    applied_graph_revision: int = Field(ge=0)
    applied_at: str


class ProvisionalSynthesisSelection(DTEBaseModel):
    selected_node_ids: list[str]
    selection_reason: str
    selection_revision: int = Field(ge=0)


class SynthesisReadinessRecord(DTEBaseModel):
    schema_version: Literal["synthesis-readiness.v1"] = "synthesis-readiness.v1"
    graph_revision: int = Field(ge=0)
    provisional_selected_node_ids: list[str]
    blocking_candidate_ids: list[str]
    unresolved_material_conflicts: list[str]
    disclosure_required_conflicts: list[str]
    unresolved_nonblocking_candidates: list[str]
    duplicate_groups: list[list[str]]
    ready: bool
    reason: str
    evaluated_at: str
