"""Strict schemas for committed epistemic provenance and researcher handoff.

The models in this module describe observable, source-labelled claims and
dependencies.  None of them is a scientific verifier or controller signal.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from .models import DTEBaseModel
from .relation_models import RelationType


EpistemicSourceType = Literal[
    "agent_reported",
    "external_artifact_backed",
    "backend_derived",
]
EpisodeEpistemicSourceType = Literal[
    "agent_reported",
    "external_artifact_backed",
]
EpistemicStatementType = Literal[
    "claim",
    "assumption",
    "evidence",
    "open_question",
    "failure_mode",
    "heuristic",
]
EpistemicRelationType = Literal[
    "supports",
    "challenges",
    "requires",
    "qualifies",
    "contradicts",
    "derived_from",
]
PathEpistemicDisposition = Literal[
    "blocked_by_assumption",
    "counterexample_found",
    "challenged",
    "contradicted",
    "inconclusive",
    "insufficient_support",
]
SearchDisposition = Literal[
    "selected",
    "not_selected",
    "merged",
    "closed",
    "out_of_budget",
    "not_explored",
]
MAX_EPISTEMIC_STATEMENTS_PER_EPISODE = 24
MAX_EPISTEMIC_EDGES_PER_EPISODE = 32
MAX_PATH_DISPOSITIONS_PER_EPISODE = 12
MAX_BASIS_REFS_PER_RECORD = 16
MAX_EPISTEMIC_REF_LENGTH = 1024


def _validate_basis_refs(refs: list[str], label: str) -> None:
    if len(refs) != len(set(refs)):
        raise ValueError(f"{label} basis_refs must be unique")
    if any(not ref.strip() for ref in refs):
        raise ValueError(f"{label} basis_refs must be non-empty")
    if any(len(ref) > MAX_EPISTEMIC_REF_LENGTH for ref in refs):
        raise ValueError(
            f"{label} basis_refs must be at most {MAX_EPISTEMIC_REF_LENGTH} characters"
        )


def stable_epistemic_id(
    prefix: str,
    *,
    run_id: str,
    episode_id: str,
    attempt_id: str,
    output_hash: str,
    local_id: str,
    record_type: str,
) -> str:
    """Bind one stable identity to its exact committed output context."""

    payload = "\x1f".join(
        (run_id, episode_id, attempt_id, output_hash, local_id, record_type)
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


class EpistemicStatementContribution(DTEBaseModel):
    local_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    statement_type: EpistemicStatementType
    text: str = Field(min_length=1, max_length=4000)
    target_node_id: str = Field(min_length=1)
    source_type: EpisodeEpistemicSourceType
    basis_refs: list[str] = Field(
        default_factory=list,
        max_length=MAX_BASIS_REFS_PER_RECORD,
    )

    @model_validator(mode="after")
    def validate_content(self) -> "EpistemicStatementContribution":
        if not self.text.strip():
            raise ValueError("epistemic statement text must be substantive")
        _validate_basis_refs(self.basis_refs, "epistemic statement")
        return self


class EpistemicEdgeContribution(DTEBaseModel):
    local_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    source_ref: str = Field(min_length=1, max_length=1024)
    target_ref: str = Field(min_length=1, max_length=1024)
    relation_type: EpistemicRelationType
    source_type: EpisodeEpistemicSourceType
    basis_refs: list[str] = Field(
        default_factory=list,
        max_length=MAX_BASIS_REFS_PER_RECORD,
    )
    explanation: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_content(self) -> "EpistemicEdgeContribution":
        if not self.source_ref.strip() or not self.target_ref.strip():
            raise ValueError("epistemic edge endpoints must be non-empty")
        if not self.explanation.strip():
            raise ValueError("epistemic edge explanation must be substantive")
        _validate_basis_refs(self.basis_refs, "epistemic edge")
        return self


class PathDispositionContribution(DTEBaseModel):
    local_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    target_node_id: str = Field(min_length=1)
    epistemic_disposition: PathEpistemicDisposition
    source_type: EpisodeEpistemicSourceType
    basis_refs: list[str] = Field(
        default_factory=list,
        max_length=MAX_BASIS_REFS_PER_RECORD,
    )
    explanation: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_disposition(self) -> "PathDispositionContribution":
        if not self.explanation.strip():
            raise ValueError("path disposition explanation must be substantive")
        _validate_basis_refs(self.basis_refs, "path disposition")
        if self.epistemic_disposition in {
            "counterexample_found",
            "contradicted",
        } and not self.basis_refs:
            raise ValueError(
                f"{self.epistemic_disposition} requires non-empty basis_refs"
            )
        return self


class EpistemicContributionBundle(DTEBaseModel):
    schema_version: Literal["dte-epistemic-contributions.v1"] = (
        "dte-epistemic-contributions.v1"
    )
    statements: list[EpistemicStatementContribution] = Field(
        default_factory=list,
        max_length=MAX_EPISTEMIC_STATEMENTS_PER_EPISODE,
    )
    edges: list[EpistemicEdgeContribution] = Field(
        default_factory=list,
        max_length=MAX_EPISTEMIC_EDGES_PER_EPISODE,
    )
    path_dispositions: list[PathDispositionContribution] = Field(
        default_factory=list,
        max_length=MAX_PATH_DISPOSITIONS_PER_EPISODE,
    )

    @model_validator(mode="after")
    def validate_bundle(self) -> "EpistemicContributionBundle":
        records = [*self.statements, *self.edges, *self.path_dispositions]
        if not records:
            raise ValueError(
                "empty epistemic_contributions must be omitted rather than submitted"
            )
        local_ids = [record.local_id for record in records]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError(
                "epistemic contribution local_id must be unique within the output"
            )
        statement_ids = {statement.local_id for statement in self.statements}
        refs = [
            ref
            for record in records
            for ref in (
                (
                    [record.source_ref, record.target_ref]
                    if isinstance(record, EpistemicEdgeContribution)
                    else []
                )
                + list(record.basis_refs)
            )
        ]
        for ref in refs:
            if not ref.startswith("local-statement:"):
                continue
            local_id = ref.removeprefix("local-statement:")
            if local_id not in statement_ids:
                raise ValueError(
                    f"local epistemic reference does not identify a statement: {ref}"
                )
        return self


class EpistemicStatementRecordV1(DTEBaseModel):
    schema_version: Literal["dte-epistemic-statement.v1"] = (
        "dte-epistemic-statement.v1"
    )
    statement_id: str = Field(min_length=1)
    local_id: str = Field(min_length=1)
    statement_type: EpistemicStatementType
    text: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    source_type: EpistemicSourceType
    basis_refs: list[str] = Field(default_factory=list)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    role: Literal["executor", "judge"]
    output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_at: str = Field(min_length=1)


class EpistemicEdgeRecordV1(DTEBaseModel):
    schema_version: Literal["dte-epistemic-edge.v1"] = "dte-epistemic-edge.v1"
    edge_id: str = Field(min_length=1)
    local_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    target_ref: str = Field(min_length=1)
    relation_type: EpistemicRelationType
    source_type: EpistemicSourceType
    basis_refs: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    role: Literal["executor", "judge"]
    output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_at: str = Field(min_length=1)


class PathDispositionRecordV1(DTEBaseModel):
    schema_version: Literal["dte-path-epistemic-disposition.v1"] = (
        "dte-path-epistemic-disposition.v1"
    )
    disposition_id: str = Field(min_length=1)
    local_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    epistemic_disposition: PathEpistemicDisposition
    source_type: EpistemicSourceType
    basis_refs: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    role: Literal["executor", "judge"]
    output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_at: str = Field(min_length=1)


class EpistemicLedgerV1(DTEBaseModel):
    schema_version: Literal["dte-epistemic-ledger.v1"] = (
        "dte-epistemic-ledger.v1"
    )
    statements: list[EpistemicStatementRecordV1] = Field(default_factory=list)
    edges: list[EpistemicEdgeRecordV1] = Field(default_factory=list)
    path_dispositions: list[PathDispositionRecordV1] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_global_ids(self) -> "EpistemicLedgerV1":
        ids = [
            *(item.statement_id for item in self.statements),
            *(item.edge_id for item in self.edges),
            *(item.disposition_id for item in self.path_dispositions),
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("epistemic ledger contains duplicate stable IDs")
        return self


class RelationEpistemicProjectionV1(DTEBaseModel):
    relation_record_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    left_node_id: str = Field(min_length=1)
    right_node_id: str = Field(min_length=1)
    relation_type: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)
    material_to_synthesis: bool
    materiality_assessment: Literal["material", "non_material", "uncertain"]
    disclosure_required: bool
    conflict_summary: str | None = None
    source_type: Literal["agent_reported"] = "agent_reported"
    episode_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)


class MergeEpistemicProjectionV1(DTEBaseModel):
    merge_application_id: str = Field(min_length=1)
    relation_record_id: str = Field(min_length=1)
    canonical_node_id: str = Field(min_length=1)
    absorbed_node_ids: list[str]
    source_node_ids: list[str]
    source_type: Literal["backend_derived"] = "backend_derived"


class NodeEpistemicSummaryV1(DTEBaseModel):
    schema_version: Literal["dte-node-epistemic-summary.v1"] = (
        "dte-node-epistemic-summary.v1"
    )
    node_id: str = Field(min_length=1)
    claim_ref: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    selected_for_synthesis: bool
    search_dispositions: list[SearchDisposition]
    search_disposition_source_type: Literal["backend_derived"] = "backend_derived"
    epistemic_dispositions: list[PathEpistemicDisposition]
    epistemic_disposition_records: list[PathDispositionRecordV1] = Field(
        default_factory=list
    )
    statement_refs: list[str] = Field(default_factory=list)
    supporting_record_refs: list[str] = Field(default_factory=list)
    challenging_record_refs: list[str] = Field(default_factory=list)
    required_assumption_refs: list[str] = Field(default_factory=list)
    derived_from_refs: list[str] = Field(default_factory=list)
    unresolved_dependency_refs: list[str] = Field(default_factory=list)
    relation_record_ids: list[str] = Field(default_factory=list)
    merge_application_ids: list[str] = Field(default_factory=list)


class EpistemicDependencyGraphV1(DTEBaseModel):
    schema_version: Literal["dte-epistemic-dependency-graph.v1"] = (
        "dte-epistemic-dependency-graph.v1"
    )
    run_id: str = Field(min_length=1)
    node_claim_refs: list[str]
    statements: list[EpistemicStatementRecordV1]
    edges: list[EpistemicEdgeRecordV1]
    path_dispositions: list[PathDispositionRecordV1]
    relation_projections: list[RelationEpistemicProjectionV1]
    merge_projections: list[MergeEpistemicProjectionV1]


class SelectedClaimHandoffV1(DTEBaseModel):
    node_id: str = Field(min_length=1)
    claim_ref: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    claim_origin: Literal["initial_node", "executor_episode"]
    claim_source_type: Literal["agent_reported"] | None = None
    claim_producing_episode_id: str | None = None
    claim_producing_attempt_id: str | None = None
    selection_reason: str = Field(min_length=1)
    search_dispositions: list[SearchDisposition]
    search_disposition_source_type: Literal["backend_derived"] = "backend_derived"
    epistemic_dispositions: list[PathEpistemicDisposition]
    epistemic_disposition_records: list[PathDispositionRecordV1] = Field(
        default_factory=list
    )
    required_assumption_refs: list[str] = Field(default_factory=list)
    supporting_record_refs: list[str] = Field(default_factory=list)
    challenging_record_refs: list[str] = Field(default_factory=list)
    conditional_dependency_refs: list[str] = Field(default_factory=list)
    derived_from_refs: list[str] = Field(default_factory=list)
    unresolved_dependency_refs: list[str] = Field(default_factory=list)
    counterexample_refs: list[str] = Field(default_factory=list)
    producing_episode_ids: list[str] = Field(default_factory=list)
    producing_attempt_ids: list[str] = Field(default_factory=list)
    referenced_artifacts: list[str] = Field(default_factory=list)
    relation_record_ids: list[str] = Field(default_factory=list)
    merge_application_ids: list[str] = Field(default_factory=list)
    source_types: list[EpistemicSourceType] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_claim_origin(self) -> "SelectedClaimHandoffV1":
        if self.claim_origin == "executor_episode":
            if (
                self.claim_source_type != "agent_reported"
                or self.claim_producing_episode_id is None
                or self.claim_producing_attempt_id is None
            ):
                raise ValueError(
                    "executor-produced selected claims require episode provenance"
                )
        elif (
            self.claim_source_type is not None
            or self.claim_producing_episode_id is not None
            or self.claim_producing_attempt_id is not None
        ):
            raise ValueError(
                "initial selected claims cannot fabricate episode provenance"
            )
        return self


class ImportantPathHandoffV1(DTEBaseModel):
    node_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    search_dispositions: list[SearchDisposition]
    search_disposition_source_type: Literal["backend_derived"] = "backend_derived"
    epistemic_dispositions: list[PathEpistemicDisposition]
    epistemic_disposition_records: list[PathDispositionRecordV1] = Field(
        default_factory=list
    )
    basis_refs: list[str] = Field(default_factory=list)
    explanation: str


class EpistemicIndependenceSummaryV1(DTEBaseModel):
    schema_version: Literal["dte-epistemic-independence-summary.v1"] = (
        "dte-epistemic-independence-summary.v1"
    )
    interpretation: Literal[
        "correlated_error_risk_indicators_not_correctness_or_reliability"
    ] = "correlated_error_risk_indicators_not_correctness_or_reliability"
    model_comparison_basis: Literal["exact_persisted_model_identifier"] = (
        "exact_persisted_model_identifier"
    )
    model_metadata_status: Literal["available", "partial", "unavailable"]
    model_metadata_available: bool
    same_model_cross_role_count: int | None = Field(default=None, ge=0)
    different_model_cross_role_count: int | None = Field(default=None, ge=0)
    runtime_profile_metadata_status: Literal["available", "partial", "unavailable"]
    same_runtime_profile_cross_role_count: int | None = Field(default=None, ge=0)
    different_runtime_profile_cross_role_count: int | None = Field(default=None, ge=0)
    same_model_support_challenge_count: int | None = Field(default=None, ge=0)
    different_model_support_challenge_count: int | None = Field(default=None, ge=0)
    selected_claim_count: int = Field(ge=0)
    agent_only_supported_selected_claim_count: int = Field(ge=0)
    external_artifact_backed_selected_claim_count: int = Field(
        ge=0,
        description=(
            "Selected claims with structured support that references an external "
            "artifact; this is reference coverage, not verification."
        ),
    )
    selected_claims_with_unresolved_assumptions: int = Field(ge=0)
    selected_claims_without_structured_support_count: int = Field(ge=0)
    self_referential_support_count: int = Field(ge=0)
    risk_flags: list[str] = Field(default_factory=list)


class EpistemicDataQualityV1(DTEBaseModel):
    schema_version: Literal["dte-epistemic-data-quality.v1"] = (
        "dte-epistemic-data-quality.v1"
    )
    epistemic_data_status: Literal["available", "partial", "unavailable"]
    structured_contribution_episode_count: int = Field(ge=0)
    missing_artifacts: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)
    inconsistent_but_recoverable_records: list[str] = Field(default_factory=list)
    model_metadata_status: Literal["available", "partial", "unavailable"]
    operational_observability_status: Literal["current", "partial_legacy"]
    operational_observability_limitations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class SourceProvenanceSummaryV1(DTEBaseModel):
    agent_reported_record_count: int = Field(ge=0)
    external_artifact_backed_record_count: int = Field(
        ge=0,
        description=(
            "Records that reference an external artifact; the backend does not "
            "verify the artifact or scientific claim."
        ),
    )
    backend_derived_record_count: int = Field(ge=0)


class TerminalEpistemicHandoffV1(DTEBaseModel):
    schema_version: Literal["dte-terminal-epistemic-handoff.v1"] = (
        "dte-terminal-epistemic-handoff.v1"
    )
    run_id: str = Field(min_length=1)
    terminal_action: str | None
    terminal_reason: str | None = None
    terminal_source: str | None = None
    selection_kind: Literal["provisional_selected_node_claims"] = (
        "provisional_selected_node_claims"
    )
    selected_claims: list[SelectedClaimHandoffV1]
    key_assumptions: list[EpistemicStatementRecordV1]
    supporting_evidence: list[EpistemicStatementRecordV1]
    challenging_evidence: list[EpistemicStatementRecordV1]
    conditional_dependencies: list[EpistemicStatementRecordV1]
    unresolved_dependencies: list[EpistemicStatementRecordV1]
    material_conflicts: list[RelationEpistemicProjectionV1]
    relation_disclosures: list[RelationEpistemicProjectionV1]
    counterexamples: list[EpistemicStatementRecordV1]
    counterexample_refs: list[str] = Field(default_factory=list)
    important_abandoned_or_inconclusive_paths: list[ImportantPathHandoffV1]
    possible_transferable_heuristics: list[EpistemicStatementRecordV1]
    transferable_failure_modes: list[EpistemicStatementRecordV1]
    source_provenance: SourceProvenanceSummaryV1
    independence_summary: EpistemicIndependenceSummaryV1
    node_summaries: list[NodeEpistemicSummaryV1]
    dependency_graph: EpistemicDependencyGraphV1
    data_quality: EpistemicDataQualityV1
