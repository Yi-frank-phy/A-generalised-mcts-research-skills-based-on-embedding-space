"""Strict, versioned schemas for the read-only DTE observability surface."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from .episode_models import (
    ComparisonProfile,
    DiagnosticsSource,
    EpisodeRole,
    UsageSource,
)
from .models import DTEBaseModel
from .relation_models import (
    RelationCandidateReason,
    RelationSchedulingClass,
    RelationType,
)


FeedbackTargetType = Literal[
    "run",
    "episode",
    "attempt",
    "node",
    "relation_record",
    "merge_application",
    "allocation_decision",
]
FeedbackSource = Literal["user", "main_agent", "external_evaluator"]
RejectionCategory = Literal[
    "schema_rejection",
    "identity_mismatch",
    "stale_revision",
    "lifecycle_rejection",
    "controller_owned_field_violation",
    "duplicate_output",
    "over_grant",
    "relation_overlap",
    "merge_provenance_conflict",
    "timeout_expire",
    "other",
]


class RunBudgetSnapshotV1(DTEBaseModel):
    max_committed_search_nodes: int | None = Field(default=None, ge=1)
    max_iterations: int | None = Field(default=None, ge=0)
    allocation_mass_per_iteration: int | None = Field(default=None, ge=0)
    max_children_per_iteration: int | None = Field(default=None, ge=0)
    max_relation_pairs_per_episode: int | None = Field(default=None, ge=0)
    max_relation_enrichment_pairs: int | None = Field(default=None, ge=0)
    max_research_iterations: int | None = Field(default=None, ge=0)
    min_iterations_before_synthesis: int | None = Field(default=None, ge=0)
    entropy_change_threshold: float | None = Field(default=None, ge=0.0)
    entropy_plateau_confirmations: int | None = Field(default=None, ge=1)
    continuation_policy: str | None = None
    t_max: float | None = Field(default=None, ge=0.0)


class RunIdentityObservabilityV1(DTEBaseModel):
    run_id: str = Field(min_length=1)
    state_schema_version: str | None = None
    controller_profile: str | None = None
    runtime_profiles: list[str] = Field(default_factory=list)
    problem_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    goal_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    budget: RunBudgetSnapshotV1
    embedding_provider: str | None = None
    embedding_dimension: int | None = Field(default=None, ge=1)
    terminal_action: str | None = None
    terminal_reason: str | None = None
    terminal_source: str | None = None
    controller_iterations: int | None = Field(default=None, ge=0)
    graph_revision: int | None = Field(default=None, ge=0)
    code_version: str | None = None
    observability_status: Literal["current", "partial_legacy"]
    created_at: str | None = None
    updated_at: str | None = None


class RuntimeAggregateDiagnosticsV1(DTEBaseModel):
    """Explicitly reported coarse diagnostics; never hidden worker detail."""

    schema_version: Literal["dte-runtime-aggregate-diagnostics.v1"] = (
        "dte-runtime-aggregate-diagnostics.v1"
    )
    adapter_name: str = Field(min_length=1)
    transport_name: str = Field(min_length=1)
    profile: ComparisonProfile
    runtime_profile: str | None = None
    model: str | None = None
    wall_clock_ms: int | None = Field(default=None, ge=0)
    queue_or_io_ms: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)
    runtime_reference: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    provider_reported_cost: float | None = Field(default=None, ge=0.0)
    estimated_cost: float | None = Field(default=None, ge=0.0)
    quota_delta: float | None = Field(default=None, ge=0.0)
    usage_source: UsageSource = "unavailable"
    internal_subagent_count: int | None = Field(default=None, ge=0)
    max_internal_parallelism: int | None = Field(default=None, ge=0)
    internal_tool_call_count: int | None = Field(default=None, ge=0)
    internal_round_count: int | None = Field(default=None, ge=0)
    internal_failure_count: int | None = Field(default=None, ge=0)
    internal_input_tokens: int | None = Field(default=None, ge=0)
    internal_output_tokens: int | None = Field(default=None, ge=0)
    diagnostics_source: DiagnosticsSource = "unavailable"

    @model_validator(mode="after")
    def validate_aggregate_diagnostics_source(
        self,
    ) -> "RuntimeAggregateDiagnosticsV1":
        aggregate_values = (
            self.internal_subagent_count,
            self.max_internal_parallelism,
            self.internal_tool_call_count,
            self.internal_round_count,
            self.internal_failure_count,
            self.internal_input_tokens,
            self.internal_output_tokens,
        )
        if self.diagnostics_source == "unavailable" and any(
            value is not None for value in aggregate_values
        ):
            raise ValueError(
                "aggregate runtime diagnostics require an explicit diagnostics_source"
            )
        return self


class AttemptObservabilityRecordV1(DTEBaseModel):
    attempt_id: str = Field(min_length=1)
    attempt_number: int = Field(ge=1)
    status: str = Field(min_length=1)
    superseded_from_status: str | None = None
    granted_at: str | None = None
    deadline_at: str | None = None
    submitted_at: str | None = None
    wall_clock_ms: int | None = Field(default=None, ge=0)
    selected_node_ids: list[str] = Field(default_factory=list)
    returned_node_count: int | None = Field(default=None, ge=0)
    accepted_node_count: int | None = Field(default=None, ge=0)
    returned_observation_count: int | None = Field(default=None, ge=0)
    rejection_reason: str | None = None
    rejection_category: RejectionCategory | None = None
    runtime_diagnostics: RuntimeAggregateDiagnosticsV1 | None = None


class EpisodeObservabilityRecordV1(DTEBaseModel):
    schema_version: Literal["dte-episode-observability.v1"] = (
        "dte-episode-observability.v1"
    )
    episode_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    role: EpisodeRole
    lifecycle_status: str = Field(min_length=1)
    attempt_count: int = Field(ge=1)
    retry_count: int = Field(ge=0)
    committed_attempt_id: str | None = None
    selected_node_ids: list[str] = Field(default_factory=list)
    created_node_ids: list[str] = Field(default_factory=list)
    relation_candidate_ids: list[str] = Field(default_factory=list)
    attempts: list[AttemptObservabilityRecordV1]


class RoleEpisodeFunnelV1(DTEBaseModel):
    episode_count: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    granted_attempt_count: int = Field(ge=0)
    started_attempt_count: int = Field(ge=0)
    in_progress_attempt_count: int = Field(ge=0)
    submitted_attempt_count: int = Field(ge=0)
    committed_attempt_count: int = Field(ge=0)
    rejected_attempt_count: int = Field(ge=0)
    failed_attempt_count: int = Field(ge=0)
    cancelled_attempt_count: int = Field(ge=0)
    expired_attempt_count: int = Field(ge=0)
    superseded_attempt_count: int = Field(ge=0)
    retried_attempt_count: int = Field(ge=0)
    retried_episode_count: int = Field(ge=0)
    commit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    rejection_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    retry_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    rate_denominator: Literal["attempt_count"] = "attempt_count"
    wall_clock_sample_count: int = Field(ge=0)
    wall_clock_total_ms: int | None = Field(default=None, ge=0)
    wall_clock_mean_ms: float | None = Field(default=None, ge=0.0)
    wall_clock_median_ms: float | None = Field(default=None, ge=0.0)


class EpisodeFunnelV1(DTEBaseModel):
    judge: RoleEpisodeFunnelV1
    executor: RoleEpisodeFunnelV1
    relation: RoleEpisodeFunnelV1
    unmodeled_role_episode_count: int = Field(default=0, ge=0)


class NodeFunnelV1(DTEBaseModel):
    node_count: int = Field(ge=0)
    unique_node_count: int = Field(ge=0)
    initial_node_count: int = Field(ge=0)
    all_committed_node_count: int = Field(ge=0)
    judged_node_count: int = Field(ge=0)
    positive_allocation_unique_node_count: int = Field(ge=0)
    parent_expansion_count: int = Field(ge=0)
    expanded_parent_unique_node_count: int = Field(ge=0)
    committed_executor_child_count: int = Field(ge=0)
    frontier_node_count: int = Field(ge=0)
    closed_node_count: int = Field(ge=0)
    merged_node_count: int = Field(ge=0)
    committed_search_node_count: int = Field(ge=0)
    remaining_search_node_slots: int = Field(ge=0)
    canonical_frontier_node_count: int = Field(ge=0)
    canonical_live_node_count: int = Field(ge=0)
    provisional_synthesis_selected_node_count: int = Field(ge=0)


class NodeAllocationHistoryRecordV1(DTEBaseModel):
    allocation_decision_id: str = Field(min_length=1)
    controller_iteration: int = Field(ge=1)
    input_graph_revision: int = Field(ge=0)
    output_graph_revision: int = Field(ge=0)
    allocated_expansion_budget: int = Field(ge=0)
    ucb_score: float | None = None
    spatial_entropy: float | None = None


class NodeRelationOutcomeRecordV1(DTEBaseModel):
    candidate_id: str = Field(min_length=1)
    relation_record_id: str | None = None
    other_node_id: str = Field(min_length=1)
    candidate_reason: RelationCandidateReason
    scheduling_class: RelationSchedulingClass
    relation_type: RelationType | None = None
    material_to_synthesis: bool
    disclosure_required: bool | None = None


class NodeLineageRecordV1(DTEBaseModel):
    schema_version: Literal["dte-node-lineage.v1"] = "dte-node-lineage.v1"
    node_id: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    parent_ids: list[str] = Field(default_factory=list)
    creation_episode_id: str | None = None
    creation_attempt_id: str | None = None
    creation_graph_revision: int | None = Field(default=None, ge=0)
    current_node_revision: int | None = Field(default=None, ge=0)
    judge_score: float | None = Field(default=None, ge=0.0, le=1.0)
    judge_reasoning: str | None = None
    judge_risks: list[str] = Field(default_factory=list)
    judge_uncertainty_evidence: list[str] = Field(default_factory=list)
    judge_episode_id: str | None = None
    judge_attempt_id: str | None = None
    allocation_history: list[NodeAllocationHistoryRecordV1] = Field(default_factory=list)
    total_expansion_budget_granted: int = Field(ge=0)
    expansion_episode_count: int = Field(ge=0)
    children_committed: list[str] = Field(default_factory=list)
    descendant_count: int = Field(ge=0)
    selected_for_synthesis: bool
    merged: bool
    canonical_target: str | None = None
    relation_outcomes: list[NodeRelationOutcomeRecordV1] = Field(default_factory=list)
    current_status: str = Field(min_length=1)


class AllocationOutcomeRecordV1(DTEBaseModel):
    schema_version: Literal["dte-allocation-outcome.v1"] = (
        "dte-allocation-outcome.v1"
    )
    allocation_decision_id: str = Field(min_length=1)
    parent_node_id: str = Field(min_length=1)
    controller_iteration: int = Field(ge=1)
    input_graph_revision: int = Field(ge=0)
    output_graph_revision: int = Field(ge=0)
    allocated_expansion_budget: int = Field(ge=0)
    actual_committed_children: int = Field(ge=0)
    unused_granted_capacity: int = Field(ge=0)
    direct_children_later_judged: int = Field(ge=0)
    direct_children_later_selected: int = Field(ge=0)
    descendant_count: int = Field(ge=0)
    selected_descendant_count: int = Field(ge=0)
    merged_descendant_count: int = Field(ge=0)
    relation_conflicts_involving_descendants: int = Field(ge=0)
    committed_child_yield: float | None = Field(default=None, ge=0.0)
    selected_descendant_yield: float | None = Field(default=None, ge=0.0, le=1.0)
    proxy_note: Literal[
        "internal_process_proxy_not_scientific_correctness"
    ] = "internal_process_proxy_not_scientific_correctness"


class DescriptiveStatsV1(DTEBaseModel):
    count: int = Field(ge=0)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None


class JudgeNodePosteriorRecordV1(DTEBaseModel):
    node_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    later_received_positive_allocation: bool
    later_expanded: bool
    later_provisional_selected: bool
    later_merged: bool
    later_involved_in_conflict: bool


class JudgeOutcomeSummaryV1(DTEBaseModel):
    interpretation: Literal[
        "observational_posterior_proxy_not_external_calibration_or_causation"
    ] = "observational_posterior_proxy_not_external_calibration_or_causation"
    score_distribution: DescriptiveStatsV1
    selected_score_distribution: DescriptiveStatsV1
    nonselected_score_distribution: DescriptiveStatsV1
    high_score_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    high_score_node_count: int = Field(ge=0)
    high_score_selected_count: int = Field(ge=0)
    posterior_records: list[JudgeNodePosteriorRecordV1] = Field(default_factory=list)


class RelationReasonYieldV1(DTEBaseModel):
    candidate_reason: RelationCandidateReason
    candidate_count: int = Field(ge=0)
    committed_relation_count: int = Field(ge=0)
    equivalent_count: int = Field(ge=0)
    complementary_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    independent_count: int = Field(ge=0)
    equivalent_yield: float | None = Field(default=None, ge=0.0, le=1.0)
    complementary_yield: float | None = Field(default=None, ge=0.0, le=1.0)
    conflict_yield: float | None = Field(default=None, ge=0.0, le=1.0)
    independent_yield: float | None = Field(default=None, ge=0.0, le=1.0)


class RelationOutcomeSummaryV1(DTEBaseModel):
    schema_version: Literal["dte-relation-outcome-summary.v1"] = (
        "dte-relation-outcome-summary.v1"
    )
    candidate_count: int = Field(ge=0)
    blocking_candidates_generated: int = Field(ge=0)
    blocking_pairs_resolved: int = Field(ge=0)
    enrichment_candidates_generated: int = Field(ge=0)
    enrichment_pairs_committed: int = Field(ge=0)
    equivalent_count: int = Field(ge=0)
    complementary_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    independent_count: int = Field(ge=0)
    merge_count: int = Field(ge=0)
    material_conflict_count: int = Field(ge=0)
    disclosure_required_count: int = Field(ge=0)
    rejected_relation_transactions: int = Field(ge=0)
    relation_budget_scope: Literal["run_enrichment_pair_budget"] = (
        "run_enrichment_pair_budget"
    )
    relation_budget_limit: int | None = Field(default=None, ge=0)
    relation_budget_consumed: int | None = Field(default=None, ge=0)
    relation_budget_remaining: int | None = Field(default=None, ge=0)
    by_candidate_reason: list[RelationReasonYieldV1] = Field(default_factory=list)


class ControllerTrajectoryRecordV1(DTEBaseModel):
    controller_iteration: int = Field(ge=1)
    input_graph_revision: int | None = Field(default=None, ge=0)
    output_graph_revision: int | None = Field(default=None, ge=0)
    frontier_size: int | None = Field(default=None, ge=0)
    judged_frontier_size: int | None = Field(default=None, ge=0)
    spatial_entropy: float | None = None
    entropy_delta: float | None = Field(default=None, ge=0.0)
    plateau_signal: bool | None = None
    consecutive_plateau_count: int | None = Field(default=None, ge=0)
    effective_child_cap: int | None = Field(default=None, ge=0)
    allocation_mass_parameter: int | None = Field(default=None, ge=0)
    allocated_child_count: int | None = Field(default=None, ge=0)
    positive_budget_parent_count: int | None = Field(default=None, ge=0)
    children_committed: int | None = Field(default=None, ge=0)
    graph_revision: int | None = Field(default=None, ge=0)
    readiness_transition: str | None = None
    terminal_transition: str | None = None


class ContinuationGateObservabilityRecordV1(DTEBaseModel):
    controller_iteration: int = Field(ge=1)
    graph_revision: int = Field(ge=0)
    committed_search_node_count: int = Field(ge=0)
    remaining_search_node_slots: int = Field(ge=0)
    canonical_frontier_count: int = Field(ge=0)
    entropy_delta: float | None = Field(default=None, ge=0.0)
    consecutive_plateau_count: int = Field(ge=0)
    plateau_confirmed: bool
    trigger_signals: list[str] = Field(default_factory=list)
    material_yield_signals: list[str] = Field(default_factory=list)
    material_epistemic_record_ids: list[str] = Field(default_factory=list)
    continuation_target_node_ids: list[str] = Field(default_factory=list)
    provisional_synthesis_node_ids: list[str] = Field(default_factory=list)
    positive_allocation_node_ids: list[str] = Field(default_factory=list)
    decision: Literal["continue", "prepare_synthesis"]
    reason: str = Field(min_length=1)


class FrontierWaitRecordV1(DTEBaseModel):
    """Read-only allocation history; this record never changes UCB."""

    node_id: str = Field(min_length=1)
    eligible_iteration_count: int = Field(ge=0)
    zero_allocation_streak: int = Field(ge=0)
    last_positive_allocation_iteration: int | None = Field(default=None, ge=1)
    current_score: float | None = Field(default=None, ge=0.0, le=1.0)
    current_uncertainty: float | None = Field(default=None, ge=0.0)
    current_ucb: float | None = None


class RejectionCategoryCountV1(DTEBaseModel):
    category: RejectionCategory
    count: int = Field(ge=0)


class RejectionSummaryV1(DTEBaseModel):
    total_rejection_or_error_count: int = Field(ge=0)
    by_category: list[RejectionCategoryCountV1]


class FeedbackRecordV1(DTEBaseModel):
    schema_version: Literal["dte-feedback.v1"] = "dte-feedback.v1"
    feedback_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    target_type: FeedbackTargetType
    target_id: str | None = None
    metric: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    label: str | None = None
    comment: str | None = None
    source: FeedbackSource
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_feedback(self) -> "FeedbackRecordV1":
        if self.target_type == "run" and self.target_id is not None:
            raise ValueError("run feedback must not include target_id")
        if self.target_type != "run" and not self.target_id:
            raise ValueError(f"{self.target_type} feedback requires target_id")
        if not any(
            (
                self.score is not None,
                bool(self.label and self.label.strip()),
                bool(self.comment and self.comment.strip()),
                bool(self.metadata),
            )
        ):
            raise ValueError(
                "feedback requires at least one substantive score, label, comment, or metadata field"
            )
        try:
            parsed = datetime.fromisoformat(self.timestamp)
        except ValueError as exc:
            raise ValueError("feedback timestamp must be ISO-8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("feedback timestamp must include a UTC offset")
        try:
            json.dumps(self.metadata, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("feedback metadata must be JSON-serializable") from exc
        return self


class FeedbackLedgerDiagnosticsV1(DTEBaseModel):
    path_present: bool
    valid_record_count: int = Field(ge=0)
    malformed_interior_line_count: int = Field(ge=0)
    corrupt_tail_detected: bool
    corrupt_tail_repaired: bool
    duplicate_feedback_ids: list[str] = Field(default_factory=list)


class ObservabilityDataQualityV1(DTEBaseModel):
    schema_version: Literal["dte-observability-data-quality.v1"] = (
        "dte-observability-data-quality.v1"
    )
    missing_event_fields: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    inconsistent_but_recoverable_records: list[str] = Field(default_factory=list)
    corrupt_telemetry_tail_detected: bool = False
    corrupt_telemetry_tail_repaired: bool = False
    usage_unavailable: bool = True
    runtime_diagnostics_unavailable: bool = True
    partial_legacy_reconstruction: bool = False
    state_validation_error: str | None = None
    telemetry_record_count: int = Field(default=0, ge=0)
    feedback_record_count: int = Field(default=0, ge=0)
    limitations: list[str] = Field(default_factory=list)


class RunObservabilitySummaryV1(DTEBaseModel):
    schema_version: Literal["dte-run-observability-summary.v1"] = (
        "dte-run-observability-summary.v1"
    )
    run: RunIdentityObservabilityV1
    episode_funnel: EpisodeFunnelV1
    node_funnel: NodeFunnelV1
    episodes: list[EpisodeObservabilityRecordV1]
    node_lineage: list[NodeLineageRecordV1]
    allocation_outcomes: list[AllocationOutcomeRecordV1]
    judge_outcomes: JudgeOutcomeSummaryV1
    relation_outcomes: RelationOutcomeSummaryV1
    controller_trajectory: list[ControllerTrajectoryRecordV1]
    continuation_gate_trajectory: list[ContinuationGateObservabilityRecordV1]
    frontier_wait: list[FrontierWaitRecordV1]
    rejections: RejectionSummaryV1
    feedback: list[FeedbackRecordV1]
    data_quality: ObservabilityDataQualityV1


class ObservabilityExportSkippedRunV1(DTEBaseModel):
    run_dir: str
    reason: str


class ObservabilityExportResultV1(DTEBaseModel):
    schema_version: Literal["dte-observability-export-result.v1"] = (
        "dte-observability-export-result.v1"
    )
    output_path: str
    processed_run_count: int = Field(ge=0)
    skipped_run_count: int = Field(ge=0)
    record_count: int = Field(ge=0)
    skipped_runs: list[ObservabilityExportSkippedRunV1] = Field(default_factory=list)
