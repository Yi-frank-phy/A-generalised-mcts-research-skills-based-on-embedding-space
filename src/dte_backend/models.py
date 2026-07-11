"""Typed data models for the DTE skill backend.

These models are intentionally small and explicit. They are the machine-facing
contract between Codex/Kimi/OpenClaw executor episodes and the DTE controller.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class DTEBaseModel(BaseModel):
    """Strict base model for DTE machine-facing contracts."""

    model_config = ConfigDict(extra="forbid")


class BudgetSpec(DTEBaseModel):
    """Hard budget limits for one DTE run."""

    max_iterations: int = Field(default=2, ge=1, le=20)
    allocation_mass_per_iteration: int = Field(default=3, ge=1, le=50)
    max_children_per_iteration: int = Field(default=5, ge=1, le=50)
    max_research_iterations: int = Field(default=1, ge=0, le=5)
    min_iterations_before_synthesis: int = Field(default=2, ge=1, le=20)
    entropy_change_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    t_max: float = Field(default=1.0, gt=0.0, le=10.0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_allocation_mass(cls, data: Any) -> Any:
        """Accept the old budget name only as a deprecated input alias."""

        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        legacy_name = "total_child_budget"
        canonical_name = "allocation_mass_per_iteration"
        if legacy_name not in values:
            return values
        if canonical_name in values and values[canonical_name] != values[legacy_name]:
            raise ValueError(
                "total_child_budget conflicts with allocation_mass_per_iteration; "
                "provide equal values or only the canonical field"
            )
        values.setdefault(canonical_name, values[legacy_name])
        del values[legacy_name]
        return values


class DTERunSpec(DTEBaseModel):
    """Top-level run specification.

    This is the source of truth. Free-form Markdown or prompt text cannot
    override this object after validation.
    """

    problem: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    mode: Literal["mandatory_frontier"] = "mandatory_frontier"
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    allow_self_organized_executor: bool = True
    require_final_synthesis: bool = True
    embedding_provider: Literal["hash", "gemini-embedding-2"] = "hash"
    embedding_dimension: int = Field(default=3072, ge=8, le=3072)


class SearchNode(DTEBaseModel):
    """A node in the DTE search graph/frontier."""

    node_id: str
    node_type: Literal[
        "candidate",
        "evidence",
        "counterexample",
        "merge",
        "synthesis",
    ] = "candidate"
    claim: str = Field(min_length=1)
    rationale: str = Field(default="")
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    local_embedding: list[float] | None = None
    judge_reasoning: str | None = None

    score: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: float | None = Field(default=None, ge=0.0)
    ucb_score: float | None = None
    expansion_budget: int = Field(default=0, ge=0)
    status: Literal["frontier", "closed", "archived", "merged", "synthesis"] = "frontier"


class AllocationResult(DTEBaseModel):
    """Expansion budget assignment for a frontier batch."""

    node_id: str
    score: float
    uncertainty: float
    ucb_score: float
    expansion_budget: int


class ExpansionRequest(DTEBaseModel):
    """Request passed from DTE Expansion to an external executor adapter."""

    parent: SearchNode
    child_count: int = Field(ge=1, le=50, validation_alias=AliasChoices("child_count", "count"))
    iteration: int = Field(ge=1)
    spec: DTERunSpec | None = None


class MergeProposal(DTEBaseModel):
    """Structured merge proposal for graph-search compression."""

    merge_type: Literal["equivalent_merge", "complementary_merge", "conflict_merge"]
    source_node_ids: list[str] = Field(min_length=2)
    target_node_id: str | None = None
    rationale: str
    merged_node: SearchNode | None = None
    absorbed_node_ids: list[str] = Field(default_factory=list)


class SynthesisControlRequest(DTEBaseModel):
    """Operator/main-agent request to stop after the current safe task."""

    action: Literal["force_synthesis_after_current_task"]
    requested_by: Literal["main_agent", "user"]
    reason: str = Field(min_length=1)
    scope: Literal["all", "node_ids"] = "all"
    node_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scope(self) -> "SynthesisControlRequest":
        if self.scope == "node_ids" and not self.node_ids:
            raise ValueError("scope='node_ids' requires at least one node_id")
        if self.scope == "all" and self.node_ids:
            raise ValueError("scope='all' must not include node_ids")
        return self


class ForcedSynthesisRecord(DTEBaseModel):
    """Recorded stop metadata for a forced synthesis."""

    stop_reason: Literal["main_agent_requested_synthesis", "user_interrupted_for_synthesis"]
    requested_by: Literal["main_agent", "user"]
    reason: str
    scope: Literal["all", "node_ids"]
    node_ids: list[str] = Field(default_factory=list)
    left_unexplored_node_ids: list[str] = Field(default_factory=list)
    control_path: str | None = None
