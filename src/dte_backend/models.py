"""Typed data models for the DTE skill backend.

These models are intentionally small and explicit. They are the machine-facing
contract between Codex/Kimi/OpenClaw executor episodes and the DTE controller.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class BudgetSpec(BaseModel):
    """Hard budget limits and auto-synthesis controls for one DTE run."""

    max_iterations: int = Field(default=2, ge=1, le=20)
    total_child_budget: int = Field(default=3, ge=1, le=50)
    max_research_iterations: int = Field(default=1, ge=0, le=5)
    min_iterations_before_synthesis: int = Field(default=2, ge=1, le=20)
    entropy_change_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    t_max: float = Field(default=1.0, gt=0.0, le=10.0)


class DTERunSpec(BaseModel):
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
    embedding_dimension: int = Field(default=64, ge=8, le=3072)


class SearchNode(BaseModel):
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

    # Local/offline prototype features. Real adapters may replace this with an
    # external embedding, but the backend can still run deterministically.
    local_embedding: list[float] | None = None
    judge_reasoning: str | None = None

    # DTE metrics. These are filled by Judge/EvolutionController.
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: float | None = Field(default=None, ge=0.0)
    ucb_score: float | None = None
    expansion_budget: int = Field(default=0, ge=0)
    status: Literal["frontier", "closed", "archived", "merged", "synthesis"] = "frontier"


class AllocationResult(BaseModel):
    """Expansion budget assignment for a frontier batch."""

    node_id: str
    score: float
    uncertainty: float
    ucb_score: float
    expansion_budget: int


class ExpansionRequest(BaseModel):
    """Request passed from DTE Expansion to an external executor adapter."""

    parent: SearchNode
    child_count: int = Field(ge=1, le=50)
    iteration: int = Field(ge=1)
    spec: DTERunSpec | None = None


class MergeProposal(BaseModel):
    """Structured merge proposal for graph-search compression."""

    merge_type: Literal["equivalent_merge", "complementary_merge", "conflict_merge"]
    source_node_ids: list[str] = Field(min_length=2)
    target_node_id: str | None = None
    rationale: str
    merged_node: SearchNode | None = None
    absorbed_node_ids: list[str] = Field(default_factory=list)
