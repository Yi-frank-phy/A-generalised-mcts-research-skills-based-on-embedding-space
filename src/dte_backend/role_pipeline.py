"""Role-isolated deterministic seed pipeline.

The old DTE backend separated decomposition, research, distillation, and strategy
generation to reduce role-mixing bias. This module restores that backend logic in
an offline deterministic form. A future model-backed implementation can replace
individual functions without changing DTE's mandatory control flow.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .models import DTERunSpec, SearchNode


@dataclass(frozen=True)
class TaskDecomposition:
    core_question: str
    subquestions: list[str]
    constraints: list[str]


@dataclass(frozen=True)
class ResearchContext:
    background: list[str]
    unknowns: list[str]
    failure_modes: list[str]


@dataclass(frozen=True)
class DistilledContext:
    summary: str
    assumptions: list[str]
    opportunities: list[str]
    risks: list[str]


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[。.!?\n]+", text) if p.strip()]
    return parts or [text.strip()]


def decompose_task(spec: DTERunSpec) -> TaskDecomposition:
    """TaskDecomposer role: isolate problem structure before strategy generation."""

    problem_parts = _sentences(spec.problem)
    goal_parts = _sentences(spec.goal)
    subquestions = [
        f"What are the minimal assumptions needed for: {problem_parts[0]}",
        f"Which constructive route can satisfy the goal: {goal_parts[0]}",
        "Which counterexamples or boundary cases would refute the proposed route?",
        "Which branches are equivalent, complementary, or in conflict?",
    ]
    return TaskDecomposition(core_question=problem_parts[0], subquestions=subquestions, constraints=list(spec.constraints))


def research_context(spec: DTERunSpec, decomposition: TaskDecomposition) -> ResearchContext:
    """Researcher role: collect what should be checked, without judging routes."""

    background = [
        f"Core problem: {decomposition.core_question}",
        f"Goal: {spec.goal}",
    ]
    unknowns = [
        "Whether the direct construction satisfies all constraints.",
        "Whether a counterexample exists in a boundary or low-dimensional case.",
        "Whether two promising branches are actually equivalent under a change of formalism.",
    ]
    failure_modes = [
        "Hidden assumption is stronger than stated constraints.",
        "Search branch is semantically redundant with an existing node.",
        "Executor produces an answer-like synthesis instead of structured evidence.",
    ]
    return ResearchContext(background=background, unknowns=unknowns, failure_modes=failure_modes)


def distill_context(decomposition: TaskDecomposition, context: ResearchContext) -> DistilledContext:
    """Distiller role: compress context before strategy generation/Judge."""

    assumptions = decomposition.constraints[:3]
    summary = " / ".join([decomposition.core_question, *context.unknowns[:2]])
    opportunities = [
        "Direct constructive derivation.",
        "Counterexample-first stress test.",
        "Alternative formalism or symmetry representation.",
        "Merge/discriminator branch for conflicting routes.",
    ]
    risks = list(context.failure_modes)
    return DistilledContext(summary=summary, assumptions=assumptions, opportunities=opportunities, risks=risks)


def generate_initial_strategies(spec: DTERunSpec, distilled: DistilledContext) -> list[SearchNode]:
    """StrategyGenerator role: produce distinct frontier nodes without ranking them."""

    base_assumptions = list(distilled.assumptions)
    return [
        SearchNode(
            node_id="seed-direct",
            claim=f"Direct constructive route for: {spec.problem}",
            rationale=f"Use the most direct derivation path. Context: {distilled.summary}",
            assumptions=base_assumptions,
            risks=[distilled.risks[0]] if distilled.risks else [],
            confidence=0.55,
        ),
        SearchNode(
            node_id="seed-counter",
            claim=f"Counterexample-first route for: {spec.problem}",
            rationale="Search boundary cases and low-dimensional reductions before trusting the constructive route.",
            assumptions=base_assumptions,
            risks=["may spend budget on a branch that only falsifies over-strong variants"],
            confidence=0.50,
        ),
        SearchNode(
            node_id="seed-formalism",
            claim=f"Alternative formalism route for: {spec.problem}",
            rationale="Re-express the candidate idea in another mathematical/physical representation to reveal hidden equivalences.",
            assumptions=base_assumptions,
            risks=["formal translation may preserve notation but not insight"],
            confidence=0.52,
        ),
        SearchNode(
            node_id="seed-merge",
            claim=f"Merge/discriminator route for: {spec.problem}",
            rationale="Prepare to distinguish equivalent, complementary, and conflicting branches instead of letting them drift independently.",
            assumptions=base_assumptions,
            risks=["requires good relation judgments to avoid premature merge"],
            confidence=0.51,
        ),
    ]


def seed_frontier_from_roles(spec: DTERunSpec) -> tuple[list[SearchNode], dict[str, object]]:
    """Run the logical role seed pipeline and return nodes plus audit metadata."""

    decomposition = decompose_task(spec)
    context = research_context(spec, decomposition)
    distilled = distill_context(decomposition, context)
    nodes = generate_initial_strategies(spec, distilled)
    audit = {
        "decomposition": decomposition.__dict__,
        "research_context": context.__dict__,
        "distilled_context": distilled.__dict__,
    }
    return nodes, audit
