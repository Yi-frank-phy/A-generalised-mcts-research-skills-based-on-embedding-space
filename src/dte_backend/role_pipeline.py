"""Role-isolated deterministic seed pipeline.

DTE keeps strategy generation separate from judging/execution to reduce bias.
The old fixed Distiller role is intentionally not part of the mandatory chain:
modern Codex-style agents can compile/summarize their own local context when it
helps. The backend only exposes a compile hint, not a mandatory distiller step.
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
class CompileHint:
    """Optional prompt-level compression hint for agents.

    This replaces the old mandatory Distiller role. A model/subagent may choose
    to compile its local context before returning a SearchNode, but DTE does not
    force a separate distiller call in the backend loop.
    """

    summary_focus: str
    preserve: list[str]
    drop: list[str]


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


def compile_hint(decomposition: TaskDecomposition, context: ResearchContext) -> CompileHint:
    """Return an optional compile instruction for Codex/subagents."""

    return CompileHint(
        summary_focus="Compile only the information needed to create or evaluate SearchNodes.",
        preserve=[
            "explicit assumptions",
            "evidence and counterexamples",
            "branch conflicts and merge opportunities",
            "uncertainty / failure modes",
        ],
        drop=[
            "long transcripts",
            "irrelevant tool logs",
            "style-only rewrites",
            "self-justifying explanations without new evidence",
        ],
    )


def generate_initial_strategies(
    spec: DTERunSpec,
    decomposition: TaskDecomposition,
    context: ResearchContext,
) -> list[SearchNode]:
    """StrategyGenerator role: produce distinct frontier nodes without ranking them."""

    base_assumptions = list(decomposition.constraints[:3])
    return [
        SearchNode(
            node_id="seed-direct",
            claim=f"Direct constructive route for: {spec.problem}",
            rationale=f"Use the most direct derivation path. Unknowns to check: {context.unknowns[0]}",
            assumptions=base_assumptions,
            risks=[context.failure_modes[0]],
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
    """Run the logical seed pipeline and return nodes plus audit metadata."""

    decomposition = decompose_task(spec)
    context = research_context(spec, decomposition)
    hint = compile_hint(decomposition, context)
    nodes = generate_initial_strategies(spec, decomposition, context)
    audit = {
        "decomposition": decomposition.__dict__,
        "research_context": context.__dict__,
        "compile_hint": hint.__dict__,
        "distiller_role": "removed: compile is an optional agent-local instruction, not a mandatory backend step",
    }
    return nodes, audit
