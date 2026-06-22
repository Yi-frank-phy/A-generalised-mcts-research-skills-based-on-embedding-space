"""Mandatory DTE frontier-search runner prototype."""

from __future__ import annotations

from dataclasses import dataclass, field

from .executor_adapter import ExecutorAdapter
from .expansion import expand_frontier
from .judge import batch_judge
from .math_engine import allocate_frontier
from .models import AllocationResult, DTERunSpec, SearchNode
from .novelty import estimate_uncertainty_from_density
from .synthesis import synthesize_report


@dataclass
class IterationTrace:
    iteration: int
    allocations: list[AllocationResult]
    notes: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    spec: DTERunSpec
    nodes: list[SearchNode]
    traces: list[IterationTrace]
    report: str


def _seed_nodes_from_spec(spec: DTERunSpec) -> list[SearchNode]:
    """Create a small initial frontier when no nodes are supplied."""

    return [
        SearchNode(
            node_id="seed-direct",
            claim=f"Direct route for: {spec.problem}",
            rationale="Start with the most direct formulation of the problem.",
            assumptions=list(spec.constraints[:2]),
            confidence=0.55,
        ),
        SearchNode(
            node_id="seed-counter",
            claim=f"Counterexample-oriented route for: {spec.problem}",
            rationale="Search for failure modes before accepting the direct route.",
            assumptions=list(spec.constraints[:2]),
            confidence=0.50,
        ),
        SearchNode(
            node_id="seed-synthesis",
            claim=f"Synthesis route for: {spec.problem}",
            rationale="Look for a formulation that merges competing perspectives.",
            assumptions=list(spec.constraints[:2]),
            confidence=0.52,
        ),
    ]


def run_frontier_search(
    spec: DTERunSpec,
    initial_nodes: list[SearchNode] | None = None,
    executor_adapter: ExecutorAdapter | None = None,
) -> RunResult:
    """Run the mandatory DTE prototype loop.

    This never bypasses Judge/Evolution/Expansion. It is deterministic and
    offline, so it is suitable for harness-free prototype validation.
    """

    nodes = list(initial_nodes) if initial_nodes else _seed_nodes_from_spec(spec)
    traces: list[IterationTrace] = []

    for iteration in range(1, spec.budget.max_iterations + 1):
        frontier = [node for node in nodes if node.status == "frontier"]
        if not frontier:
            traces.append(IterationTrace(iteration=iteration, allocations=[], notes=["no frontier nodes; stopped"]))
            break

        # Batch judge.
        for result in batch_judge(frontier):
            for node in frontier:
                if node.node_id == result.node_id:
                    node.score = result.score
                    node.judge_reasoning = result.reasoning
                    break

        # Entropy/novelty proxy.
        uncertainties = estimate_uncertainty_from_density(nodes)
        for node in frontier:
            node.uncertainty = uncertainties.get(node.node_id, 0.0)

        allocations = allocate_frontier(
            nodes,
            total_budget=spec.budget.total_child_budget,
            tau=1.0,
            c_explore=1.0,
            temperature=1.0,
        )
        for allocation in allocations:
            for node in frontier:
                if node.node_id == allocation.node_id:
                    node.ucb_score = allocation.ucb_score
                    node.expansion_budget = allocation.expansion_budget
                    break

        traces.append(
            IterationTrace(
                iteration=iteration,
                allocations=allocations,
                notes=[f"frontier={len(frontier)}", f"budget={spec.budget.total_child_budget}"],
            )
        )

        budget_map = {a.node_id: a.expansion_budget for a in allocations if a.expansion_budget > 0}
        nodes = expand_frontier(
            nodes,
            budget_map,
            iteration=iteration,
            spec=spec,
            executor_adapter=executor_adapter,
        )

    report = synthesize_report(spec, nodes)
    return RunResult(spec=spec, nodes=nodes, traces=traces, report=report)
