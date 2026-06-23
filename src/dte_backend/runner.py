"""Mandatory DTE frontier-search runner prototype."""

from __future__ import annotations

from dataclasses import dataclass, field

from .adapter import ExecutorAdapter
from .cache import DTECache
from .embedding import get_embedding_provider
from .entropy import EntropyState, evaluate_entropy_state
from .expansion import expand_frontier
from .human import HumanQuestion, maybe_create_human_question
from .judge import batch_judge
from .math_engine import allocate_frontier
from .merge import apply_equivalent_merges
from .models import AllocationResult, DTERunSpec, MergeProposal, SearchNode
from .novelty import estimate_frontier_kde_state
from .role_pipeline import seed_frontier_from_roles
from .synthesis import synthesize_report


@dataclass
class IterationTrace:
    iteration: int
    allocations: list[AllocationResult]
    notes: list[str] = field(default_factory=list)
    merges: list[MergeProposal] = field(default_factory=list)
    entropy_state: EntropyState | None = None
    human_question: HumanQuestion | None = None


@dataclass
class RunResult:
    spec: DTERunSpec
    nodes: list[SearchNode]
    traces: list[IterationTrace]
    report: str
    cache: DTECache
    role_audit: dict[str, object] = field(default_factory=dict)


def _seed_nodes_from_spec(spec: DTERunSpec) -> tuple[list[SearchNode], dict[str, object]]:
    """Seed frontier through the old role-isolated backend logic."""

    return seed_frontier_from_roles(spec)


def run_frontier_search(
    spec: DTERunSpec,
    initial_nodes: list[SearchNode] | None = None,
    executor_adapter: ExecutorAdapter | None = None,
    cache: DTECache | None = None,
) -> RunResult:
    """Run the mandatory DTE loop.

    This preserves the old backend structure at the logical level:
    role-isolated seeding -> Judge -> embedding/KDE/entropy -> UCB/Boltzmann ->
    Expansion/Executor -> merge/synthesis/HIL artifacts.
    """

    role_audit: dict[str, object] = {}
    if initial_nodes:
        nodes = list(initial_nodes)
    else:
        nodes, role_audit = _seed_nodes_from_spec(spec)

    traces: list[IterationTrace] = []
    cache = cache or DTECache()
    embedding_provider = get_embedding_provider(spec.embedding_provider, dim=spec.embedding_dimension)
    previous_entropy: float | None = None

    for iteration in range(1, spec.budget.max_iterations + 1):
        frontier = [node for node in nodes if node.status == "frontier"]
        if not frontier:
            traces.append(IterationTrace(iteration=iteration, allocations=[], notes=["no frontier nodes; stopped"]))
            break

        merges = apply_equivalent_merges(nodes)
        frontier = [node for node in nodes if node.status == "frontier"]
        if not frontier:
            traces.append(
                IterationTrace(
                    iteration=iteration,
                    allocations=[],
                    notes=["all frontier nodes merged; stopped"],
                    merges=merges,
                )
            )
            break

        for result in batch_judge(frontier, cache=cache):
            for node in frontier:
                if node.node_id == result.node_id:
                    node.score = result.score
                    node.judge_reasoning = result.reasoning
                    break

        frontier, kde_state = estimate_frontier_kde_state(nodes, cache=cache, provider=embedding_provider)
        for node, uncertainty in zip(frontier, kde_state.uncertainty):
            node.uncertainty = uncertainty

        entropy_state = evaluate_entropy_state(
            spatial_entropy=kde_state.spatial_entropy,
            previous_entropy=previous_entropy,
            iteration=iteration,
            min_iterations=spec.budget.min_iterations_before_synthesis,
            entropy_change_threshold=spec.budget.entropy_change_threshold,
            t_max=spec.budget.t_max,
        )
        previous_entropy = kde_state.spatial_entropy

        allocations = allocate_frontier(
            nodes,
            total_budget=spec.budget.total_child_budget,
            tau=max(entropy_state.normalized_temperature, 0.05),
            c_explore=1.0,
            temperature=max(entropy_state.effective_temperature, 0.05),
        )
        for allocation in allocations:
            for node in frontier:
                if node.node_id == allocation.node_id:
                    node.ucb_score = allocation.ucb_score
                    node.expansion_budget = allocation.expansion_budget
                    break

        human_question = maybe_create_human_question(
            frontier,
            entropy_plateau=entropy_state.should_synthesize,
        )
        traces.append(
            IterationTrace(
                iteration=iteration,
                allocations=allocations,
                merges=merges,
                notes=[
                    f"frontier={len(frontier)}",
                    f"budget={spec.budget.total_child_budget}",
                    f"judge_cache_hits={cache.stats.judge_hits}",
                    f"embedding_cache_hits={cache.stats.embedding_hits}",
                    f"spatial_entropy={entropy_state.spatial_entropy:.4f}",
                    f"normalized_temperature={entropy_state.normalized_temperature:.4f}",
                    f"kde_bandwidth2={kde_state.bandwidth2:.4f}",
                ],
                entropy_state=entropy_state,
                human_question=human_question,
            )
        )

        if entropy_state.should_synthesize:
            traces[-1].notes.append(f"auto_synthesis_trigger={entropy_state.stop_reason}")
            break

        budget_map = {a.node_id: a.expansion_budget for a in allocations if a.expansion_budget > 0}
        nodes = expand_frontier(
            nodes,
            budget_map,
            iteration=iteration,
            spec=spec,
            executor_adapter=executor_adapter,
        )

    report = synthesize_report(spec, nodes)
    return RunResult(spec=spec, nodes=nodes, traces=traces, report=report, cache=cache, role_audit=role_audit)
