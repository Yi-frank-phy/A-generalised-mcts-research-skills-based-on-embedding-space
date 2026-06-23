"""Mandatory DTE frontier-search runner prototype."""

from __future__ import annotations

from dataclasses import dataclass, field

from .adapter import ExecutorAdapter
from .cache import DTECache
from .embedding import get_embedding_provider
from .entropy import EntropyState, evaluate_entropy_state, spatial_entropy_from_embeddings
from .expansion import expand_frontier
from .judge import batch_judge
from .math_engine import allocate_frontier
from .merge import apply_equivalent_merges
from .models import AllocationResult, DTERunSpec, MergeProposal, SearchNode
from .novelty import estimate_uncertainty_from_density
from .synthesis import synthesize_report


@dataclass
class IterationTrace:
    iteration: int
    allocations: list[AllocationResult]
    notes: list[str] = field(default_factory=list)
    merges: list[MergeProposal] = field(default_factory=list)
    entropy_state: EntropyState | None = None


@dataclass
class RunResult:
    spec: DTERunSpec
    nodes: list[SearchNode]
    traces: list[IterationTrace]
    report: str
    cache: DTECache


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
    cache: DTECache | None = None,
) -> RunResult:
    """Run the mandatory DTE prototype loop.

    This never bypasses Judge/Evolution/Expansion. It is deterministic and
    offline by default, so it is suitable for harness-free prototype validation.
    """

    nodes = list(initial_nodes) if initial_nodes else _seed_nodes_from_spec(spec)
    traces: list[IterationTrace] = []
    cache = cache or DTECache()
    embedding_provider = get_embedding_provider(spec.embedding_provider, dim=spec.embedding_dimension)
    previous_entropy: float | None = None

    for iteration in range(1, spec.budget.max_iterations + 1):
        frontier = [node for node in nodes if node.status == "frontier"]
        if not frontier:
            traces.append(IterationTrace(iteration=iteration, allocations=[], notes=["no frontier nodes; stopped"]))
            break

        # Conservative graph-search compression before scoring/expansion.
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

        # Batch judge with cache.
        for result in batch_judge(frontier, cache=cache):
            for node in frontier:
                if node.node_id == result.node_id:
                    node.score = result.score
                    node.judge_reasoning = result.reasoning
                    break

        # Entropy/novelty proxy with embedding cache/provider.
        uncertainties = estimate_uncertainty_from_density(nodes, cache=cache, provider=embedding_provider)
        for node in frontier:
            node.uncertainty = uncertainties.get(node.node_id, 0.0)

        frontier_embeddings = [node.local_embedding or [] for node in frontier]
        spatial_entropy = spatial_entropy_from_embeddings(frontier_embeddings)
        entropy_state = evaluate_entropy_state(
            spatial_entropy=spatial_entropy,
            previous_entropy=previous_entropy,
            iteration=iteration,
            min_iterations=spec.budget.min_iterations_before_synthesis,
            entropy_change_threshold=spec.budget.entropy_change_threshold,
            t_max=spec.budget.t_max,
        )
        previous_entropy = spatial_entropy

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
                ],
                entropy_state=entropy_state,
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
    return RunResult(spec=spec, nodes=nodes, traces=traces, report=report, cache=cache)
