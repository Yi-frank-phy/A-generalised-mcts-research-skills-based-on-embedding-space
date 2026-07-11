"""Mandatory DTE frontier-search runner prototype."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .adapter import ExecutorAdapter
from .cache import DTECache
from .control import record_forced_synthesis
from .embedding import get_embedding_provider
from .entropy import EntropyState, evaluate_entropy_state
from .expansion import expand_frontier
from .human import HumanQuestion, maybe_create_human_question
from .judge import batch_judge
from .math_engine import allocate_frontier
from .merge import apply_equivalent_merges
from .models import AllocationResult, DTERunSpec, ForcedSynthesisRecord, MergeProposal, SearchNode, SynthesisControlRequest
from .novelty import estimate_frontier_kde_state
from .oracle_validation import validate_judge_output
from .subprocess_oracles import JudgeAdapter
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
    stop_reason: str | None = None
    forced_synthesis: ForcedSynthesisRecord | None = None


ControlCallback = Callable[[DTERunSpec, list[SearchNode], list[IterationTrace]], SynthesisControlRequest | None]
CheckpointCallback = Callable[[RunResult], None]


def _seed_nodes_from_spec(spec: DTERunSpec) -> tuple[list[SearchNode], dict[str, object]]:
    """Seed frontier through the old role-isolated backend logic."""

    return seed_frontier_from_roles(spec)


def _validated_judge_results(frontier: list[SearchNode], judge_adapter: JudgeAdapter | None, cache: DTECache):
    """Return Judge results after enforcing the observable oracle contract."""

    raw_results = judge_adapter(frontier) if judge_adapter else batch_judge(frontier, cache=cache)
    normalized = []
    for result in raw_results:
        if isinstance(result, dict):
            normalized.append(result)
        else:
            normalized.append(
                {
                    "node_id": result.node_id,
                    "score": result.score,
                    "reasoning": result.reasoning,
                    "risks": getattr(result, "risks", []),
                }
            )
    return validate_judge_output(frontier, {"results": normalized})


def run_frontier_search(
    spec: DTERunSpec,
    initial_nodes: list[SearchNode] | None = None,
    executor_adapter: ExecutorAdapter | None = None,
    judge_adapter: JudgeAdapter | None = None,
    cache: DTECache | None = None,
    control_callback: ControlCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    control_path: str | Path | None = None,
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
    stop_reason: str | None = None
    forced_synthesis: ForcedSynthesisRecord | None = None

    def maybe_checkpoint(current_nodes: list[SearchNode]) -> None:
        if checkpoint_callback is None:
            return
        checkpoint_callback(
            RunResult(
                spec=spec,
                nodes=current_nodes,
                traces=traces,
                report="",
                cache=cache,
                role_audit=role_audit,
                stop_reason=stop_reason,
                forced_synthesis=forced_synthesis,
            )
        )

    def maybe_force_synthesis(current_nodes: list[SearchNode]) -> bool:
        nonlocal forced_synthesis, stop_reason
        if control_callback is None:
            return False
        request = control_callback(spec, current_nodes, traces)
        if request is None:
            return False
        forced_synthesis = record_forced_synthesis(request, current_nodes, control_path=control_path)
        stop_reason = forced_synthesis.stop_reason
        if traces:
            traces[-1].notes.append(f"forced_synthesis_trigger={stop_reason}")
        return True

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

        judge_results = _validated_judge_results(frontier, judge_adapter, cache)
        for result in judge_results:
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
        maybe_checkpoint(nodes)

        if entropy_state.should_synthesize:
            traces[-1].notes.append(f"auto_synthesis_trigger={entropy_state.stop_reason}")
            stop_reason = entropy_state.stop_reason
            break
        if maybe_force_synthesis(nodes):
            maybe_checkpoint(nodes)
            break

        budget_map = {a.node_id: a.expansion_budget for a in allocations if a.expansion_budget > 0}
        nodes = expand_frontier(
            nodes,
            budget_map,
            iteration=iteration,
            spec=spec,
            executor_adapter=executor_adapter,
            after_node_expanded=maybe_force_synthesis,
        )
        maybe_checkpoint(nodes)
        if forced_synthesis is not None:
            break

    if stop_reason is None and traces and traces[-1].iteration >= spec.budget.max_iterations:
        stop_reason = "max_iterations"
        traces[-1].notes.append("auto_synthesis_trigger=max_iterations")

    report = synthesize_report(spec, nodes, forced_synthesis=forced_synthesis)
    return RunResult(
        spec=spec,
        nodes=nodes,
        traces=traces,
        report=report,
        cache=cache,
        role_audit=role_audit,
        stop_reason=stop_reason,
        forced_synthesis=forced_synthesis,
    )
