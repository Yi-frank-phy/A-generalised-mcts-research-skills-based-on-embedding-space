"""Mandatory DTE frontier-search runner prototype."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .adapter import ExecutorAdapter
from .episode_adapter import AgentEpisodeAdapter
from .episode_commit import EpisodeGraph
from .cache import DTECache
from .control import authorize_synthesis_control, record_forced_synthesis
from .continuation import (
    ContinuationGateRecord,
    count_committed_search_nodes,
    evaluate_continuation_gate,
    remaining_search_node_slots,
)
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
from .synthesis import select_provisional_synthesis_nodes, synthesize_report
from .telemetry import EpisodeEventLog


@dataclass
class IterationTrace:
    iteration: int
    allocations: list[AllocationResult]
    notes: list[str] = field(default_factory=list)
    merges: list[MergeProposal] = field(default_factory=list)
    entropy_state: EntropyState | None = None
    continuation_gate: ContinuationGateRecord | None = None
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
    run_id: str = "local-run"


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
    episode_adapter: AgentEpisodeAdapter | None = None,
    judge_adapter: JudgeAdapter | None = None,
    cache: DTECache | None = None,
    control_callback: ControlCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    control_path: str | Path | None = None,
    episode_event_log: EpisodeEventLog | None = None,
    run_id: str = "local-run",
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

    if (
        spec.budget.continuation_policy == "bounded_node_yield_v1"
        and count_committed_search_nodes(nodes)
        > spec.budget.max_committed_search_nodes
    ):
        raise ValueError("initial search nodes exceed max_committed_search_nodes")

    traces: list[IterationTrace] = []
    cache = cache or DTECache()
    embedding_provider = get_embedding_provider(spec.embedding_provider, dim=spec.embedding_dimension)
    previous_entropy: float | None = None
    previous_plateau_count = 0
    previous_frontier_node_ids: set[str] = set()
    previous_positive_allocation_node_ids: set[str] = set()
    previous_provisional_synthesis_node_ids: set[str] = set()
    considered_epistemic_record_ids: set[str] = set()
    stop_reason: str | None = None
    forced_synthesis: ForcedSynthesisRecord | None = None
    episode_graph = EpisodeGraph(nodes=nodes)

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
                run_id=run_id,
            )
        )

    def maybe_apply_operator_command(current_nodes: list[SearchNode]) -> bool:
        nonlocal forced_synthesis, stop_reason
        if control_callback is None:
            return False
        request = control_callback(spec, current_nodes, traces)
        if request is None:
            return False
        authorize_synthesis_control(spec, request)
        forced_synthesis = record_forced_synthesis(request, current_nodes, control_path=control_path)
        stop_reason = forced_synthesis.stop_reason
        if traces:
            traces[-1].notes.append(f"operator_command_trigger={stop_reason}")
        return True

    def checkpoint_then_maybe_apply_operator_command(current_nodes: list[SearchNode]) -> bool:
        """Persist a complete node-level commit before polling operator control."""

        maybe_checkpoint(current_nodes)
        return maybe_apply_operator_command(current_nodes)

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

        frontier, kde_state = estimate_frontier_kde_state(
            nodes,
            cache=cache,
            provider=embedding_provider,
            expected_dimension=spec.embedding_dimension,
        )
        for node, uncertainty in zip(frontier, kde_state.uncertainty):
            node.uncertainty = uncertainty

        entropy_state = evaluate_entropy_state(
            spatial_entropy=kde_state.spatial_entropy,
            previous_entropy=previous_entropy,
            iteration=iteration,
            min_iterations=spec.budget.min_iterations_before_synthesis,
            entropy_change_threshold=spec.budget.entropy_change_threshold,
            t_max=spec.budget.t_max,
            previous_plateau_count=previous_plateau_count,
            plateau_confirmations=spec.budget.entropy_plateau_confirmations,
        )
        previous_entropy = kde_state.spatial_entropy
        previous_plateau_count = entropy_state.consecutive_plateau_count

        if (
            spec.budget.continuation_policy == "bounded_node_yield_v1"
            and remaining_search_node_slots(
                nodes,
                spec.budget.max_committed_search_nodes,
            )
            == 0
        ):
            traces.append(
                IterationTrace(
                    iteration=iteration,
                    allocations=[],
                    merges=merges,
                    notes=[
                        f"frontier={len(frontier)}",
                        "auto_synthesis_trigger=max_search_nodes",
                    ],
                    entropy_state=entropy_state,
                )
            )
            stop_reason = "max_search_nodes"
            maybe_checkpoint(nodes)
            break

        effective_child_cap = spec.budget.max_children_per_iteration
        if spec.budget.continuation_policy == "bounded_node_yield_v1":
            effective_child_cap = min(
                effective_child_cap,
                remaining_search_node_slots(
                    nodes,
                    spec.budget.max_committed_search_nodes,
                ),
            )

        allocations = allocate_frontier(
            nodes,
            allocation_mass_per_iteration=spec.budget.allocation_mass_per_iteration,
            max_children_per_iteration=effective_child_cap,
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
            entropy_plateau=entropy_state.plateau_signal,
        )
        continuation_gate: ContinuationGateRecord | None = None
        if spec.budget.continuation_policy == "bounded_node_yield_v1":
            provisional = select_provisional_synthesis_nodes(
                nodes,
                graph_revision=episode_graph.revision,
            )
            allocation_map = {
                allocation.node_id: allocation.expansion_budget
                for allocation in allocations
            }
            continuation_gate = evaluate_continuation_gate(
                iteration=iteration,
                graph_revision=episode_graph.revision,
                nodes=nodes,
                max_committed_search_nodes=spec.budget.max_committed_search_nodes,
                entropy_delta=entropy_state.entropy_delta,
                consecutive_plateau_count=entropy_state.consecutive_plateau_count,
                plateau_confirmed=(
                    iteration >= spec.budget.min_iterations_before_synthesis
                    and entropy_state.consecutive_plateau_count
                    >= spec.budget.entropy_plateau_confirmations
                ),
                allocations=allocation_map,
                previous_frontier_node_ids=previous_frontier_node_ids,
                previous_positive_allocation_node_ids=(
                    previous_positive_allocation_node_ids
                ),
                previous_provisional_synthesis_node_ids=(
                    previous_provisional_synthesis_node_ids
                ),
                provisional_synthesis_node_ids=provisional.selected_node_ids,
                ledger=episode_graph.epistemic_ledger,
                previously_considered_epistemic_ids=(
                    considered_epistemic_record_ids
                ),
            )
            considered_epistemic_record_ids.update(
                continuation_gate.considered_epistemic_record_ids
            )
            previous_frontier_node_ids = {node.node_id for node in frontier}
            previous_positive_allocation_node_ids = {
                allocation.node_id
                for allocation in allocations
                if allocation.expansion_budget > 0
            }
            previous_provisional_synthesis_node_ids = set(
                provisional.selected_node_ids
            )
        traces.append(
            IterationTrace(
                iteration=iteration,
                allocations=allocations,
                merges=merges,
                notes=[
                    f"frontier={len(frontier)}",
                    f"allocation_mass={spec.budget.allocation_mass_per_iteration}",
                    f"max_children={effective_child_cap}",
                    f"judge_cache_hits={cache.stats.judge_hits}",
                    f"embedding_cache_hits={cache.stats.embedding_hits}",
                    f"spatial_entropy={entropy_state.spatial_entropy:.4f}",
                    f"normalized_temperature={entropy_state.normalized_temperature:.4f}",
                    f"kde_bandwidth2={kde_state.bandwidth2:.4f}",
                ],
                entropy_state=entropy_state,
                continuation_gate=continuation_gate,
                human_question=human_question,
            )
        )
        maybe_checkpoint(nodes)

        if (
            continuation_gate is not None
            and continuation_gate.decision == "prepare_synthesis"
        ):
            traces[-1].notes.append("auto_synthesis_trigger=continuation_gate")
            stop_reason = "continuation_gate"
            break
        if (
            spec.budget.continuation_policy == "legacy_entropy_v1"
            and entropy_state.should_synthesize
        ):
            traces[-1].notes.append(f"auto_synthesis_trigger={entropy_state.stop_reason}")
            stop_reason = entropy_state.stop_reason
            break
        if maybe_apply_operator_command(nodes):
            maybe_checkpoint(nodes)
            break

        budget_map = {a.node_id: a.expansion_budget for a in allocations if a.expansion_budget > 0}
        nodes = expand_frontier(
            nodes,
            budget_map,
            iteration=iteration,
            spec=spec,
            executor_adapter=executor_adapter,
            episode_adapter=episode_adapter,
            episode_graph=episode_graph,
            episode_event_log=episode_event_log,
            run_id=run_id,
            after_node_expanded=checkpoint_then_maybe_apply_operator_command,
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
        run_id=run_id,
    )
