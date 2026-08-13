"""Expansion boundary for the new release line."""

from __future__ import annotations
from collections.abc import Callable
from .adapter import ExecutorAdapter, validate_search_node_output
from .episode_adapter import AgentEpisodeAdapter, LegacyExecutorEpisodeAdapter, build_executor_episode_request, run_and_commit_episode
from .episode_commit import EpisodeGraph
from .models import DTERunSpec, ExpansionRequest, SearchNode
from .telemetry import EpisodeEventLog
from .transition_children import deterministic_transition_children


def deterministic_expand_node(parent: SearchNode, count: int, iteration: int) -> list[SearchNode]:
    return deterministic_transition_children(parent, count, iteration)


def expand_node(parent: SearchNode, count: int, iteration: int, spec: DTERunSpec | None = None, executor_adapter: ExecutorAdapter | None = None) -> list[SearchNode]:
    if count <= 0:
        return []
    if executor_adapter is None:
        return deterministic_transition_children(parent, count, iteration)
    request = ExpansionRequest(parent=parent, child_count=count, iteration=iteration, spec=spec)
    children = executor_adapter.expand(request) if hasattr(executor_adapter, "expand") else executor_adapter(request)
    return validate_search_node_output(parent, count, children)


def expand_frontier(
    nodes: list[SearchNode],
    budgets: dict[str, int],
    iteration: int,
    spec: DTERunSpec | None = None,
    executor_adapter: ExecutorAdapter | None = None,
    episode_adapter: AgentEpisodeAdapter | None = None,
    episode_graph: EpisodeGraph | None = None,
    episode_event_log: EpisodeEventLog | None = None,
    run_id: str = "local-run",
    after_node_expanded: Callable[[list[SearchNode]], bool] | None = None,
) -> list[SearchNode]:
    graph = episode_graph or EpisodeGraph(nodes=list(nodes))
    if episode_adapter is not None and executor_adapter is not None:
        raise ValueError("provide either episode_adapter or executor_adapter, not both")
    effective = episode_adapter or (LegacyExecutorEpisodeAdapter(executor_adapter) if executor_adapter is not None else None)
    for node_id, budget in budgets.items():
        parent = graph.node_by_id(node_id)
        if parent is None or parent.status != "frontier" or budget <= 0:
            continue
        if effective is None:
            children = expand_node(parent, budget, iteration, spec)
            parent.status = "closed"
            parent.expansion_budget = 0
            graph.nodes.extend(children)
        else:
            request = build_executor_episode_request(
                graph,
                parent,
                run_id=run_id,
                iteration=iteration,
                max_returned_children=budget,
                objective=parent.claim if spec is None else f"{spec.goal}: continue {parent.claim}",
                constraints=[] if spec is None else list(spec.constraints),
                native_orchestration_allowed=True if spec is None else spec.allow_self_organized_executor,
            )
            outcome = run_and_commit_episode(graph, request, effective, telemetry=episode_event_log)
            if not outcome.accepted:
                raise ValueError(outcome.rejection_reason or "Executor episode result rejected")
        if after_node_expanded is not None and after_node_expanded(graph.nodes):
            break
    return graph.nodes
