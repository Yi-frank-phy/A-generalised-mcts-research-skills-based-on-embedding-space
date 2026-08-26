from tests.helpers import completed_node
import math

from dte_backend.app_driver import (
    app_run_status,
    create_app_run,
    next_app_episode,
    submit_app_episode_result,
)
from dte_backend.episode_models import EpisodeResult, RuntimeDiagnostics, compute_output_hash
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode, SynthesisControlRequest
from dte_backend.runner import run_frontier_search
from dte_backend.synthesis import synthesize_report
import pytest


def test_run_frontier_search_minimal_loop():
    spec = DTERunSpec(
        problem="test problem",
        goal="test goal",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=2),
    )
    nodes = [
        completed_node(node_id="a", claim="route A", rationale="direct", confidence=0.6),
        completed_node(node_id="b", claim="route B", rationale="counter", confidence=0.5),
    ]
    result = run_frontier_search(spec, nodes)
    assert result.traces
    assert len(result.nodes) >= 2
    assert "DTE Prototype Report" in result.report
    assert any(node.status == "closed" for node in result.nodes)


def test_runner_temperature_uses_current_frontier_diversity():
    spec = DTERunSpec(
        problem="temperature mapping",
        goal="trace canonical controller state",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=2),
    )
    nodes = [
        completed_node(node_id="a", claim="route A", confidence=0.6),
        completed_node(node_id="b", claim="route B", confidence=0.5),
    ]

    result = run_frontier_search(spec, nodes)

    entropy_state = result.traces[0].entropy_state
    assert entropy_state is not None
    assert entropy_state.normalized_temperature == pytest.approx(
        entropy_state.spatial_entropy / math.log(2)
    )


def test_run_seeds_when_no_nodes():
    spec = DTERunSpec(
        problem="seed me",
        goal="report",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=1),
    )
    result = run_frontier_search(spec)
    assert len(result.nodes) >= 3
    assert result.report.startswith("# DTE Prototype Report")






def test_run_frontier_search_accepts_user_interruption_after_checkpoint():
    spec = DTERunSpec(
        problem="force synthesis",
        goal="stop after reviewed checkpoint",
        operator_policy={"main_agent_may_request_synthesis": False},
        budget=BudgetSpec(
            max_iterations=5,
            allocation_mass_per_iteration=2,
            min_iterations_before_synthesis=5,
        ),
    )
    nodes = [
        completed_node(node_id="a", claim="route A", confidence=0.7),
        completed_node(node_id="b", claim="route B", confidence=0.6),
    ]

    def control_callback(spec, nodes, traces):
        return SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="user",
            reason="user reviewed the checkpoint and requested synthesis",
            scope="node_ids",
            node_ids=["a"],
        )

    result = run_frontier_search(spec, nodes, control_callback=control_callback)

    assert len(result.traces) == 1
    assert result.stop_reason == "user_interrupted_for_synthesis"
    assert result.forced_synthesis is not None
    assert result.forced_synthesis.node_ids == ["a"]
    assert next(node for node in result.nodes if node.node_id == "a").status == "frontier"
    assert next(node for node in result.nodes if node.node_id == "b").status == "frontier"
    assert "User-Interrupted Synthesis" in result.report
    assert "user_interrupted_for_synthesis" in result.report
    assert "left unexplored" in result.report
    assert "- stop reason: `entropy_plateau`" not in result.report


def test_run_frontier_search_accepts_authorized_main_agent_request():
    spec = DTERunSpec(
        problem="operator request",
        goal="stop through backend policy",
        budget=BudgetSpec(max_iterations=3, allocation_mass_per_iteration=1, min_iterations_before_synthesis=3),
    )

    def control_callback(spec, nodes, traces):
        return SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="operator proxy found sufficient coverage",
        )

    result = run_frontier_search(
        spec,
        [completed_node(node_id="a", claim="route A")],
        control_callback=control_callback,
    )

    assert result.stop_reason == "main_agent_requested_synthesis"
    assert result.forced_synthesis is not None
    assert result.forced_synthesis.requested_by == "main_agent"
    assert "Main-Agent-Requested Synthesis" in result.report
    assert "main_agent_requested_synthesis" in result.report
    assert "`entropy_plateau` convergence or algorithmic convergence" in result.report


def test_legacy_controller_natural_entropy_stop_is_unchanged():
    spec = DTERunSpec(
        problem="natural stop",
        goal="stop only when the controller converges",
        budget=BudgetSpec(
            max_iterations=5,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=1,
            min_iterations_before_synthesis=2,
            entropy_change_threshold=0.05,
            continuation_policy="legacy_entropy_v1",
            entropy_plateau_confirmations=1,
        ),
    )

    result = run_frontier_search(spec, [completed_node(node_id="a", claim="route A")])

    assert result.stop_reason == "entropy_plateau"
    assert result.forced_synthesis is None
    assert any("auto_synthesis_trigger=entropy_plateau" in note for note in result.traces[-1].notes)


def test_bounded_runner_rejects_initial_nodes_above_cap():
    spec = DTERunSpec(
        problem="bounded",
        goal="reject excess seeds",
        budget=BudgetSpec(max_committed_search_nodes=1),
    )

    with pytest.raises(ValueError, match="initial search nodes exceed"):
        run_frontier_search(
            spec,
            [
                completed_node(node_id="a", claim="route A"),
                completed_node(node_id="b", claim="route B"),
            ],
        )






def test_bounded_runner_trims_allocation_to_remaining_node_slots():
    spec = DTERunSpec(
        problem="bounded",
        goal="grant only remaining slots",
        budget=BudgetSpec(
            max_committed_search_nodes=4,
            max_iterations=1,
            allocation_mass_per_iteration=5,
            max_children_per_iteration=5,
        ),
    )
    initial = [
        completed_node(node_id="a", claim="route A"),
        completed_node(node_id="b", claim="route B"),
    ]

    result = run_frontier_search(spec, initial)

    assert sum(item.expansion_budget for item in result.traces[0].allocations) == 2
    assert len(result.nodes) == 4


