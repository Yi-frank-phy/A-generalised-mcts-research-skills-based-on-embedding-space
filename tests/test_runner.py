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
        SearchNode(node_id="a", claim="route A", rationale="direct", confidence=0.6),
        SearchNode(node_id="b", claim="route B", rationale="counter", confidence=0.5),
    ]
    result = run_frontier_search(spec, nodes)
    assert result.traces
    assert len(result.nodes) >= 2
    assert "DTE Prototype Report" in result.report
    assert any(node.status == "closed" for node in result.nodes)


def test_run_seeds_when_no_nodes():
    spec = DTERunSpec(
        problem="seed me",
        goal="report",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=1),
    )
    result = run_frontier_search(spec)
    assert len(result.nodes) >= 3
    assert result.report.startswith("# DTE Prototype Report")


def test_run_frontier_search_uses_supplied_judge_adapter():
    spec = DTERunSpec(
        problem="judge me",
        goal="report",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=1),
    )
    nodes = [
        SearchNode(node_id="a", claim="strong route", confidence=0.1),
        SearchNode(node_id="b", claim="weak route", confidence=0.9),
    ]

    def judge_adapter(frontier):
        return [
            {"node_id": node.node_id, "score": 0.91, "reasoning": "external judge", "risks": []}
            for node in frontier
        ]

    result = run_frontier_search(spec, nodes, judge_adapter=judge_adapter)

    assert {node.score for node in result.nodes if node.node_id in {"a", "b"}} == {0.91}
    assert all(node.judge_reasoning == "external judge" for node in result.nodes if node.node_id in {"a", "b"})


def test_run_frontier_search_validates_supplied_judge_adapter_output():
    spec = DTERunSpec(
        problem="judge me",
        goal="report",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=1),
    )
    nodes = [SearchNode(node_id="a", claim="route")]

    def bad_judge_adapter(frontier):
        return [{"node_id": frontier[0].node_id, "score": 0.91, "reasoning": "bad", "ucb_score": 99}]

    with pytest.raises(ValueError, match="forbidden"):
        run_frontier_search(spec, nodes, judge_adapter=bad_judge_adapter)


def test_run_frontier_search_can_force_synthesis_after_checkpoint():
    spec = DTERunSpec(
        problem="force synthesis",
        goal="stop after reviewed checkpoint",
        budget=BudgetSpec(
            max_iterations=5,
            allocation_mass_per_iteration=2,
            min_iterations_before_synthesis=5,
        ),
    )
    nodes = [
        SearchNode(node_id="a", claim="route A", confidence=0.7),
        SearchNode(node_id="b", claim="route B", confidence=0.6),
    ]

    def control_callback(spec, nodes, traces):
        return SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="checkpoint shows enough coverage",
            scope="node_ids",
            node_ids=["a"],
        )

    result = run_frontier_search(spec, nodes, control_callback=control_callback)

    assert len(result.traces) == 1
    assert result.stop_reason == "main_agent_requested_synthesis"
    assert result.forced_synthesis is not None
    assert result.forced_synthesis.node_ids == ["a"]
    assert next(node for node in result.nodes if node.node_id == "a").status == "frontier"
    assert next(node for node in result.nodes if node.node_id == "b").status == "frontier"
    assert "Forced Synthesis" in result.report
    assert "main_agent_requested_synthesis" in result.report
    assert "left unexplored" in result.report
    assert "- stop reason: `entropy_plateau`" not in result.report


def test_synthesis_mentions_protocol():
    spec = DTERunSpec(problem="p", goal="g")
    report = synthesize_report(spec, [SearchNode(node_id="n", claim="claim", score=0.8)])
    assert "Judge/Evolution/Expansion" in report
