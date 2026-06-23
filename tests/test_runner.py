from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.runner import run_frontier_search
from dte_backend.synthesis import synthesize_report
import pytest


def test_run_frontier_search_minimal_loop():
    spec = DTERunSpec(problem="test problem", goal="test goal", budget=BudgetSpec(max_iterations=1, total_child_budget=2))
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
    spec = DTERunSpec(problem="seed me", goal="report", budget=BudgetSpec(max_iterations=1, total_child_budget=1))
    result = run_frontier_search(spec)
    assert len(result.nodes) >= 3
    assert result.report.startswith("# DTE Prototype Report")


def test_run_frontier_search_uses_supplied_judge_adapter():
    spec = DTERunSpec(problem="judge me", goal="report", budget=BudgetSpec(max_iterations=1, total_child_budget=1))
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
    spec = DTERunSpec(problem="judge me", goal="report", budget=BudgetSpec(max_iterations=1, total_child_budget=1))
    nodes = [SearchNode(node_id="a", claim="route")]

    def bad_judge_adapter(frontier):
        return [{"node_id": frontier[0].node_id, "score": 0.91, "reasoning": "bad", "ucb_score": 99}]

    with pytest.raises(ValueError, match="forbidden"):
        run_frontier_search(spec, nodes, judge_adapter=bad_judge_adapter)


def test_synthesis_mentions_protocol():
    spec = DTERunSpec(problem="p", goal="g")
    report = synthesize_report(spec, [SearchNode(node_id="n", claim="claim", score=0.8)])
    assert "Judge/Evolution/Expansion" in report
