from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.runner import run_frontier_search
from dte_backend.synthesis import synthesize_report


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


def test_synthesis_mentions_protocol():
    spec = DTERunSpec(problem="p", goal="g")
    report = synthesize_report(spec, [SearchNode(node_id="n", claim="claim", score=0.8)])
    assert "Judge/Evolution/Expansion" in report
