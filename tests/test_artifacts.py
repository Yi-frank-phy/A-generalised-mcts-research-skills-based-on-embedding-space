from dte_backend.artifacts import render_entropy_trace_markdown, render_frontier_markdown, render_main_agent_status
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.runner import run_frontier_search


def test_codex_app_artifacts_render():
    spec = DTERunSpec(problem="p", goal="g", budget=BudgetSpec(max_iterations=1, total_child_budget=1))
    result = run_frontier_search(spec, [SearchNode(node_id="n", claim="claim")])
    assert "DTE Frontier" in render_frontier_markdown(result)
    assert "Entropy" in render_entropy_trace_markdown(result)
    assert "Main Agent Status" in render_main_agent_status(result)
