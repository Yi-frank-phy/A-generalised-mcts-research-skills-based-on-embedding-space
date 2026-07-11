from dte_backend.artifacts import (
    render_checkpoint_summary_markdown,
    render_entropy_trace_markdown,
    render_frontier_markdown,
    render_main_agent_status,
)
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode, SynthesisControlRequest
from dte_backend.runner import run_frontier_search


def test_codex_app_artifacts_render():
    spec = DTERunSpec(problem="p", goal="g", budget=BudgetSpec(max_iterations=1, total_child_budget=1))
    result = run_frontier_search(spec, [SearchNode(node_id="n", claim="claim")])
    assert "DTE Frontier" in render_frontier_markdown(result)
    assert "Entropy" in render_entropy_trace_markdown(result)
    assert "Main Agent Status" in render_main_agent_status(result)


def test_checkpoint_summary_mentions_forced_synthesis_state():
    spec = DTERunSpec(problem="p", goal="g", budget=BudgetSpec(max_iterations=5, total_child_budget=1))
    nodes = [SearchNode(node_id="n", claim="claim")]

    def control_callback(spec, nodes, traces):
        return SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="user",
            reason="reviewed checkpoint",
        )

    result = run_frontier_search(spec, nodes, control_callback=control_callback)
    summary = render_checkpoint_summary_markdown(result)
    status = render_main_agent_status(result)

    assert "DTE Checkpoint Summary" in summary
    assert "Run-level stop reason: user_interrupted_for_synthesis" in summary
    assert "user_interrupted_for_synthesis" in status
