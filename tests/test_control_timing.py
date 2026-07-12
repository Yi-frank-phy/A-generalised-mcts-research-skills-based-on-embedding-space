import pytest

from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode, SynthesisControlRequest
from dte_backend.runner import run_frontier_search


def _spec() -> DTERunSpec:
    return DTERunSpec(
        problem="safe-point timing",
        goal="preserve completed validated work",
        budget=BudgetSpec(
            max_iterations=3,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=1,
            min_iterations_before_synthesis=3,
        ),
    )


def test_operator_control_is_polled_only_after_checkpoint_and_complete_expansion():
    events: list[str] = []

    class RecordingExecutor:
        def expand(self, request):
            events.append("executor_started")
            child = SearchNode(node_id="child", claim="validated child", parent_ids=[request.parent.node_id])
            events.append("executor_returned")
            return [child]

    def checkpoint_callback(result):
        events.append("controller_checkpoint")

    polls = 0

    def control_callback(spec, nodes, traces):
        nonlocal polls
        polls += 1
        if polls == 1:
            events.append("control_after_controller_checkpoint")
            assert all(node.status == "frontier" for node in nodes)
            return None
        events.append("control_after_validated_expansion")
        assert {node.node_id for node in nodes} == {"parent", "child"}
        assert next(node for node in nodes if node.node_id == "parent").status == "closed"
        return SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="operator requested synthesis after the completed node expansion",
        )

    result = run_frontier_search(
        _spec(),
        [SearchNode(node_id="parent", claim="parent")],
        executor_adapter=RecordingExecutor(),
        checkpoint_callback=checkpoint_callback,
        control_callback=control_callback,
    )

    assert events == [
        "controller_checkpoint",
        "control_after_controller_checkpoint",
        "executor_started",
        "executor_returned",
        "controller_checkpoint",
        "control_after_validated_expansion",
        "controller_checkpoint",
    ]
    assert result.stop_reason == "main_agent_requested_synthesis"
    assert {node.node_id for node in result.nodes} == {"parent", "child"}


def test_pending_control_does_not_bypass_executor_validation():
    polls = 0

    class InvalidExecutor:
        def expand(self, request):
            return [
                SearchNode(
                    node_id="bad-child",
                    claim="invalid because it pre-fills a controller field",
                    parent_ids=[request.parent.node_id],
                    score=1.0,
                )
            ]

    def control_callback(spec, nodes, traces):
        nonlocal polls
        polls += 1
        if polls == 1:
            return None
        return SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="user",
            reason="must not be consumed before validation",
        )

    parent = SearchNode(node_id="parent", claim="parent")
    with pytest.raises(ValueError, match="controller-owned field: score"):
        run_frontier_search(
            _spec(),
            [parent],
            executor_adapter=InvalidExecutor(),
            control_callback=control_callback,
        )

    assert polls == 1
    assert parent.status == "frontier"


def test_invalid_control_after_valid_expansion_preserves_only_the_completed_checkpoint():
    snapshots: list[dict[str, str]] = []
    polls = 0

    class ValidExecutor:
        def expand(self, request):
            return [SearchNode(node_id="child", claim="validated child", parent_ids=[request.parent.node_id])]

    def checkpoint_callback(result):
        snapshots.append({node.node_id: node.status for node in result.nodes})

    def invalid_control_callback(spec, nodes, traces):
        nonlocal polls
        polls += 1
        if polls == 1:
            return None
        raise ValueError("invalid synthesis control request")

    parent = SearchNode(node_id="parent", claim="parent")
    with pytest.raises(ValueError, match="invalid synthesis control request"):
        run_frontier_search(
            _spec(),
            [parent],
            executor_adapter=ValidExecutor(),
            checkpoint_callback=checkpoint_callback,
            control_callback=invalid_control_callback,
        )

    assert snapshots == [
        {"parent": "frontier"},
        {"parent": "closed", "child": "frontier"},
    ]
    assert parent.status == "closed"
