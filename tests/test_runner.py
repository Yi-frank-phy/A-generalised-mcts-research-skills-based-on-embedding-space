from dte_backend.app_driver import (
    app_run_status,
    create_app_run,
    next_app_episode,
    submit_app_episode_result,
)
from dte_backend.episode_models import (
    EpisodeResult,
    JudgeEpisodeOutput,
    JudgeObservation,
    RuntimeDiagnostics,
    compute_output_hash,
)
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
        SearchNode(node_id="a", claim="route A", confidence=0.7),
        SearchNode(node_id="b", claim="route B", confidence=0.6),
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
        [SearchNode(node_id="a", claim="route A")],
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

    result = run_frontier_search(spec, [SearchNode(node_id="a", claim="route A")])

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
                SearchNode(node_id="a", claim="route A"),
                SearchNode(node_id="b", claim="route B"),
            ],
        )


def test_bounded_runner_judges_equal_cap_then_stops_without_expansion():
    spec = DTERunSpec(
        problem="bounded",
        goal="judge before node-cap terminal",
        budget=BudgetSpec(
            max_committed_search_nodes=1,
            max_iterations=10,
        ),
    )

    result = run_frontier_search(
        spec,
        [SearchNode(node_id="a", claim="route A")],
    )

    assert result.stop_reason == "max_search_nodes"
    assert len(result.nodes) == 1
    assert result.nodes[0].score is not None
    assert result.traces[0].allocations == []


def test_app_native_and_strict_runner_share_equal_cap_stop(tmp_path):
    spec = DTERunSpec(
        problem="bounded",
        goal="share node-cap stopping semantics",
        budget=BudgetSpec(
            max_committed_search_nodes=1,
            max_iterations=10,
        ),
        embedding_provider="hash",
        embedding_dimension=8,
    )
    strict = run_frontier_search(
        spec,
        [SearchNode(node_id="a", claim="route A")],
    )

    run_dir = tmp_path / "shared-cap"
    create_app_run(
        run_dir,
        spec,
        [SearchNode(node_id="a", claim="route A")],
        run_id="shared-cap",
    )
    request = next_app_episode(run_dir).request
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id="a",
                score=0.8,
                reasoning="bounded Judge observation",
                risks=[],
            )
        ]
    )
    submit_app_episode_result(
        run_dir,
        EpisodeResult(
            episode_id=request.episode_id,
            attempt_id=request.attempt_id,
            run_id=request.run_id,
            role="judge",
            input_graph_revision=request.input_graph_revision,
            selected_node_revisions=request.selected_node_revisions,
            status="completed",
            structured_output=output,
            runtime_diagnostics=RuntimeDiagnostics(
                adapter_name="test",
                transport_name="test",
                profile="native-guided",
                usage_source="unavailable",
            ),
            output_hash=compute_output_hash(output, request.output_schema_version),
            schema_version=request.output_schema_version,
        ),
    )
    app_outcome = next_app_episode(run_dir)
    app_state = app_run_status(run_dir)

    assert strict.stop_reason == "max_search_nodes"
    assert app_outcome.controller_action == "ready_for_synthesis"
    assert app_state.terminal_record.source == strict.stop_reason
    assert app_state.nodes[0].score is not None


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
        SearchNode(node_id="a", claim="route A"),
        SearchNode(node_id="b", claim="route B"),
    ]

    result = run_frontier_search(spec, initial)

    assert sum(item.expansion_budget for item in result.traces[0].allocations) == 2
    assert len(result.nodes) == 4


def test_synthesis_mentions_protocol():
    spec = DTERunSpec(problem="p", goal="g")
    report = synthesize_report(spec, [SearchNode(node_id="n", claim="claim", score=0.8)])
    assert "Judge/Evolution/Expansion" in report
