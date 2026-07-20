import json
import hashlib
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

import dte_backend.app_driver as app_driver
from dte_backend.app_driver import (
    app_run_status,
    cancel_app_episode,
    create_app_run,
    fail_app_episode,
    next_app_episode,
    request_app_synthesis,
    retry_app_episode,
    submit_app_episode_result,
)
from dte_backend.episode_models import (
    EpisodeResult,
    ExecutorEpisodeOutput,
    ExecutorNodeCandidate,
    JudgeEpisodeOutput,
    JudgeObservation,
    RuntimeDiagnostics,
    RuntimeLimits,
    compute_output_hash,
    canonical_json_bytes,
)
from dte_backend.control import OperatorAuthorizationError
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode, SynthesisControlRequest
from dte_backend.telemetry import EpisodeEventLog


def spec() -> DTERunSpec:
    return DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(
            max_iterations=2,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=2,
            max_relation_enrichment_pairs=0,
        ),
        embedding_provider="hash",
        embedding_dimension=8,
    )


def parent() -> SearchNode:
    return SearchNode(node_id="parent", claim="committed parent")


def create_run(tmp_path):
    run_dir = tmp_path / "run"
    create_app_run(run_dir, spec(), [parent()], run_id="run-1")
    judge = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result_for(judge))
    state = app_driver.load_app_run(run_dir)
    action, _ = app_driver._progress_controller(
        run_dir,
        state,
        embedding_provider=HashEmbeddingProvider(dim=state.spec.embedding_dimension),
    )
    state.controller_action = action
    app_driver._save_state(run_dir, state)
    return run_dir


def test_restart_preserves_bounded_continuation_gate_decision(tmp_path):
    run_dir = create_run(tmp_path)
    before = app_driver.load_app_run(run_dir)

    after = app_driver.load_app_run(run_dir)

    assert [
        record.model_dump(mode="json")
        for record in after.continuation_gate_records
    ] == [
        record.model_dump(mode="json")
        for record in before.continuation_gate_records
    ]


def test_restart_rejects_tampered_continuation_gate_decision(tmp_path):
    run_dir = create_run(tmp_path)
    state_path = run_dir / "app_run_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    original = payload["continuation_gate_records"][0]["decision"]
    payload["continuation_gate_records"][0]["decision"] = (
        "prepare_synthesis" if original == "continue" else "continue"
    )
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="continuation-gate record"):
        app_driver.load_app_run(run_dir)


def test_restart_replays_and_rejects_tampered_material_yield(tmp_path):
    run_dir = create_run(tmp_path)
    state_path = run_dir / "app_run_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["continuation_gate_records"][0]["material_yield_signals"].append(
        "epistemic_disposition:counterexample_found:forged"
    )
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="continuation-gate record"):
        app_driver.load_app_run(run_dir)


def test_legacy_app_state_keeps_legacy_entropy_policy(tmp_path):
    run_dir = create_run(tmp_path)
    state_path = run_dir / "app_run_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    budget = payload["spec"]["budget"]
    budget.pop("max_committed_search_nodes")
    budget.pop("entropy_plateau_confirmations")
    budget.pop("continuation_policy")
    payload["continuation_gate_records"] = []
    payload["spec_hash"] = hashlib.sha256(
        canonical_json_bytes(payload["spec"])
    ).hexdigest()
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = app_driver.load_app_run(run_dir)

    assert restored.spec.budget.continuation_policy == "legacy_entropy_v1"
    assert restored.spec.budget.entropy_plateau_confirmations == 1
    assert restored.continuation_gate_records == []


def test_app_create_rejects_initial_nodes_above_search_node_cap(tmp_path):
    bounded = spec().model_copy(
        update={
            "budget": spec().budget.model_copy(
                update={"max_committed_search_nodes": 1}
            )
        }
    )
    with pytest.raises(
        ValueError,
        match="initial committed search nodes exceed max_committed_search_nodes",
    ):
        create_app_run(
            tmp_path / "over-cap",
            bounded,
            [
                SearchNode(node_id="one", claim="one"),
                SearchNode(node_id="two", claim="two"),
            ],
        )


def test_app_equal_search_node_cap_judges_before_terminal(tmp_path):
    bounded = spec().model_copy(
        update={
            "budget": spec().budget.model_copy(
                update={
                    "max_committed_search_nodes": 1,
                    "max_iterations": 10,
                }
            )
        }
    )
    run_dir = tmp_path / "equal-cap"
    create_app_run(run_dir, bounded, [parent()], run_id="equal-cap")
    judge = next_app_episode(run_dir).request
    assert judge.role == "judge"
    submit_app_episode_result(run_dir, judge_result_for(judge))

    terminal = next_app_episode(run_dir)
    assert terminal.controller_action == "ready_for_synthesis"
    state = app_run_status(run_dir)
    assert state.nodes[0].score == pytest.approx(0.8)
    assert state.terminal_record.source == "max_search_nodes"


def test_app_allocation_and_executor_grant_share_remaining_node_slots(tmp_path):
    bounded = spec().model_copy(
        update={
            "budget": spec().budget.model_copy(
                update={
                    "max_committed_search_nodes": 4,
                    "max_iterations": 10,
                    "allocation_mass_per_iteration": 5,
                    "max_children_per_iteration": 5,
                }
            )
        }
    )
    run_dir = tmp_path / "two-slots"
    create_app_run(
        run_dir,
        bounded,
        [
            SearchNode(node_id="one", claim="one"),
            SearchNode(node_id="two", claim="two"),
        ],
        run_id="two-slots",
    )
    while True:
        request = next_app_episode(
            run_dir, embedding_provider=HashEmbeddingProvider(dim=8)
        ).request
        if request.role == "executor":
            executor = request
            break
        assert request.role == "judge"
        submit_app_episode_result(run_dir, judge_result_for(request))

    state = app_run_status(run_dir)
    controller = state.controller_iteration_records[-1]
    assert sum(controller.allocations.values()) == 2
    assert controller.effective_child_cap == 2
    assert executor.max_returned_children <= 2

    rejected = submit_app_episode_result(
        run_dir,
        result_for(executor, children=executor.max_returned_children + 1),
    )
    assert rejected.commit_outcome.accepted is False
    assert rejected.commit_outcome.rejection_reason == (
        "returned child count exceeds grant"
    )
    assert len(app_run_status(run_dir).nodes) == 2


def judge_result_for(request):
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id=node_id,
                score=0.8,
                reasoning="bounded Judge observation",
                risks=[],
            )
            for node_id in request.selected_node_revisions
        ]
    )
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role="judge",
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status="completed",
        structured_output=output,
        runtime_diagnostics=RuntimeDiagnostics(
            adapter_name="codex-app-main-agent",
            transport_name="current-app-runtime",
            profile="native-autonomous",
            usage_source="unavailable",
        ),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def result_for(request, *, children=1, node_id_prefix="child", status="completed"):
    output = None
    if status == "completed":
        output = ExecutorEpisodeOutput(
            nodes=[
                ExecutorNodeCandidate(
                    node_id=f"{node_id_prefix}-{index}",
                    claim=f"candidate {index}",
                    parent_ids=[request.parent_node_id],
                )
                for index in range(children)
            ],
            episode_summary="App-native episode completed",
        )
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role=request.role,
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status=status,
        structured_output=output,
        runtime_diagnostics=RuntimeDiagnostics(
            adapter_name="codex-app-main-agent",
            transport_name="current-app-runtime",
            profile="native-autonomous",
            usage_source="unavailable",
        ),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def graph_snapshot(run_dir):
    state = app_run_status(run_dir)
    return {
        "revision": state.graph_revision,
        "node_revisions": dict(state.node_revisions),
        "nodes": [node.model_dump(mode="json") for node in state.nodes],
    }


def lifecycle_for(state, episode_id):
    return next(episode for episode in state.episodes if episode.episode_id == episode_id)


def test_next_episode_creates_one_bounded_persistent_grant_without_subprocess(monkeypatch, tmp_path):
    run_dir = create_run(tmp_path)

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("App-native next-episode must not launch a subprocess")

    monkeypatch.setattr("subprocess.run", forbidden_subprocess)
    outcome = next_app_episode(run_dir)
    assert outcome.controller_action == "episode_required"
    assert outcome.request.role == "executor"
    assert outcome.request.max_returned_children == 1
    assert outcome.request.transport_hints == {
        "profile": "native-autonomous",
        "runtime": "current-codex-app",
    }
    request_path = (
        run_dir
        / "episodes"
        / outcome.request.episode_id
        / outcome.request.attempt_id
        / "request.json"
    )
    assert request_path.exists()
    assert [
        event["event_type"]
        for event in EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    ][-2:] == ["episode_granted", "episode_started"]


def test_next_episode_resumes_existing_attempt_instead_of_double_grant(tmp_path):
    run_dir = create_run(tmp_path)
    first = next_app_episode(run_dir)
    second = next_app_episode(run_dir)
    assert second.resumed_existing_attempt is True
    assert second.request.attempt_id == first.request.attempt_id
    events = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    assert [event["event_type"] for event in events].count("episode_granted") == 2


def test_valid_app_result_commits_and_backend_selects_next_action(tmp_path):
    run_dir = create_run(tmp_path)
    request = next_app_episode(run_dir).request
    revision_before = app_run_status(run_dir).graph_revision
    outcome = submit_app_episode_result(run_dir, result_for(request))
    assert outcome.commit_outcome.accepted is True
    assert outcome.next_controller_action == "continue_controller"
    state = app_run_status(run_dir)
    assert state.graph_revision == revision_before + 1
    assert [node.node_id for node in state.nodes] == ["parent", "child-0"]
    assert state.nodes[0].status == "closed"
    assert lifecycle_for(state, request.episode_id).committed_attempt_id == request.attempt_id


def test_valid_zero_child_app_result(tmp_path):
    run_dir = create_run(tmp_path)
    request = next_app_episode(run_dir).request
    outcome = submit_app_episode_result(run_dir, result_for(request, children=0))
    assert outcome.commit_outcome.accepted is True
    assert outcome.commit_outcome.accepted_node_count == 0
    assert app_run_status(run_dir).nodes[0].status == "closed"


@pytest.mark.parametrize("transition", ["failed", "cancelled"])
def test_failed_or_cancelled_attempt_cannot_commit_and_graph_is_unchanged(tmp_path, transition):
    run_dir = create_run(tmp_path)
    request = next_app_episode(run_dir).request
    before = graph_snapshot(run_dir)
    if transition == "failed":
        fail_app_episode(run_dir, request.episode_id, request.attempt_id, "runtime failed")
    else:
        cancel_app_episode(run_dir, request.episode_id, request.attempt_id, "operator cancelled")
    outcome = submit_app_episode_result(run_dir, result_for(request))
    assert outcome.commit_outcome.accepted is False
    assert f"status={transition}" in outcome.commit_outcome.rejection_reason
    assert graph_snapshot(run_dir) == before


def test_expired_attempt_cannot_commit(monkeypatch, tmp_path):
    run_dir = create_run(tmp_path)
    request = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(wall_clock_seconds=1, selected_by="main_agent"),
    ).request
    before = graph_snapshot(run_dir)
    original_now = app_driver._now()
    monkeypatch.setattr(app_driver, "_now", lambda: original_now + timedelta(seconds=10))
    outcome = submit_app_episode_result(run_dir, result_for(request))
    assert outcome.commit_outcome.accepted is False
    assert "expired" in outcome.commit_outcome.rejection_reason
    assert graph_snapshot(run_dir) == before
    state = app_run_status(run_dir)
    assert lifecycle_for(state, request.episode_id).attempts[0].status == "expired"


def test_retry_creates_new_attempt_and_supersedes_old_result(tmp_path):
    run_dir = create_run(tmp_path)
    first = next_app_episode(run_dir).request
    fail_app_episode(run_dir, first.episode_id, first.attempt_id, "retryable")
    retry = retry_app_episode(run_dir, first.episode_id)
    assert retry.request.episode_id == first.episode_id
    assert retry.attempt_id != first.attempt_id
    assert retry.request.attempt_id == retry.attempt_id
    before = graph_snapshot(run_dir)
    old_outcome = submit_app_episode_result(run_dir, result_for(first, node_id_prefix="late"))
    assert old_outcome.commit_outcome.accepted is False
    assert "superseded" in old_outcome.commit_outcome.rejection_reason
    assert graph_snapshot(run_dir) == before

    new_outcome = submit_app_episode_result(run_dir, result_for(retry.request))
    assert new_outcome.commit_outcome.accepted is True
    state = app_run_status(run_dir)
    episode = lifecycle_for(state, first.episode_id)
    assert episode.attempts[0].status == "superseded"
    assert episode.attempts[1].status == "committed"


def test_retry_limit_is_enforced(tmp_path):
    run_dir = create_run(tmp_path)
    first = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(max_retries=1),
    ).request
    fail_app_episode(run_dir, first.episode_id, first.attempt_id, "first failure")
    second = retry_app_episode(run_dir, first.episode_id)
    fail_app_episode(run_dir, second.episode_id, second.attempt_id, "second failure")
    with pytest.raises(ValueError, match="retry limit exhausted"):
        retry_app_episode(run_dir, first.episode_id)


def test_only_one_attempt_can_commit(tmp_path):
    run_dir = create_run(tmp_path)
    request = next_app_episode(run_dir).request
    first = submit_app_episode_result(run_dir, result_for(request))
    snapshot = graph_snapshot(run_dir)
    second = submit_app_episode_result(run_dir, result_for(request))
    assert first.commit_outcome.accepted is True
    assert second.commit_outcome.accepted is False
    assert "status=committed" in second.commit_outcome.rejection_reason
    assert graph_snapshot(run_dir) == snapshot
    state = app_run_status(run_dir)
    assert lifecycle_for(state, request.episode_id).attempts[0].commit_outcome.accepted is True


def test_result_artifact_is_not_graph_state_until_submit(tmp_path):
    run_dir = create_run(tmp_path)
    request = next_app_episode(run_dir).request
    before = graph_snapshot(run_dir)
    result_path = run_dir / "episodes" / request.episode_id / request.attempt_id / "result.json"
    result_path.write_text(result_for(request).model_dump_json(indent=2), encoding="utf-8")
    assert graph_snapshot(run_dir) == before


def test_app_telemetry_is_coarse_and_usage_is_unavailable(tmp_path):
    run_dir = create_run(tmp_path)
    request = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, result_for(request))
    events = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    assert {event["event_type"] for event in events}.issuperset(
        {"episode_granted", "episode_started", "episode_submitted", "episode_completed", "nodes_committed"}
    )
    episode_events = [event for event in events if event["episode_id"]]
    assert all(event["usage_source"] == "unavailable" for event in episode_events)
    serialized = json.dumps(episode_events)
    assert "subagent_count" not in serialized
    assert "subagent_names" not in serialized


def test_app_driver_preserves_operator_policy_authority(tmp_path):
    run_dir = tmp_path / "run"
    restricted_data = spec().model_dump()
    restricted_data["operator_policy"] = {"main_agent_may_request_synthesis": False}
    restricted = DTERunSpec.model_validate(restricted_data)
    create_app_run(run_dir, restricted, [SearchNode(node_id="closed", claim="done")])
    main_request = SynthesisControlRequest(
        action="force_synthesis_after_current_task",
        requested_by="main_agent",
        reason="operator requests synthesis",
    )
    with pytest.raises(OperatorAuthorizationError):
        request_app_synthesis(run_dir, main_request)
    judge = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result_for(judge))
    executor = next_app_episode(
        run_dir,
        embedding_provider=HashEmbeddingProvider(dim=restricted.embedding_dimension),
    ).request
    submit_app_episode_result(run_dir, result_for(executor, children=0))
    user_request = main_request.model_copy(update={"requested_by": "user"})
    state = request_app_synthesis(run_dir, user_request)
    assert state.controller_action == "continue_controller"
    assert state.synthesis_request.requested_by == "user"
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"


def test_skill_and_agents_define_current_app_loop_without_sdk_primary_path():
    root = Path(__file__).resolve().parents[1]
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    combined = skill + agents
    for command in ("create-run", "next-episode", "submit-episode-result", "retry-episode"):
        assert command in combined
    assert "current App main agent performs the episode" in combined
    assert "Do not launch another Codex process" in combined
    assert "CodexSdkEpisodeAdapter" not in combined
    assert "subagent count" in combined


def test_app_driver_cli_round_trip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "cli-run"
    spec_path = tmp_path / "spec.json"
    nodes_path = tmp_path / "nodes.json"
    result_path = tmp_path / "result.json"
    spec_path.write_text(spec().model_dump_json(indent=2), encoding="utf-8")
    nodes_path.write_text(json.dumps([parent().model_dump(mode="json")]), encoding="utf-8")

    def command(*args):
        completed = subprocess.run(
            [sys.executable, "-m", "dte_backend", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    command(
        "create-run",
        "--run-dir",
        str(run_dir),
        "--spec",
        str(spec_path),
        "--nodes",
        str(nodes_path),
        "--run-id",
        "cli-run",
    )
    granted = command("next-episode", "--run-dir", str(run_dir))
    request = next_app_episode(run_dir).request
    assert request.role == "judge"
    assert granted["request"]["attempt_id"] == request.attempt_id
    result_path.write_text(judge_result_for(request).model_dump_json(indent=2), encoding="utf-8")
    submitted = command(
        "submit-episode-result",
        "--run-dir",
        str(run_dir),
        "--result",
        str(result_path),
    )
    assert submitted["commit_outcome"]["accepted"] is True
    status = command("run-status", "--run-dir", str(run_dir))
    assert status["graph_revision"] == 1
