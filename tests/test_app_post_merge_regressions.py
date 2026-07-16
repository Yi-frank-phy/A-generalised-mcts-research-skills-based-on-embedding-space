import json
import copy
import uuid
from datetime import timedelta

import pytest

import dte_backend.app_driver as app_driver
from dte_backend.app_driver import (
    TerminalRecord,
    app_run_status,
    create_app_run,
    fail_app_episode,
    next_app_episode,
    request_app_synthesis,
    retry_app_episode,
    submit_app_episode_result,
)
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.episode_models import (
    EpisodeResult,
    ExecutorEpisodeOutput,
    ExecutorNodeCandidate,
    JudgeEpisodeOutput,
    JudgeObservation,
    RuntimeDiagnostics,
    RuntimeLimits,
    compute_output_hash,
)
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode, SynthesisControlRequest
from dte_backend.relation_models import RelationEpisodeOutput, RelationObservation
from dte_backend.telemetry import EpisodeEventLog


def run_spec(*, final=True, iterations=2, enrichment=0):
    return DTERunSpec(
        problem="post-merge controller regression",
        goal="preserve authority, atomicity, and liveness",
        budget=BudgetSpec(
            max_iterations=iterations,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=2,
            max_relation_pairs_per_episode=2,
            max_relation_enrichment_pairs=enrichment,
            min_iterations_before_synthesis=2,
        ),
        require_final_synthesis=final,
        embedding_provider="hash",
        embedding_dimension=8,
    )


def judge_result(request, score=0.8):
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id=node_id,
                score=score,
                reasoning="bounded observation",
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


def executor_result(request, child_id="child"):
    output = ExecutorEpisodeOutput(
        nodes=[
            ExecutorNodeCandidate(
                node_id=child_id,
                claim="bounded executor child",
                parent_ids=[request.parent_node_id],
            )
        ]
    )
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role="executor",
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


def relation_result(request, relation_type="independent"):
    observations = []
    for pair in request.relation_payload.candidate_pairs:
        fields = dict(
            candidate_id=pair.candidate_id,
            left_node_id=pair.left.node_id,
            right_node_id=pair.right.node_id,
            relation_type=relation_type,
            confidence=0.9,
            rationale=f"classified as {relation_type}",
            evidence_refs=[],
            materiality_assessment="material" if pair.material_to_synthesis else "non_material",
        )
        if relation_type == "equivalent":
            fields.update(merge_recommended=True)
        elif relation_type == "complementary":
            fields.update(complementarity_summary="the branches work together")
        elif relation_type == "conflict":
            fields.update(
                conflict_summary="the branches conflict",
                disclosure_required=True,
            )
        else:
            fields.update(independence_summary="the branches address separate questions")
        observations.append(RelationObservation(**fields))
    output = RelationEpisodeOutput(observations=observations)
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role="relation",
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


def write_raw_state(run_dir, payload):
    (run_dir / "app_run_state.json").write_text(
        json.dumps(payload, allow_nan=False),
        encoding="utf-8",
    )


def create_duplicate_gate(run_dir, *, final=True):
    create_app_run(
        run_dir,
        run_spec(final=final, iterations=1),
        [
            SearchNode(node_id="a", claim="duplicate"),
            SearchNode(node_id="b", claim=" DUPLICATE "),
        ],
    )
    judge = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(judge))
    outcome = next_app_episode(
        run_dir,
        embedding_provider=HashEmbeddingProvider(dim=8),
    )
    while outcome.request is not None and outcome.request.role == "executor":
        executor = outcome.request
        output = ExecutorEpisodeOutput(nodes=[])
        submit_app_episode_result(
            run_dir,
            EpisodeResult(
                episode_id=executor.episode_id,
                attempt_id=executor.attempt_id,
                run_id=executor.run_id,
                role="executor",
                input_graph_revision=executor.input_graph_revision,
                selected_node_revisions=executor.selected_node_revisions,
                status="completed",
                structured_output=output,
                runtime_diagnostics=RuntimeDiagnostics(
                    adapter_name="codex-app-main-agent",
                    transport_name="current-app-runtime",
                    profile="native-autonomous",
                    usage_source="unavailable",
                ),
                output_hash=compute_output_hash(output, executor.output_schema_version),
                schema_version=executor.output_schema_version,
            ),
        )
        state = app_driver.load_app_run(run_dir)
        if app_driver._select_executor_parent(state) is None:
            break
        outcome = next_app_episode(run_dir)


@pytest.mark.parametrize(
    "update",
    [
        {"score": 0.9},
        {"local_embedding": [0.0] * 8},
        {"density": 1.0},
        {"uncertainty": 0.2},
        {"ucb_score": 5.0},
        {"expansion_budget": 1},
        {"judge_reasoning": "prefilled"},
        {"judge_risks": ["prefilled"]},
        {"judge_uncertainty_evidence": ["prefilled"]},
        {"judge_result_provenance": {"episode_id": "forged"}},
        {"status": "closed"},
        {"node_type": "synthesis"},
    ],
)
def test_create_run_rejects_controller_owned_initial_state(tmp_path, update):
    run_dir = tmp_path / next(iter(update))
    node = SearchNode(node_id="seed", claim="producer node").model_copy(update=update)
    with pytest.raises(ValueError, match="controller-owned"):
        create_app_run(run_dir, run_spec(), [node])
    assert not (run_dir / "app_run_state.json").exists()


def test_clean_create_run_always_grants_judge_first(tmp_path):
    run_dir = tmp_path / "clean"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    outcome = next_app_episode(run_dir)
    assert outcome.request is not None
    assert outcome.request.role == "judge"


@pytest.mark.parametrize("revision_kind", ["graph", "node"])
def test_load_rejects_revision_without_a_committed_transition(tmp_path, revision_kind):
    run_dir = tmp_path / f"forged-{revision_kind}-revision"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    if revision_kind == "graph":
        payload["graph_revision"] = 50
        expected = "graph revision is not backed"
    else:
        payload["node_revisions"]["seed"] = 50
        expected = "node revisions disagree"
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match=expected):
        app_driver.load_app_run(run_dir)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["episodes"][0]["attempts"][0].__setitem__(
            "deadline_at", None
        ),
        lambda payload: payload["episodes"][0]["attempts"][0].__setitem__(
            "deadline_at", "2099-01-01T00:00:00+00:00"
        ),
    ],
)
def test_load_binds_deadline_to_runtime_grant(tmp_path, mutate):
    run_dir = tmp_path / "deadline-binding"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(wall_clock_seconds=1),
    )
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    mutate(payload)
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="deadline disagrees with its runtime grant"):
        app_driver.load_app_run(run_dir)


@pytest.mark.parametrize("granted_at", ["not-a-time", "2026-01-01T00:00:00"])
def test_load_rejects_invalid_or_naive_attempt_timestamp(tmp_path, granted_at):
    run_dir = tmp_path / "invalid-time"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    next_app_episode(run_dir)
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["episodes"][0]["attempts"][0]["granted_at"] = granted_at
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="timestamp"):
        app_driver.load_app_run(run_dir)


def test_submit_uses_one_receipt_timestamp_at_deadline_boundary(tmp_path, monkeypatch):
    run_dir = tmp_path / "deadline-receipt"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    request = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(wall_clock_seconds=1),
    ).request
    attempt = app_driver.load_app_run(run_dir).episodes[0].attempts[0]
    deadline = app_driver._parse_time(attempt.deadline_at)
    before_deadline = deadline - timedelta(microseconds=1)
    after_deadline = deadline + timedelta(microseconds=1)
    ticks = [before_deadline, after_deadline]

    def crossing_clock():
        return ticks.pop(0) if ticks else after_deadline

    monkeypatch.setattr(app_driver, "_now", crossing_clock)
    outcome = submit_app_episode_result(run_dir, judge_result(request))

    assert outcome.commit_outcome.accepted is True
    committed = app_run_status(run_dir).episodes[0].attempts[0]
    assert app_driver._parse_time(committed.submitted_at) == before_deadline


def test_synthesis_intent_waits_for_judge_and_controller_safe_point(tmp_path):
    run_dir = tmp_path / "safe-point"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    request_app_synthesis(
        run_dir,
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="enough coverage after the next safe checkpoint",
        ),
    )
    judge = next_app_episode(run_dir)
    assert judge.request is not None and judge.request.role == "judge"
    assert submit_app_episode_result(run_dir, judge_result(judge.request)).commit_outcome.accepted
    terminal = next_app_episode(
        run_dir,
        embedding_provider=HashEmbeddingProvider(dim=8),
    )
    while terminal.controller_action == "episode_required":
        assert terminal.request is not None
        if terminal.request.role == "executor":
            output = ExecutorEpisodeOutput(nodes=[])
            result = EpisodeResult(
                episode_id=terminal.request.episode_id,
                attempt_id=terminal.request.attempt_id,
                run_id=terminal.request.run_id,
                role="executor",
                input_graph_revision=terminal.request.input_graph_revision,
                selected_node_revisions=terminal.request.selected_node_revisions,
                status="completed",
                structured_output=output,
                runtime_diagnostics=RuntimeDiagnostics(
                    adapter_name="codex-app-main-agent",
                    transport_name="current-app-runtime",
                    profile="native-autonomous",
                    usage_source="unavailable",
                ),
                output_hash=compute_output_hash(
                    output,
                    terminal.request.output_schema_version,
                ),
                schema_version=terminal.request.output_schema_version,
            )
        elif terminal.request.role == "judge":
            result = judge_result(terminal.request)
        else:
            result = relation_result(terminal.request)
        assert submit_app_episode_result(run_dir, result).commit_outcome.accepted
        terminal = next_app_episode(run_dir)
    assert terminal.controller_action == "ready_for_synthesis"
    state = app_run_status(run_dir)
    assert state.controller_iteration == 1
    assert state.provisional_synthesis_selection.selected_node_ids == ["seed"]


def test_targeted_synthesis_rejects_unknown_node(tmp_path):
    run_dir = tmp_path / "unknown-target"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    with pytest.raises(ValueError, match="unknown node IDs"):
        request_app_synthesis(
            run_dir,
            SynthesisControlRequest(
                action="force_synthesis_after_current_task",
                requested_by="main_agent",
                reason="typo",
                scope="node_ids",
                node_ids=["ghost"],
            ),
        )


def test_run_complete_is_sticky_preserves_reason_and_emits_once(tmp_path):
    run_dir = tmp_path / "terminal"
    create_app_run(
        run_dir,
        run_spec(final=False, iterations=1),
        [SearchNode(node_id="seed", claim="trusted checkpoint")],
    )
    judge = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(judge))
    executor = next_app_episode(
        run_dir,
        embedding_provider=HashEmbeddingProvider(dim=8),
    ).request
    submit_app_episode_result(run_dir, executor_result(executor, "terminal-child"))
    child_judge = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(child_judge))
    first = next_app_episode(run_dir)
    second = next_app_episode(run_dir)
    assert first.controller_action == second.controller_action == "run_complete"
    assert first.reason == second.reason == "maximum controller iterations reached"
    state = app_run_status(run_dir)
    assert state.terminal_record is not None
    assert state.terminal_record.reason == first.reason
    with pytest.raises(ValueError, match="terminal action"):
        request_app_synthesis(
            run_dir,
            SynthesisControlRequest(
                action="force_synthesis_after_current_task",
                requested_by="main_agent",
                reason="must not reopen",
            ),
        )
    events = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    assert [event["event_type"] for event in events].count("run_completed") == 1


def test_retry_cannot_reopen_terminal_state(tmp_path):
    run_dir = tmp_path / "terminal-retry"
    create_app_run(
        run_dir,
        run_spec(iterations=1),
        [SearchNode(node_id="seed", claim="clean")],
    )
    judge = next_app_episode(run_dir).request
    fail_app_episode(run_dir, judge.episode_id, judge.attempt_id, "failed before terminal")
    retried_judge = retry_app_episode(run_dir, judge.episode_id).request
    submit_app_episode_result(run_dir, judge_result(retried_judge))
    executor = next_app_episode(
        run_dir,
        embedding_provider=HashEmbeddingProvider(dim=8),
    ).request
    submit_app_episode_result(run_dir, executor_result(executor, "terminal-retry-child"))
    child_judge = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(child_judge))
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"
    before = (run_dir / "app_run_state.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="terminal action"):
        retry_app_episode(run_dir, judge.episode_id)
    assert (run_dir / "app_run_state.json").read_text(encoding="utf-8") == before


def test_relation_retry_exhaustion_releases_blocker_for_new_logical_episode(tmp_path):
    run_dir = tmp_path / "relation-recovery"
    create_duplicate_gate(run_dir)

    first = next_app_episode(run_dir, runtime_limits=RuntimeLimits(max_retries=1)).request
    assert first.role == "relation"
    fail_app_episode(run_dir, first.episode_id, first.attempt_id, "first failure")
    second = retry_app_episode(run_dir, first.episode_id).request
    fail_app_episode(run_dir, second.episode_id, second.attempt_id, "second failure")
    exhausted = retry_app_episode(run_dir, first.episode_id)
    assert exhausted.controller_action == "continue_controller"
    assert "retry limit exhausted" in exhausted.reason
    assert all(candidate.status == "pending" for candidate in app_run_status(run_dir).relation_candidates)

    request_app_synthesis(
        run_dir,
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="explicitly re-enter the readiness gate",
        ),
    )
    recovered = next_app_episode(run_dir)
    assert recovered.request is not None and recovered.request.role == "relation"
    assert recovered.request.episode_id != first.episode_id


def test_synthesis_request_rejects_outstanding_rejected_relation_without_mutation(tmp_path):
    run_dir = tmp_path / "rejected-relation-synthesis"
    create_duplicate_gate(run_dir, final=False)
    request = next_app_episode(run_dir).request
    valid = relation_result(request)
    forged_observation = valid.structured_output.observations[0].model_copy(
        update={"candidate_id": "not-the-granted-candidate"}
    )
    forged_output = RelationEpisodeOutput(observations=[forged_observation])
    forged = valid.model_copy(
        update={
            "structured_output": forged_output,
            "output_hash": compute_output_hash(
                forged_output,
                request.output_schema_version,
            ),
        }
    )
    rejected = submit_app_episode_result(run_dir, forged)
    assert rejected.commit_outcome.accepted is False
    before = (run_dir / "app_run_state.json").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="unresolved Relation attempt must be retried"):
        request_app_synthesis(
            run_dir,
            SynthesisControlRequest(
                action="force_synthesis_after_current_task",
                requested_by="main_agent",
                reason="must not strand the rejected Relation grant",
            ),
        )

    assert (run_dir / "app_run_state.json").read_text(encoding="utf-8") == before
    assert retry_app_episode(run_dir, request.episode_id).request is not None


@pytest.mark.parametrize(
    ("final", "natural_action", "forged_action"),
    [
        (True, "ready_for_synthesis", "run_complete"),
        (False, "run_complete", "ready_for_synthesis"),
    ],
)
def test_load_rejects_terminal_action_flip_against_policy(
    tmp_path,
    final,
    natural_action,
    forged_action,
):
    run_dir = tmp_path / natural_action
    create_duplicate_gate(run_dir, final=final)
    relation = next_app_episode(run_dir).request
    assert submit_app_episode_result(run_dir, relation_result(relation)).commit_outcome.accepted
    assert next_app_episode(run_dir).controller_action == natural_action
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["controller_action"] = forged_action
    payload["terminal_record"]["action"] = forged_action
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="synthesis policy"):
        app_driver.load_app_run(run_dir)


def test_load_replays_equivalent_merge_semantics_for_canonical_node(tmp_path):
    run_dir = tmp_path / "merge-semantic-replay"
    create_duplicate_gate(run_dir)
    relation = next_app_episode(run_dir).request
    assert submit_app_episode_result(
        run_dir,
        relation_result(relation, "equivalent"),
    ).commit_outcome.accepted
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    canonical_id = payload["merge_applications"][0]["canonical_node_id"]
    canonical = next(node for node in payload["nodes"] if node["node_id"] == canonical_id)
    canonical["risks"] = ["forged synthesis risk"]
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="authoritative producer output"):
        app_driver.load_app_run(run_dir)


def test_state_write_failure_cannot_publish_false_commit(tmp_path, monkeypatch):
    run_dir = tmp_path / "state-failure"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    request = next_app_episode(run_dir).request
    status_path = run_dir / "episodes" / request.episode_id / request.attempt_id / "status.json"

    def fail_save(*_args, **_kwargs):
        raise OSError("injected state failure")

    monkeypatch.setattr(app_driver, "_save_state", fail_save)
    with pytest.raises(OSError, match="injected state failure"):
        submit_app_episode_result(run_dir, judge_result(request))

    persisted = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    assert persisted["graph_revision"] == 0
    assert persisted["episodes"][0]["attempts"][0]["status"] == "in_progress"
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "in_progress"
    event_types = {
        event["event_type"]
        for event in EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    }
    assert "judge_observations_committed" not in event_types
    assert "episode_completed" not in event_types


def test_commit_event_outbox_recovers_after_telemetry_failure(tmp_path, monkeypatch):
    run_dir = tmp_path / "outbox"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    request = next_app_episode(run_dir).request
    original_emit = EpisodeEventLog.emit
    failed = {"done": False}

    def flaky_emit(self, event_type, **fields):
        if event_type == "judge_observations_committed" and not failed["done"]:
            failed["done"] = True
            raise OSError("injected telemetry failure")
        return original_emit(self, event_type, **fields)

    monkeypatch.setattr(EpisodeEventLog, "emit", flaky_emit)
    outcome = submit_app_episode_result(run_dir, judge_result(request))
    assert outcome.commit_outcome.accepted
    persisted = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    assert persisted["graph_revision"] == 1
    assert persisted["pending_telemetry_events"]

    monkeypatch.setattr(EpisodeEventLog, "emit", original_emit)
    recovered = app_driver.load_app_run(run_dir)
    assert recovered.pending_telemetry_events == []
    events = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    assert [event["event_type"] for event in events].count("judge_observations_committed") == 1
    assert [event["event_type"] for event in events].count("episode_completed") == 1


def test_restart_repairs_stale_attempt_status_mirror_after_commit(tmp_path, monkeypatch):
    run_dir = tmp_path / "attempt-mirror-repair"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    request = next_app_episode(run_dir).request
    status_path = run_dir / "episodes" / request.episode_id / request.attempt_id / "status.json"
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    original_write = app_driver._write_attempt_artifacts

    def fail_mirror(*_args, **_kwargs):
        raise OSError("injected derived artifact failure")

    monkeypatch.setattr(app_driver, "_write_attempt_artifacts", fail_mirror)
    assert submit_app_episode_result(run_dir, judge_result(request)).commit_outcome.accepted
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    monkeypatch.setattr(app_driver, "_write_attempt_artifacts", original_write)
    recovered = app_driver.load_app_run(run_dir)
    assert recovered.episodes[0].attempts[0].status == "committed"
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "committed"


def test_load_rejects_persisted_self_ancestry(tmp_path):
    run_dir = tmp_path / "invalid-persisted"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="a", claim="clean")])
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["nodes"][0]["parent_ids"] = ["a"]
    write_raw_state(run_dir, payload)
    with pytest.raises(ValueError, match="authoritative producer output"):
        app_driver.load_app_run(run_dir)


def test_load_rejects_partial_active_attempt_identity(tmp_path):
    run_dir = tmp_path / "partial-active"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="a", claim="clean")])
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["active_episode_id"] = "orphaned-episode"
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="partial active-attempt identity"):
        app_driver.load_app_run(run_dir)


@pytest.mark.parametrize("identity_field", ["episode_id", "attempt_id"])
def test_persisted_attempt_identity_cannot_escape_artifact_directory(tmp_path, identity_field):
    run_dir = tmp_path / f"path-{identity_field}"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="a", claim="clean")])
    request = next_app_episode(run_dir).request
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    forged = f"../../../escaped-{identity_field}"
    lifecycle = payload["episodes"][0]
    attempt = lifecycle["attempts"][0]
    result_payload = judge_result(request).model_dump(mode="json")
    if identity_field == "episode_id":
        lifecycle["episode_id"] = forged
        attempt["request"]["episode_id"] = forged
        payload["active_episode_id"] = forged
        result_payload["episode_id"] = forged
    else:
        attempt["attempt_id"] = forged
        attempt["request"]["attempt_id"] = forged
        payload["active_attempt_id"] = forged
        result_payload["attempt_id"] = forged
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match=f"unsafe {identity_field}"):
        submit_app_episode_result(run_dir, result_payload)
    assert not (tmp_path.parent / f"escaped-{identity_field}").exists()


def test_load_rejects_reopened_terminal_record(tmp_path):
    run_dir = tmp_path / "reopened-terminal"
    create_duplicate_gate(run_dir, final=False)
    relation = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, relation_result(relation))
    assert next_app_episode(run_dir).controller_action == "run_complete"
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["controller_action"] = "continue_controller"
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="active gate or terminal|terminal record disagrees"):
        app_driver.load_app_run(run_dir)


def test_load_recomputes_terminal_relation_inventory_instead_of_trusting_ready_flag(tmp_path):
    run_dir = tmp_path / "forged-ready"
    create_duplicate_gate(run_dir)
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["controller_action"] = "ready_for_synthesis"
    payload["relation_readiness_status"] = "evaluated"
    payload["provisional_synthesis_selection"] = {
        "selected_node_ids": ["a", "b"],
        "selection_reason": "deterministic report ranking over committed non-merged branches",
        "selection_revision": payload["graph_revision"],
    }
    payload["synthesis_readiness"] = {
        "schema_version": "synthesis-readiness.v2",
        "graph_revision": payload["graph_revision"],
        "provisional_selected_node_ids": ["a", "b"],
        "blocking_inventory_complete": True,
        "blocking_pair_count": 0,
        "resolved_blocking_pair_count": 0,
        "unresolved_blocking_pair_count": 0,
        "blocking_candidate_ids": [],
        "unresolved_material_conflicts": [],
        "disclosure_required_conflicts": [],
        "unresolved_nonblocking_candidates": [],
        "duplicate_groups": [],
        "enrichment_budget_limit": 0,
        "enrichment_pairs_committed": 0,
        "enrichment_pairs_remaining": 0,
        "eligible_enrichment_candidate_ids": [],
        "enrichment_pending": False,
        "ready": True,
        "reason": "all required blocking Relation obligations are resolved",
        "evaluated_at": payload["updated_at"],
    }
    payload["terminal_record"] = {
        "action": "ready_for_synthesis",
        "source": "max_iterations",
        "reason": "maximum controller iterations reached",
        "graph_revision": payload["graph_revision"],
        "controller_iteration": payload["controller_iteration"],
        "committed_at": payload["updated_at"],
    }
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="cannot be reproduced"):
        app_driver.load_app_run(run_dir)


def test_load_binds_relation_ledger_to_committed_attempt(tmp_path):
    run_dir = tmp_path / "ledger-attempt-binding"
    create_duplicate_gate(run_dir)
    request = next_app_episode(run_dir).request
    assert submit_app_episode_result(run_dir, relation_result(request)).commit_outcome.accepted
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["episodes"] = [
        episode for episode in payload["episodes"] if episode["role"] != "relation"
    ]
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="missing attempt"):
        app_driver.load_app_run(run_dir)


def test_load_recomputes_relation_output_hash_from_ledger_observations(tmp_path):
    run_dir = tmp_path / "ledger-output-hash"
    create_duplicate_gate(run_dir)
    request = next_app_episode(run_dir).request
    assert submit_app_episode_result(run_dir, relation_result(request)).commit_outcome.accepted
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    record = payload["relation_ledger"][0]
    record["relation_type"] = "conflict"
    record["disclosure_required"] = True
    record["observation"]["relation_type"] = "conflict"
    record["observation"]["conflict_summary"] = "forged after commit"
    record["observation"]["disclosure_required"] = True
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="output hash"):
        app_driver.load_app_run(run_dir)


def test_load_rejects_forged_merge_application_revision(tmp_path):
    run_dir = tmp_path / "merge-provenance"
    create_duplicate_gate(run_dir)
    request = next_app_episode(run_dir).request
    assert submit_app_episode_result(
        run_dir,
        relation_result(request, relation_type="equivalent"),
    ).commit_outcome.accepted
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["merge_applications"][0]["applied_graph_revision"] += 50
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="stable commit provenance"):
        app_driver.load_app_run(run_dir)


def test_create_run_rejects_empty_initial_frontier_without_writing_state(tmp_path):
    run_dir = tmp_path / "empty-ingress"
    with pytest.raises(ValueError, match="at least one initial node"):
        create_app_run(run_dir, run_spec(), [])
    assert not (run_dir / "app_run_state.json").exists()


def test_submit_uses_one_detached_payload_for_commit_artifact_and_hash(tmp_path):
    run_dir = tmp_path / "single-result-snapshot"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    request = next_app_episode(run_dir).request
    first = judge_result(request, score=0.81)
    second = judge_result(request, score=0.12)

    class StatefulResult:
        def __init__(self):
            self.calls = 0

        def model_dump(self, mode="json"):
            self.calls += 1
            selected = first if self.calls == 1 else second
            return selected.model_dump(mode=mode)

    raw = StatefulResult()
    outcome = submit_app_episode_result(run_dir, raw)
    assert outcome.commit_outcome.accepted
    assert raw.calls == 1
    state = app_run_status(run_dir)
    artifact = json.loads(
        (
            run_dir
            / "episodes"
            / request.episode_id
            / request.attempt_id
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    attempt = state.episodes[0].attempts[0]
    assert state.nodes[0].score == artifact["structured_output"]["observations"][0]["score"] == 0.81
    assert attempt.result_hash == artifact["output_hash"] == first.output_hash


def test_result_json_detachment_failure_is_audited_without_consuming_attempt(tmp_path):
    run_dir = tmp_path / "non-finite-result"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    request = next_app_episode(run_dir).request
    payload = judge_result(request).model_dump(mode="json")
    payload["structured_output"]["observations"][0]["score"] = float("nan")
    before = (run_dir / "app_run_state.json").read_text(encoding="utf-8")

    outcome = submit_app_episode_result(run_dir, payload)

    assert not outcome.commit_outcome.accepted
    assert "JSON detachment failed" in outcome.commit_outcome.rejection_reason
    assert (run_dir / "app_run_state.json").read_text(encoding="utf-8") == before
    state = app_run_status(run_dir)
    assert state.active_attempt_id == request.attempt_id
    assert state.episodes[0].attempts[0].status == "in_progress"
    events = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    assert events[-1]["event_type"] == "output_rejected"
    assert events[-1]["episode_id"] == request.episode_id


def test_active_relation_retry_exhaustion_does_not_release_live_grant(tmp_path):
    run_dir = tmp_path / "active-relation-retry-limit"
    create_duplicate_gate(run_dir)
    request = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(max_retries=0),
    ).request
    before = (run_dir / "app_run_state.json").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="retry limit exhausted"):
        retry_app_episode(run_dir, request.episode_id)

    assert (run_dir / "app_run_state.json").read_text(encoding="utf-8") == before
    state = app_run_status(run_dir)
    assert state.active_attempt_id == request.attempt_id
    assert all(
        candidate.status == "granted"
        for candidate in state.relation_candidates
        if candidate.candidate_id
        in {pair.candidate_id for pair in request.relation_payload.candidate_pairs}
    )
    assert submit_app_episode_result(run_dir, relation_result(request)).commit_outcome.accepted


def test_rejected_relation_retry_exhaustion_releases_uncommittable_grant(tmp_path):
    run_dir = tmp_path / "rejected-relation-retry-limit"
    create_duplicate_gate(run_dir)
    request = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(max_retries=0),
    ).request
    invalid = relation_result(request).model_dump(mode="json")
    invalid["output_hash"] = "0" * 64
    assert not submit_app_episode_result(run_dir, invalid).commit_outcome.accepted
    assert any(
        candidate.status == "granted"
        for candidate in app_run_status(run_dir).relation_candidates
    )

    exhausted = retry_app_episode(run_dir, request.episode_id)
    assert exhausted.controller_action == "continue_controller"
    assert "retry limit exhausted" in exhausted.reason

    state = app_run_status(run_dir)
    assert all(candidate.status == "pending" for candidate in state.relation_candidates)
    assert state.controller_action == "continue_controller"
    recovered = next_app_episode(run_dir)
    assert recovered.request is not None and recovered.request.role == "relation"
    assert recovered.request.episode_id != request.episode_id


def test_failed_relation_retry_exhaustion_reenters_controller_after_restart(tmp_path):
    run_dir = tmp_path / "failed-relation-retry-limit"
    create_duplicate_gate(run_dir)
    request = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(max_retries=0),
    ).request
    fail_app_episode(run_dir, request.episode_id, request.attempt_id, "failed")

    exhausted = retry_app_episode(run_dir, request.episode_id)
    assert exhausted.controller_action == "continue_controller"
    assert "retry limit exhausted" in exhausted.reason

    assert app_driver.load_app_run(run_dir).controller_action == "continue_controller"
    recovered = next_app_episode(run_dir)
    assert recovered.request is not None and recovered.request.role == "relation"
    assert recovered.request.episode_id != request.episode_id


def test_authorized_synthesis_replaces_pending_run_complete_intent(tmp_path):
    run_dir = tmp_path / "override-run-complete"
    create_duplicate_gate(run_dir, final=False)
    first = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(max_retries=0),
    ).request
    fail_app_episode(run_dir, first.episode_id, first.attempt_id, "operator decision")
    exhausted = retry_app_episode(run_dir, first.episode_id)
    assert exhausted.controller_action == "continue_controller"

    state = request_app_synthesis(
        run_dir,
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="user",
            reason="synthesize after Relation recovery",
        ),
    )
    assert state.pending_terminal_action == "ready_for_synthesis"
    assert state.pending_terminal_reason == "authorized synthesis request is pending"

    recovered = next_app_episode(run_dir)
    assert recovered.request is not None and recovered.request.role == "relation"
    assert submit_app_episode_result(
        run_dir,
        relation_result(recovered.request),
    ).commit_outcome.accepted
    terminal = next_app_episode(run_dir)
    assert terminal.controller_action == "ready_for_synthesis"
    assert terminal.reason == "authorized synthesis request is pending"
    events = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    assert not any(event["event_type"] == "run_completed" for event in events)


def test_mutated_runtime_limits_are_rejected_before_grant_persistence(tmp_path):
    run_dir = tmp_path / "mutated-limits"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    limits = RuntimeLimits(max_retries=1)
    limits.max_retries = -7
    limits.wall_clock_seconds = 0
    before = (run_dir / "app_run_state.json").read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        next_app_episode(run_dir, runtime_limits=limits)

    assert (run_dir / "app_run_state.json").read_text(encoding="utf-8") == before
    assert app_run_status(run_dir).episodes == []
    assert next_app_episode(run_dir).request.role == "judge"


def test_retry_runtime_override_is_validated_without_superseding_attempt(tmp_path):
    run_dir = tmp_path / "invalid-retry-limits"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    first = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(max_retries=1),
    ).request
    fail_app_episode(run_dir, first.episode_id, first.attempt_id, "retryable")
    before = (run_dir / "app_run_state.json").read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        retry_app_episode(run_dir, first.episode_id, wall_clock_seconds=0)

    assert (run_dir / "app_run_state.json").read_text(encoding="utf-8") == before
    state = app_run_status(run_dir)
    assert state.episodes[0].attempts[0].status == "failed"
    assert retry_app_episode(run_dir, first.episode_id).request is not None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["episodes"][0]["attempts"][0]["request"].__setitem__(
                "run_id", "foreign-run"
            ),
            "request run_id disagrees",
        ),
        (
            lambda payload: payload["episodes"][0].__setitem__("run_id", "foreign-run"),
            "episode run_id disagrees",
        ),
        (
            lambda payload: payload["episodes"][0]["attempts"][0]["request"].__setitem__(
                "attempt_id", "foreign-attempt"
            ),
            "request attempt_id disagrees",
        ),
    ],
)
def test_load_rejects_cross_envelope_lifecycle_identity(tmp_path, mutate, message):
    run_dir = tmp_path / message.replace(" ", "-")
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    next_app_episode(run_dir)
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    mutate(payload)
    attempt = payload["episodes"][0]["attempts"][0]
    request = app_driver.EpisodeRequest.model_validate(attempt["request"])
    attempt["request_hash"] = app_driver._episode_request_hash(request)
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match=message):
        app_driver.load_app_run(run_dir)


def test_load_rejects_relation_grant_not_owned_by_active_request(tmp_path):
    run_dir = tmp_path / "orphaned-relation-grant"
    create_duplicate_gate(run_dir)
    next_app_episode(run_dir)
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    granted = next(candidate for candidate in payload["relation_candidates"] if candidate["status"] == "granted")
    granted["granted_attempt_id"] = "foreign-attempt"
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="missing attempt"):
        app_driver.load_app_run(run_dir)


def test_save_state_revalidates_assignment_mutations_before_install(tmp_path):
    run_dir = tmp_path / "save-boundary"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    state = app_driver.load_app_run(run_dir)
    state.pending_terminal_reason = "orphaned reason"
    before = (run_dir / "app_run_state.json").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="present together"):
        app_driver._save_state(run_dir, state)

    assert (run_dir / "app_run_state.json").read_text(encoding="utf-8") == before
    assert app_run_status(run_dir).pending_terminal_reason is None


def test_load_rejects_await_without_a_durable_blocking_fact(tmp_path):
    run_dir = tmp_path / "forged-await"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["controller_action"] = "await_operator_decision"
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="lacks a durable blocking fact"):
        app_driver.load_app_run(run_dir)


def test_load_rejects_continue_that_bypasses_failed_attempt(tmp_path):
    run_dir = tmp_path / "bypass-failed"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    request = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(max_retries=0),
    ).request
    fail_app_episode(run_dir, request.episode_id, request.attempt_id, "failed")
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["controller_action"] = "continue_controller"
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="bypasses an unresolved operator decision"):
        app_driver.load_app_run(run_dir)


def test_load_rejects_hand_appended_attempt_beyond_retry_grant(tmp_path):
    run_dir = tmp_path / "forged-retry"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    request = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(max_retries=0),
    ).request
    fail_app_episode(run_dir, request.episode_id, request.attempt_id, "failed")
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    lifecycle = payload["episodes"][0]
    first = lifecycle["attempts"][0]
    first["status"] = "superseded"
    first["superseded_from_status"] = "failed"
    second = copy.deepcopy(first)
    second["attempt_id"] = "forged-second-attempt"
    second["attempt_number"] = 2
    second["status"] = "in_progress"
    second["superseded_from_status"] = None
    second["request"]["attempt_id"] = second["attempt_id"]
    second_request = app_driver.EpisodeRequest.model_validate(second["request"])
    second["request_hash"] = app_driver._episode_request_hash(second_request)
    lifecycle["attempts"].append(second)
    payload["active_episode_id"] = lifecycle["episode_id"]
    payload["active_attempt_id"] = second["attempt_id"]
    payload["controller_action"] = "episode_required"
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="exceed or rewrite the retry grant"):
        app_driver.load_app_run(run_dir)


def test_load_rejects_accepted_outcome_on_noncommitted_attempt(tmp_path):
    run_dir = tmp_path / "forged-outcome"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    next_app_episode(run_dir)
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    attempt = payload["episodes"][0]["attempts"][0]
    attempt["commit_outcome"] = {
        "accepted": True,
        "episode_id": payload["episodes"][0]["episode_id"],
        "accepted_node_ids": ["seed"],
        "accepted_node_count": 1,
        "graph_revision_before": 0,
        "graph_revision_after": 1,
        "rejection_reason": None,
    }
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="non-committed attempt claims"):
        app_driver.load_app_run(run_dir)


def test_load_rejects_forged_nonterminal_synthesis_readiness(tmp_path):
    run_dir = tmp_path / "forged-readiness"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["relation_readiness_status"] = "evaluated"
    payload["provisional_synthesis_selection"] = {
        "selected_node_ids": ["seed"],
        "selection_reason": "forged",
        "selection_revision": 0,
    }
    payload["synthesis_readiness"] = {
        "schema_version": "synthesis-readiness.v2",
        "graph_revision": 0,
        "provisional_selected_node_ids": ["seed"],
        "blocking_inventory_complete": True,
        "blocking_pair_count": 0,
        "resolved_blocking_pair_count": 0,
        "unresolved_blocking_pair_count": 0,
        "blocking_candidate_ids": [],
        "unresolved_material_conflicts": [],
        "disclosure_required_conflicts": [],
        "unresolved_nonblocking_candidates": [],
        "duplicate_groups": [],
        "enrichment_budget_limit": 0,
        "enrichment_pairs_committed": 0,
        "enrichment_pairs_remaining": 0,
        "eligible_enrichment_candidate_ids": [],
        "enrichment_pending": False,
        "ready": True,
        "reason": "forged ready",
        "evaluated_at": payload["created_at"],
    }
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="lacks an active gate or terminal"):
        app_driver.load_app_run(run_dir)


def test_load_rejects_commit_telemetry_without_commit_fact(tmp_path):
    run_dir = tmp_path / "forged-outbox"
    create_app_run(run_dir, run_spec(), [SearchNode(node_id="seed", claim="clean")])
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["pending_telemetry_events"] = [
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "episode_completed",
            "fields": {
                "run_id": payload["run_id"],
                "episode_id": "ghost",
                "attempt_id": "ghost-attempt",
                "role": "executor",
                "status": "committed",
                "input_graph_revision": 0,
                "graph_revision": 1,
                "accepted_node_count": 99,
                "usage_source": "unavailable",
            },
        }
    ]
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="references a missing attempt"):
        app_driver.load_app_run(run_dir)
    assert not any(
        event["event_type"] == "episode_completed"
        for event in EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    )
