import json
from datetime import timedelta

import pytest
from pydantic import ValidationError

import dte_backend.app_driver as app_driver
from dte_backend.app_driver import (
    app_run_status,
    cancel_app_episode,
    create_app_run,
    fail_app_episode,
    next_app_episode,
    retry_app_episode,
    submit_app_episode_result,
)
from dte_backend.cache import EmbeddingCacheNamespace
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.episode_adapter import build_judge_episode_request
from dte_backend.episode_commit import EpisodeGraph, commit_episode_result
from dte_backend.episode_models import (
    EpisodeRequest,
    EpisodeResult,
    ExecutorEpisodeOutput,
    ExecutorNodeCandidate,
    JudgeEpisodeOutput,
    JudgeObservation,
    RuntimeDiagnostics,
    RuntimeLimits,
    compute_output_hash,
)
from dte_backend.file_cache import FileDTECache
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.telemetry import EpisodeEventLog


def run_spec(*, cap=2, max_iterations=2, require_final_synthesis=True):
    return DTERunSpec(
        problem="ordinary unscored frontier",
        goal="advance through Judge and controller to Executor",
        constraints=["preserve controller ownership"],
        budget=BudgetSpec(
            max_iterations=max_iterations,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=cap,
        ),
        embedding_provider="hash",
        embedding_dimension=8,
        require_final_synthesis=require_final_synthesis,
    )


def make_run(
    tmp_path,
    *,
    node_count=1,
    cap=2,
    max_iterations=2,
    require_final_synthesis=True,
):
    run_dir = tmp_path / "run"
    nodes = [SearchNode(node_id=f"n{index}", claim=f"claim {index}") for index in range(node_count)]
    create_app_run(
        run_dir,
        run_spec(
            cap=cap,
            max_iterations=max_iterations,
            require_final_synthesis=require_final_synthesis,
        ),
        nodes,
        run_id="judge-run",
    )
    return run_dir


def diagnostics():
    return RuntimeDiagnostics(
        adapter_name="codex-app-main-agent",
        transport_name="current-app-runtime",
        profile="native-autonomous",
        usage_source="unavailable",
    )


def judge_result(request, *, observations=None, status="completed"):
    output = None
    if status == "completed":
        output = JudgeEpisodeOutput(
            observations=observations
            or [
                JudgeObservation(
                    node_id=node_id,
                    score=0.75,
                    reasoning=f"observable judgment for {node_id}",
                    risks=["material risk"],
                    uncertainty_evidence=["evidence remains incomplete"],
                )
                for node_id in request.selected_node_revisions
            ]
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
        runtime_diagnostics=diagnostics(),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def executor_result(request, *, child_id="child", child_claim="bounded child"):
    output = ExecutorEpisodeOutput(
        nodes=[
            ExecutorNodeCandidate(
                node_id=child_id,
                claim=child_claim,
                parent_ids=[request.parent_node_id],
            )
        ]
    )
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role=request.role,
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status="completed",
        structured_output=output,
        runtime_diagnostics=diagnostics(),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


class TrackingEmbeddingProvider:
    def __init__(self, *, name="tracking", model="tracking-v1", dim=8):
        self.name = name
        self.model = model
        self.dim = dim
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += len(texts)
        return [[float(index + 1) for index in range(self.dim)] for _ in texts]


def graph_snapshot(run_dir):
    state = app_run_status(run_dir)
    return {
        "graph_revision": state.graph_revision,
        "node_revisions": dict(state.node_revisions),
        "nodes": [node.model_dump(mode="json") for node in state.nodes],
    }


def test_unscored_frontier_grants_one_bounded_strict_judge_episode(monkeypatch, tmp_path):
    run_dir = make_run(tmp_path, node_count=3, cap=2)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("App path launched subprocess")),
    )
    first = next_app_episode(run_dir)
    second = next_app_episode(run_dir)
    assert first.controller_action == "episode_required"
    assert first.request.role == "judge"
    assert len(first.request.selected_node_revisions) == 2
    assert first.request.allowed_output_types == []
    assert first.request.judge_payload.rubric_version == "research-potential.v1"
    assert second.resumed_existing_attempt is True
    assert second.request.attempt_id == first.request.attempt_id


def test_judge_request_and_result_schemas_are_strict(tmp_path):
    request = next_app_episode(make_run(tmp_path)).request
    request_data = request.model_dump(mode="json")
    request_data["judge_payload"]["allocation"] = 9
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EpisodeRequest.model_validate(request_data)

    result_data = judge_result(request).model_dump(mode="json")
    result_data["structured_output"]["observations"][0]["ucb_score"] = 9
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EpisodeResult.model_validate(result_data)

    result_data = judge_result(request).model_dump(mode="json")
    del result_data["structured_output"]["observations"][0]["risks"]
    with pytest.raises(ValidationError, match="Field required"):
        EpisodeResult.model_validate(result_data)


def test_valid_judge_commit_records_observation_provenance_and_revisions(tmp_path):
    run_dir = make_run(tmp_path)
    request = next_app_episode(run_dir).request
    outcome = submit_app_episode_result(run_dir, judge_result(request))
    state = app_run_status(run_dir)
    node = state.nodes[0]
    assert outcome.commit_outcome.accepted is True
    assert state.graph_revision == 1
    assert state.node_revisions == {"n0": 1}
    assert node.score == 0.75
    assert node.judge_reasoning.startswith("observable judgment")
    assert node.judge_risks == ["material risk"]
    assert node.judge_result_provenance["attempt_id"] == request.attempt_id


def test_judge_score_range_is_rejected_without_mutation(tmp_path):
    run_dir = make_run(tmp_path)
    request = next_app_episode(run_dir).request
    before = graph_snapshot(run_dir)
    raw = judge_result(request).model_dump(mode="json")
    raw["structured_output"]["observations"][0]["score"] = 1.1
    outcome = submit_app_episode_result(run_dir, raw)
    assert outcome.commit_outcome.accepted is False
    assert "schema validation failed" in outcome.commit_outcome.rejection_reason
    assert graph_snapshot(run_dir) == before


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate"])
def test_judge_grant_membership_is_exact_and_atomic(tmp_path, case):
    run_dir = make_run(tmp_path, node_count=2)
    request = next_app_episode(run_dir).request
    before = graph_snapshot(run_dir)
    observations = [
        JudgeObservation(node_id=node_id, score=0.6, reasoning="reason", risks=[])
        for node_id in request.selected_node_revisions
    ]
    if case == "missing":
        observations.pop()
    elif case == "extra":
        observations.append(JudgeObservation(node_id="outside", score=0.6, reasoning="reason", risks=[]))
    else:
        observations.append(observations[0].model_copy(deep=True))
    outcome = submit_app_episode_result(run_dir, judge_result(request, observations=observations))
    assert outcome.commit_outcome.accepted is False
    expected = {"missing": "omitted", "extra": "ungranted", "duplicate": "duplicate"}[case]
    assert expected in outcome.commit_outcome.rejection_reason.lower()
    assert graph_snapshot(run_dir) == before


def test_judge_stale_graph_and_selected_node_revisions_are_atomic():
    graph = EpisodeGraph(nodes=[SearchNode(node_id="n", claim="claim")])
    request = build_judge_episode_request(
        graph,
        graph.nodes,
        run_id="run",
        problem="p",
        goal="g",
    )
    result = judge_result(request)
    before = graph.snapshot()
    graph.revision += 1
    stale_graph_before = graph.snapshot()
    outcome = commit_episode_result(graph, request, result)
    assert outcome.accepted is False
    assert "stale graph revision" in outcome.rejection_reason
    assert graph.snapshot() == stale_graph_before

    graph.revision = before["revision"]
    graph.node_revisions["n"] += 1
    stale_node_before = graph.snapshot()
    outcome = commit_episode_result(graph, request, result)
    assert outcome.accepted is False
    assert "stale selected-node revision" in outcome.rejection_reason
    assert graph.snapshot() == stale_node_before


def test_judge_controller_owned_pollution_and_hidden_metadata_do_not_become_graph_facts(tmp_path):
    run_dir = make_run(tmp_path)
    request = next_app_episode(run_dir).request
    before = graph_snapshot(run_dir)
    polluted = judge_result(request).model_dump(mode="json")
    polluted["structured_output"]["observations"][0]["expansion_budget"] = 50
    outcome = submit_app_episode_result(run_dir, polluted)
    assert outcome.commit_outcome.accepted is False
    assert graph_snapshot(run_dir) == before

    retry = retry_app_episode(run_dir, request.episode_id).request
    clean = judge_result(retry)
    clean.runtime_diagnostics.internal_subagent_metadata = {
        "agents": [{"name": "hidden", "claim": "must not enter graph"}]
    }
    assert submit_app_episode_result(run_dir, clean).commit_outcome.accepted is True
    serialized = json.dumps(graph_snapshot(run_dir), sort_keys=True)
    assert "hidden" not in serialized
    assert "must not enter graph" not in serialized


@pytest.mark.parametrize("transition", ["failed", "cancelled"])
def test_failed_or_cancelled_judge_attempt_cannot_commit(tmp_path, transition):
    run_dir = make_run(tmp_path)
    request = next_app_episode(run_dir).request
    before = graph_snapshot(run_dir)
    if transition == "failed":
        fail_app_episode(run_dir, request.episode_id, request.attempt_id, "runtime failed")
    else:
        cancel_app_episode(run_dir, request.episode_id, request.attempt_id, "cancelled")
    outcome = submit_app_episode_result(run_dir, judge_result(request))
    assert outcome.commit_outcome.accepted is False
    assert f"status={transition}" in outcome.commit_outcome.rejection_reason
    assert graph_snapshot(run_dir) == before
    blocked = next_app_episode(run_dir)
    assert blocked.controller_action == "await_operator_decision"
    assert blocked.request is None


def test_expired_and_superseded_judge_attempts_cannot_commit(monkeypatch, tmp_path):
    run_dir = make_run(tmp_path)
    first = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(wall_clock_seconds=1, max_retries=1),
    ).request
    before = graph_snapshot(run_dir)
    original_now = app_driver._now()
    monkeypatch.setattr(app_driver, "_now", lambda: original_now + timedelta(seconds=10))
    expired = submit_app_episode_result(run_dir, judge_result(first))
    assert expired.commit_outcome.accepted is False
    assert "expired" in expired.commit_outcome.rejection_reason
    assert graph_snapshot(run_dir) == before

    monkeypatch.setattr(app_driver, "_now", lambda: original_now)
    retry = retry_app_episode(run_dir, first.episode_id)
    assert retry.attempt_id != first.attempt_id
    late = submit_app_episode_result(run_dir, judge_result(first))
    assert late.commit_outcome.accepted is False
    assert "superseded" in late.commit_outcome.rejection_reason
    assert submit_app_episode_result(run_dir, judge_result(retry.request)).commit_outcome.accepted is True


def test_only_one_judge_attempt_may_commit(tmp_path):
    run_dir = make_run(tmp_path)
    request = next_app_episode(run_dir).request
    assert submit_app_episode_result(run_dir, judge_result(request)).commit_outcome.accepted is True
    before = graph_snapshot(run_dir)
    second = submit_app_episode_result(run_dir, judge_result(request))
    assert second.commit_outcome.accepted is False
    assert "status=committed" in second.commit_outcome.rejection_reason
    assert graph_snapshot(run_dir) == before


def test_judge_to_controller_to_executor_to_child_end_to_end(tmp_path):
    run_dir = make_run(tmp_path)
    judge_request = next_app_episode(run_dir).request
    assert submit_app_episode_result(run_dir, judge_result(judge_request)).commit_outcome.accepted is True

    executor = next_app_episode(
        run_dir,
        embedding_provider=HashEmbeddingProvider(dim=8),
    )
    assert executor.controller_action == "episode_required"
    assert executor.request.role == "executor"
    assert 0 < executor.request.max_returned_children <= run_spec().budget.max_children_per_iteration
    progressed = app_run_status(run_dir)
    node = progressed.nodes[0]
    assert progressed.controller_iteration == 1
    assert progressed.graph_revision == 2
    assert node.local_embedding is not None
    assert node.density is not None
    assert node.uncertainty is not None
    assert node.ucb_score is not None
    assert node.expansion_budget == executor.request.max_returned_children

    committed = submit_app_episode_result(run_dir, executor_result(executor.request))
    assert committed.commit_outcome.accepted is True
    state = app_run_status(run_dir)
    assert [node.node_id for node in state.nodes] == ["n0", "child"]
    assert state.nodes[0].status == "closed"
    assert state.nodes[1].status == "frontier"


def test_unjudged_frontier_never_returns_continue_controller_to_main_agent(tmp_path):
    run_dir = make_run(tmp_path)
    first = next_app_episode(run_dir)
    assert first.request.role == "judge"
    submit_app_episode_result(run_dir, judge_result(first.request))
    second = next_app_episode(run_dir, embedding_provider=HashEmbeddingProvider(dim=8))
    assert second.controller_action == "episode_required"
    assert second.request.role == "executor"


def test_judge_telemetry_is_coarse_with_unavailable_usage(tmp_path):
    run_dir = make_run(tmp_path)
    request = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(request))
    events = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    judge_events = [event for event in events if event["role"] == "judge"]
    assert {event["event_type"] for event in judge_events}.issuperset(
        {
            "episode_granted",
            "episode_started",
            "episode_submitted",
            "episode_completed",
            "judge_observations_committed",
        }
    )
    assert all(event["usage_source"] == "unavailable" for event in judge_events)
    committed = next(event for event in events if event["event_type"] == "judge_observations_committed")
    assert committed["selected_node_count"] == 1
    assert committed["returned_observation_count"] == 1
    assert committed["accepted_observation_count"] == 1
    serialized = json.dumps(events)
    assert "subagent_names" not in serialized
    assert "hidden_reasoning" not in serialized


@pytest.mark.parametrize("terminal_action", ["ready_for_synthesis", "run_complete"])
def test_terminal_controller_actions_are_sticky_even_with_positive_budget(tmp_path, terminal_action):
    run_dir = make_run(tmp_path)
    state = app_driver.load_app_run(run_dir)
    state.nodes[0].expansion_budget = 1
    state.nodes[0].ucb_score = 1.0
    state.controller_action = terminal_action
    app_driver._save_state(run_dir, state)

    state_path = run_dir / "app_run_state.json"
    event_path = run_dir / "episode_events.jsonl"
    state_before = state_path.read_text(encoding="utf-8")
    events_before = event_path.read_text(encoding="utf-8")

    for _ in range(3):
        outcome = next_app_episode(run_dir)
        assert outcome.controller_action == terminal_action
        assert outcome.request is None

    assert state_path.read_text(encoding="utf-8") == state_before
    assert event_path.read_text(encoding="utf-8") == events_before
    persisted = app_run_status(run_dir)
    assert persisted.graph_revision == state.graph_revision
    assert persisted.node_revisions == state.node_revisions
    assert persisted.nodes[0].expansion_budget == 1
    assert persisted.episodes == []


@pytest.mark.parametrize(
    ("require_final_synthesis", "terminal_action"),
    [(True, "ready_for_synthesis"), (False, "run_complete")],
)
def test_max_iterations_stops_before_judging_final_executor_child(
    tmp_path,
    require_final_synthesis,
    terminal_action,
):
    run_dir = make_run(
        tmp_path,
        max_iterations=1,
        require_final_synthesis=require_final_synthesis,
    )
    initial_judge = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(initial_judge))
    executor = next_app_episode(run_dir, embedding_provider=HashEmbeddingProvider(dim=8)).request
    assert executor.role == "executor"
    submit_app_episode_result(run_dir, executor_result(executor, child_id="final-child"))

    events_before = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    judge_grants_before = [
        event
        for event in events_before
        if event["role"] == "judge" and event["event_type"] in {"episode_granted", "episode_started"}
    ]
    outcome = next_app_episode(run_dir)
    assert outcome.controller_action == terminal_action
    assert outcome.request is None
    assert app_run_status(run_dir).nodes[-1].node_id == "final-child"
    assert app_run_status(run_dir).nodes[-1].score is None

    events_after = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    judge_grants_after = [
        event
        for event in events_after
        if event["role"] == "judge" and event["event_type"] in {"episode_granted", "episode_started"}
    ]
    assert judge_grants_after == judge_grants_before


def test_app_progression_reuses_file_backed_embedding_cache_across_calls(tmp_path):
    run_dir = make_run(tmp_path, max_iterations=2)
    initial_judge = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(initial_judge))

    first_provider = TrackingEmbeddingProvider()
    executor = next_app_episode(run_dir, embedding_provider=first_provider).request
    assert first_provider.calls == 1
    assert (run_dir / "dte_cache.json").exists()
    submit_app_episode_result(
        run_dir,
        executor_result(executor, child_id="same-semantics", child_claim="claim 0"),
    )

    child_judge = next_app_episode(run_dir).request
    assert child_judge.role == "judge"
    submit_app_episode_result(run_dir, judge_result(child_judge))

    second_provider = TrackingEmbeddingProvider()
    next_grant = next_app_episode(run_dir, embedding_provider=second_provider)
    assert next_grant.controller_action == "ready_for_synthesis"
    assert next_grant.request is None
    assert second_provider.calls == 0

    cache = FileDTECache(run_dir / "dte_cache.json")
    semantic_twin = SearchNode(node_id="other-id", claim="claim 0")
    matching = EmbeddingCacheNamespace("tracking", "tracking-v1", 8, "embedding-v1")
    assert cache.get_embedding(semantic_twin, namespace=matching) is not None
    for changed in (
        EmbeddingCacheNamespace("other-provider", "tracking-v1", 8, "embedding-v1"),
        EmbeddingCacheNamespace("tracking", "tracking-v2", 8, "embedding-v1"),
        EmbeddingCacheNamespace("tracking", "tracking-v1", 16, "embedding-v1"),
    ):
        assert cache.get_embedding(semantic_twin, namespace=changed) is None


def test_cache_failure_cannot_partially_commit_controller_state(monkeypatch, tmp_path):
    run_dir = make_run(tmp_path)
    request = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(request))
    state_path = run_dir / "app_run_state.json"
    event_path = run_dir / "episode_events.jsonl"
    state_before = state_path.read_text(encoding="utf-8")
    events_before = event_path.read_text(encoding="utf-8")

    def fail_cache_write(*args, **kwargs):
        raise OSError("cache write failed")

    monkeypatch.setattr(FileDTECache, "set_embedding", fail_cache_write)
    with pytest.raises(OSError, match="cache write failed"):
        next_app_episode(run_dir, embedding_provider=TrackingEmbeddingProvider())

    assert state_path.read_text(encoding="utf-8") == state_before
    assert event_path.read_text(encoding="utf-8") == events_before
