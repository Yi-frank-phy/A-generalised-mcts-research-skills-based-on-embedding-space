from tests.helpers import completed_node, completed_candidate
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
from dte_backend.episode_models import EpisodeResult, ExecutorEpisodeOutput, ExecutorNodeCandidate, RuntimeDiagnostics, RuntimeLimits, compute_output_hash
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




def executor_result(request, child_id="child"):
    output = ExecutorEpisodeOutput(
        nodes=[
            completed_candidate(
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
    node = completed_node(node_id="seed", claim="producer node").model_copy(update=update)
    with pytest.raises(ValueError, match="controller-owned"):
        create_app_run(run_dir, run_spec(), [node])
    assert not (run_dir / "app_run_state.json").exists()




@pytest.mark.parametrize("revision_kind", ["graph", "node"])
def test_load_rejects_revision_without_a_committed_transition(tmp_path, revision_kind):
    run_dir = tmp_path / f"forged-{revision_kind}-revision"
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
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
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
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
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
    next_app_episode(run_dir)
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["episodes"][0]["attempts"][0]["granted_at"] = granted_at
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="timestamp"):
        app_driver.load_app_run(run_dir)






def test_targeted_synthesis_rejects_unknown_node(tmp_path):
    run_dir = tmp_path / "unknown-target"
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
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




















def test_load_rejects_persisted_self_ancestry(tmp_path):
    run_dir = tmp_path / "invalid-persisted"
    create_app_run(run_dir, run_spec(), [completed_node(node_id="a", claim="clean")])
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["nodes"][0]["parent_ids"] = ["a"]
    write_raw_state(run_dir, payload)
    with pytest.raises(ValueError, match="authoritative producer output"):
        app_driver.load_app_run(run_dir)


def test_load_rejects_partial_active_attempt_identity(tmp_path):
    run_dir = tmp_path / "partial-active"
    create_app_run(run_dir, run_spec(), [completed_node(node_id="a", claim="clean")])
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["active_episode_id"] = "orphaned-episode"
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="partial active-attempt identity"):
        app_driver.load_app_run(run_dir)














def test_create_run_rejects_empty_initial_frontier_without_writing_state(tmp_path):
    run_dir = tmp_path / "empty-ingress"
    with pytest.raises(ValueError, match="at least one initial node"):
        create_app_run(run_dir, run_spec(), [])
    assert not (run_dir / "app_run_state.json").exists()
















def test_retry_runtime_override_is_validated_without_superseding_attempt(tmp_path):
    run_dir = tmp_path / "invalid-retry-limits"
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
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
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
    next_app_episode(run_dir)
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    mutate(payload)
    attempt = payload["episodes"][0]["attempts"][0]
    request = app_driver.EpisodeRequest.model_validate(attempt["request"])
    attempt["request_hash"] = app_driver._episode_request_hash(request)
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match=message):
        app_driver.load_app_run(run_dir)




def test_save_state_revalidates_assignment_mutations_before_install(tmp_path):
    run_dir = tmp_path / "save-boundary"
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
    state = app_driver.load_app_run(run_dir)
    state.pending_terminal_reason = "orphaned reason"
    before = (run_dir / "app_run_state.json").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="present together"):
        app_driver._save_state(run_dir, state)

    assert (run_dir / "app_run_state.json").read_text(encoding="utf-8") == before
    assert app_run_status(run_dir).pending_terminal_reason is None


def test_load_rejects_await_without_a_durable_blocking_fact(tmp_path):
    run_dir = tmp_path / "forged-await"
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
    payload = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    payload["controller_action"] = "await_operator_decision"
    write_raw_state(run_dir, payload)

    with pytest.raises(ValueError, match="lacks a durable blocking fact"):
        app_driver.load_app_run(run_dir)


def test_load_rejects_continue_that_bypasses_failed_attempt(tmp_path):
    run_dir = tmp_path / "bypass-failed"
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
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
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
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
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
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
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
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
    create_app_run(run_dir, run_spec(), [completed_node(node_id="seed", claim="clean")])
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
