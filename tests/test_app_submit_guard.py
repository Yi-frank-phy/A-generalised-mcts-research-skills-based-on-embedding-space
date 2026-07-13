import dte_backend
import dte_backend.app_driver as app_driver

from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.telemetry import EpisodeEventLog


def _create_run(tmp_path):
    run_dir = tmp_path / "guarded-run"
    spec = DTERunSpec(
        problem="submission identity guard",
        goal="reject invalid App episode identities without mutation",
        budget=BudgetSpec(
            max_iterations=2,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=1,
        ),
    )
    parent = SearchNode(
        node_id="parent",
        claim="committed parent",
        expansion_budget=1,
        ucb_score=0.8,
    )
    app_driver.create_app_run(run_dir, spec, [parent], run_id="guard-run")
    request = app_driver.next_app_episode(run_dir).request
    assert request is not None
    return run_dir, request


def _state_snapshot(run_dir):
    return app_driver.app_run_status(run_dir).model_dump(mode="json")


def test_public_and_app_driver_submission_entrypoints_are_the_same_guard():
    assert app_driver.submit_app_episode_result is dte_backend.submit_app_episode_result


def test_unknown_missing_and_blank_episode_identities_fail_closed(tmp_path):
    run_dir, request = _create_run(tmp_path)
    before = _state_snapshot(run_dir)
    cases = [
        (
            {"episode_id": "unknown-episode", "attempt_id": request.attempt_id},
            "unknown episode_id",
        ),
        (
            {"episode_id": request.episode_id, "attempt_id": "unknown-attempt"},
            "unknown attempt_id",
        ),
        (
            {"attempt_id": request.attempt_id},
            "missing a non-empty episode_id",
        ),
        (
            {"episode_id": request.episode_id, "attempt_id": "   "},
            "missing a non-empty attempt_id",
        ),
    ]

    for payload, expected_reason in cases:
        outcome = app_driver.submit_app_episode_result(run_dir, payload)
        assert outcome.commit_outcome.accepted is False
        assert expected_reason in (outcome.commit_outcome.rejection_reason or "")
        assert outcome.commit_outcome.graph_revision_before == before["graph_revision"]
        assert outcome.commit_outcome.graph_revision_after == before["graph_revision"]
        assert _state_snapshot(run_dir) == before

    rejected = [
        event
        for event in EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
        if event["event_type"] == "output_rejected"
    ]
    assert len(rejected) == len(cases)
    assert all(event["accepted_node_count"] == 0 for event in rejected)
    assert all(event["schema_valid"] is False for event in rejected)
    assert all(event["usage_source"] == "unavailable" for event in rejected)


def test_non_mapping_submission_fails_closed(tmp_path):
    run_dir, _ = _create_run(tmp_path)
    before = _state_snapshot(run_dir)

    outcome = dte_backend.submit_app_episode_result(run_dir, object())

    assert outcome.commit_outcome.accepted is False
    assert "must be a mapping" in (outcome.commit_outcome.rejection_reason or "")
    assert _state_snapshot(run_dir) == before
    event = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()[-1]
    assert event["event_type"] == "output_rejected"
    assert event["episode_id"] is None
    assert event["attempt_id"] is None
