from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

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
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.observability import (
    DuplicateFeedbackError,
    build_run_observability_summary,
    classify_rejection_reason,
    export_observability_jsonl,
    read_feedback_ledger,
    record_feedback,
    render_observability_text,
)
from dte_backend.observability_models import RunObservabilitySummaryV1
from dte_backend.relation_models import (
    RelationEpisodeOutput,
    RelationObservation,
)


def run_spec(*, enrichment=0, cap=2, iterations=1) -> DTERunSpec:
    return DTERunSpec(
        problem="observe a bounded DTE run",
        goal="reconstruct decisions and later outcomes",
        constraints=["observability is read-only"],
        budget=BudgetSpec(
            max_iterations=iterations,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=cap,
            max_relation_pairs_per_episode=cap,
            max_relation_enrichment_pairs=enrichment,
            min_iterations_before_synthesis=2,
        ),
        embedding_provider="hash",
        embedding_dimension=8,
    )


def diagnostics(**updates) -> RuntimeDiagnostics:
    payload = {
        "adapter_name": "codex-app-main-agent",
        "transport_name": "current-app-runtime",
        "profile": "native-autonomous",
        "usage_source": "unavailable",
        "diagnostics_source": "unavailable",
    }
    payload.update(updates)
    return RuntimeDiagnostics(**payload)


def judge_result(request, *, runtime=None, score=0.8) -> EpisodeResult:
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id=node_id,
                score=score,
                reasoning=f"observable judgment for {node_id}",
                risks=["known risk"],
                uncertainty_evidence=["external correctness remains unknown"],
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
        status="completed",
        structured_output=output,
        runtime_diagnostics=runtime or diagnostics(),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def executor_result(
    request,
    *,
    child_id: str | None = None,
    claim: str = "child claim",
    evidence: list[str] | None = None,
) -> EpisodeResult:
    nodes = []
    if child_id is not None:
        nodes.append(
            ExecutorNodeCandidate(
                node_id=child_id,
                claim=claim,
                evidence=list(evidence or []),
                parent_ids=[request.parent_node_id],
            )
        )
    output = ExecutorEpisodeOutput(nodes=nodes)
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


def relation_result(
    request,
    relation_type: str,
    *,
    disclosure_required: bool = False,
) -> EpisodeResult:
    observations = []
    for pair in request.relation_payload.candidate_pairs:
        payload = {
            "candidate_id": pair.candidate_id,
            "left_node_id": pair.left.node_id,
            "right_node_id": pair.right.node_id,
            "relation_type": relation_type,
            "confidence": 0.9,
            "rationale": f"classified as {relation_type}",
            "evidence_refs": (
                [pair.left.evidence[0].evidence_ref] if pair.left.evidence else []
            ),
            "materiality_assessment": (
                "material" if pair.material_to_synthesis else "non_material"
            ),
        }
        if relation_type == "equivalent":
            payload.update(
                merge_recommended=True,
                canonicality_factors=["evidence completeness"],
            )
        elif relation_type == "complementary":
            payload.update(
                complementarity_summary="the branches contribute different pieces",
                recommended_joint_use="retain both",
            )
        elif relation_type == "conflict":
            payload.update(
                conflict_summary="the conclusions conflict",
                disclosure_required=disclosure_required,
                conflicting_claims=[pair.left.claim, pair.right.claim],
            )
        else:
            payload.update(independence_summary="the branches are independent")
        observations.append(RelationObservation(**payload))
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
        runtime_diagnostics=diagnostics(),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def drive_relation_run(
    tmp_path: Path,
    *,
    relation_type: str,
    claims: tuple[str, str],
    evidence: tuple[list[str], list[str]],
) -> Path:
    run_dir = tmp_path / f"run-{relation_type}"
    create_app_run(
        run_dir,
        run_spec(),
        [
            SearchNode(node_id="p0", claim="parent zero"),
            SearchNode(node_id="p1", claim="parent one"),
        ],
        run_id=f"run-{relation_type}",
    )
    judge = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(judge))
    child_ids = []
    for index in range(2):
        executor = next_app_episode(
            run_dir, embedding_provider=HashEmbeddingProvider(dim=8)
        ).request
        assert executor.role == "executor"
        child_id = f"child-{index}"
        submit_app_episode_result(
            run_dir,
            executor_result(
                executor,
                child_id=child_id,
                claim=claims[index],
                evidence=evidence[index],
            ),
        )
        child_ids.append(child_id)
    final_judge = next_app_episode(run_dir).request
    assert final_judge.role == "judge"
    assert set(final_judge.selected_node_revisions) == set(child_ids)
    submit_app_episode_result(run_dir, judge_result(final_judge))
    relation = next_app_episode(run_dir).request
    assert relation.role == "relation"
    submit_app_episode_result(
        run_dir,
        relation_result(relation, relation_type),
    )
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"
    return run_dir


def drive_enrichment_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run-enrichment"
    create_app_run(
        run_dir,
        run_spec(enrichment=1),
        [
            SearchNode(node_id="a", claim="alpha route"),
            SearchNode(node_id="b", claim="beta route"),
        ],
        run_id="run-enrichment",
    )
    judge = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(judge))
    while True:
        outcome = next_app_episode(
            run_dir, embedding_provider=HashEmbeddingProvider(dim=8)
        )
        assert outcome.request is not None
        if outcome.request.role != "executor":
            relation = outcome.request
            break
        submit_app_episode_result(run_dir, executor_result(outcome.request))
    assert {
        pair.scheduling_class for pair in relation.relation_payload.candidate_pairs
    } == {"enrichment"}
    submit_app_episode_result(run_dir, relation_result(relation, "independent"))
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"
    return run_dir


def file_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_summary_is_deterministic_strict_round_trip_and_read_only(tmp_path):
    run_dir = drive_relation_run(
        tmp_path,
        relation_type="equivalent",
        claims=("same result", " SAME   RESULT "),
        evidence=(["left evidence"], ["right evidence"]),
    )
    before = file_snapshot(run_dir)
    first = build_run_observability_summary(run_dir)
    second = build_run_observability_summary(run_dir)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert RunObservabilitySummaryV1.model_validate_json(
        first.model_dump_json()
    ).model_dump(mode="json") == first.model_dump(mode="json")
    assert file_snapshot(run_dir) == before


def test_complete_lineage_allocation_judge_relation_and_trajectory(tmp_path):
    run_dir = drive_relation_run(
        tmp_path,
        relation_type="equivalent",
        claims=("same result", " SAME   RESULT "),
        evidence=(["left evidence"], ["right evidence"]),
    )
    summary = build_run_observability_summary(run_dir)

    assert summary.run.terminal_action == "ready_for_synthesis"
    assert summary.node_funnel.initial_node_count == 2
    assert summary.node_funnel.all_committed_node_count == 4
    assert summary.node_funnel.committed_executor_child_count == 2
    assert summary.node_funnel.merged_node_count == 1
    children = {
        node.node_id: node
        for node in summary.node_lineage
        if node.node_id.startswith("child-")
    }
    assert all(node.creation_episode_id and node.creation_attempt_id for node in children.values())
    assert {tuple(node.parent_ids) for node in children.values()} == {("p0",), ("p1",)}
    absorbed = next(node for node in children.values() if node.merged)
    assert absorbed.canonical_target in children
    assert all(node.judge_score == pytest.approx(0.8) for node in children.values())
    assert len(summary.allocation_outcomes) == 2
    assert all(item.actual_committed_children == 1 for item in summary.allocation_outcomes)
    assert all(item.unused_granted_capacity == 0 for item in summary.allocation_outcomes)
    assert summary.judge_outcomes.interpretation.endswith("calibration_or_causation")
    assert summary.relation_outcomes.blocking_candidates_generated == 1
    assert summary.relation_outcomes.blocking_pairs_resolved == 1
    assert summary.relation_outcomes.equivalent_count == 1
    assert summary.relation_outcomes.merge_count == 1
    exact = next(
        row
        for row in summary.relation_outcomes.by_candidate_reason
        if row.candidate_reason == "exact_duplicate"
    )
    assert exact.equivalent_yield == 1.0
    assert summary.controller_trajectory[0].positive_budget_parent_count == 2
    assert summary.controller_trajectory[0].children_committed == 2


def test_material_conflict_and_disclosure_are_reported_without_verifier_claim(tmp_path):
    run_dir = drive_relation_run(
        tmp_path,
        relation_type="conflict",
        claims=("condition is sufficient", "condition is not sufficient"),
        evidence=(["shared evidence"], ["shared evidence"]),
    )
    summary = build_run_observability_summary(run_dir)

    assert summary.relation_outcomes.conflict_count == 1
    assert summary.relation_outcomes.material_conflict_count == 1
    assert summary.relation_outcomes.disclosure_required_count == 1
    assert summary.relation_outcomes.merge_count == 0
    assert any(
        record.later_involved_in_conflict
        for record in summary.judge_outcomes.posterior_records
    )


def test_blocking_and_enrichment_are_counted_separately(tmp_path):
    blocking = drive_relation_run(
        tmp_path,
        relation_type="equivalent",
        claims=("same result", " SAME RESULT "),
        evidence=(["e1"], ["e2"]),
    )
    enrichment = drive_enrichment_run(tmp_path)
    blocking_summary = build_run_observability_summary(blocking)
    enrichment_summary = build_run_observability_summary(enrichment)

    assert blocking_summary.relation_outcomes.blocking_pairs_resolved == 1
    assert blocking_summary.relation_outcomes.enrichment_pairs_committed == 0
    assert enrichment_summary.relation_outcomes.blocking_candidates_generated == 0
    assert enrichment_summary.relation_outcomes.enrichment_pairs_committed == 1
    close = next(
        row
        for row in enrichment_summary.relation_outcomes.by_candidate_reason
        if row.candidate_reason == "embedding_close"
    )
    assert close.independent_yield == 1.0


def test_rejected_late_and_retried_attempts_do_not_duplicate_committed_work(tmp_path):
    run_dir = tmp_path / "retry-run"
    create_app_run(
        run_dir,
        run_spec(cap=1, iterations=2),
        [SearchNode(node_id="n", claim="candidate")],
        run_id="retry-run",
    )
    first = next_app_episode(run_dir).request
    invalid = judge_result(first).model_dump(mode="json")
    invalid["run_id"] = "wrong-run"
    rejected = submit_app_episode_result(run_dir, invalid)
    assert rejected.commit_outcome.accepted is False
    retry = retry_app_episode(run_dir, first.episode_id).request
    late = submit_app_episode_result(run_dir, judge_result(first))
    assert late.commit_outcome.accepted is False
    assert submit_app_episode_result(run_dir, judge_result(retry)).commit_outcome.accepted

    summary = build_run_observability_summary(run_dir)
    funnel = summary.episode_funnel.judge
    assert funnel.attempt_count == 2
    assert funnel.committed_attempt_count == 1
    assert funnel.rejected_attempt_count == 1
    assert funnel.superseded_attempt_count == 1
    assert funnel.retried_attempt_count == 1
    assert summary.node_funnel.judged_node_count == 1
    categories = {
        row.category: row.count for row in summary.rejections.by_category
    }
    assert categories["identity_mismatch"] == 1
    assert categories["lifecycle_rejection"] == 1


@pytest.mark.parametrize(
    ("transition", "expected_field"),
    [("failed", "failed_attempt_count"), ("cancelled", "cancelled_attempt_count")],
)
def test_failed_and_cancelled_attempts_are_visible(tmp_path, transition, expected_field):
    run_dir = tmp_path / transition
    create_app_run(
        run_dir,
        run_spec(),
        [SearchNode(node_id="n", claim="candidate")],
        run_id=transition,
    )
    request = next_app_episode(run_dir).request
    if transition == "failed":
        fail_app_episode(run_dir, request.episode_id, request.attempt_id, "runtime failed")
    else:
        cancel_app_episode(run_dir, request.episode_id, request.attempt_id, "operator cancelled")
    funnel = build_run_observability_summary(run_dir).episode_funnel.judge
    assert getattr(funnel, expected_field) == 1


def test_expired_attempt_is_visible(tmp_path, monkeypatch):
    run_dir = tmp_path / "expired"
    create_app_run(
        run_dir,
        run_spec(),
        [SearchNode(node_id="n", claim="candidate")],
        run_id="expired",
    )
    request = next_app_episode(
        run_dir, runtime_limits=RuntimeLimits(wall_clock_seconds=1)
    ).request
    granted = app_driver._parse_time(
        app_run_status(run_dir).episodes[0].attempts[0].granted_at
    )
    monkeypatch.setattr(app_driver, "_now", lambda: granted + timedelta(seconds=2))
    assert app_run_status(run_dir).episodes[0].attempts[0].status == "expired"
    assert (
        build_run_observability_summary(run_dir)
        .episode_funnel.judge.expired_attempt_count
        == 1
    )


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("episode result schema validation failed", "schema_rejection"),
        ("attempt ID mismatch", "identity_mismatch"),
        ("stale graph revision", "stale_revision"),
        ("attempt lifecycle forbids commit", "lifecycle_rejection"),
        ("controller-owned field violation", "controller_owned_field_violation"),
        ("duplicate node ID inside result", "duplicate_output"),
        ("returned child count exceeds grant", "over_grant"),
        ("Relation episode candidate pairs are not node-disjoint", "relation_overlap"),
        ("merge provenance conflict", "merge_provenance_conflict"),
        ("attempt deadline elapsed", "timeout_expire"),
        ("unclassified backend error", "other"),
    ],
)
def test_rejection_classification_is_explicit_and_deterministic(reason, category):
    assert classify_rejection_reason(reason) == category


def test_corrupt_telemetry_tail_is_recovered_logically_without_repair(tmp_path):
    run_dir = tmp_path / "corrupt-telemetry"
    create_app_run(
        run_dir,
        run_spec(),
        [SearchNode(node_id="n", claim="candidate")],
        run_id="corrupt-telemetry",
    )
    telemetry = run_dir / "episode_events.jsonl"
    with telemetry.open("ab") as handle:
        handle.write(b'{"event_id":"cut-off"')
    before = telemetry.read_bytes()

    summary = build_run_observability_summary(run_dir)

    assert summary.data_quality.corrupt_telemetry_tail_detected is True
    assert telemetry.read_bytes() == before
    assert not telemetry.with_suffix(".jsonl.corrupt").exists()


def test_missing_legacy_fields_and_artifacts_are_reported_not_guessed(tmp_path):
    run_dir = tmp_path / "legacy"
    create_app_run(
        run_dir,
        run_spec(),
        [SearchNode(node_id="n", claim="candidate")],
        run_id="legacy",
    )
    state_path = run_dir / "app_run_state.json"
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw.pop("state_schema_version")
    raw.pop("spec_hash")
    raw.pop("initial_nodes_hash")
    raw["spec"]["budget"].pop("max_children_per_iteration")
    state_path.write_text(json.dumps(raw), encoding="utf-8")
    (run_dir / "episode_events.jsonl").unlink()
    before = state_path.read_bytes()

    summary = build_run_observability_summary(run_dir)

    assert summary.run.observability_status == "partial_legacy"
    assert summary.run.state_schema_version is None
    assert summary.run.budget.max_children_per_iteration is None
    assert summary.data_quality.partial_legacy_reconstruction is True
    assert "episode_events.jsonl" in summary.data_quality.missing_artifacts
    assert state_path.read_bytes() == before


def test_unprovenanced_legacy_score_is_not_guessed_as_a_judge_result(tmp_path):
    run_dir = tmp_path / "legacy-score"
    create_app_run(
        run_dir,
        run_spec(),
        [SearchNode(node_id="n", claim="candidate")],
        run_id="legacy-score",
    )
    state_path = run_dir / "app_run_state.json"
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw["nodes"][0]["score"] = 0.95
    state_path.write_text(json.dumps(raw), encoding="utf-8")

    summary = build_run_observability_summary(run_dir)

    assert summary.run.observability_status == "partial_legacy"
    assert summary.node_lineage[0].judge_score is None
    assert summary.node_funnel.judged_node_count == 0
    assert summary.judge_outcomes.score_distribution.count == 0
    assert any(
        "without a committed Judge observation" in issue
        for issue in summary.data_quality.inconsistent_but_recoverable_records
    )


def test_feedback_append_target_validation_duplicate_and_controller_isolation(tmp_path):
    run_dir = drive_relation_run(
        tmp_path,
        relation_type="equivalent",
        claims=("same result", " SAME RESULT "),
        evidence=(["e1"], ["e2"]),
    )
    state_before = (run_dir / "app_run_state.json").read_bytes()
    telemetry_before = (run_dir / "episode_events.jsonl").read_bytes()
    summary = build_run_observability_summary(run_dir)
    allocation_id = summary.allocation_outcomes[0].allocation_decision_id

    run_feedback = record_feedback(
        run_dir,
        target_type="run",
        metric="architecture_effectiveness",
        score=0.8,
        source="user",
        comment="found a route I had not considered",
        feedback_id="feedback-run",
    )
    allocation_feedback = record_feedback(
        run_dir,
        target_type="allocation_decision",
        target_id=allocation_id,
        metric="branch_usefulness",
        label="useful",
        source="main_agent",
        feedback_id="feedback-allocation",
    )
    assert run_feedback.source == "user"
    assert allocation_feedback.source == "main_agent"
    records, diagnostics_record = read_feedback_ledger(run_dir)
    assert [record.feedback_id for record in records] == [
        "feedback-run",
        "feedback-allocation",
    ]
    assert diagnostics_record.valid_record_count == 2
    assert (run_dir / "app_run_state.json").read_bytes() == state_before
    assert (run_dir / "episode_events.jsonl").read_bytes() == telemetry_before

    with pytest.raises(ValueError, match="target does not exist"):
        record_feedback(
            run_dir,
            target_type="node",
            target_id="missing-node",
            metric="usefulness",
            score=0.2,
            source="external_evaluator",
        )
    with pytest.raises(ValidationError, match="substantive"):
        record_feedback(
            run_dir,
            target_type="run",
            metric="empty",
            source="main_agent",
        )
    with pytest.raises(DuplicateFeedbackError):
        record_feedback(
            run_dir,
            target_type="run",
            metric="architecture_effectiveness",
            score=0.8,
            source="user",
            feedback_id="feedback-run",
        )
    assert build_run_observability_summary(run_dir).data_quality.feedback_record_count == 2


def test_feedback_accepts_every_supported_non_run_target_type(tmp_path):
    run_dir = drive_relation_run(
        tmp_path,
        relation_type="equivalent",
        claims=("same result", " SAME RESULT "),
        evidence=(["e1"], ["e2"]),
    )
    summary = build_run_observability_summary(run_dir)
    state = json.loads((run_dir / "app_run_state.json").read_text(encoding="utf-8"))
    targets = {
        "episode": summary.episodes[0].episode_id,
        "attempt": summary.episodes[0].attempts[0].attempt_id,
        "node": summary.node_lineage[0].node_id,
        "relation_record": state["relation_ledger"][0]["relation_record_id"],
        "merge_application": state["merge_applications"][0]["merge_application_id"],
        "allocation_decision": summary.allocation_outcomes[0].allocation_decision_id,
    }

    for target_type, target_id in targets.items():
        record_feedback(
            run_dir,
            target_type=target_type,
            target_id=target_id,
            metric="target_binding",
            label="observed",
            source="external_evaluator",
            feedback_id=f"feedback-{target_type}",
        )

    records, diagnostics_record = read_feedback_ledger(run_dir)
    assert diagnostics_record.valid_record_count == len(targets)
    assert {(record.target_type, record.target_id) for record in records} == set(
        targets.items()
    )


def test_feedback_append_repairs_only_its_own_corrupt_tail(tmp_path):
    run_dir = tmp_path / "feedback-tail"
    create_app_run(
        run_dir,
        run_spec(),
        [SearchNode(node_id="n", claim="candidate")],
        run_id="feedback-tail",
    )
    record_feedback(
        run_dir,
        target_type="run",
        metric="usefulness",
        score=0.5,
        source="user",
        feedback_id="first",
    )
    path = run_dir / "observability" / "feedback.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"feedback_id":')

    record_feedback(
        run_dir,
        target_type="run",
        metric="usefulness",
        score=0.6,
        source="user",
        feedback_id="second",
    )
    records, ledger_quality = read_feedback_ledger(run_dir)
    assert [record.feedback_id for record in records] == ["first", "second"]
    assert ledger_quality.corrupt_tail_repaired is True
    assert path.with_suffix(".jsonl.corrupt").exists()


def test_runtime_aggregate_diagnostics_are_optional_nullable_and_source_labelled(tmp_path):
    with pytest.raises(ValidationError, match="diagnostics_source"):
        diagnostics(internal_subagent_count=2)

    run_dir = tmp_path / "diagnostics"
    create_app_run(
        run_dir,
        run_spec(),
        [SearchNode(node_id="n", claim="candidate")],
        run_id="diagnostics",
    )
    request = next_app_episode(run_dir).request
    reported = diagnostics(
        diagnostics_source="main_agent_reported",
        internal_subagent_count=2,
        max_internal_parallelism=2,
        internal_tool_call_count=4,
        internal_round_count=2,
        internal_failure_count=0,
        internal_subagent_metadata={
            "agents": [{"name": "hidden-worker", "transcript": "must stay opaque"}]
        },
    )
    submit_app_episode_result(run_dir, judge_result(request, runtime=reported))
    summary = build_run_observability_summary(run_dir)
    stored = summary.episodes[0].attempts[0].runtime_diagnostics
    assert stored.diagnostics_source == "main_agent_reported"
    assert stored.internal_subagent_count == 2
    assert not hasattr(stored, "internal_subagent_metadata")
    assert "hidden-worker" not in summary.model_dump_json()
    assert summary.data_quality.runtime_diagnostics_unavailable is False
    assert summary.data_quality.usage_unavailable is True


def test_legacy_runtime_diagnostics_defaults_do_not_create_false_mirror_drift(tmp_path):
    run_dir = tmp_path / "legacy-runtime-diagnostics"
    create_app_run(
        run_dir,
        run_spec(),
        [SearchNode(node_id="n", claim="candidate")],
        run_id="legacy-runtime-diagnostics",
    )
    request = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, judge_result(request))
    state_path = run_dir / "app_run_state.json"
    raw_state = json.loads(state_path.read_text(encoding="utf-8"))
    attempt = raw_state["episodes"][0]["attempts"][0]
    episode_id = raw_state["episodes"][0]["episode_id"]
    attempt_id = attempt["attempt_id"]
    status_path = run_dir / "episodes" / episode_id / attempt_id / "status.json"
    result_path = run_dir / "episodes" / episode_id / attempt_id / "result.json"
    raw_status = json.loads(status_path.read_text(encoding="utf-8"))
    raw_result = json.loads(result_path.read_text(encoding="utf-8"))
    aggregate_fields = {
        "internal_subagent_count",
        "max_internal_parallelism",
        "internal_tool_call_count",
        "internal_round_count",
        "internal_failure_count",
        "internal_input_tokens",
        "internal_output_tokens",
        "diagnostics_source",
    }
    diagnostics_payloads = (
        attempt["committed_result"]["runtime_diagnostics"],
        raw_status["committed_result"]["runtime_diagnostics"],
        raw_result["runtime_diagnostics"],
    )
    for payload in diagnostics_payloads:
        for field_name in aggregate_fields:
            payload.pop(field_name)
    state_path.write_text(json.dumps(raw_state), encoding="utf-8")
    status_path.write_text(json.dumps(raw_status), encoding="utf-8")
    result_path.write_text(json.dumps(raw_result), encoding="utf-8")

    summary = build_run_observability_summary(run_dir)

    stored = summary.episodes[0].attempts[0].runtime_diagnostics
    assert stored.diagnostics_source == "unavailable"
    assert stored.internal_subagent_count is None
    assert not any(
        "episode mirror disagrees" in issue
        for issue in summary.data_quality.inconsistent_but_recoverable_records
    )


def test_text_report_contains_core_metrics_without_changing_json(tmp_path):
    run_dir = drive_relation_run(
        tmp_path,
        relation_type="equivalent",
        claims=("same", " SAME "),
        evidence=(["e1"], ["e2"]),
    )
    summary = build_run_observability_summary(run_dir)
    canonical = summary.model_dump(mode="json")
    text = render_observability_text(summary)
    assert "judge:" in text
    assert "nodes: 2 initial -> 4 committed" in text
    assert "allocations:" in text
    assert "terminal:" in text
    assert "internal process metrics" in text
    assert summary.model_dump(mode="json") == canonical


def test_cross_run_export_skips_bad_run_and_types_every_record(tmp_path):
    runs_root = tmp_path / "runs"
    good = drive_relation_run(
        runs_root,
        relation_type="equivalent",
        claims=("same", " SAME "),
        evidence=(["e1"], ["e2"]),
    )
    record_feedback(
        good,
        target_type="run",
        metric="usefulness",
        score=0.9,
        source="user",
        feedback_id="exported-feedback",
    )
    bad = runs_root / "bad"
    bad.mkdir(parents=True)
    (bad / "app_run_state.json").write_text("{bad", encoding="utf-8")
    output = tmp_path / "observability.jsonl"

    result = export_observability_jsonl(runs_root, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert result.processed_run_count == 1
    assert result.skipped_run_count == 1
    assert result.record_count == len(rows)
    assert {row["record_type"] for row in rows}.issuperset(
        {"run", "episode", "node", "allocation", "feedback"}
    )
    assert all(row["schema_version"] == "dte-observability-export.v1" for row in rows)


def test_valid_but_stale_mirror_is_reported_as_recoverable(tmp_path):
    run_dir = tmp_path / "stale-mirror"
    create_app_run(
        run_dir,
        run_spec(),
        [SearchNode(node_id="n", claim="candidate")],
        run_id="stale-mirror",
    )
    mirror = run_dir / "relations" / "candidates.json"
    payload = json.loads(mirror.read_text(encoding="utf-8"))
    payload["blocking_candidate_count"] = 99
    mirror.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_run_observability_summary(run_dir)

    assert any(
        "derived Relation artifact disagrees" in issue
        for issue in summary.data_quality.inconsistent_but_recoverable_records
    )


def test_feedback_refuses_to_append_after_invalid_complete_record(tmp_path):
    run_dir = tmp_path / "invalid-feedback-record"
    create_app_run(
        run_dir,
        run_spec(),
        [SearchNode(node_id="n", claim="candidate")],
        run_id="invalid-feedback-record",
    )
    path = run_dir / "observability" / "feedback.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"feedback_id": "not-a-valid-record"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid existing records"):
        record_feedback(
            run_dir,
            target_type="run",
            metric="usefulness",
            score=0.5,
            source="user",
        )


def test_observability_and_feedback_cli_smoke(tmp_path):
    run_dir = drive_relation_run(
        tmp_path,
        relation_type="equivalent",
        claims=("same", " SAME "),
        evidence=(["e1"], ["e2"]),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    json_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "observability-summary",
            "--run-dir",
            str(run_dir),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(json_result.stdout)["schema_version"] == (
        "dte-run-observability-summary.v1"
    )
    text_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "observability-summary",
            "--run-dir",
            str(run_dir),
            "--format",
            "text",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "DTE observability summary" in text_result.stdout
    feedback_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "record-feedback",
            "--run-dir",
            str(run_dir),
            "--target-type",
            "run",
            "--metric",
            "architecture_effectiveness",
            "--score",
            "0.8",
            "--source",
            "user",
            "--comment",
            "useful route",
            "--feedback-id",
            "cli-feedback",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(feedback_result.stdout)["feedback_id"] == "cli-feedback"
    export_path = tmp_path / "cli-export.jsonl"
    export_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "observability-export",
            "--runs-root",
            str(tmp_path),
            "--format",
            "jsonl",
            "--output",
            str(export_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(export_result.stdout)["processed_run_count"] == 1
    assert any(
        json.loads(line)["record_type"] == "feedback"
        for line in export_path.read_text(encoding="utf-8").splitlines()
    )


def test_skill_and_agents_require_terminal_summary_but_not_hidden_topology():
    root = Path(__file__).resolve().parents[1]
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    combined = skill + agents

    assert "observability-summary --run-dir <run-dir> --format json" in combined
    assert "record-feedback" in skill
    assert "No fixed subagent count or topology is required" in skill
    assert "Relation compares only the granted pairs; it is not a verifier" in agents
    assert "do not prove" in combined
    assert "require a complete hidden subagent topology" not in combined
