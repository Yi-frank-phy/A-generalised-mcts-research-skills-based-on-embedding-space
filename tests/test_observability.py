from __future__ import annotations
from tests.helpers import completed_node, completed_candidate

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
from dte_backend.episode_models import EpisodeResult, ExecutorEpisodeOutput, ExecutorNodeCandidate, RuntimeDiagnostics, RuntimeLimits, compute_output_hash
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


def run_spec(*, enrichment=0, cap=2, iterations=1, allocation_mass=1) -> DTERunSpec:
    return DTERunSpec(
        problem="observe a bounded DTE run",
        goal="reconstruct decisions and later outcomes",
        constraints=["observability is read-only"],
        budget=BudgetSpec(
            max_iterations=iterations,
            allocation_mass_per_iteration=allocation_mass,
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
            completed_candidate(
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






def file_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


















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
        [completed_node(node_id="n", claim="candidate")],
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
        [completed_node(node_id="n", claim="candidate")],
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








def test_feedback_append_repairs_only_its_own_corrupt_tail(tmp_path):
    run_dir = tmp_path / "feedback-tail"
    create_app_run(
        run_dir,
        run_spec(),
        [completed_node(node_id="n", claim="candidate")],
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










def test_valid_but_stale_mirror_is_reported_as_recoverable(tmp_path):
    run_dir = tmp_path / "stale-mirror"
    create_app_run(
        run_dir,
        run_spec(),
        [completed_node(node_id="n", claim="candidate")],
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
        [completed_node(node_id="n", claim="candidate")],
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




def test_skill_and_agents_require_terminal_summary_but_not_hidden_topology():
    root = Path(__file__).resolve().parents[1]
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    combined = skill + agents

    assert "hook-driver handoff" in combined
    assert "observability-summary.json" in combined
    assert "terminal-handoff.json" in combined
    assert "record-feedback" in skill
    assert "No fixed subagent count or topology is required" in skill
    assert '"max_committed_search_nodes": 20' in skill
    assert '"max_iterations": 10' in skill
    assert "Default to Sol High" in skill
    assert "Recommend XHigh or Max" in skill
    assert "Recommend Ultra" in skill
    assert "Do not automatically downgrade research episodes to Terra" in skill
    assert "reasoning effort changed" in skill
    assert "Relation compares only the granted pairs; it is not a verifier" in agents
    assert "do not prove" in combined
    assert "require a complete hidden subagent topology" not in combined
