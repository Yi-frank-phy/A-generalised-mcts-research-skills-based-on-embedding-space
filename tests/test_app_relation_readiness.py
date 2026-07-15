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
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.episode_adapter import build_relation_episode_request
from dte_backend.episode_commit import EpisodeGraph
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
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.relation_candidates import (
    generate_relation_candidates,
    generate_relation_enrichment_candidates,
)
from dte_backend.relation_models import (
    RelationCandidate,
    RelationEpisodeOutput,
    RelationObservation,
)
from dte_backend.relation_readiness import evaluate_synthesis_readiness
from dte_backend.telemetry import EpisodeEventLog


def spec(*, cap=3, pair_cap=3, max_iterations=1, enrichment_cap=0):
    return DTERunSpec(
        problem="relation readiness",
        goal="reach synthesis without duplicate or undisclosed material-conflict ambiguity",
        constraints=["Relation is observation-only"],
        budget=BudgetSpec(
            max_iterations=max_iterations,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=cap,
            max_relation_pairs_per_episode=pair_cap,
            max_relation_enrichment_pairs=enrichment_cap,
            min_iterations_before_synthesis=2,
        ),
        embedding_provider="hash",
        embedding_dimension=8,
    )


def diagnostics():
    return RuntimeDiagnostics(
        adapter_name="codex-app-main-agent",
        transport_name="current-app-runtime",
        profile="native-autonomous",
        usage_source="unavailable",
    )


def force_stop_intent(run_dir):
    state = app_driver.load_app_run(run_dir)
    state.controller_iteration = state.spec.budget.max_iterations
    app_driver._save_state(run_dir, state)


def relation_result(request, relation_type="independent", *, disclosure_required=False):
    observations = []
    for pair in request.relation_payload.candidate_pairs:
        common = dict(
            candidate_id=pair.candidate_id,
            left_node_id=pair.left.node_id,
            right_node_id=pair.right.node_id,
            relation_type=relation_type,
            confidence=0.9,
            rationale=f"classified as {relation_type}",
            evidence_refs=(
                [pair.left.evidence[0].evidence_ref] if pair.left.evidence else []
            ),
            materiality_assessment="material" if pair.material_to_synthesis else "non_material",
        )
        if relation_type == "equivalent":
            common.update(merge_recommended=True, canonicality_factors=["evidence completeness"])
        elif relation_type == "complementary":
            common.update(
                complementarity_summary="distinct contributions support joint use",
                recommended_joint_use="retain both branches",
                distinct_contributions=["left route", "right route"],
            )
        elif relation_type == "conflict":
            common.update(
                conflict_summary="the conclusions disagree under shared evidence",
                disclosure_required=disclosure_required,
                conflicting_claims=[pair.left.claim, pair.right.claim],
            )
        else:
            common.update(independence_summary="the branches address separate questions")
        observations.append(RelationObservation(**common))
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


def state_snapshot(run_dir):
    state = app_run_status(run_dir)
    return {
        "graph_revision": state.graph_revision,
        "node_revisions": dict(state.node_revisions),
        "nodes": [node.model_dump(mode="json") for node in state.nodes],
        "candidates": [item.model_dump(mode="json") for item in state.relation_candidates],
        "ledger": [item.model_dump(mode="json") for item in state.relation_ledger],
        "merges": [item.model_dump(mode="json") for item in state.merge_applications],
    }


def make_duplicate_gate_run(tmp_path, *, pair_cap=3, nodes=None):
    run_dir = tmp_path / "run"
    nodes = nodes or [
        SearchNode(node_id="a", claim="Same claim", score=0.8, evidence=["source a"]),
        SearchNode(node_id="b", claim=" same   CLAIM ", score=0.7, evidence=["source b"]),
    ]
    create_app_run(run_dir, spec(pair_cap=pair_cap), nodes, run_id="relation-run")
    force_stop_intent(run_dir)
    return run_dir


def test_candidate_generation_is_canonical_stable_bounded_and_prioritized():
    nodes = [
        SearchNode(node_id="b", claim="same", score=0.7),
        SearchNode(node_id="a", claim=" SAME ", score=0.8),
        SearchNode(node_id="far", claim="unrelated", score=0.1),
    ]
    revisions = {node.node_id: 0 for node in nodes}
    first = generate_relation_candidates(
        nodes,
        node_revisions=revisions,
        graph_revision=4,
        provisional_synthesis_node_ids=["a", "b"],
    )
    second = generate_relation_candidates(
        list(reversed(nodes)),
        node_revisions=revisions,
        graph_revision=4,
        provisional_synthesis_node_ids=["a", "b"],
    )
    duplicate = next(item for item in first if item.candidate_reason == "exact_duplicate")
    assert (duplicate.left_node_id, duplicate.right_node_id) == ("a", "b")
    assert duplicate.priority == "critical"
    assert duplicate.material_to_synthesis is True
    assert [item.candidate_id for item in first] == [item.candidate_id for item in second]
    assert len({(item.left_node_id, item.right_node_id) for item in first}) == len(first)
    assert all("far" not in (item.left_node_id, item.right_node_id) for item in first)


def test_material_conflict_candidate_uses_selected_shared_evidence_only():
    nodes = [
        SearchNode(node_id="a", claim="condition is sufficient", evidence=["paper-1"], score=0.8),
        SearchNode(node_id="b", claim="condition is not sufficient", evidence=["paper-1"], score=0.79),
        SearchNode(node_id="c", claim="remote branch", evidence=["other"], score=0.2),
    ]
    candidates = generate_relation_candidates(
        nodes,
        node_revisions={node.node_id: 0 for node in nodes},
        graph_revision=2,
        provisional_synthesis_node_ids=["a", "b"],
    )
    material = next(item for item in candidates if item.candidate_reason == "potential_material_conflict")
    assert (material.left_node_id, material.right_node_id) == ("a", "b")
    assert material.material_to_synthesis is True
    assert not any(
        item.candidate_reason == "potential_material_conflict" and "c" in (item.left_node_id, item.right_node_id)
        for item in candidates
    )


def test_candidate_generation_never_expands_to_global_all_pairs():
    nodes = [SearchNode(node_id=f"n{i:03d}", claim="same", score=0.5) for i in range(100)]
    candidates = generate_relation_candidates(
        nodes,
        node_revisions={node.node_id: 0 for node in nodes},
        graph_revision=1,
        provisional_synthesis_node_ids=[node.node_id for node in nodes[:8]],
        max_candidates=5,
    )
    assert len(candidates) == 28
    assert len(candidates) < len(nodes) * (len(nodes) - 1) // 2


def test_selected_duplicate_obligation_is_not_hidden_by_nonselected_alias():
    nodes = [
        SearchNode(node_id="a", claim="same", score=0.9),
        SearchNode(node_id="b", claim="same", score=0.1),
        SearchNode(node_id="c", claim="same", score=0.8),
    ]
    candidates = generate_relation_candidates(
        nodes,
        node_revisions={node.node_id: 0 for node in nodes},
        graph_revision=1,
        provisional_synthesis_node_ids=["a", "c"],
        max_candidates=3,
    )
    selected_pair = next(
        item for item in candidates if (item.left_node_id, item.right_node_id) == ("a", "c")
    )
    assert selected_pair.candidate_reason == "exact_duplicate"
    assert selected_pair.material_to_synthesis is True


def test_entropy_plateau_only_changes_candidate_reason_priority_not_relation_type():
    nodes = [SearchNode(node_id="a", claim="A", score=0.8), SearchNode(node_id="b", claim="B", score=0.79)]
    candidates = generate_relation_candidates(
        nodes,
        node_revisions={"a": 0, "b": 0},
        graph_revision=1,
        provisional_synthesis_node_ids=["a", "b"],
        entropy_plateau=True,
    )
    assert any(item.candidate_reason in {"entropy_plateau", "high_score_near_tie"} for item in candidates)
    assert all(not hasattr(item, "relation_type") for item in candidates)


def test_relation_grant_is_strict_bounded_and_persistent(tmp_path):
    nodes = [SearchNode(node_id=f"n{i}", claim="duplicate", score=0.9 - i * 0.01) for i in range(4)]
    run_dir = make_duplicate_gate_run(tmp_path, pair_cap=2, nodes=nodes)
    outcome = next_app_episode(run_dir)
    assert outcome.request.role == "relation"
    assert len(outcome.request.relation_payload.candidate_pairs) == 2
    assert outcome.request.allowed_output_types == []
    assert outcome.request.required_parent_id_on_children is False
    assert all(item.status in {"granted", "pending"} for item in app_run_status(run_dir).relation_candidates)
    assert (run_dir / "relations" / "candidates.json").exists()
    assert (run_dir / "relations" / "relation_ledger.json").exists()
    assert (run_dir / "relations" / "synthesis_readiness.json").exists()
    reloaded = app_driver.load_app_run(run_dir)
    assert reloaded.relation_candidates
    assert reloaded.synthesis_readiness.ready is False


def test_relation_grant_launches_no_subprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Relation launched subprocess")),
    )
    outcome = next_app_episode(make_duplicate_gate_run(tmp_path))
    assert outcome.request.role == "relation"


def test_relation_request_and_result_schemas_are_strict(tmp_path):
    request = next_app_episode(make_duplicate_gate_run(tmp_path)).request
    raw_request = request.model_dump(mode="json")
    raw_request["relation_payload"]["candidate_pairs"][0]["allocation"] = 1
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EpisodeRequest.model_validate(raw_request)
    raw_result = relation_result(request).model_dump(mode="json")
    raw_result["structured_output"]["observations"][0]["ucb_score"] = 9
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EpisodeResult.model_validate(raw_result)


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "reversed"])
def test_relation_exact_grant_membership_is_atomic(tmp_path, case):
    nodes = [SearchNode(node_id=f"n{i}", claim="duplicate", score=0.8) for i in range(3)]
    run_dir = make_duplicate_gate_run(tmp_path, pair_cap=2, nodes=nodes)
    request = next_app_episode(run_dir).request
    before = state_snapshot(run_dir)
    raw = relation_result(request).model_dump(mode="json")
    observations = raw["structured_output"]["observations"]
    if case == "missing":
        observations.pop()
    elif case == "extra":
        extra = dict(observations[0])
        extra["candidate_id"] = "ungranted"
        observations.append(extra)
    elif case == "duplicate":
        observations.append(dict(observations[0]))
    else:
        observations[0]["left_node_id"], observations[0]["right_node_id"] = (
            observations[0]["right_node_id"], observations[0]["left_node_id"],
        )
    try:
        parsed = RelationEpisodeOutput.model_validate(raw["structured_output"])
    except ValidationError:
        parsed = None
    if parsed is not None and case != "reversed":
        raw["output_hash"] = compute_output_hash(parsed, request.output_schema_version)
    outcome = submit_app_episode_result(run_dir, raw)
    assert outcome.commit_outcome.accepted is False
    assert state_snapshot(run_dir) == before


@pytest.mark.parametrize("case", ["enum", "confidence", "evidence", "pollution"])
def test_relation_semantic_validation_rejects_whole_result(tmp_path, case):
    run_dir = make_duplicate_gate_run(tmp_path)
    request = next_app_episode(run_dir).request
    before = state_snapshot(run_dir)
    raw = relation_result(request).model_dump(mode="json")
    observation = raw["structured_output"]["observations"][0]
    if case == "enum":
        observation["relation_type"] = "similar"
    elif case == "confidence":
        observation["confidence"] = 1.5
    elif case == "evidence":
        observation["evidence_refs"] = ["not-granted:evidence:9"]
    else:
        observation["discriminator_task_proposal"] = {
            "task_type": "formal_derivation",
            "objective": "check",
            "rationale": "needed",
            "material_to_synthesis": True,
            "graph_revision": 99,
        }
    try:
        parsed = RelationEpisodeOutput.model_validate(raw["structured_output"])
    except ValidationError:
        parsed = None
    if parsed is not None:
        raw["output_hash"] = compute_output_hash(parsed, request.output_schema_version)
    outcome = submit_app_episode_result(run_dir, raw)
    assert outcome.commit_outcome.accepted is False
    assert state_snapshot(run_dir) == before


@pytest.mark.parametrize("transition", ["failed", "cancelled", "expired", "superseded"])
def test_relation_attempt_lifecycle_rejects_late_results(monkeypatch, tmp_path, transition):
    run_dir = make_duplicate_gate_run(tmp_path)
    request = next_app_episode(
        run_dir,
        runtime_limits=RuntimeLimits(wall_clock_seconds=1, max_retries=1),
    ).request
    if transition == "failed":
        fail_app_episode(run_dir, request.episode_id, request.attempt_id, "failed")
    elif transition == "cancelled":
        cancel_app_episode(run_dir, request.episode_id, request.attempt_id, "cancelled")
    elif transition == "expired":
        now = app_driver._now()
        monkeypatch.setattr(app_driver, "_now", lambda: now + timedelta(seconds=5))
    else:
        fail_app_episode(run_dir, request.episode_id, request.attempt_id, "retry")
        retry = retry_app_episode(run_dir, request.episode_id)
        assert retry.attempt_id != request.attempt_id
    before = state_snapshot(run_dir)
    outcome = submit_app_episode_result(run_dir, relation_result(request))
    assert outcome.commit_outcome.accepted is False
    assert state_snapshot(run_dir) == before


def test_relation_retry_gets_new_attempt_and_only_one_can_commit(tmp_path):
    run_dir = make_duplicate_gate_run(tmp_path)
    first = next_app_episode(run_dir, runtime_limits=RuntimeLimits(max_retries=1)).request
    fail_app_episode(run_dir, first.episode_id, first.attempt_id, "retry")
    retry = retry_app_episode(run_dir, first.episode_id)
    assert retry.attempt_id != first.attempt_id
    assert submit_app_episode_result(run_dir, relation_result(retry.request)).commit_outcome.accepted
    snapshot = state_snapshot(run_dir)
    late = submit_app_episode_result(run_dir, relation_result(retry.request))
    assert late.commit_outcome.accepted is False
    assert state_snapshot(run_dir) == snapshot


@pytest.mark.parametrize("side", ["left", "right"])
def test_stale_relation_node_revision_rejected_without_ledger_mutation(tmp_path, side):
    run_dir = make_duplicate_gate_run(tmp_path)
    request = next_app_episode(run_dir).request
    state = app_driver.load_app_run(run_dir)
    target = getattr(request.relation_payload.candidate_pairs[0], side).node_id
    state.node_revisions[target] += 1
    app_driver._save_state(run_dir, state)
    before = state_snapshot(run_dir)
    outcome = submit_app_episode_result(run_dir, relation_result(request))
    assert outcome.commit_outcome.accepted is False
    assert "stale selected-node revision" in outcome.commit_outcome.rejection_reason
    assert state_snapshot(run_dir) == before


def test_stale_relation_graph_revision_rejected_without_ledger_mutation(tmp_path):
    run_dir = make_duplicate_gate_run(tmp_path)
    request = next_app_episode(run_dir).request
    state = app_driver.load_app_run(run_dir)
    state.graph_revision += 1
    app_driver._save_state(run_dir, state)
    before = state_snapshot(run_dir)
    outcome = submit_app_episode_result(run_dir, relation_result(request))
    assert outcome.commit_outcome.accepted is False
    assert "stale graph revision" in outcome.commit_outcome.rejection_reason
    assert state_snapshot(run_dir) == before


def test_equivalent_commit_uses_backend_canonical_merge_and_preserves_provenance(tmp_path):
    nodes = [
        SearchNode(node_id="a", claim="same", score=0.95, evidence=[]),
        SearchNode(
            node_id="b",
            claim=" SAME ",
            score=0.6,
            assumptions=["unique assumption"],
            evidence=["unique evidence"],
            risks=["unique risk"],
            parent_ids=["origin"],
        ),
    ]
    run_dir = make_duplicate_gate_run(tmp_path, nodes=nodes)
    request = next_app_episode(run_dir).request
    outcome = submit_app_episode_result(run_dir, relation_result(request, "equivalent"))
    assert outcome.commit_outcome.accepted is True
    state = app_run_status(run_dir)
    assert state.graph_revision == 2
    assert len(state.relation_ledger) == 1
    assert len(state.merge_applications) == 1
    merge = state.merge_applications[0]
    assert merge.canonical_node_id == "b"  # information completeness outranks score alone
    assert next(node for node in state.nodes if node.node_id == "a").status == "merged"
    canonical = next(node for node in state.nodes if node.node_id == "b")
    assert canonical.evidence == ["unique evidence"]
    assert merge.source_node_revisions == {"a": 0, "b": 0}
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"
    assert app_run_status(run_dir).provisional_synthesis_selection.selected_node_ids == ["b"]


@pytest.mark.parametrize("relation_type", ["complementary", "independent"])
def test_nonmerge_relations_preserve_nodes_and_permit_readiness(tmp_path, relation_type):
    nodes = [
        SearchNode(node_id="a", claim="route A", evidence=["shared"], score=0.8),
        SearchNode(node_id="b", claim="route B", evidence=["shared"], score=0.79),
    ]
    run_dir = make_duplicate_gate_run(tmp_path, nodes=nodes)
    request = next_app_episode(run_dir).request
    before_revisions = dict(app_run_status(run_dir).node_revisions)
    submit = submit_app_episode_result(run_dir, relation_result(request, relation_type))
    assert submit.commit_outcome.accepted
    state = app_run_status(run_dir)
    assert state.graph_revision == 1
    assert state.node_revisions == before_revisions
    assert all(node.status == "frontier" for node in state.nodes)
    assert not state.merge_applications
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"
    assert set(app_run_status(run_dir).provisional_synthesis_selection.selected_node_ids) == {"a", "b"}


def test_material_conflict_is_preserved_as_explicit_disclosure_obligation(tmp_path):
    nodes = [
        SearchNode(node_id="a", claim="condition is sufficient", evidence=["paper"], score=0.8),
        SearchNode(node_id="b", claim="condition is not sufficient", evidence=["paper"], score=0.79),
    ]
    run_dir = make_duplicate_gate_run(tmp_path, nodes=nodes)
    grant = next_app_episode(run_dir)
    assert grant.request.role == "relation"
    assert app_run_status(run_dir).synthesis_readiness.ready is False
    submit_app_episode_result(run_dir, relation_result(grant.request, "conflict", disclosure_required=False))
    state = app_run_status(run_dir)
    assert state.relation_ledger[0].relation_type == "conflict"
    assert state.relation_ledger[0].disclosure_required is True
    assert all(node.status == "frontier" for node in state.nodes)
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"
    readiness = app_run_status(run_dir).synthesis_readiness
    assert readiness.unresolved_material_conflicts == []
    assert readiness.disclosure_required_conflicts == [state.relation_ledger[0].relation_record_id]


def test_nonmaterial_unresolved_candidate_does_not_block_readiness():
    candidate = RelationCandidate(
        candidate_id="c",
        left_node_id="a",
        right_node_id="b",
        left_node_revision=0,
        right_node_revision=0,
        candidate_reason="embedding_close",
        scheduling_class="enrichment",
        priority="medium",
        material_to_synthesis=False,
        created_from_graph_revision=0,
    )
    readiness = evaluate_synthesis_readiness(
        graph_revision=0,
        provisional_selected_node_ids=["a"],
        candidates=[candidate],
        relation_ledger=[],
        merge_applications=[],
        evaluated_at="2026-01-01T00:00:00+00:00",
    )
    assert readiness.ready is True
    assert readiness.blocking_candidate_ids == []
    assert readiness.unresolved_nonblocking_candidates == ["c"]


def test_selected_exact_duplicate_blocks_but_nonselected_duplicate_does_not():
    candidate = RelationCandidate(
        candidate_id="duplicate",
        left_node_id="a",
        right_node_id="b",
        left_node_revision=0,
        right_node_revision=0,
        candidate_reason="exact_duplicate",
        scheduling_class="blocking",
        priority="critical",
        material_to_synthesis=True,
        created_from_graph_revision=0,
    )
    selected = evaluate_synthesis_readiness(
        graph_revision=0,
        provisional_selected_node_ids=["a", "b"],
        candidates=[candidate],
        relation_ledger=[],
        merge_applications=[],
        evaluated_at="2026-01-01T00:00:00+00:00",
    )
    nonselected = evaluate_synthesis_readiness(
        graph_revision=0,
        provisional_selected_node_ids=["a"],
        candidates=[candidate],
        relation_ledger=[],
        merge_applications=[],
        evaluated_at="2026-01-01T00:00:00+00:00",
    )
    assert selected.ready is False
    assert selected.blocking_candidate_ids == ["duplicate"]
    assert selected.duplicate_groups == [["a", "b"]]
    assert nonselected.ready is True


def test_confirmed_unapplied_merge_blocks_readiness(tmp_path):
    run_dir = make_duplicate_gate_run(tmp_path)
    request = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, relation_result(request, "equivalent"))
    state = app_run_status(run_dir)
    record = state.relation_ledger[0]
    readiness = evaluate_synthesis_readiness(
        graph_revision=state.graph_revision,
        provisional_selected_node_ids=[record.left_node_id, record.right_node_id],
        candidates=state.relation_candidates,
        relation_ledger=state.relation_ledger,
        merge_applications=[],
        evaluated_at="2026-01-01T00:00:00+00:00",
    )
    assert readiness.ready is False
    assert record.candidate_id in readiness.blocking_candidate_ids


def test_material_conflict_requires_resolution_or_disclosure(tmp_path):
    nodes = [
        SearchNode(node_id="a", claim="yes", evidence=["shared"], score=0.8),
        SearchNode(node_id="b", claim="no", evidence=["shared"], score=0.79),
    ]
    run_dir = make_duplicate_gate_run(tmp_path, nodes=nodes)
    request = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, relation_result(request, "conflict"))
    state = app_run_status(run_dir)
    disclosed = state.relation_ledger[0]
    unresolved = disclosed.model_copy(update={"disclosure_required": False})
    blocked = evaluate_synthesis_readiness(
        graph_revision=state.graph_revision,
        provisional_selected_node_ids=["a", "b"],
        candidates=state.relation_candidates,
        relation_ledger=[unresolved],
        merge_applications=[],
        evaluated_at="2026-01-01T00:00:00+00:00",
    )
    ready = evaluate_synthesis_readiness(
        graph_revision=state.graph_revision,
        provisional_selected_node_ids=["a", "b"],
        candidates=state.relation_candidates,
        relation_ledger=[disclosed],
        merge_applications=[],
        evaluated_at="2026-01-01T00:00:00+00:00",
    )
    assert blocked.ready is False
    assert blocked.unresolved_material_conflicts == [unresolved.relation_record_id]
    assert ready.ready is True
    assert ready.disclosure_required_conflicts == [disclosed.relation_record_id]


def test_legacy_persisted_terminal_is_sticky_and_marked_unchecked(tmp_path):
    run_dir = tmp_path / "legacy"
    create_app_run(run_dir, spec(), [SearchNode(node_id="a", claim="A")])
    state = app_driver.load_app_run(run_dir)
    state.controller_action = "ready_for_synthesis"
    state.synthesis_readiness = None
    app_driver._save_state(run_dir, state)
    before = (run_dir / "app_run_state.json").read_text(encoding="utf-8")
    outcome = next_app_episode(run_dir)
    assert outcome.controller_action == "ready_for_synthesis"
    assert app_run_status(run_dir).relation_readiness_status == "legacy_unchecked"
    assert (run_dir / "app_run_state.json").read_text(encoding="utf-8") == before


def judge_result(request):
    output = JudgeEpisodeOutput(
        observations=[
            JudgeObservation(node_id=node_id, score=0.8, reasoning="worth exploring", risks=[])
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
        runtime_diagnostics=diagnostics(),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def executor_result(request, *, child_id, claim, evidence):
    output = ExecutorEpisodeOutput(
        nodes=[
            ExecutorNodeCandidate(
                node_id=child_id,
                claim=claim,
                evidence=list(evidence),
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
        runtime_diagnostics=diagnostics(),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )


def drive_two_executor_children(run_dir, *, claims, evidence):
    judge = next_app_episode(run_dir).request
    assert judge.role == "judge"
    submit_app_episode_result(run_dir, judge_result(judge))
    children = []
    for index in range(2):
        executor = next_app_episode(run_dir, embedding_provider=HashEmbeddingProvider(dim=8)).request
        assert executor.role == "executor"  # positive grants precede Relation scheduling
        child_id = f"child-{index}"
        submit_app_episode_result(
            run_dir,
            executor_result(executor, child_id=child_id, claim=claims[index], evidence=evidence[index]),
        )
        children.append(child_id)
    return children


def test_end_to_end_judge_executor_relation_equivalent_to_ready(tmp_path):
    run_dir = tmp_path / "e2e-equivalent"
    create_app_run(
        run_dir,
        spec(cap=2, pair_cap=2),
        [SearchNode(node_id="p0", claim="parent A"), SearchNode(node_id="p1", claim="parent B")],
    )
    children = drive_two_executor_children(
        run_dir,
        claims=["same conclusion", " SAME   CONCLUSION "],
        evidence=[["e1"], ["e2"]],
    )
    assert app_run_status(run_dir).controller_iteration == 1
    relation = next_app_episode(run_dir)
    assert relation.request.role == "relation"
    submit_app_episode_result(run_dir, relation_result(relation.request, "equivalent"))
    terminal = next_app_episode(run_dir)
    assert terminal.controller_action == "ready_for_synthesis"
    state = app_run_status(run_dir)
    assert sum(node.status != "merged" for node in state.nodes if node.node_id in children) == 1
    assert state.controller_iteration == 1


def test_end_to_end_judge_executor_material_conflict_to_disclosed_ready(tmp_path):
    run_dir = tmp_path / "e2e-conflict"
    create_app_run(
        run_dir,
        spec(cap=2, pair_cap=2),
        [SearchNode(node_id="p0", claim="parent A"), SearchNode(node_id="p1", claim="parent B")],
    )
    drive_two_executor_children(
        run_dir,
        claims=["condition is sufficient", "condition is not sufficient"],
        evidence=[["shared source"], ["shared source"]],
    )
    relation = next_app_episode(run_dir)
    assert relation.request.role == "relation"
    assert app_run_status(run_dir).synthesis_readiness.ready is False
    submit_app_episode_result(run_dir, relation_result(relation.request, "conflict"))
    terminal = next_app_episode(run_dir)
    assert terminal.controller_action == "ready_for_synthesis"
    readiness = app_run_status(run_dir).synthesis_readiness
    assert readiness.disclosure_required_conflicts
    assert readiness.ready is True


def test_relation_telemetry_is_coarse_and_contains_no_hidden_topology(tmp_path):
    run_dir = make_duplicate_gate_run(tmp_path)
    request = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, relation_result(request, "equivalent"))
    next_app_episode(run_dir)
    events = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    relation_events = [event for event in events if event["role"] == "relation"]
    assert {event["event_type"] for event in relation_events}.issuperset(
        {"relation_episode_granted", "relation_observations_committed", "merge_applied"}
    )
    assert all(event["usage_source"] == "unavailable" for event in relation_events)
    committed = next(event for event in events if event["event_type"] == "relation_observations_committed")
    assert committed["selected_pair_count"] == 1
    assert committed["equivalent_count"] == 1
    serialized = json.dumps(events)
    assert "hidden_reasoning" not in serialized
    assert "subagent_names" not in serialized


def test_writing_relation_result_file_alone_cannot_mutate_ledger(tmp_path):
    run_dir = make_duplicate_gate_run(tmp_path)
    request = next_app_episode(run_dir).request
    before = state_snapshot(run_dir)
    result_path = run_dir / "episodes" / request.episode_id / request.attempt_id / "result.json"
    result_path.write_text(relation_result(request).model_dump_json(indent=2), encoding="utf-8")
    assert state_snapshot(run_dir) == before


def conflict_nodes(count=8):
    return [
        SearchNode(
            node_id=f"n{i}",
            claim=f"material conclusion {i}",
            evidence=["shared source"],
            score=0.9 - i * 0.01,
        )
        for i in range(count)
    ]


def test_complete_blocking_inventory_covers_all_28_pairs_across_bounded_episodes(tmp_path):
    run_dir = tmp_path / "blocking-28"
    create_app_run(
        run_dir,
        spec(pair_cap=3, enrichment_cap=0),
        conflict_nodes(),
        run_id="blocking-28",
    )
    force_stop_intent(run_dir)
    first = next_app_episode(run_dir)
    assert first.request.role == "relation"
    assert len(first.request.relation_payload.candidate_pairs) == 3
    state = app_run_status(run_dir)
    blockers = [item for item in state.relation_candidates if item.scheduling_class == "blocking"]
    assert len(blockers) == 28
    assert state.synthesis_readiness.blocking_inventory_complete is True
    assert state.synthesis_readiness.blocking_pair_count == 28

    granted_pairs = []
    current = first
    while current.controller_action == "episode_required":
        granted_pairs.extend(
            (pair.left.node_id, pair.right.node_id)
            for pair in current.request.relation_payload.candidate_pairs
        )
        submit_app_episode_result(run_dir, relation_result(current.request, "conflict"))
        current = next_app_episode(run_dir)

    assert current.controller_action == "ready_for_synthesis"
    assert len(granted_pairs) == 28
    assert len(set(granted_pairs)) == 28
    assert len(app_run_status(run_dir).relation_ledger) == 28
    readiness = app_run_status(run_dir).synthesis_readiness
    assert readiness.blocking_pair_count == 28
    assert readiness.resolved_blocking_pair_count == 28
    assert readiness.unresolved_blocking_pair_count == 0
    assert readiness.ready is True


def test_all_selected_duplicate_pairs_are_inventoried_before_any_merge(tmp_path):
    run_dir = tmp_path / "duplicates-28"
    nodes = [SearchNode(node_id=f"n{i}", claim="same", score=0.9 - i * 0.01) for i in range(8)]
    create_app_run(run_dir, spec(pair_cap=3, enrichment_cap=0), nodes)
    force_stop_intent(run_dir)
    grant = next_app_episode(run_dir)
    state = app_run_status(run_dir)
    blockers = [item for item in state.relation_candidates if item.scheduling_class == "blocking"]
    assert len(blockers) == 28
    assert all(item.candidate_reason == "exact_duplicate" for item in blockers)
    submit_app_episode_result(run_dir, relation_result(grant.request, "equivalent"))
    next_app_episode(run_dir)
    state = app_run_status(run_dir)
    assert len(state.provisional_synthesis_selection.selected_node_ids) == 5
    assert all(
        candidate.status == "invalidated"
        for candidate in state.relation_candidates
        if any(
            node.status == "merged" and node.node_id in (candidate.left_node_id, candidate.right_node_id)
            for node in state.nodes
        )
        and candidate.status != "resolved"
    )


def test_enrichment_generation_filters_known_pairs_before_window_truncation():
    nodes = [SearchNode(node_id=f"n{i}", claim=f"claim {i}", score=0.8) for i in range(8)]
    revisions = {node.node_id: 0 for node in nodes}
    first = generate_relation_enrichment_candidates(
        nodes,
        node_revisions=revisions,
        graph_revision=1,
        provisional_synthesis_node_ids=list(revisions),
        existing=[],
        relation_ledger=[],
        max_candidates=16,
    )
    known = [
        item.model_copy(update={"status": "resolved", "resolved_relation_record_id": f"r{i}"})
        for i, item in enumerate(first)
    ]
    second = generate_relation_enrichment_candidates(
        nodes,
        node_revisions=revisions,
        graph_revision=2,
        provisional_synthesis_node_ids=list(revisions),
        existing=known,
        relation_ledger=[],
        max_candidates=16,
    )
    assert len(first) == 16
    assert len(second) == 12
    assert set(item.candidate_id for item in first).isdisjoint(
        item.candidate_id for item in second
    )


def test_relation_identity_ignores_graph_revision_but_tracks_node_revision():
    nodes = [SearchNode(node_id="a", claim="A", score=0.8), SearchNode(node_id="b", claim="B", score=0.8)]
    first = generate_relation_enrichment_candidates(
        nodes,
        node_revisions={"a": 0, "b": 0},
        graph_revision=1,
        provisional_synthesis_node_ids=["a", "b"],
        existing=[],
        relation_ledger=[],
    )[0]
    graph_only = generate_relation_enrichment_candidates(
        nodes,
        node_revisions={"a": 0, "b": 0},
        graph_revision=9,
        provisional_synthesis_node_ids=["a", "b"],
        existing=[],
        relation_ledger=[],
    )[0]
    node_changed = generate_relation_enrichment_candidates(
        nodes,
        node_revisions={"a": 1, "b": 0},
        graph_revision=10,
        provisional_synthesis_node_ids=["a", "b"],
        existing=[],
        relation_ledger=[],
    )[0]
    assert first.candidate_id == graph_only.candidate_id
    assert first.candidate_id != node_changed.candidate_id


def test_invalidated_candidates_do_not_occupy_enrichment_window():
    nodes = [SearchNode(node_id=f"n{i}", claim=f"claim {i}", score=0.8) for i in range(8)]
    revisions = {node.node_id: 0 for node in nodes}
    first = generate_relation_enrichment_candidates(
        nodes,
        node_revisions=revisions,
        graph_revision=1,
        provisional_synthesis_node_ids=list(revisions),
        existing=[],
        relation_ledger=[],
        max_candidates=16,
    )
    invalidated = [item.model_copy(update={"status": "invalidated"}) for item in first]
    regenerated = generate_relation_enrichment_candidates(
        nodes,
        node_revisions=revisions,
        graph_revision=2,
        provisional_synthesis_node_ids=list(revisions),
        existing=invalidated,
        relation_ledger=[],
        max_candidates=16,
    )
    assert [item.candidate_id for item in regenerated] == [item.candidate_id for item in first]


def test_nonselected_unrelated_pairs_are_not_scheduled_for_enrichment():
    nodes = [
        SearchNode(node_id="a", claim="selected A", score=0.8),
        SearchNode(node_id="b", claim="selected B", score=0.8),
        SearchNode(node_id="x", claim="unrelated X", score=0.8),
        SearchNode(node_id="y", claim="unrelated Y", score=0.8),
    ]
    candidates = generate_relation_enrichment_candidates(
        nodes,
        node_revisions={node.node_id: 0 for node in nodes},
        graph_revision=1,
        provisional_synthesis_node_ids=["a", "b"],
        existing=[],
        relation_ledger=[],
    )
    assert [(item.left_node_id, item.right_node_id) for item in candidates] == [("a", "b")]


def make_enrichment_run(tmp_path, *, budget=3, pair_cap=2, node_count=5):
    run_dir = tmp_path / f"enrichment-{budget}-{pair_cap}-{node_count}"
    nodes = [
        SearchNode(node_id=f"n{i}", claim=f"distinct claim {i}", score=0.8)
        for i in range(node_count)
    ]
    create_app_run(
        run_dir,
        spec(pair_cap=pair_cap, enrichment_cap=budget),
        nodes,
        run_id="enrichment-run",
    )
    force_stop_intent(run_dir)
    return run_dir


def test_zero_enrichment_budget_goes_directly_terminal_after_blockers_clear(tmp_path):
    run_dir = make_enrichment_run(tmp_path, budget=0)
    outcome = next_app_episode(run_dir)
    assert outcome.controller_action == "ready_for_synthesis"
    readiness = app_run_status(run_dir).synthesis_readiness
    assert readiness.ready is True
    assert readiness.enrichment_pairs_remaining == 0


def test_run_level_enrichment_budget_is_bounded_across_episodes_and_restart(tmp_path):
    run_dir = make_enrichment_run(tmp_path, budget=3, pair_cap=2)
    grants = []
    while True:
        outcome = next_app_episode(run_dir)
        if outcome.controller_action != "episode_required":
            assert outcome.controller_action == "ready_for_synthesis"
            break
        assert {pair.scheduling_class for pair in outcome.request.relation_payload.candidate_pairs} == {
            "enrichment"
        }
        grants.append(len(outcome.request.relation_payload.candidate_pairs))
        submit_app_episode_result(run_dir, relation_result(outcome.request, "independent"))
        app_driver.load_app_run(run_dir)  # explicit restart/reload boundary
    assert grants == [2, 1]
    state = app_run_status(run_dir)
    assert len([record for record in state.relation_ledger if record.scheduling_class == "enrichment"]) == 3
    assert state.synthesis_readiness.enrichment_pairs_committed == 3
    assert state.synthesis_readiness.enrichment_pairs_remaining == 0


def test_failed_enrichment_attempt_and_retry_consume_only_successful_pairs(tmp_path):
    run_dir = make_enrichment_run(tmp_path, budget=3, pair_cap=2)
    first = next_app_episode(run_dir, runtime_limits=RuntimeLimits(max_retries=1)).request
    fail_app_episode(run_dir, first.episode_id, first.attempt_id, "retry")
    assert app_driver._relation_enrichment_pairs_committed(app_run_status(run_dir)) == 0
    retry = retry_app_episode(run_dir, first.episode_id)
    submit_app_episode_result(run_dir, relation_result(retry.request, "independent"))
    state = app_run_status(run_dir)
    assert app_driver._relation_enrichment_pairs_committed(state) == len(
        retry.request.relation_payload.candidate_pairs
    )


@pytest.mark.parametrize("transition", ["cancelled", "expired"])
def test_cancelled_or_expired_enrichment_attempt_does_not_consume_budget(
    monkeypatch, tmp_path, transition
):
    run_dir = make_enrichment_run(tmp_path, budget=3, pair_cap=1)
    request = next_app_episode(
        run_dir, runtime_limits=RuntimeLimits(wall_clock_seconds=1, max_retries=1)
    ).request
    if transition == "cancelled":
        cancel_app_episode(run_dir, request.episode_id, request.attempt_id, "cancel")
    else:
        now = app_driver._now()
        monkeypatch.setattr(app_driver, "_now", lambda: now + timedelta(seconds=5))
        next_app_episode(run_dir)
    state = app_run_status(run_dir)
    assert app_driver._relation_enrichment_pairs_committed(state) == 0
    ledger = json.loads(
        (run_dir / "relations" / "relation_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["enrichment_pairs_committed"] == 0
    assert ledger["enrichment_pairs_remaining"] == 3


def test_blocking_candidates_are_always_granted_before_enrichment(tmp_path):
    run_dir = tmp_path / "blocking-first"
    nodes = [
        SearchNode(node_id="a", claim="yes", evidence=["shared"], score=0.8),
        SearchNode(node_id="b", claim="no", evidence=["shared"], score=0.8),
        SearchNode(node_id="c", claim="other", evidence=["different"], score=0.8),
    ]
    create_app_run(run_dir, spec(enrichment_cap=3), nodes)
    force_stop_intent(run_dir)
    request = next_app_episode(run_dir).request
    assert {pair.scheduling_class for pair in request.relation_payload.candidate_pairs} == {"blocking"}


@pytest.mark.parametrize("relation_type", ["equivalent", "complementary", "conflict", "independent"])
def test_enrichment_preserves_the_four_relation_semantics(tmp_path, relation_type):
    run_dir = make_enrichment_run(tmp_path, budget=1, pair_cap=1, node_count=2)
    request = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, relation_result(request, relation_type))
    state = app_run_status(run_dir)
    assert state.relation_ledger[0].relation_type == relation_type
    if relation_type == "equivalent":
        assert len(state.merge_applications) == 1
    else:
        assert state.merge_applications == []
    if relation_type == "conflict":
        assert state.relation_ledger[0].disclosure_required is True
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"


def test_discriminator_proposal_is_persisted_but_never_scheduled(tmp_path):
    run_dir = make_enrichment_run(tmp_path, budget=1, pair_cap=1, node_count=2)
    request = next_app_episode(run_dir).request
    raw = relation_result(request, "conflict").model_dump(mode="json")
    raw["structured_output"]["observations"][0]["discriminator_task_proposal"] = {
        "task_type": "formal_derivation",
        "objective": "compare assumptions",
        "rationale": "preserve a future research question",
        "material_to_synthesis": True,
    }
    output = RelationEpisodeOutput.model_validate(raw["structured_output"])
    raw["output_hash"] = compute_output_hash(output, request.output_schema_version)
    assert submit_app_episode_result(run_dir, raw).commit_outcome.accepted
    state = app_run_status(run_dir)
    assert state.relation_ledger[0].observation.discriminator_task_proposal is not None
    outcome = next_app_episode(run_dir)
    assert outcome.controller_action == "ready_for_synthesis"
    assert outcome.request is None


def test_relation_inventory_and_enrichment_telemetry_are_auditable(tmp_path):
    run_dir = make_enrichment_run(tmp_path, budget=1, pair_cap=1, node_count=2)
    request = next_app_episode(run_dir).request
    submit_app_episode_result(run_dir, relation_result(request, "independent"))
    next_app_episode(run_dir)
    events = EpisodeEventLog(run_dir / "episode_events.jsonl").read_events()
    types = {event["event_type"] for event in events}
    assert {
        "relation_blocking_inventory_evaluated",
        "relation_blocking_inventory_completed",
        "relation_enrichment_granted",
        "relation_enrichment_committed",
        "relation_enrichment_budget_exhausted",
    }.issubset(types)
    inventory = next(event for event in events if event["event_type"] == "relation_blocking_inventory_evaluated")
    assert inventory["blocking_inventory_complete"] is True
    assert inventory["role"] == "relation"
    assert inventory["usage_source"] == "unavailable"
