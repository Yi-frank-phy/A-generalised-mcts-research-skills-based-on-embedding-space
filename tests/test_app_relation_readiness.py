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
    request_app_synthesis,
    retry_app_episode,
    submit_app_episode_result,
)
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.context_envelope import semantic_embedding_text
from dte_backend.episode_adapter import build_relation_episode_request
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
from dte_backend.merge import (
    apply_relation_equivalent_merge,
    resolve_merge_aliases,
    validate_alias_projected_node_ancestry,
    validate_merge_application_relation_provenance,
)
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode, SynthesisControlRequest
from dte_backend.relation_candidates import (
    generate_relation_candidates,
    generate_relation_enrichment_candidates,
    promote_pending_enrichment_materiality,
    select_node_disjoint_relation_batch,
)
from dte_backend.relation_models import (
    MergeApplicationRecord,
    RelationCandidate,
    RelationEpisodeOutput,
    RelationObservation,
    RelationRecord,
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
    assert state.controller_iteration >= state.spec.budget.max_iterations


def create_controller_checkpoint_run(run_dir, run_spec, nodes, *, run_id="relation-run"):
    """Reach a real Judge/controller checkpoint for gate-only tests."""

    producer_nodes = [
        node.model_copy(
            update={
                "local_embedding": None,
                "judge_reasoning": None,
                "judge_risks": [],
                "judge_uncertainty_evidence": [],
                "judge_result_provenance": None,
                "score": None,
                "density": None,
                "uncertainty": None,
                "ucb_score": None,
                "expansion_budget": 0,
                "status": "frontier",
            },
            deep=True,
        )
        for node in nodes
    ]
    create_app_run(run_dir, run_spec, producer_nodes, run_id=run_id)
    source_by_id = {node.node_id: node for node in nodes}
    while any(node.score is None for node in app_run_status(run_dir).nodes):
        judge = next_app_episode(run_dir).request
        assert judge.role == "judge"
        judge_output = JudgeEpisodeOutput(
            observations=[
                JudgeObservation(
                    node_id=node_id,
                    score=(
                        source_by_id[node_id].score
                        if source_by_id[node_id].score is not None
                        else source_by_id[node_id].confidence
                    ),
                    reasoning="trusted gate fixture Judge observation",
                    risks=[],
                )
                for node_id in judge.selected_node_revisions
            ]
        )
        submit_app_episode_result(
            run_dir,
            EpisodeResult(
                episode_id=judge.episode_id,
                attempt_id=judge.attempt_id,
                run_id=judge.run_id,
                role="judge",
                input_graph_revision=judge.input_graph_revision,
                selected_node_revisions=judge.selected_node_revisions,
                status="completed",
                structured_output=judge_output,
                runtime_diagnostics=diagnostics(),
                output_hash=compute_output_hash(judge_output, judge.output_schema_version),
                schema_version=judge.output_schema_version,
            ),
        )

    desired_embeddings = {
        semantic_embedding_text(producer): list(source.local_embedding)
        for producer, source in zip(producer_nodes, nodes)
        if source.local_embedding is not None
    }

    class FixtureEmbeddingProvider:
        dim = run_spec.embedding_dimension
        name = "fixture"

        def embed_texts(self, texts):
            fallback = HashEmbeddingProvider(dim=self.dim).embed_texts(texts)
            return [
                list(desired_embeddings.get(text, fallback[index]))
                for index, text in enumerate(texts)
            ]

    state = app_driver.load_app_run(run_dir)
    app_driver._progress_controller(
        run_dir,
        state,
        embedding_provider=FixtureEmbeddingProvider(),
    )
    state.controller_action = "continue_controller"
    app_driver._save_state(run_dir, state)
    while app_driver._select_executor_parent(app_driver.load_app_run(run_dir)) is not None:
        outcome = next_app_episode(run_dir)
        assert outcome.request is not None and outcome.request.role == "executor"
        request = outcome.request
        output = ExecutorEpisodeOutput(nodes=[])
        submit_app_episode_result(
            run_dir,
            EpisodeResult(
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
            ),
        )
    assert app_run_status(run_dir).controller_iteration == run_spec.budget.max_iterations


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
    create_controller_checkpoint_run(
        run_dir,
        spec(pair_cap=pair_cap),
        nodes,
        run_id="relation-run",
    )
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
    before_submitted_at = next(
        attempt.submitted_at
        for episode in app_driver.load_app_run(run_dir).episodes
        if episode.episode_id == request.episode_id
        for attempt in episode.attempts
        if attempt.attempt_id == request.attempt_id
    )
    outcome = submit_app_episode_result(run_dir, relation_result(request))
    assert outcome.commit_outcome.accepted is False
    assert state_snapshot(run_dir) == before
    after_submitted_at = next(
        attempt.submitted_at
        for episode in app_driver.load_app_run(run_dir).episodes
        if episode.episode_id == request.episode_id
        for attempt in episode.attempts
        if attempt.attempt_id == request.attempt_id
    )
    assert after_submitted_at == before_submitted_at


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
    state_path = run_dir / "app_run_state.json"
    durable_before = state_path.read_text(encoding="utf-8")
    state = app_driver.load_app_run(run_dir)
    target = getattr(request.relation_payload.candidate_pairs[0], side).node_id
    state.node_revisions[target] += 1
    with pytest.raises(ValueError, match="node revisions disagree"):
        app_driver._save_state(run_dir, state)
    assert state_path.read_text(encoding="utf-8") == durable_before


def test_stale_relation_graph_revision_rejected_without_ledger_mutation(tmp_path):
    run_dir = make_duplicate_gate_run(tmp_path)
    next_app_episode(run_dir)
    state_path = run_dir / "app_run_state.json"
    durable_before = state_path.read_text(encoding="utf-8")
    state = app_driver.load_app_run(run_dir)
    state.graph_revision += 1
    with pytest.raises(ValueError, match="graph revision is not backed"):
        app_driver._save_state(run_dir, state)
    assert state_path.read_text(encoding="utf-8") == durable_before


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
        ),
    ]
    run_dir = make_duplicate_gate_run(tmp_path, nodes=nodes)
    request = next_app_episode(run_dir).request
    revision_before = app_run_status(run_dir).graph_revision
    outcome = submit_app_episode_result(run_dir, relation_result(request, "equivalent"))
    assert outcome.commit_outcome.accepted is True
    state = app_run_status(run_dir)
    assert state.graph_revision == revision_before + 2
    assert len(state.relation_ledger) == 1
    assert len(state.merge_applications) == 1
    merge = state.merge_applications[0]
    assert merge.canonical_node_id == "b"  # information completeness outranks score alone
    assert next(node for node in state.nodes if node.node_id == "a").status == "merged"
    canonical = next(node for node in state.nodes if node.node_id == "b")
    assert canonical.evidence == ["unique evidence"]
    assert merge.source_node_revisions == request.selected_node_revisions
    validate_merge_application_relation_provenance(merge, state.relation_ledger[0])
    bad_revision_provenance = merge.model_copy(
        update={"source_node_revisions": {"a": 1, "b": 0}}
    )
    with pytest.raises(ValueError, match="source revisions disagree"):
        validate_merge_application_relation_provenance(
            bad_revision_provenance,
            state.relation_ledger[0],
        )
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"
    assert app_run_status(run_dir).provisional_synthesis_selection.selected_node_ids == ["b"]


def test_parent_child_equivalent_merge_removes_internal_parent_links(tmp_path):
    nodes = [
        SearchNode(
            node_id="a",
            claim="same",
            score=0.9,
            evidence=["canonical information"],
        ),
        SearchNode(
            node_id="b",
            claim=" SAME ",
            score=0.8,
            parent_ids=["a"],
        ),
    ]
    run_dir = make_duplicate_gate_run(tmp_path, nodes=nodes)
    request = next_app_episode(run_dir).request
    assert submit_app_episode_result(
        run_dir, relation_result(request, "equivalent")
    ).commit_outcome.accepted

    state = app_run_status(run_dir)
    merge = state.merge_applications[0]
    canonical = next(
        node for node in state.nodes if node.node_id == merge.canonical_node_id
    )
    assert canonical.status in {"frontier", "closed"}
    assert canonical.parent_ids == []
    assert not set(state.merge_applications[0].source_node_ids).intersection(canonical.parent_ids)


def test_chained_equivalent_merge_cleans_alias_projected_self_parent_atomically():
    nodes = [
        SearchNode(node_id="a", claim="same"),
        SearchNode(node_id="b", claim="same", rationale="first canonical"),
        SearchNode(
            node_id="c",
            claim="same",
            parent_ids=["a"],
            evidence=["second canonical"],
        ),
    ]
    revisions = {node.node_id: 0 for node in nodes}
    first = apply_relation_equivalent_merge(
        nodes,
        revisions,
        source_node_ids=["a", "b"],
        relation_record_id="relation-a-b",
        applied_graph_revision=2,
        applied_at="2026-01-01T00:00:00+00:00",
    )
    graph = EpisodeGraph(
        nodes=nodes,
        revision=2,
        node_revisions=revisions,
        merge_applications=[first],
    )
    candidate = RelationCandidate(
        candidate_id="candidate-b-c",
        left_node_id="b",
        right_node_id="c",
        left_node_revision=revisions["b"],
        right_node_revision=revisions["c"],
        candidate_reason="exact_duplicate",
        scheduling_class="blocking",
        priority="critical",
        material_to_synthesis=True,
        created_from_graph_revision=2,
    )
    request = build_relation_request_for_test(graph, [candidate], pair_cap=1)
    grant_relation_candidates(graph, [candidate], request)

    outcome = commit_episode_result(graph, request, relation_result(request, "equivalent"))

    assert outcome.accepted is True
    validate_alias_projected_node_ancestry(graph.nodes, graph.merge_applications)
    aliases = resolve_merge_aliases(
        graph.merge_applications,
        committed_node_ids={node.node_id for node in graph.nodes},
    )
    canonical = next(
        node
        for node in graph.nodes
        if node.node_id == graph.merge_applications[-1].canonical_node_id
    )
    assert canonical.node_id not in [aliases.get(parent, parent) for parent in canonical.parent_ids]


def test_chained_equivalent_merge_rejects_alias_projected_cycle_atomically():
    nodes = [
        SearchNode(node_id="a", claim="same"),
        SearchNode(node_id="b", claim="same", rationale="first canonical"),
        SearchNode(node_id="c", claim="same", parent_ids=["x"], evidence=["more"]),
        SearchNode(node_id="x", claim="bridge", parent_ids=["b"]),
    ]
    revisions = {node.node_id: 0 for node in nodes}
    first = apply_relation_equivalent_merge(
        nodes,
        revisions,
        source_node_ids=["a", "b"],
        relation_record_id="relation-a-b",
        applied_graph_revision=2,
        applied_at="2026-01-01T00:00:00+00:00",
    )
    graph = EpisodeGraph(
        nodes=nodes,
        revision=2,
        node_revisions=revisions,
        merge_applications=[first],
    )
    candidate = RelationCandidate(
        candidate_id="candidate-b-c",
        left_node_id="b",
        right_node_id="c",
        left_node_revision=revisions["b"],
        right_node_revision=revisions["c"],
        candidate_reason="exact_duplicate",
        scheduling_class="blocking",
        priority="critical",
        material_to_synthesis=True,
        created_from_graph_revision=2,
    )
    request = build_relation_request_for_test(graph, [candidate], pair_cap=1)
    grant_relation_candidates(graph, [candidate], request)
    before = graph.snapshot()

    outcome = commit_episode_result(graph, request, relation_result(request, "equivalent"))

    assert outcome.accepted is False
    assert "merge-projected ancestry contains a cycle" in outcome.rejection_reason
    assert graph.snapshot() == before


@pytest.mark.parametrize("relation_type", ["complementary", "independent"])
def test_nonmerge_relations_preserve_nodes_and_permit_readiness(tmp_path, relation_type):
    nodes = [
        SearchNode(node_id="a", claim="route A", evidence=["shared"], score=0.8),
        SearchNode(node_id="b", claim="route B", evidence=["shared"], score=0.79),
    ]
    run_dir = make_duplicate_gate_run(tmp_path, nodes=nodes)
    request = next_app_episode(run_dir).request
    before_state = app_run_status(run_dir)
    before_revisions = dict(before_state.node_revisions)
    before_statuses = {node.node_id: node.status for node in before_state.nodes}
    revision_before = before_state.graph_revision
    submit = submit_app_episode_result(run_dir, relation_result(request, relation_type))
    assert submit.commit_outcome.accepted
    state = app_run_status(run_dir)
    assert state.graph_revision == revision_before + 1
    assert state.node_revisions == before_revisions
    assert {node.node_id: node.status for node in state.nodes} == before_statuses
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
    statuses_before = {
        node.node_id: node.status for node in app_run_status(run_dir).nodes
    }
    submit_app_episode_result(run_dir, relation_result(grant.request, "conflict", disclosure_required=False))
    state = app_run_status(run_dir)
    assert state.relation_ledger[0].relation_type == "conflict"
    assert state.relation_ledger[0].disclosure_required is True
    assert {node.node_id: node.status for node in state.nodes} == statuses_before
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"
    readiness = app_run_status(run_dir).synthesis_readiness
    assert readiness.unresolved_material_conflicts == []
    assert readiness.disclosure_required_conflicts == [state.relation_ledger[0].relation_record_id]


def test_pending_enrichment_becomes_material_when_merge_expands_selection(tmp_path):
    run_dir = tmp_path / "pending-materiality-promotion"
    nodes = [
        SearchNode(
            node_id="a",
            claim="route A",
            score=0.99,
            local_embedding=[1.0] + [0.0] * 7,
        ),
        SearchNode(node_id="c", claim="duplicate", score=0.89),
        SearchNode(
            node_id="d",
            claim=" DUPLICATE ",
            score=0.79,
            evidence=["richer canonical"],
        ),
        SearchNode(node_id="e", claim="e", score=0.69),
        SearchNode(node_id="f", claim="f", score=0.59),
        SearchNode(node_id="g", claim="g", score=0.49),
        SearchNode(node_id="h", claim="h", score=0.39),
        SearchNode(node_id="i", claim="i", score=0.29),
        SearchNode(
            node_id="b",
            claim="route B",
            score=0.19,
            parent_ids=["a"],
            local_embedding=[1.0] + [0.0] * 7,
        ),
    ]
    create_controller_checkpoint_run(
        run_dir,
        spec(pair_cap=1, max_iterations=1, enrichment_cap=1),
        nodes,
    )
    force_stop_intent(run_dir)

    blocking = next_app_episode(run_dir).request
    assert {
        blocking.relation_payload.candidate_pairs[0].left.node_id,
        blocking.relation_payload.candidate_pairs[0].right.node_id,
    } == {"c", "d"}
    assert submit_app_episode_result(
        run_dir,
        relation_result(blocking, "equivalent"),
    ).commit_outcome.accepted

    enrichment = next_app_episode(run_dir).request
    pair = enrichment.relation_payload.candidate_pairs[0]
    assert {pair.left.node_id, pair.right.node_id} == {"a", "b"}
    assert pair.scheduling_class == "enrichment"
    assert pair.material_to_synthesis is True
    assert submit_app_episode_result(
        run_dir,
        relation_result(enrichment, "conflict"),
    ).commit_outcome.accepted

    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"
    state = app_run_status(run_dir)
    assert {"a", "b"}.issubset(state.provisional_synthesis_selection.selected_node_ids)
    assert state.synthesis_readiness.disclosure_required_conflicts == [
        next(
            record.relation_record_id
            for record in state.relation_ledger
            if {record.left_node_id, record.right_node_id} == {"a", "b"}
        )
    ]


def test_resolved_nonmaterial_conflict_is_disclosed_if_both_endpoints_later_selected():
    nodes = [
        SearchNode(
            node_id="a",
            claim="route A",
            score=0.8,
            local_embedding=[1.0, 0.0],
        ),
        SearchNode(
            node_id="b",
            claim="route B",
            score=0.7,
            parent_ids=["a"],
            local_embedding=[1.0, 0.0],
        ),
    ]
    candidate = generate_relation_enrichment_candidates(
        nodes,
        node_revisions={"a": 0, "b": 0},
        graph_revision=1,
        provisional_synthesis_node_ids=["a"],
        existing=[],
        relation_ledger=[],
    )[0]
    assert candidate.material_to_synthesis is False
    observation = RelationObservation(
        candidate_id=candidate.candidate_id,
        left_node_id="a",
        right_node_id="b",
        relation_type="conflict",
        confidence=0.9,
        rationale="the routes conflict",
        materiality_assessment="non_material",
        conflict_summary="the routes conflict",
        disclosure_required=False,
    )
    record = RelationRecord(
        relation_record_id="relation-a-b",
        candidate_id=candidate.candidate_id,
        left_node_id="a",
        right_node_id="b",
        relation_type="conflict",
        scheduling_class="enrichment",
        confidence=0.9,
        rationale="the routes conflict",
        material_to_synthesis=False,
        materiality_assessment="non_material",
        observation=observation,
        disclosure_required=False,
        episode_id="episode",
        attempt_id="attempt",
        input_graph_revision=1,
        selected_node_revisions={"a": 0, "b": 0},
        output_hash="hash",
        schema_version="relation-output.v1",
        committed_at="2026-01-01T00:00:00+00:00",
    )
    candidate = candidate.model_copy(
        update={
            "status": "resolved",
            "resolved_relation_record_id": record.relation_record_id,
        }
    )

    readiness = evaluate_synthesis_readiness(
        graph_revision=2,
        provisional_selected_node_ids=["a", "b"],
        candidates=[candidate],
        relation_ledger=[record],
        merge_applications=[],
        evaluated_at="2026-01-01T00:01:00+00:00",
        blocking_inventory_candidate_ids=[],
        enrichment_budget_limit=1,
        enrichment_pairs_committed=1,
    )

    assert readiness.ready is True
    assert readiness.unresolved_material_conflicts == []
    assert readiness.disclosure_required_conflicts == [record.relation_record_id]


def test_materiality_promotion_helper_changes_only_pending_enrichment():
    base = RelationCandidate(
        candidate_id="candidate-a-b",
        left_node_id="a",
        right_node_id="b",
        left_node_revision=0,
        right_node_revision=0,
        candidate_reason="embedding_close",
        scheduling_class="enrichment",
        priority="high",
        material_to_synthesis=False,
        created_from_graph_revision=1,
    )
    resolved = base.model_copy(
        update={
            "candidate_id": "candidate-c-d",
            "left_node_id": "c",
            "right_node_id": "d",
            "status": "resolved",
            "resolved_relation_record_id": "relation-c-d",
        }
    )

    promoted = promote_pending_enrichment_materiality(
        [base, resolved],
        provisional_synthesis_node_ids=["a", "b", "c", "d"],
    )

    assert promoted[0].material_to_synthesis is True
    assert promoted[1].material_to_synthesis is False
    assert base.material_to_synthesis is False

    entropy_only = base.model_copy(update={"candidate_reason": "entropy_plateau"})
    updates = generate_relation_enrichment_candidates(
        [
            SearchNode(node_id="a", claim="A", score=0.9),
            SearchNode(node_id="b", claim="B", score=0.1),
        ],
        node_revisions={"a": 0, "b": 0},
        graph_revision=2,
        provisional_synthesis_node_ids=["a", "b"],
        existing=[entropy_only],
        relation_ledger=[],
        entropy_plateau=False,
    )
    assert [(item.candidate_id, item.material_to_synthesis) for item in updates] == [
        (entropy_only.candidate_id, True)
    ]


def test_enrichment_record_does_not_cover_a_new_blocking_obligation(tmp_path):
    run_dir = tmp_path / "enrichment-promoted-to-blocking"
    nodes = [
        SearchNode(node_id="root", claim="shared committed parent", score=0.1),
        SearchNode(
            node_id="a",
            claim="condition is sufficient",
            evidence=["shared source"],
            score=0.8,
            local_embedding=[1.0] + [0.0] * 7,
            parent_ids=["root"],
        ),
        SearchNode(
            node_id="b",
            claim="condition is not sufficient",
            evidence=["shared source"],
            score=0.7,
            local_embedding=[1.0] + [0.0] * 7,
            parent_ids=["root"],
        ),
    ]
    create_controller_checkpoint_run(run_dir, spec(pair_cap=1, enrichment_cap=1), nodes)
    force_stop_intent(run_dir)
    request_app_synthesis(
        run_dir,
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="first inspect only a",
            scope="node_ids",
            node_ids=["a"],
        ),
    )
    enrichment = next_app_episode(run_dir).request
    first_pair = enrichment.relation_payload.candidate_pairs[0]
    assert first_pair.scheduling_class == "enrichment"
    assert first_pair.material_to_synthesis is False
    submit_app_episode_result(run_dir, relation_result(enrichment, "conflict"))

    request_app_synthesis(
        run_dir,
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="include both now-selected branches",
            scope="node_ids",
            node_ids=["a", "b"],
        ),
    )
    blocking = next_app_episode(run_dir)
    assert blocking.request.role == "relation"
    second_pair = blocking.request.relation_payload.candidate_pairs[0]
    assert second_pair.scheduling_class == "blocking"
    assert second_pair.candidate_reason == "potential_material_conflict"
    assert second_pair.candidate_id != first_pair.candidate_id

    submit_app_episode_result(run_dir, relation_result(blocking.request, "conflict"))
    assert next_app_episode(run_dir).controller_action == "ready_for_synthesis"
    state = app_run_status(run_dir)
    blocking_record = next(
        record for record in state.relation_ledger if record.scheduling_class == "blocking"
    )
    assert state.synthesis_readiness.disclosure_required_conflicts == [
        blocking_record.relation_record_id
    ]


def test_material_conflict_disclosure_survives_later_equivalent_alias_merge(tmp_path):
    run_dir = tmp_path / "conflict-then-alias"
    nodes = [
        SearchNode(node_id="a", claim="route A", score=0.80, local_embedding=[1.0] + [0.0] * 7),
        SearchNode(node_id="b", claim="route A", score=0.79),
        SearchNode(
            node_id="c",
            claim="richer equivalent of A",
            score=0.20,
            local_embedding=[1.0] + [0.0] * 7,
            assumptions=["more complete canonical context"],
        ),
    ]
    create_controller_checkpoint_run(run_dir, spec(pair_cap=1, enrichment_cap=2), nodes)
    force_stop_intent(run_dir)

    conflict = next_app_episode(run_dir).request
    assert [
        (pair.left.node_id, pair.right.node_id)
        for pair in conflict.relation_payload.candidate_pairs
    ] == [("a", "b")]
    submit_app_episode_result(run_dir, relation_result(conflict, "conflict"))
    conflict_record_id = app_run_status(run_dir).relation_ledger[0].relation_record_id

    equivalent = next_app_episode(run_dir).request
    assert [
        (pair.left.node_id, pair.right.node_id)
        for pair in equivalent.relation_payload.candidate_pairs
    ] == [("a", "c")]
    submit_app_episode_result(run_dir, relation_result(equivalent, "equivalent"))

    terminal = next_app_episode(run_dir)
    while terminal.controller_action == "episode_required":
        assert terminal.request.role == "relation"
        submit_app_episode_result(
            run_dir,
            relation_result(terminal.request, "independent"),
        )
        terminal = next_app_episode(run_dir)
    assert terminal.controller_action == "ready_for_synthesis"
    state = app_run_status(run_dir)
    assert next(node for node in state.nodes if node.node_id == "a").status == "merged"
    assert state.merge_applications[-1].canonical_node_id == "c"
    assert state.synthesis_readiness.disclosure_required_conflicts == [conflict_record_id]


def test_targeted_synthesis_follows_equivalent_merge_alias(tmp_path):
    run_dir = tmp_path / "targeted-alias"
    nodes = [
        SearchNode(node_id="root", claim="shared parent", score=0.10),
        SearchNode(
            node_id="a",
            claim="targeted route",
            parent_ids=["root"],
            score=0.80,
            local_embedding=[1.0] + [0.0] * 7,
        ),
        SearchNode(
            node_id="c",
            claim="richer equivalent route",
            parent_ids=["root"],
            score=0.20,
            local_embedding=[1.0] + [0.0] * 7,
            assumptions=["canonical context retained"],
        ),
    ]
    create_controller_checkpoint_run(
        run_dir,
        spec(pair_cap=1, enrichment_cap=1),
        nodes,
    )
    force_stop_intent(run_dir)
    request_app_synthesis(
        run_dir,
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="synthesize the explicitly selected route",
            scope="node_ids",
            node_ids=["a"],
        ),
    )

    relation = next_app_episode(run_dir).request
    assert relation.role == "relation"
    assert [
        (pair.left.node_id, pair.right.node_id)
        for pair in relation.relation_payload.candidate_pairs
    ] == [("a", "c")]
    submit_app_episode_result(run_dir, relation_result(relation, "equivalent"))

    terminal = next_app_episode(run_dir)
    assert terminal.controller_action == "ready_for_synthesis"
    state = app_run_status(run_dir)
    assert state.synthesis_request.node_ids == ["c"]
    assert state.provisional_synthesis_selection.selected_node_ids == ["c"]


def test_load_rejects_resolved_candidate_with_stale_ledger_identity(tmp_path):
    run_dir = make_duplicate_gate_run(tmp_path)
    request = next_app_episode(run_dir).request
    assert submit_app_episode_result(
        run_dir,
        relation_result(request, "equivalent"),
    ).commit_outcome.accepted
    state = app_driver.load_app_run(run_dir)
    record = state.relation_ledger[0]
    record.selected_node_revisions[record.left_node_id] += 1
    # Bypass the validated writer to simulate a legacy/hand-edited artifact.
    (run_dir / "app_run_state.json").write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inconsistent ledger link"):
        app_driver.load_app_run(run_dir)


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


def test_legacy_persisted_terminal_without_audit_record_is_rejected(tmp_path):
    run_dir = tmp_path / "legacy"
    create_app_run(run_dir, spec(), [SearchNode(node_id="a", claim="A")])
    state_path = run_dir / "app_run_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["controller_action"] = "ready_for_synthesis"
    payload["synthesis_readiness"] = None
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal App state lacks"):
        app_driver.load_app_run(run_dir)


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
    final_children_judge = next_app_episode(run_dir)
    assert final_children_judge.request.role == "judge"
    assert set(final_children_judge.request.selected_node_revisions) == set(children)
    submit_app_episode_result(run_dir, judge_result(final_children_judge.request))
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
    children = drive_two_executor_children(
        run_dir,
        claims=["condition is sufficient", "condition is not sufficient"],
        evidence=[["shared source"], ["shared source"]],
    )
    final_children_judge = next_app_episode(run_dir)
    assert final_children_judge.request.role == "judge"
    assert set(final_children_judge.request.selected_node_revisions) == set(children)
    submit_app_episode_result(run_dir, judge_result(final_children_judge.request))
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
    create_controller_checkpoint_run(
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
    create_controller_checkpoint_run(run_dir, spec(pair_cap=3, enrichment_cap=0), nodes)
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


def test_enrichment_node_window_rotates_past_covered_related_nodes():
    selected = [
        SearchNode(
            node_id=f"s{i}",
            claim=f"selected {i}",
            score=1.0 - i * 0.05,
            local_embedding=([1.0, 0.0] if i == 0 else None),
        )
        for i in range(8)
    ]
    related = [
        SearchNode(
            node_id=f"r{i}",
            claim=f"related {i}",
            score=0.4 - i * 0.01,
            parent_ids=["s0"],
            local_embedding=[1.0, 0.0],
        )
        for i in range(5)
    ]
    nodes = selected + related
    revisions = {node.node_id: 0 for node in nodes}
    selected_ids = [node.node_id for node in selected]
    first = generate_relation_enrichment_candidates(
        nodes,
        node_revisions=revisions,
        graph_revision=1,
        provisional_synthesis_node_ids=selected_ids,
        existing=[],
        relation_ledger=[],
        max_candidates=16,
    )
    assert all("r4" not in (item.left_node_id, item.right_node_id) for item in first)
    covered = [
        item.model_copy(update={"status": "resolved", "resolved_relation_record_id": f"r{i}"})
        for i, item in enumerate(first)
    ]

    second = generate_relation_enrichment_candidates(
        nodes,
        node_revisions=revisions,
        graph_revision=2,
        provisional_synthesis_node_ids=selected_ids,
        existing=covered,
        relation_ledger=[],
        max_candidates=16,
    )

    assert any("r4" in (item.left_node_id, item.right_node_id) for item in second)
    assert len(second) <= 16


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
    create_controller_checkpoint_run(
        run_dir,
        spec(pair_cap=pair_cap, enrichment_cap=budget),
        nodes,
        run_id="enrichment-run",
    )
    force_stop_intent(run_dir)
    return run_dir


def relation_candidates_for_pairs(pairs, *, scheduling_class="enrichment"):
    return [
        RelationCandidate(
            candidate_id=f"candidate-{left}-{right}",
            left_node_id=left,
            right_node_id=right,
            left_node_revision=0,
            right_node_revision=0,
            candidate_reason=(
                "potential_material_conflict"
                if scheduling_class == "blocking"
                else "high_score_near_tie"
            ),
            scheduling_class=scheduling_class,
            priority="critical" if scheduling_class == "blocking" else "high",
            material_to_synthesis=scheduling_class == "blocking",
            created_from_graph_revision=0,
        )
        for left, right in pairs
    ]


def build_relation_request_for_test(graph, candidates, *, pair_cap=3):
    return build_relation_episode_request(
        graph,
        candidates,
        run_id="relation-test",
        problem="relation merge safety",
        goal="preserve graph consistency",
        constraints=["Relation is not a verifier"],
        provisional_synthesis_node_ids=[node.node_id for node in graph.nodes],
        max_relation_pairs_per_episode=pair_cap,
    )


def grant_relation_candidates(graph, candidates, request):
    graph.relation_candidates = [
        candidate.model_copy(
            update={
                "status": "granted",
                "granted_episode_id": request.episode_id,
                "granted_attempt_id": request.attempt_id,
            }
        )
        for candidate in candidates
    ]


def test_node_disjoint_batch_skips_overlaps_but_preserves_order_for_later_progression():
    candidates = relation_candidates_for_pairs([("a", "b"), ("b", "c"), ("c", "d")])
    selected = select_node_disjoint_relation_batch(candidates, max_pairs=3)
    assert [(item.left_node_id, item.right_node_id) for item in selected] == [
        ("a", "b"),
        ("c", "d"),
    ]
    assert [item.candidate_id for item in candidates if item not in selected] == ["candidate-b-c"]


def test_node_disjoint_batch_allows_independent_pairs_and_obeys_pair_cap():
    candidates = relation_candidates_for_pairs([("a", "b"), ("c", "d"), ("e", "f")])
    assert len(select_node_disjoint_relation_batch(candidates, max_pairs=3)) == 3
    assert len(select_node_disjoint_relation_batch(candidates, max_pairs=2)) == 2
    assert select_node_disjoint_relation_batch(candidates, max_pairs=0) == []


def test_enrichment_grants_are_node_disjoint_and_obey_remaining_run_budget(tmp_path):
    run_dir = make_enrichment_run(tmp_path, budget=2, pair_cap=5, node_count=6)
    request = next_app_episode(run_dir).request
    granted_node_ids = [
        node_id
        for pair in request.relation_payload.candidate_pairs
        for node_id in (pair.left.node_id, pair.right.node_id)
    ]
    assert len(granted_node_ids) == len(set(granted_node_ids))
    assert len(request.relation_payload.candidate_pairs) == 2


def test_overlapping_enrichment_candidate_can_be_granted_in_a_later_episode(tmp_path):
    run_dir = make_enrichment_run(tmp_path, budget=3, pair_cap=2, node_count=4)
    first = next_app_episode(run_dir).request
    first_pairs = first.relation_payload.candidate_pairs
    assert len(first_pairs) == 2
    submit_app_episode_result(run_dir, relation_result(first, "independent"))
    second = next_app_episode(run_dir).request
    first_nodes = {pair.left.node_id for pair in first_pairs}.union(
        pair.right.node_id for pair in first_pairs
    )
    assert any(
        pair.left.node_id in first_nodes or pair.right.node_id in first_nodes
        for pair in second.relation_payload.candidate_pairs
    )


def test_relation_request_builder_rejects_overlapping_pairs_without_dropping_them():
    graph = EpisodeGraph(nodes=[SearchNode(node_id=node_id, claim=node_id) for node_id in "abc"])
    candidates = relation_candidates_for_pairs([("a", "b"), ("b", "c")])
    with pytest.raises(ValueError, match="candidate pairs must be node-disjoint"):
        build_relation_request_for_test(graph, candidates)


def test_relation_request_builder_accepts_node_disjoint_pairs():
    graph = EpisodeGraph(nodes=[SearchNode(node_id=node_id, claim=node_id) for node_id in "abcd"])
    candidates = relation_candidates_for_pairs([("a", "b"), ("c", "d")])
    request = build_relation_request_for_test(graph, candidates)
    assert [pair.candidate_id for pair in request.relation_payload.candidate_pairs] == [
        candidate.candidate_id for candidate in candidates
    ]


@pytest.mark.parametrize("relation_type", ["equivalent", "complementary", "independent"])
def test_commit_rejects_old_overlapping_relation_request_atomically(relation_type):
    graph = EpisodeGraph(nodes=[SearchNode(node_id=node_id, claim=node_id) for node_id in "abc"])
    candidates = relation_candidates_for_pairs([("a", "b"), ("b", "c")])
    left_request = build_relation_request_for_test(graph, candidates[:1])
    right_request = build_relation_request_for_test(graph, candidates[1:])
    old_payload = left_request.relation_payload.model_copy(
        update={
            "candidate_pairs": [
                left_request.relation_payload.candidate_pairs[0],
                right_request.relation_payload.candidate_pairs[0],
            ]
        }
    )
    old_request = left_request.model_copy(
        update={
            "relation_payload": old_payload,
            "selected_node_revisions": {"a": 0, "b": 0, "c": 0},
        }
    )
    grant_relation_candidates(graph, candidates, old_request)
    before = graph.snapshot()
    outcome = commit_episode_result(graph, old_request, relation_result(old_request, relation_type))
    assert outcome.accepted is False
    assert outcome.rejection_reason == "Relation episode candidate pairs are not node-disjoint"
    assert graph.snapshot() == before


def test_merge_provenance_conflict_rejects_the_whole_relation_commit():
    nodes = [
        SearchNode(node_id="a", claim="canonical A"),
        SearchNode(node_id="b", claim="absorbed B", status="merged"),
        SearchNode(node_id="c", claim="canonical C"),
    ]
    candidate = relation_candidates_for_pairs([("b", "c")])[0]
    graph = EpisodeGraph(
        nodes=nodes,
        revision=2,
        merge_applications=[
            MergeApplicationRecord(
                merge_application_id="merge-b-a",
                relation_record_id="relation-b-a",
                canonical_node_id="a",
                absorbed_node_ids=["b"],
                source_node_ids=["a", "b"],
                source_node_revisions={"a": 0, "b": 0},
                applied_graph_revision=2,
                applied_at="2026-01-01T00:00:00+00:00",
            )
        ],
    )
    request = build_relation_request_for_test(graph, [candidate], pair_cap=1)
    grant_relation_candidates(graph, [candidate], request)
    before = graph.snapshot()
    outcome = commit_episode_result(graph, request, relation_result(request, "equivalent"))
    assert outcome.accepted is False
    assert outcome.rejection_reason == (
        "merge provenance conflict: absorbed node b already maps to canonical a"
    )
    assert graph.snapshot() == before


def test_merge_alias_resolver_is_transitive_and_rejects_cycles_or_missing_nodes():
    a_to_c = MergeApplicationRecord(
        merge_application_id="merge-a-c",
        relation_record_id="relation-a-c",
        canonical_node_id="c",
        absorbed_node_ids=["a"],
        source_node_ids=["a", "c"],
        source_node_revisions={"a": 0, "c": 0},
        applied_graph_revision=2,
        applied_at="2026-01-01T00:00:00+00:00",
    )
    c_to_d = MergeApplicationRecord(
        merge_application_id="merge-c-d",
        relation_record_id="relation-c-d",
        canonical_node_id="d",
        absorbed_node_ids=["c"],
        source_node_ids=["c", "d"],
        source_node_revisions={"c": 1, "d": 0},
        applied_graph_revision=4,
        applied_at="2026-01-01T00:01:00+00:00",
    )
    assert resolve_merge_aliases(
        [a_to_c, c_to_d], committed_node_ids={"a", "c", "d"}
    ) == {"a": "d", "c": "d"}

    c_to_a = c_to_d.model_copy(
        update={
            "merge_application_id": "merge-c-a",
            "canonical_node_id": "a",
            "source_node_ids": ["a", "c"],
            "source_node_revisions": {"a": 0, "c": 1},
        }
    )
    with pytest.raises(ValueError, match="alias cycle"):
        resolve_merge_aliases([a_to_c, c_to_a])
    with pytest.raises(ValueError, match="missing committed node"):
        resolve_merge_aliases([a_to_c], committed_node_ids={"a"})


def test_merge_alias_resolver_rejects_noop_or_unaccounted_sources():
    noop = MergeApplicationRecord(
        merge_application_id="merge-noop",
        relation_record_id="relation-a-b",
        canonical_node_id="a",
        absorbed_node_ids=[],
        source_node_ids=["a", "b"],
        source_node_revisions={"a": 0, "b": 0},
        applied_graph_revision=2,
        applied_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="absorb at least one"):
        resolve_merge_aliases([noop], committed_node_ids={"a", "b"})

    unaccounted = noop.model_copy(
        update={
            "absorbed_node_ids": ["b"],
            "source_node_ids": ["a", "b", "c"],
            "source_node_revisions": {"a": 0, "b": 0, "c": 0},
        }
    )
    with pytest.raises(ValueError, match="equal the canonical plus absorbed"):
        resolve_merge_aliases([unaccounted], committed_node_ids={"a", "b", "c"})

    duplicate_source = unaccounted.model_copy(
        update={
            "source_node_ids": ["a", "a", "b"],
            "source_node_revisions": {"a": 0, "b": 0},
        }
    )
    with pytest.raises(ValueError, match="duplicate source"):
        resolve_merge_aliases([duplicate_source], committed_node_ids={"a", "b"})


def test_two_node_disjoint_equivalent_merges_commit_atomically():
    graph = EpisodeGraph(nodes=[SearchNode(node_id=node_id, claim=node_id) for node_id in "abcd"])
    candidates = relation_candidates_for_pairs([("a", "b"), ("c", "d")])
    request = build_relation_request_for_test(graph, candidates)
    grant_relation_candidates(graph, candidates, request)
    outcome = commit_episode_result(graph, request, relation_result(request, "equivalent"))
    assert outcome.accepted is True
    assert graph.revision == 2
    assert graph.node_revisions == {"a": 1, "b": 1, "c": 1, "d": 1}
    assert len(graph.relation_ledger) == 2
    assert len(graph.merge_applications) == 2
    assert {item.canonical_node_id for item in graph.merge_applications} == {"a", "c"}
    absorbed_targets = {
        absorbed: application.canonical_node_id
        for application in graph.merge_applications
        for absorbed in application.absorbed_node_ids
    }
    assert absorbed_targets == {"b": "a", "d": "c"}


def test_rejected_overlapping_enrichment_retry_rebatches_without_consuming_budget(tmp_path):
    run_dir = make_enrichment_run(tmp_path, budget=3, pair_cap=3, node_count=4)
    granted = next_app_episode(run_dir, runtime_limits=RuntimeLimits(max_retries=1)).request
    state = app_driver.load_app_run(run_dir)
    first_pair = granted.relation_payload.candidate_pairs[0]
    first_candidate = next(
        item for item in state.relation_candidates if item.candidate_id == first_pair.candidate_id
    )
    overlap = next(
        item
        for item in state.relation_candidates
        if item.status == "pending"
        and first_candidate.left_node_id in (item.left_node_id, item.right_node_id)
    )
    overlap_request = build_relation_request_for_test(state.graph(), [overlap], pair_cap=1)
    old_payload = granted.relation_payload.model_copy(
        update={
            "candidate_pairs": [
                first_pair,
                overlap_request.relation_payload.candidate_pairs[0],
            ]
        }
    )
    old_request = granted.model_copy(
        update={
            "relation_payload": old_payload,
            "selected_node_revisions": {
                node_id: state.node_revisions[node_id]
                for node_id in {
                    first_candidate.left_node_id,
                    first_candidate.right_node_id,
                    overlap.left_node_id,
                    overlap.right_node_id,
                }
            },
        }
    )
    episode = app_driver._find_episode(state, granted.episode_id)
    episode.attempts[-1].request = old_request
    episode.attempts[-1].request_hash = app_driver._episode_request_hash(old_request)
    for candidate in state.relation_candidates:
        if candidate.candidate_id in {first_candidate.candidate_id, overlap.candidate_id}:
            candidate.status = "granted"
            candidate.granted_episode_id = old_request.episode_id
            candidate.granted_attempt_id = old_request.attempt_id
        elif candidate.status == "granted":
            candidate.status = "pending"
            candidate.granted_episode_id = None
            candidate.granted_attempt_id = None
    durable_before = (run_dir / "app_run_state.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="not node-disjoint and bounded"):
        app_driver._save_state(run_dir, state)
    assert (run_dir / "app_run_state.json").read_text(encoding="utf-8") == durable_before


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
    create_controller_checkpoint_run(run_dir, spec(enrichment_cap=3), nodes)
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
