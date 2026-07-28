import json
from pathlib import Path

from dte_backend import app_driver
from dte_backend import epistemic as epistemic_module
from dte_backend.app_driver import (
    app_run_status,
    create_app_run,
    next_app_episode,
    submit_app_episode_result,
)
from dte_backend.continuation import evaluate_continuation_gate
from dte_backend.episode_adapter import (
    build_executor_episode_request,
    build_judge_episode_request,
    build_relation_episode_request,
)
from dte_backend.episode_commit import EpisodeGraph, commit_episode_result
from dte_backend.episode_models import (
    EpisodeResult,
    ExecutorEpisodeOutput,
    JudgeEpisodeOutput,
    JudgeObservation,
    RoleIsolationAttestation,
    RuntimeDiagnostics,
    compute_output_hash,
)
from dte_backend.epistemic_models import EpistemicLedgerV1
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.relation_candidates import (
    select_node_disjoint_relation_batch,
)
from dte_backend.relation_models import (
    ProvisionalSynthesisSelection,
    RelationCandidate,
    RelationEpisodeOutput,
    RelationObservation,
)
from dte_backend.relation_readiness import evaluate_synthesis_readiness
from dte_backend.synthesis import select_provisional_synthesis_nodes


FIXTURE = Path(__file__).parent / "fixtures" / "subgroup_stator_selection.json"


def diagnostics() -> RuntimeDiagnostics:
    return RuntimeDiagnostics(
        adapter_name="test-runtime",
        transport_name="test-transport",
        profile="native-autonomous",
        usage_source="unavailable",
    )


def result_for(
    request,
    output,
    *,
    session_id: str | None = None,
    manifest_hash: str | None = None,
    isolation_mode: str | None = None,
):
    mode = isolation_mode or request.role_execution_contract.isolation_mode
    strict = mode == "strict_fresh_context"
    attestation = RoleIsolationAttestation(
        isolation_mode=mode,
        role_session_id=session_id,
        context_manifest_hash=(
            manifest_hash
            if manifest_hash is not None
            else request.role_execution_contract.context_manifest_hash
            if strict
            else None
        ),
        isolation_attestation_source=(
            "runtime_reported"
            if strict
            else "backend_fallback"
            if mode == "shared_context_single_agent"
            else "legacy_unverified"
        ),
        isolation_verified=strict,
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
        role_isolation_attestation=attestation,
    )


def judge_output(request, score: float = 0.8) -> JudgeEpisodeOutput:
    return JudgeEpisodeOutput(
        observations=[
            JudgeObservation(
                node_id=node_id,
                score=score,
                reasoning="bounded material assessment",
                risks=[],
            )
            for node_id in request.selected_node_revisions
        ]
    )


def strict_spec() -> DTERunSpec:
    return DTERunSpec(
        problem="bounded isolation regression",
        goal="exercise strict role execution",
        role_isolation_mode="strict_fresh_context",
        budget=BudgetSpec(
            max_committed_search_nodes=3,
            max_iterations=2,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=1,
            max_relation_pairs_per_episode=1,
            max_relation_enrichment_pairs=0,
            min_iterations_before_synthesis=2,
        ),
        embedding_dimension=8,
    )


def candidate(
    candidate_id: str,
    left: str,
    right: str,
    *,
    reason: str,
    scheduling_class: str,
    priority: str,
    material: bool,
) -> RelationCandidate:
    return RelationCandidate(
        candidate_id=candidate_id,
        left_node_id=left,
        right_node_id=right,
        left_node_revision=0,
        right_node_revision=0,
        candidate_reason=reason,
        scheduling_class=scheduling_class,
        priority=priority,
        material_to_synthesis=material,
        created_from_graph_revision=0,
    )


def all_keys(value):
    if isinstance(value, dict):
        return set(value).union(
            *(all_keys(item) for item in value.values()),
        )
    if isinstance(value, list):
        return set().union(*(all_keys(item) for item in value))
    return set()


def test_strict_isolation_rejects_reused_role_session_across_roles(tmp_path):
    run_dir = tmp_path / "strict-reuse"
    create_app_run(
        run_dir,
        strict_spec(),
        [SearchNode(node_id="seed", claim="seed claim")],
        run_id="strict-reuse",
    )
    judge = next_app_episode(run_dir).request
    assert judge is not None and judge.role == "judge"
    accepted = submit_app_episode_result(
        run_dir,
        result_for(judge, judge_output(judge), session_id="fresh-session-1"),
    )
    assert accepted.commit_outcome.accepted

    executor = next_app_episode(run_dir).request
    assert executor is not None and executor.role == "executor"
    rejected = submit_app_episode_result(
        run_dir,
        result_for(
            executor,
            ExecutorEpisodeOutput(nodes=[]),
            session_id="fresh-session-1",
        ),
    )
    assert not rejected.commit_outcome.accepted
    assert "reused role_session_id" in (rejected.commit_outcome.rejection_reason or "")


def test_strict_isolation_rejects_context_manifest_mismatch(tmp_path):
    run_dir = tmp_path / "strict-hash"
    create_app_run(
        run_dir,
        strict_spec(),
        [SearchNode(node_id="seed", claim="seed claim")],
        run_id="strict-hash",
    )
    request = next_app_episode(run_dir).request
    assert request is not None
    rejected = submit_app_episode_result(
        run_dir,
        result_for(
            request,
            judge_output(request),
            session_id="fresh-session",
            manifest_hash="f" * 64,
        ),
    )
    assert not rejected.commit_outcome.accepted
    assert "isolation attestation" in (
        rejected.commit_outcome.rejection_reason or ""
    )


def test_shared_context_requires_unverified_fallback_and_discloses_risk(tmp_path):
    spec = strict_spec().model_copy(
        update={"role_isolation_mode": "shared_context_single_agent"}
    )
    run_dir = tmp_path / "shared"
    create_app_run(
        run_dir,
        spec,
        [SearchNode(node_id="seed", claim="seed claim")],
        run_id="shared",
    )
    request = next_app_episode(run_dir).request
    assert request is not None
    accepted = submit_app_episode_result(
        run_dir,
        result_for(
            request,
            judge_output(request),
            session_id="shared-main-agent",
            isolation_mode="shared_context_single_agent",
        ),
    )
    assert accepted.commit_outcome.accepted
    state = app_run_status(run_dir)
    assert state.correlated_error_risk is True
    assert state.role_session_registry[-1].isolation_verified is False

    second_dir = tmp_path / "shared-false-claim"
    create_app_run(
        second_dir,
        spec,
        [SearchNode(node_id="seed", claim="seed claim")],
        run_id="shared-false-claim",
    )
    second = next_app_episode(second_dir).request
    assert second is not None
    raw = result_for(
        second,
        judge_output(second),
        session_id="shared-main-agent",
        isolation_mode="shared_context_single_agent",
    ).model_dump(mode="json")
    raw["role_isolation_attestation"]["isolation_verified"] = True
    rejected = submit_app_episode_result(second_dir, raw)
    assert not rejected.commit_outcome.accepted


def test_executor_and_judge_payloads_exclude_controller_and_cross_role_state():
    parent = SearchNode(
        node_id="canonical-parent",
        claim="parent claim",
        judge_reasoning="CANARY_JUDGE_REASONING",
        judge_risks=["CANARY_JUDGE_RISK"],
        score=0.99,
        ucb_score=4.2,
        expansion_budget=7,
    )
    hidden = SearchNode(node_id="hidden-frontier", claim="CANARY_HIDDEN_FRONTIER")
    graph = EpisodeGraph(nodes=[parent, hidden])
    executor = build_executor_episode_request(
        graph,
        parent,
        run_id="payload",
        iteration=1,
        max_returned_children=1,
        objective="expand only the parent",
        isolation_mode="shared_context_single_agent",
    )
    executor_json = executor.model_dump_json()
    assert "CANARY_JUDGE_REASONING" not in executor_json
    assert "CANARY_HIDDEN_FRONTIER" not in executor_json
    assert {
        "score",
        "ucb_score",
        "expansion_budget",
        "judge_reasoning",
        "relation_payload",
        "judge_payload",
    }.isdisjoint(all_keys(executor.model_dump(mode="json")["executor_payload"]))

    judge = build_judge_episode_request(
        graph,
        [parent],
        run_id="payload",
        problem="problem",
        goal="goal",
        isolation_mode="shared_context_single_agent",
    )
    judge_json = judge.model_dump_json()
    assert "canonical-parent" not in judge_json
    assert "CANARY_JUDGE_REASONING" not in judge_json
    assert "CANARY_HIDDEN_FRONTIER" not in judge_json
    assert {
        "score",
        "ucb_score",
        "density",
        "uncertainty",
        "expansion_budget",
        "provisional_synthesis_node_ids",
    }.isdisjoint(all_keys(judge.model_dump(mode="json")["judge_payload"]))


def test_relation_v2_payload_is_blind_and_backend_reattaches_metadata():
    nodes = [
        SearchNode(
            node_id="a",
            claim="claim a",
            evidence=["direct a"],
            judge_reasoning="CANARY_JUDGE_A",
            judge_risks=["CANARY_RISK_A"],
            score=0.99,
        ),
        SearchNode(
            node_id="b",
            claim="claim b",
            evidence=["direct b"],
            judge_reasoning="CANARY_JUDGE_B",
            score=0.95,
        ),
    ]
    relation_candidate = candidate(
        "candidate-critical",
        "a",
        "b",
        reason="potential_material_conflict",
        scheduling_class="blocking",
        priority="critical",
        material=True,
    )
    graph = EpisodeGraph(nodes=nodes, relation_candidates=[relation_candidate])
    request = build_relation_episode_request(
        graph,
        [relation_candidate],
        run_id="relation-blind",
        problem="problem",
        goal="goal",
        constraints=[],
        provisional_synthesis_node_ids=["a"],
        max_relation_pairs_per_episode=1,
        isolation_mode="shared_context_single_agent",
    )
    relation_candidate.status = "granted"
    relation_candidate.granted_episode_id = request.episode_id
    relation_candidate.granted_attempt_id = request.attempt_id
    payload = request.model_dump(mode="json")["relation_payload"]
    forbidden = {
        "provisional_synthesis_node_ids",
        "material_to_synthesis",
        "priority",
        "candidate_reason",
        "judge_reasoning",
        "judge_risks",
        "judge_uncertainty_evidence",
        "judge_result_provenance",
        "selection_status",
    }
    assert forbidden.isdisjoint(all_keys(payload))
    serialized = json.dumps(payload)
    assert "CANARY_JUDGE_A" not in serialized
    assert '"a"' not in serialized and '"b"' not in serialized

    pair = request.relation_payload.candidate_pairs[0]
    output = RelationEpisodeOutput(
        observations=[
            RelationObservation(
                candidate_id=pair.candidate_id,
                left_node_id=pair.left.node_id,
                right_node_id=pair.right.node_id,
                relation_type="independent",
                confidence=0.8,
                rationale="claims are semantically independent",
                evidence_refs=[],
                materiality_assessment="material",
                independence_summary="no shared conclusion",
            )
        ]
    )
    result = result_for(
        request,
        output,
        session_id="shared-main",
        isolation_mode="shared_context_single_agent",
    )
    canonical_request, canonical_payload = app_driver._canonicalize_blinded_contract(
        request,
        result.model_dump(mode="json"),
        request._canonical_node_id_map,
    )
    outcome = commit_episode_result(graph, canonical_request, canonical_payload)
    assert outcome.accepted
    record = graph.relation_ledger[0]
    assert (record.left_node_id, record.right_node_id) == ("a", "b")
    assert record.scheduling_class == "blocking"
    assert record.material_to_synthesis is True


def test_terminal_handoff_visibility_excludes_undisclosed_unselected_content(tmp_path):
    state = create_app_run(
        tmp_path / "visibility",
        strict_spec().model_copy(
            update={"role_isolation_mode": "shared_context_single_agent"}
        ),
        [
            SearchNode(node_id="material", claim="material claim"),
            SearchNode(
                node_id="hidden",
                claim="CANARY_UNSELECTED_CONTENT",
            ),
        ],
        run_id="visibility",
    )
    next(node for node in state.nodes if node.node_id == "material").score = 0.9
    next(node for node in state.nodes if node.node_id == "hidden").score = 0.2
    selection = ProvisionalSynthesisSelection(
        selected_node_ids=["material"],
        material_scope_node_ids=["material"],
        selection_reason="test material scope",
        selection_revision=state.graph_revision,
    )
    state.provisional_synthesis_selection = selection
    state.unselected_node_dispositions = app_driver._build_unselected_dispositions(
        state, selection
    )
    visible = epistemic_module._handoff_visible_node_ids(state)
    assert visible == {"material"}
    graph = epistemic_module._dependency_graph(state, visible)
    assert graph.node_claim_refs == ["node-claim:material"]


def test_subgroup_stator_fixture_preserves_coverage_dependencies_and_headline_cap():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    nodes = [SearchNode.model_validate(item) for item in fixture["nodes"]]
    selection = select_provisional_synthesis_nodes(
        nodes,
        graph_revision=7,
        max_nodes=8,
        required_coverage_ids=fixture["required_coverage_ids"],
    )
    material = set(selection.material_scope_node_ids)
    assert "small-case-required" in material
    assert "resource-minimality-central" in material
    assert "novelty-replacement" in material
    assert "root-novelty" in material
    assert "root-small-case" in selection.support_dependency_node_ids
    assert len(selection.selected_node_ids) == 8
    assert len(selection.material_scope_node_ids) > len(selection.selected_node_ids)
    represented = {
        coverage_id
        for node in nodes
        if node.node_id in material
        for coverage_id in node.coverage_ids
    }
    assert set(fixture["required_coverage_ids"]).issubset(represented)
    assert selection.unresolved_coverage_ids == []


def test_missing_required_coverage_is_durable_and_score_cannot_displace_it():
    nodes = [
        SearchNode(
            node_id="peripheral",
            claim="peripheral",
            score=0.99,
            coverage_ids=["coverage:peripheral"],
        ),
        SearchNode(
            node_id="required",
            claim="required",
            score=0.95,
            coverage_ids=["coverage:required"],
        ),
    ]
    selection = select_provisional_synthesis_nodes(
        nodes,
        graph_revision=1,
        max_nodes=1,
        required_coverage_ids=[
            "coverage:required",
            "coverage:missing",
        ],
    )
    assert "required" in selection.material_scope_node_ids
    assert selection.selected_node_ids == ["required"]
    assert selection.unresolved_coverage_ids == ["coverage:missing"]


def test_every_unselected_fixture_node_gets_counterfactual_disposition(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    nodes = [SearchNode.model_validate(item) for item in fixture["nodes"]]
    spec = DTERunSpec(
        problem="subgroup stator structural regression",
        goal="preserve material branches",
        role_isolation_mode="shared_context_single_agent",
        coverage_obligations=[
            {"coverage_id": coverage_id}
            for coverage_id in fixture["required_coverage_ids"]
        ],
        embedding_dimension=8,
    )
    state = create_app_run(
        tmp_path / "dispositions",
        spec,
        [
            node.model_copy(update={"score": None, "status": "frontier"})
            for node in nodes
        ],
        run_id="dispositions",
    )
    fixture_by_id = {node.node_id: node for node in nodes}
    for node in state.nodes:
        source = fixture_by_id[node.node_id]
        node.score = source.score
        node.status = source.status
    selection = select_provisional_synthesis_nodes(
        state.nodes,
        graph_revision=state.graph_revision,
        required_coverage_ids=fixture["required_coverage_ids"],
    )
    dispositions = app_driver._build_unselected_dispositions(state, selection)
    eligible_unselected = {
        node.node_id
        for node in state.nodes
        if node.status in {"frontier", "closed"}
        and node.node_id not in selection.material_scope_node_ids
    }
    assert {item.node_id for item in dispositions} == eligible_unselected
    by_id = {item.node_id: item for item in dispositions}
    assert by_id["controller-low-confidence"].disposition == (
        "future_work_unverified"
    )
    assert by_id["peripheral-098"].disposition == "unresolved_high_value"
    assert by_id["peripheral-098"].disclosure_required is True


def test_material_unselected_or_missing_provenance_blocks_readiness():
    readiness = evaluate_synthesis_readiness(
        graph_revision=1,
        provisional_selected_node_ids=["selected"],
        candidates=[],
        relation_ledger=[],
        merge_applications=[],
        evaluated_at="2026-01-01T00:00:00+00:00",
        undisposed_material_node_ids=["material-unselected"],
        provenance_incomplete_node_ids=["selected"],
    )
    assert readiness.ready is False
    assert readiness.undisposed_material_node_ids == ["material-unselected"]
    assert readiness.provenance_incomplete_node_ids == ["selected"]


def test_empty_material_epistemic_contributions_do_not_force_nonmaterial_filler(
    tmp_path,
):
    state = create_app_run(
        tmp_path / "provenance",
        strict_spec().model_copy(
            update={"role_isolation_mode": "shared_context_single_agent"}
        ),
        [
            SearchNode(node_id="material", claim="material"),
            SearchNode(node_id="nonmaterial", claim="nonmaterial"),
        ],
        run_id="provenance",
    )
    next(node for node in state.nodes if node.node_id == "material").score = 0.9
    next(node for node in state.nodes if node.node_id == "nonmaterial").score = 0.2
    selection = ProvisionalSynthesisSelection(
        selected_node_ids=["material"],
        material_scope_node_ids=["material"],
        selection_reason="test",
        selection_revision=state.graph_revision,
    )
    incomplete = app_driver._provenance_incomplete_material_nodes(
        state, selection
    )
    assert incomplete == ["material"]
    assert "nonmaterial" not in incomplete


def test_relation_batch_prioritizes_material_conflict_over_early_similarity():
    similarity = candidate(
        "similarity",
        "a",
        "c",
        reason="embedding_close",
        scheduling_class="enrichment",
        priority="high",
        material=False,
    )
    conflict = candidate(
        "conflict",
        "a",
        "b",
        reason="potential_material_conflict",
        scheduling_class="blocking",
        priority="critical",
        material=True,
    )
    selected = select_node_disjoint_relation_batch(
        [similarity, conflict],
        max_pairs=1,
        maximize_material_priority=True,
    )
    assert [item.candidate_id for item in selected] == ["conflict"]


def test_triggered_continuation_rejects_selection_change_as_epistemic_yield():
    nodes = [
        SearchNode(
            node_id="n1",
            claim="single frontier",
            coverage_ids=["coverage:one"],
        )
    ]
    gate = evaluate_continuation_gate(
        iteration=1,
        graph_revision=1,
        nodes=nodes,
        max_committed_search_nodes=3,
        entropy_delta=0.0,
        consecutive_plateau_count=2,
        plateau_confirmed=True,
        allocations={"n1": 1},
        previous_frontier_node_ids={"n1"},
        previous_positive_allocation_node_ids=set(),
        previous_provisional_synthesis_node_ids=set(),
        provisional_synthesis_node_ids=["n1"],
        ledger=EpistemicLedgerV1(),
        previously_considered_epistemic_ids=set(),
        legacy_process_yield_allowed=False,
    )
    assert "provisional_synthesis_membership_changed" in gate.process_yield_signals
    assert gate.epistemic_yield_signals == []
    assert gate.decision == "continue"
    assert gate.coverage_yield_signals == ["required_coverage_gain:coverage:one"]

    no_coverage = evaluate_continuation_gate(
        iteration=1,
        graph_revision=1,
        nodes=[SearchNode(node_id="n1", claim="single frontier")],
        max_committed_search_nodes=3,
        entropy_delta=0.0,
        consecutive_plateau_count=2,
        plateau_confirmed=True,
        allocations={"n1": 1},
        previous_frontier_node_ids={"n1"},
        previous_positive_allocation_node_ids=set(),
        previous_provisional_synthesis_node_ids=set(),
        provisional_synthesis_node_ids=["n1"],
        ledger=EpistemicLedgerV1(),
        previously_considered_epistemic_ids=set(),
        legacy_process_yield_allowed=False,
    )
    assert no_coverage.process_yield_signals
    assert no_coverage.material_yield_signals == []
    assert no_coverage.decision == "prepare_synthesis"


def test_direct_legacy_selection_keeps_historical_top_eight_scope():
    nodes = [
        SearchNode(node_id=f"n{index}", claim=f"claim {index}", score=index / 10)
        for index in range(9)
    ]
    selection = select_provisional_synthesis_nodes(
        nodes,
        graph_revision=1,
    )
    assert len(selection.selected_node_ids) == 8
    assert selection.material_scope_node_ids == sorted(selection.selected_node_ids)
