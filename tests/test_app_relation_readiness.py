from tests.helpers import completed_node, completed_candidate
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
from dte_backend.episode_models import EpisodeRequest, EpisodeResult, ExecutorEpisodeOutput, ExecutorNodeCandidate, RuntimeDiagnostics, RuntimeLimits, compute_output_hash
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


def spec(*, cap=3, pair_cap=3, max_iterations=1, enrichment_cap=0, allocation_mass=1):
    return DTERunSpec(
        problem="relation readiness",
        goal="reach synthesis without duplicate or undisclosed material-conflict ambiguity",
        constraints=["Relation is observation-only"],
        budget=BudgetSpec(
            max_iterations=max_iterations,
            allocation_mass_per_iteration=allocation_mass,
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




def test_candidate_generation_is_canonical_stable_bounded_and_prioritized():
    nodes = [
        completed_node(node_id="b", claim="same", score=0.7),
        completed_node(node_id="a", claim=" SAME ", score=0.8),
        completed_node(node_id="far", claim="unrelated", score=0.1),
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


def test_shared_evidence_requires_explicit_claim_divergence_to_block():
    nodes = [
        completed_node(node_id="a", claim="condition is sufficient", evidence=["paper-1"], score=0.8),
        completed_node(node_id="b", claim="condition is not sufficient", evidence=["paper-1"], score=0.79),
        completed_node(node_id="c", claim="remote branch", evidence=["other"], score=0.2),
    ]
    candidates = generate_relation_candidates(
        nodes,
        node_revisions={node.node_id: 0 for node in nodes},
        graph_revision=2,
        provisional_synthesis_node_ids=["a", "b"],
    )
    material = next(
        item
        for item in candidates
        if item.candidate_reason == "shared_evidence_divergence"
    )
    assert (material.left_node_id, material.right_node_id) == ("a", "b")
    assert material.material_to_synthesis is True
    assert not any(
        item.candidate_reason == "potential_material_conflict" and "c" in (item.left_node_id, item.right_node_id)
        for item in candidates
    )


def test_candidate_generation_never_expands_to_global_all_pairs():
    nodes = [completed_node(node_id=f"n{i:03d}", claim="same", score=0.5) for i in range(100)]
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
        completed_node(node_id="a", claim="same", score=0.9),
        completed_node(node_id="b", claim="same", score=0.1),
        completed_node(node_id="c", claim="same", score=0.8),
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
    nodes = [completed_node(node_id="a", claim="A", score=0.8), completed_node(node_id="b", claim="B", score=0.79)]
    candidates = generate_relation_candidates(
        nodes,
        node_revisions={"a": 0, "b": 0},
        graph_revision=1,
        provisional_synthesis_node_ids=["a", "b"],
        entropy_plateau=True,
    )
    assert any(item.candidate_reason in {"entropy_plateau", "high_score_near_tie"} for item in candidates)
    assert all(not hasattr(item, "relation_type") for item in candidates)
























def test_chained_equivalent_merge_cleans_alias_projected_self_parent_atomically():
    nodes = [
        completed_node(node_id="a", claim="same"),
        completed_node(node_id="b", claim="same", rationale="first canonical"),
        completed_node(
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
        completed_node(node_id="a", claim="same"),
        completed_node(node_id="b", claim="same", rationale="first canonical"),
        completed_node(node_id="c", claim="same", parent_ids=["x"], evidence=["more"]),
        completed_node(node_id="x", claim="bridge", parent_ids=["b"]),
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








def test_resolved_nonmaterial_conflict_is_disclosed_if_both_endpoints_later_selected():
    nodes = [
        completed_node(
            node_id="a",
            claim="route A",
            score=0.8,
            local_embedding=[1.0, 0.0],
        ),
        completed_node(
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
            completed_node(node_id="a", claim="A", score=0.9),
            completed_node(node_id="b", claim="B", score=0.1),
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






def test_legacy_persisted_terminal_without_audit_record_is_rejected(tmp_path):
    run_dir = tmp_path / "legacy"
    create_app_run(run_dir, spec(), [completed_node(node_id="a", claim="A")])
    state_path = run_dir / "app_run_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["controller_action"] = "ready_for_synthesis"
    payload["synthesis_readiness"] = None
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal App state lacks"):
        app_driver.load_app_run(run_dir)




def executor_result(request, *, child_id, claim, evidence):
    output = ExecutorEpisodeOutput(
        nodes=[
            completed_candidate(
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












def conflict_nodes(count=8):
    return [
        completed_node(
            node_id=f"n{i}",
            claim=f"material conclusion {i}",
            coverage_ids=["seed:shared"],
            score=0.9 - i * 0.01,
        )
        for i in range(count)
    ]






def test_enrichment_generation_filters_known_pairs_before_window_truncation():
    nodes = [completed_node(node_id=f"n{i}", claim=f"claim {i}", score=0.8) for i in range(8)]
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
        completed_node(
            node_id=f"s{i}",
            claim=f"selected {i}",
            score=1.0 - i * 0.05,
            local_embedding=([1.0, 0.0] if i == 0 else None),
        )
        for i in range(8)
    ]
    related = [
        completed_node(
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
    nodes = [completed_node(node_id="a", claim="A", score=0.8), completed_node(node_id="b", claim="B", score=0.8)]
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
    nodes = [completed_node(node_id=f"n{i}", claim=f"claim {i}", score=0.8) for i in range(8)]
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
        completed_node(node_id="a", claim="selected A", score=0.8),
        completed_node(node_id="b", claim="selected B", score=0.8),
        completed_node(node_id="x", claim="unrelated X", score=0.8),
        completed_node(node_id="y", claim="unrelated Y", score=0.8),
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






def test_relation_request_builder_rejects_overlapping_pairs_without_dropping_them():
    graph = EpisodeGraph(nodes=[completed_node(node_id=node_id, claim=node_id) for node_id in "abc"])
    candidates = relation_candidates_for_pairs([("a", "b"), ("b", "c")])
    with pytest.raises(ValueError, match="candidate pairs must be node-disjoint"):
        build_relation_request_for_test(graph, candidates)


def test_relation_request_builder_accepts_node_disjoint_pairs():
    graph = EpisodeGraph(nodes=[completed_node(node_id=node_id, claim=node_id) for node_id in "abcd"])
    candidates = relation_candidates_for_pairs([("a", "b"), ("c", "d")])
    request = build_relation_request_for_test(graph, candidates)
    assert [pair.candidate_id for pair in request.relation_payload.candidate_pairs] == [
        candidate.candidate_id for candidate in candidates
    ]


@pytest.mark.parametrize("relation_type", ["equivalent", "complementary", "independent"])
def test_commit_rejects_old_overlapping_relation_request_atomically(relation_type):
    graph = EpisodeGraph(nodes=[completed_node(node_id=node_id, claim=node_id) for node_id in "abc"])
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
        completed_node(node_id="a", claim="canonical A"),
        completed_node(node_id="b", claim="absorbed B", status="merged"),
        completed_node(node_id="c", claim="canonical C"),
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
    graph = EpisodeGraph(nodes=[completed_node(node_id=node_id, claim=node_id) for node_id in "abcd"])
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


















