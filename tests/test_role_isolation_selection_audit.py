from tests.helpers import completed_node
import json
import time
from pathlib import Path

from dte_backend import app_driver
from dte_backend import epistemic as epistemic_module
from dte_backend.app_driver import (
    app_run_status,
    create_app_run,
    next_app_episode,
    request_app_synthesis,
    submit_app_episode_result,
)
from dte_backend.continuation import evaluate_continuation_gate
from dte_backend.episode_adapter import build_executor_episode_request, build_relation_episode_request
from dte_backend.episode_commit import EpisodeGraph, commit_episode_result
from dte_backend.episode_models import EpisodeResult, ExecutorEpisodeOutput, RoleIsolationAttestation, RuntimeDiagnostics, compute_output_hash
from dte_backend.epistemic_models import (
    EpistemicContributionBundle,
    EpistemicLedgerV1,
    EpistemicStatementContribution,
)
from dte_backend.models import (
    BudgetSpec,
    CoverageObligation,
    DTERunSpec,
    SearchNode,
    SynthesisControlRequest,
)
from dte_backend.relation_candidates import (
    generate_blocking_relation_obligations,
    generate_relation_enrichment_candidates,
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














def test_terminal_handoff_visibility_excludes_undisclosed_unselected_content(tmp_path):
    state = create_app_run(
        tmp_path / "visibility",
        strict_spec().model_copy(
            update={"role_isolation_mode": "shared_context_single_agent"}
        ),
        [
            completed_node(node_id="material", claim="material claim"),
            completed_node(
                node_id="hidden",
                claim=" material   CLAIM ",
            ),
        ],
        run_id="visibility",
    )
    next(node for node in state.nodes if node.node_id == "material").score = 0.9
    next(node for node in state.nodes if node.node_id == "hidden").score = 0.2
    for node in state.nodes:
        node.coverage_ids = ["shared:test"]
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
        completed_node(
            node_id="peripheral",
            claim="peripheral",
            score=0.99,
            coverage_ids=["coverage:peripheral"],
        ),
        completed_node(
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
    assert (
        by_id["controller-low-confidence"].disposition
        == "unresolved_high_value"
    )
    assert by_id["controller-low-confidence"].disclosure_required is True
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
            completed_node(node_id="material", claim="material"),
            completed_node(node_id="nonmaterial", claim="nonmaterial"),
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


def _force_after_current_task(run_dir, *, node_ids=None):
    request_app_synthesis(
        run_dir,
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="exercise bounded terminal policy",
            scope="all" if node_ids is None else "node_ids",
            node_ids=[] if node_ids is None else node_ids,
        ),
    )




def _shared_provenance_spec(policy: str) -> DTERunSpec:
    return DTERunSpec(
        problem="provenance terminal recovery",
        goal="terminate without treating provenance as truth verification",
        role_isolation_mode="shared_context_single_agent",
        material_provenance_policy=policy,
        budget=BudgetSpec(
            max_committed_search_nodes=2,
            max_iterations=2,
            allocation_mass_per_iteration=1,
            max_children_per_iteration=1,
            max_relation_pairs_per_episode=1,
            max_relation_enrichment_pairs=0,
            min_iterations_before_synthesis=2,
        ),
        embedding_dimension=8,
    )












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


def _dense_relation_candidates(node_count: int) -> list[RelationCandidate]:
    candidates = []
    for left in range(node_count):
        for right in range(left + 1, node_count):
            candidates.append(
                candidate(
                    f"dense-{left:03d}-{right:03d}",
                    f"n{left:03d}",
                    f"n{right:03d}",
                    reason=(
                        "potential_material_conflict"
                        if (left + right) % 11 == 0
                        else "high_score_near_tie"
                    ),
                    scheduling_class=(
                        "blocking" if (left + right) % 11 == 0 else "enrichment"
                    ),
                    priority="critical" if (left + right) % 11 == 0 else "high",
                    material=(left + right) % 11 == 0,
                )
            )
    return candidates


def test_dense_relation_batch_is_bounded_deterministic_and_node_disjoint():
    started = time.perf_counter()
    for node_count in (20, 50, 100):
        candidates = _dense_relation_candidates(node_count)
        for max_pairs in (1, 3, 20):
            first = select_node_disjoint_relation_batch(
                candidates,
                max_pairs=max_pairs,
                maximize_material_priority=True,
            )
            second = select_node_disjoint_relation_batch(
                list(reversed(candidates)),
                max_pairs=max_pairs,
                maximize_material_priority=True,
            )
            assert [item.candidate_id for item in first] == [
                item.candidate_id for item in second
            ]
            endpoints = [
                node_id
                for item in first
                for node_id in (item.left_node_id, item.right_node_id)
            ]
            assert len(endpoints) == len(set(endpoints))
            assert len(first) <= min(max_pairs, node_count // 2)
            if any(item.priority == "critical" for item in candidates):
                assert first[0].priority == "critical"
    # Exhaustive subset enumeration on the 4,950-edge case cannot satisfy this
    # guard.  The generous threshold avoids depending on exact CI hardware.
    assert time.perf_counter() - started < 8.0


def test_relation_batch_bounded_improvement_recovers_one_for_two_matching():
    candidates = [
        candidate(
            "ab",
            "a",
            "b",
            reason="potential_material_conflict",
            scheduling_class="blocking",
            priority="critical",
            material=True,
        ),
        candidate(
            "ac",
            "a",
            "c",
            reason="potential_material_conflict",
            scheduling_class="blocking",
            priority="critical",
            material=True,
        ),
        candidate(
            "bd",
            "b",
            "d",
            reason="potential_material_conflict",
            scheduling_class="blocking",
            priority="critical",
            material=True,
        ),
    ]
    selected = select_node_disjoint_relation_batch(
        candidates,
        max_pairs=2,
        maximize_material_priority=True,
    )
    assert {item.candidate_id for item in selected} == {"ac", "bd"}


def test_shared_seed_coverage_is_enrichment_not_all_pairs_blocking():
    nodes = [
        completed_node(
            node_id=f"child-{index:02d}",
            claim=f"independent alternative {index}",
            coverage_ids=["seed:one"],
            score=0.8 - index / 100,
        )
        for index in range(20)
    ]
    revisions = {node.node_id: 0 for node in nodes}
    blockers = generate_blocking_relation_obligations(
        nodes,
        node_revisions=revisions,
        graph_revision=1,
        provisional_synthesis_node_ids=[node.node_id for node in nodes],
        include_material_review_pool=True,
    )
    assert blockers == []
    enrichment = generate_relation_enrichment_candidates(
        nodes,
        node_revisions=revisions,
        graph_revision=1,
        provisional_synthesis_node_ids=[nodes[0].node_id],
        existing=[],
        relation_ledger=[],
        max_candidates=6,
    )
    assert 0 < len(enrichment) <= 6
    assert all(
        item.candidate_reason == "shared_coverage_alternative"
        and item.scheduling_class == "enrichment"
        for item in enrichment
    )


def test_duplicate_and_direct_counterexample_remain_blocking():
    nodes = [
        completed_node(node_id="claim", claim="base claim"),
        completed_node(node_id="duplicate", claim=" base   CLAIM "),
        completed_node(
            node_id="counterexample",
            node_type="counterexample",
            claim="explicit failing case",
            parent_ids=["claim"],
        ),
    ]
    blockers = generate_blocking_relation_obligations(
        nodes,
        node_revisions={node.node_id: 0 for node in nodes},
        graph_revision=1,
        provisional_synthesis_node_ids=[node.node_id for node in nodes],
        include_material_review_pool=True,
    )
    by_pair = {
        (item.left_node_id, item.right_node_id): item.candidate_reason
        for item in blockers
    }
    assert by_pair[("claim", "duplicate")] == "exact_duplicate"
    assert (
        by_pair[("claim", "counterexample")]
        == "potential_material_conflict"
    )


def test_triggered_continuation_rejects_selection_change_as_epistemic_yield():
    nodes = [
        completed_node(
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
        nodes=[completed_node(node_id="n1", claim="single frontier")],
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
        completed_node(node_id=f"n{index}", claim=f"claim {index}", score=index / 10)
        for index in range(9)
    ]
    selection = select_provisional_synthesis_nodes(
        nodes,
        graph_revision=1,
    )
    assert len(selection.selected_node_ids) == 8
    assert selection.material_scope_node_ids == sorted(selection.selected_node_ids)
