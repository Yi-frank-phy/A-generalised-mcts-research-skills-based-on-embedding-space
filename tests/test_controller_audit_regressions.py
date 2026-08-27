import pytest

import dte_backend.runner as runner
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.merge import apply_relation_equivalent_merge
from dte_backend.models import BudgetSpec, DTERunSpec
from dte_backend.novelty import estimate_frontier_kde_state
from dte_backend.reference_atlas import combined_reference_nodes, packaged_reference_nodes
from tests.helpers import completed_node


def test_combined_reference_nodes_deduplicates_exact_transition_cells():
    packaged = packaged_reference_nodes()
    duplicate = completed_node(
        node_id="duplicate-root",
        retrospective_method=packaged[0].retrospective_method,
        epistemic_change_kind=packaged[0].epistemic_change_kind,
        epistemic_change=packaged[0].epistemic_change,
    )
    distinct = completed_node(
        node_id="distinct-root",
        retrospective_method="a run-specific method not present in the packaged atlas",
        epistemic_change_kind="sharper_unknown",
        epistemic_change="isolated a run-specific unresolved assumption",
    )

    combined = combined_reference_nodes([duplicate, distinct])

    assert len(combined) == len(packaged) + 1
    assert combined[-1].node_id == "distinct-root"
    assert all(node.node_id != "duplicate-root" for node in combined)


def test_singleton_matching_packaged_cell_has_feasible_entropy_match():
    packaged = packaged_reference_nodes()
    root = completed_node(
        node_id="matching-root",
        claim="singleton matching one packaged transition cell",
        retrospective_method=packaged[0].retrospective_method,
        epistemic_change_kind=packaged[0].epistemic_change_kind,
        epistemic_change=packaged[0].epistemic_change,
    )

    frontier, state = estimate_frontier_kde_state(
        [root],
        provider=HashEmbeddingProvider(dim=8),
        expected_dimension=8,
    )

    assert [node.node_id for node in frontier] == ["matching-root"]
    assert state.occupancy_fractions[0] == pytest.approx(1.0)
    assert state.standard_deviations[0] >= 0.0


def test_relation_equivalent_merge_invalidates_stale_semantic_embedding():
    canonical = completed_node(
        node_id="a",
        claim="same claim",
        evidence=["evidence-a"],
        local_embedding=[1.0, 0.0],
    )
    absorbed = completed_node(
        node_id="b",
        claim="same claim",
        assumptions=["assumption-b"],
        local_embedding=[0.0, 1.0],
    )
    nodes = [canonical, absorbed]
    revisions = {"a": 0, "b": 0}

    application = apply_relation_equivalent_merge(
        nodes,
        revisions,
        source_node_ids=["a", "b"],
        relation_record_id="relation-1",
        applied_graph_revision=1,
        applied_at="2026-08-27T00:00:00+00:00",
    )

    merged = next(node for node in nodes if node.node_id == application.canonical_node_id)
    assert merged.local_embedding is None
    assert "assumption-b" in merged.assumptions
    assert "evidence-a" in merged.evidence


def test_runner_forwards_run_spec_geometry_knobs(monkeypatch):
    spec = DTERunSpec(
        problem="verify controller configuration wiring",
        goal="stop immediately after observing scoring arguments",
        budget=BudgetSpec(
            max_iterations=1,
            controller_graph_k=7,
            volume_bandwidth=0.25,
        ),
        embedding_provider="hash",
        embedding_dimension=8,
    )
    initial = [completed_node(node_id="seed", claim="seed")]
    captured = {}

    class ScoringObserved(RuntimeError):
        pass

    def fake_estimate(nodes, cache=None, provider=None, **kwargs):
        captured.update(kwargs)
        raise ScoringObserved

    monkeypatch.setattr(runner, "estimate_frontier_kde_state", fake_estimate)

    with pytest.raises(ScoringObserved):
        runner.run_frontier_search(spec, initial_nodes=initial)

    assert captured["expected_dimension"] == 8
    assert captured["graph_k"] == 7
    assert captured["volume_bandwidth"] == 0.25
