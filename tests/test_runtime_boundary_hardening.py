import pytest
from pydantic import ValidationError

from dte_backend.episode_adapter import (
    build_executor_episode_request,
    build_relation_episode_request,
    run_and_commit_episode,
)
from dte_backend.episode_commit import EpisodeGraph
from dte_backend.episode_models import (
    EpisodeResult,
    RuntimeDiagnostics,
    compute_output_hash,
)
from dte_backend.models import SearchNode
from dte_backend.oracle_validation import validate_relation_episode_output
from dte_backend.relation_models import (
    RelationCandidate,
    RelationEpisodeOutput,
    RelationObservation,
)


def executor_grant():
    parent = SearchNode(node_id="parent", claim="committed")
    graph = EpisodeGraph(nodes=[parent], revision=0, node_revisions={"parent": 0})
    request = build_executor_episode_request(
        graph,
        parent,
        run_id="run",
        iteration=1,
        max_returned_children=1,
        objective="expand",
    )
    return graph, request


def test_legacy_runtime_revalidates_request_before_adapter_side_effects():
    graph, request = executor_grant()

    class CountingAdapter:
        adapter_name = "must-not-run"
        transport_name = "test"
        calls = 0

        def run_episode(self, _request):
            self.calls += 1
            raise AssertionError("invalid request reached runtime")

    adapter = CountingAdapter()
    request.max_returned_children = -1
    before = graph.snapshot()
    outcome = run_and_commit_episode(graph, request, adapter)

    assert outcome.accepted is False
    assert "invalid episode request before runtime call" in outcome.rejection_reason
    assert adapter.calls == 0
    assert graph.snapshot() == before


def relation_envelope():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    graph = EpisodeGraph(nodes=nodes, revision=0, node_revisions={"a": 0, "b": 0})
    candidate = RelationCandidate(
        candidate_id="candidate",
        left_node_id="a",
        right_node_id="b",
        left_node_revision=0,
        right_node_revision=0,
        candidate_reason="embedding_close",
        scheduling_class="enrichment",
        priority="high",
        material_to_synthesis=False,
        created_from_graph_revision=0,
    )
    request = build_relation_episode_request(
        graph,
        [candidate],
        run_id="run",
        problem="p",
        goal="g",
        constraints=[],
        provisional_synthesis_node_ids=["a"],
        max_relation_pairs_per_episode=1,
    )
    observation = RelationObservation(
        candidate_id="candidate",
        left_node_id="a",
        right_node_id="b",
        relation_type="independent",
        confidence=0.9,
        rationale="separate",
        evidence_refs=[],
        materiality_assessment="non_material",
        independence_summary="separate questions",
    )
    output = RelationEpisodeOutput(observations=[observation])
    result = EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role="relation",
        input_graph_revision=request.input_graph_revision,
        selected_node_revisions=request.selected_node_revisions,
        status="completed",
        structured_output=output,
        runtime_diagnostics=RuntimeDiagnostics(
            adapter_name="test",
            transport_name="in-process",
            profile="native-autonomous",
        ),
        output_hash=compute_output_hash(output, request.output_schema_version),
        schema_version=request.output_schema_version,
    )
    return request, result


def test_relation_guard_revalidates_mutated_pydantic_instances():
    request, result = relation_envelope()
    result.structured_output.observations[0].relation_type = "verifier"

    with pytest.warns(UserWarning), pytest.raises(ValidationError, match="relation_type"):
        validate_relation_episode_output(request, result)
