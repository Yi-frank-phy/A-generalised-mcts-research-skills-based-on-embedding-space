import json
import sys

import pytest
from pydantic import ValidationError

from dte_backend.episode_adapter import (
    CommandAgentEpisodeAdapter,
    NativeStubEpisodeAdapter,
    build_executor_episode_request,
    run_and_commit_episode,
)
from dte_backend.episode_commit import CONTROLLER_OWNED_FIELDS, EpisodeGraph, commit_episode_result
from dte_backend.episode_models import (
    EpisodeRequest,
    EpisodeResult,
    ExecutorEpisodeOutput,
    ExecutorNodeCandidate,
    RuntimeDiagnostics,
    compute_output_hash,
)
from dte_backend.models import SearchNode
from dte_backend.models import BudgetSpec, DTERunSpec
from dte_backend.runner import run_frontier_search
from dte_backend.strict_runner import strict_run
from dte_backend.telemetry import EpisodeEventLog


def make_graph() -> EpisodeGraph:
    return EpisodeGraph(nodes=[SearchNode(node_id="parent", claim="assigned parent")])


def make_request(graph: EpisodeGraph, *, grant: int = 2) -> EpisodeRequest:
    return build_executor_episode_request(
        graph,
        graph.nodes[0],
        run_id="run-1",
        iteration=1,
        max_returned_children=grant,
        objective="expand the assigned parent",
        coverage_requirements=["test a boundary case"],
    )


def make_output(*nodes: ExecutorNodeCandidate) -> ExecutorEpisodeOutput:
    return ExecutorEpisodeOutput(nodes=list(nodes), episode_summary="bounded work")


def make_result(request: EpisodeRequest, output: ExecutorEpisodeOutput) -> EpisodeResult:
    return EpisodeResult(
        episode_id=request.episode_id,
        attempt_id=request.attempt_id,
        run_id=request.run_id,
        role=request.role,
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


def child(node_id: str = "child", parent_id: str = "parent") -> ExecutorNodeCandidate:
    return ExecutorNodeCandidate(node_id=node_id, claim="candidate", parent_ids=[parent_id])


def assert_rejected_unchanged(graph, request, result, match):
    before = graph.snapshot()
    outcome = commit_episode_result(graph, request, result)
    assert outcome.accepted is False
    assert match in (outcome.rejection_reason or "")
    assert graph.snapshot() == before


def test_episode_request_schema_is_strict():
    graph = make_graph()
    data = make_request(graph).model_dump()
    data["unknown"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EpisodeRequest.model_validate(data)


def test_episode_result_schema_is_strict():
    request = make_request(make_graph())
    data = make_result(request, make_output()).model_dump()
    data["unknown"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EpisodeResult.model_validate(data)


def test_valid_executor_commit_is_atomic_and_revises_parent_and_graph():
    graph = make_graph()
    request = make_request(graph)
    outcome = commit_episode_result(graph, request, make_result(request, make_output(child())))
    assert outcome.accepted is True
    assert outcome.accepted_node_ids == ["child"]
    assert graph.revision == 1
    assert graph.node_revisions == {"parent": 1, "child": 0}
    assert graph.node_by_id("parent").status == "closed"
    assert graph.node_by_id("child").status == "frontier"


def test_valid_zero_child_result_closes_assigned_parent():
    graph = make_graph()
    request = make_request(graph, grant=0)
    outcome = commit_episode_result(graph, request, make_result(request, make_output()))
    assert outcome.accepted is True
    assert outcome.accepted_node_count == 0
    assert graph.revision == 1
    assert graph.node_by_id("parent").status == "closed"


def test_child_count_exceeds_grant():
    graph = make_graph()
    request = make_request(graph, grant=1)
    result = make_result(request, make_output(child("a"), child("b")))
    assert_rejected_unchanged(graph, request, result, "exceeds grant")


def test_stale_graph_revision():
    graph = make_graph()
    request = make_request(graph)
    result = make_result(request, make_output(child()))
    graph.revision += 1
    assert_rejected_unchanged(graph, request, result, "stale graph revision")


def test_stale_parent_revision():
    graph = make_graph()
    request = make_request(graph)
    result = make_result(request, make_output(child()))
    graph.node_revisions["parent"] += 1
    assert_rejected_unchanged(graph, request, result, "stale parent revision")


def test_collision_with_committed_node_id():
    graph = make_graph()
    request = make_request(graph)
    result = make_result(request, make_output(child("parent")))
    assert_rejected_unchanged(graph, request, result, "collision")


def test_duplicate_ids_inside_result():
    graph = make_graph()
    request = make_request(graph)
    result = make_result(request, make_output(child("same"), child("same")))
    assert_rejected_unchanged(graph, request, result, "duplicate node ID")


def test_child_must_reference_assigned_parent():
    graph = make_graph()
    request = make_request(graph)
    result = make_result(request, make_output(child(parent_id="other")))
    assert_rejected_unchanged(graph, request, result, "assigned parent")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("node_type", "synthesis"),
        ("node_type", "unknown"),
        ("status", "closed"),
    ],
)
def test_forbidden_synthesis_status_and_node_type_are_rejected(field, value):
    graph = make_graph()
    request = make_request(graph)
    raw = make_result(request, make_output(child())).model_dump(mode="json")
    raw["structured_output"]["nodes"][0][field] = value
    assert_rejected_unchanged(graph, request, raw, "schema validation failed")


@pytest.mark.parametrize("field", sorted(CONTROLLER_OWNED_FIELDS))
def test_every_controller_owned_field_is_rejected(field):
    graph = make_graph()
    request = make_request(graph)
    raw = make_result(request, make_output(child())).model_dump(mode="json")
    raw["structured_output"]["nodes"][0][field] = [0.1] if "embedding" in field else "pollution"
    assert_rejected_unchanged(graph, request, raw, "schema validation failed")


@pytest.mark.parametrize("status", ["failed", "timed_out", "cancelled"])
def test_non_completed_results_are_rejected_without_mutation(status):
    graph = make_graph()
    request = make_request(graph)
    raw = make_result(request, make_output()).model_dump(mode="json")
    raw["status"] = status
    raw["structured_output"] = None
    raw["output_hash"] = compute_output_hash(None, request.output_schema_version)
    assert_rejected_unchanged(graph, request, EpisodeResult.model_validate(raw), f"status is {status}")


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("episode_id", "other", "episode ID mismatch"),
        ("attempt_id", "other", "attempt ID mismatch"),
        ("run_id", "other", "run ID mismatch"),
        ("role", "seed", "role mismatch"),
        ("schema_version", "other", "schema version mismatch"),
        ("output_hash", "0" * 64, "output hash mismatch"),
    ],
)
def test_envelope_mismatches_are_rejected(field, value, reason):
    graph = make_graph()
    request = make_request(graph)
    raw = make_result(request, make_output(child())).model_dump(mode="json")
    raw[field] = value
    assert_rejected_unchanged(graph, request, EpisodeResult.model_validate(raw), reason)


def test_event_emission_for_grant_start_completion_and_commit(tmp_path):
    graph = make_graph()
    request = make_request(graph)
    log = EpisodeEventLog(tmp_path / "episode_events.jsonl")
    adapter = NativeStubEpisodeAdapter(lambda _: make_output(child()))
    outcome = run_and_commit_episode(graph, request, adapter, telemetry=log)
    assert outcome.accepted is True
    assert [event["event_type"] for event in log.read_events()] == [
        "episode_granted",
        "episode_started",
        "episode_completed",
        "nodes_committed",
    ]
    assert log.read_events()[-1]["accepted_node_count"] == 1
    assert log.read_events()[-1]["usage_source"] == "unavailable"


def test_event_emission_for_rejection(tmp_path):
    graph = make_graph()
    request = make_request(graph, grant=0)
    log = EpisodeEventLog(tmp_path / "episode_events.jsonl")
    adapter = NativeStubEpisodeAdapter(lambda _: make_output(child()))
    outcome = run_and_commit_episode(graph, request, adapter, telemetry=log)
    assert outcome.accepted is False
    assert log.read_events()[-1]["event_type"] == "output_rejected"
    assert "exceeds grant" in log.read_events()[-1]["rejection_reason"]


def test_runtime_failure_emits_failed_and_rejected_events(tmp_path):
    class BrokenAdapter:
        adapter_name = "broken"
        transport_name = "test"

        def run_episode(self, request):
            raise RuntimeError("transport unavailable")

    graph = make_graph()
    request = make_request(graph)
    log = EpisodeEventLog(tmp_path / "episode_events.jsonl")
    before = graph.snapshot()
    outcome = run_and_commit_episode(graph, request, BrokenAdapter(), telemetry=log)
    assert outcome.accepted is False
    assert graph.snapshot() == before
    assert [event["event_type"] for event in log.read_events()] == [
        "episode_granted",
        "episode_started",
        "episode_failed",
        "output_rejected",
    ]


def test_strict_run_emits_run_lifecycle_events(tmp_path):
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=1),
    )
    strict_run(
        spec=spec,
        mode="smoke",
        out_dir=tmp_path / "out",
        cache_path=None,
        initial_nodes=[SearchNode(node_id="p", claim="parent")],
        control_path=None,
    )
    events = EpisodeEventLog(tmp_path / "out" / "episode_events.jsonl").read_events()
    assert events[0]["event_type"] == "run_created"
    assert events[-1]["event_type"] == "run_completed"
    assert events[0]["run_id"] == events[-1]["run_id"]


def test_command_adapter_conforms_to_agent_episode_protocol(tmp_path):
    script = tmp_path / "executor.py"
    script.write_text(
        "import json, sys\n"
        "r=json.loads(sys.stdin.read())\n"
        "p=r['parent']['node_id']\n"
        "print(json.dumps({'nodes':[{'node_id':'cmd-child','claim':'command','parent_ids':[p]}]}))\n",
        encoding="utf-8",
    )
    graph = make_graph()
    request = make_request(graph)
    adapter = CommandAgentEpisodeAdapter([sys.executable, str(script)])
    result = adapter.run_episode(request)
    assert result.status == "completed"
    assert result.structured_output.nodes[0].node_id == "cmd-child"
    assert result.runtime_diagnostics.transport_name == "subprocess"


def test_native_stub_conforms_without_minimum_subagents():
    graph = make_graph()
    request = make_request(graph)
    request.native_orchestration_allowed = True
    adapter = NativeStubEpisodeAdapter(lambda _: make_output(child()))
    result = adapter.run_episode(request)
    assert result.status == "completed"
    assert result.runtime_diagnostics.internal_subagent_metadata is None


def test_native_stub_runs_inside_dte_executor_vertical_slice(tmp_path):
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=1),
    )

    def produce(request):
        return make_output(child("native-child", request.parent_node_id))

    log = EpisodeEventLog(tmp_path / "episode_events.jsonl")
    result = run_frontier_search(
        spec,
        [SearchNode(node_id="parent", claim="parent")],
        episode_adapter=NativeStubEpisodeAdapter(produce),
        episode_event_log=log,
        run_id="vertical-slice",
    )
    assert [node.node_id for node in result.nodes] == ["parent", "native-child"]
    assert result.nodes[0].status == "closed"
    assert result.nodes[1].status == "frontier"
    assert log.read_events()[-1]["event_type"] == "nodes_committed"


def test_internal_subagent_metadata_cannot_become_graph_facts():
    graph = make_graph()
    request = make_request(graph)
    result = make_result(request, make_output(child()))
    result.runtime_diagnostics.internal_subagent_metadata = {
        "agents": [{"name": "private-worker", "claim": "must not enter graph"}]
    }
    outcome = commit_episode_result(graph, request, result)
    assert outcome.accepted is True
    dumped_graph = json.dumps(graph.snapshot(), sort_keys=True)
    assert "private-worker" not in dumped_graph
    assert "must not enter graph" not in dumped_graph
