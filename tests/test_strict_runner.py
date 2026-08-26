from tests.helpers import completed_node
import json

import pytest

import dte_backend.strict_runner as strict_runner_module
from dte_backend.control import OperatorAuthorizationError
from dte_backend.embedding import HashEmbeddingProvider
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.strict_runner import StrictRunError, enforce_strict_policy, policy_for_mode, strict_run
















def test_strict_run_reads_control_file_and_records_forced_synthesis(tmp_path):
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(
            max_iterations=5,
            allocation_mass_per_iteration=2,
            min_iterations_before_synthesis=5,
        ),
    )
    nodes = [
        completed_node(node_id="a", claim="route A", confidence=0.7),
        completed_node(node_id="b", claim="route B", confidence=0.6),
    ]
    control_path = tmp_path / "strict_run_control.json"
    control_path.write_text(
        """
{
  "action": "force_synthesis_after_current_task",
  "requested_by": "user",
  "reason": "reviewed checkpoint in chat",
  "scope": "all"
}
""".strip(),
        encoding="utf-8",
    )

    result = strict_run(
        spec=spec,
        mode="smoke",
        out_dir=tmp_path / "out",
        cache_path=None,
        initial_nodes=nodes,
        control_path=control_path,
    )

    assert result.stop_reason == "user_interrupted_for_synthesis"
    status = json.loads((tmp_path / "out" / "strict_run_status.json").read_text(encoding="utf-8"))
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert status["stop_reason"] == "user_interrupted_for_synthesis"
    assert status["finalized"] is True
    assert status["forced_synthesis"]["control_path"] == str(control_path)
    assert status["control_path"] == str(control_path)
    assert "User-Interrupted Synthesis" in report
    assert "user_interrupted_for_synthesis" in report


def test_strict_run_rejects_invalid_control_file(tmp_path):
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(
            max_iterations=5,
            allocation_mass_per_iteration=1,
            min_iterations_before_synthesis=5,
        ),
    )
    nodes = [completed_node(node_id="a", claim="route A")]
    control_path = tmp_path / "strict_run_control.json"
    control_path.write_text(
        json.dumps(
            {
                "action": "force_synthesis_after_current_task",
                "requested_by": "user",
                "reason": "bad node id",
                "scope": "node_ids",
                "node_ids": ["missing"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown node ids"):
        strict_run(
            spec=spec,
            mode="smoke",
            out_dir=tmp_path / "out",
            cache_path=None,
            initial_nodes=nodes,
            control_path=control_path,
        )

    assert [node.node_id for node in nodes] == ["a"]
    assert nodes[0].status == "frontier"
    checkpoint_nodes = json.loads((tmp_path / "out" / "nodes.json").read_text(encoding="utf-8"))
    checkpoint_status = json.loads((tmp_path / "out" / "strict_run_status.json").read_text(encoding="utf-8"))
    assert [node["node_id"] for node in checkpoint_nodes] == ["a"]
    assert checkpoint_nodes[0]["status"] == "frontier"
    assert checkpoint_status["finalized"] is False
    assert not (tmp_path / "out" / "report.md").exists()


def test_strict_run_rejects_malformed_control_json_fail_closed(tmp_path):
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(max_iterations=2, allocation_mass_per_iteration=1, min_iterations_before_synthesis=2),
    )
    control_path = tmp_path / "operator-control.json"
    control_path.write_text('{"action":', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        strict_run(
            spec=spec,
            mode="smoke",
            out_dir=tmp_path / "out",
            cache_path=None,
            initial_nodes=[completed_node(node_id="a", claim="route A")],
            control_path=control_path,
        )

    status = json.loads((tmp_path / "out" / "strict_run_status.json").read_text(encoding="utf-8"))
    assert status["finalized"] is False
    assert status["stop_reason"] is None
    assert (tmp_path / "out" / "checkpoint_summary.md").exists()
    assert not (tmp_path / "out" / "report.md").exists()


def test_strict_run_without_control_path_registers_no_callback_and_finalizes_once(monkeypatch, tmp_path):
    original_search = strict_runner_module.run_frontier_search
    original_write = strict_runner_module.write_run_artifacts
    final_flags = []

    def recording_search(*args, **kwargs):
        assert kwargs["control_callback"] is None
        return original_search(*args, **kwargs)

    def recording_write(*args, **kwargs):
        final = kwargs["final"]
        final_flags.append(final)
        if not final:
            assert not (tmp_path / "out" / "report.md").exists()
        return original_write(*args, **kwargs)

    monkeypatch.setattr(strict_runner_module, "run_frontier_search", recording_search)
    monkeypatch.setattr(strict_runner_module, "write_run_artifacts", recording_write)
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=1, min_iterations_before_synthesis=1),
    )

    result = strict_run(
        spec=spec,
        mode="smoke",
        out_dir=tmp_path / "out",
        cache_path=None,
        initial_nodes=[completed_node(node_id="a", claim="route A")],
        control_path=None,
    )

    status = json.loads((tmp_path / "out" / "strict_run_status.json").read_text(encoding="utf-8"))
    assert result.stop_reason == "max_iterations"
    assert final_flags[:-1] and all(flag is False for flag in final_flags[:-1])
    assert final_flags[-1] is True
    assert status["control_path"] is None
    assert status["finalized"] is True
    assert (tmp_path / "out" / "checkpoint_summary.md").exists()
    assert (tmp_path / "out" / "report.md").exists()


def test_strict_run_rejects_main_agent_when_operator_policy_disables_it(tmp_path):
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(max_iterations=5, allocation_mass_per_iteration=1, min_iterations_before_synthesis=5),
        operator_policy={"main_agent_may_request_synthesis": False},
    )
    nodes = [completed_node(node_id="a", claim="route A")]
    control_path = tmp_path / "strict_run_control.json"
    control_path.write_text(
        json.dumps(
            {
                "action": "force_synthesis_after_current_task",
                "requested_by": "main_agent",
                "reason": "operator proxy requested synthesis",
                "scope": "all",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        OperatorAuthorizationError,
        match="main-agent synthesis request is disabled by operator_policy",
    ):
        strict_run(
            spec=spec,
            mode="smoke",
            out_dir=tmp_path / "out",
            cache_path=None,
            initial_nodes=nodes,
            control_path=control_path,
        )

    checkpoint_nodes = json.loads((tmp_path / "out" / "nodes.json").read_text(encoding="utf-8"))
    checkpoint_status = json.loads((tmp_path / "out" / "strict_run_status.json").read_text(encoding="utf-8"))
    assert [node.model_dump() for node in nodes] == checkpoint_nodes
    assert checkpoint_status["stop_reason"] is None
    assert checkpoint_status["forced_synthesis"] is None
    assert checkpoint_status["finalized"] is False
    assert not (tmp_path / "out" / "report.md").exists()


