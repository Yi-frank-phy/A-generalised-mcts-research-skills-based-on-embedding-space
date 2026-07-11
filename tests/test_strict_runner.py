import pytest
import json

from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.strict_runner import StrictRunError, enforce_strict_policy, policy_for_mode, strict_run


def test_real_mode_rejects_hash_geometry():
    spec = DTERunSpec(problem="p", goal="g", embedding_provider="hash")
    with pytest.raises(StrictRunError, match="hash embedding"):
        enforce_strict_policy(
            spec,
            policy=policy_for_mode("real"),
            cache_path=".dte_cache/cache.json",
            judge_command="python real_judge.py",
        )


def test_real_mode_requires_judge_command(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    spec = DTERunSpec(problem="p", goal="g", embedding_provider="gemini-embedding-2", embedding_dimension=3072)
    with pytest.raises(StrictRunError, match="requires --judge-command"):
        enforce_strict_policy(
            spec,
            policy=policy_for_mode("real"),
            cache_path=".dte_cache/cache.json",
            judge_command=None,
        )


def test_real_mode_rejects_mock_judge(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    spec = DTERunSpec(problem="p", goal="g", embedding_provider="gemini-embedding-2", embedding_dimension=3072)
    with pytest.raises(StrictRunError, match="mock Judge"):
        enforce_strict_policy(
            spec,
            policy=policy_for_mode("real"),
            cache_path=".dte_cache/cache.json",
            judge_command="python examples/mock_judge_adapter.py",
        )


def test_real_mode_rejects_mock_executor(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    spec = DTERunSpec(problem="p", goal="g", embedding_provider="gemini-embedding-2", embedding_dimension=3072)
    with pytest.raises(StrictRunError, match="mock Executor"):
        enforce_strict_policy(
            spec,
            policy=policy_for_mode("real"),
            cache_path=".dte_cache/cache.json",
            judge_command="python real_judge.py",
            executor_command="python examples/mock_executor_adapter.py",
        )


def test_real_mode_requires_cache_path(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    spec = DTERunSpec(problem="p", goal="g", embedding_provider="gemini-embedding-2", embedding_dimension=3072)
    with pytest.raises(StrictRunError, match="requires --cache-path"):
        enforce_strict_policy(
            spec,
            policy=policy_for_mode("real"),
            cache_path=None,
            judge_command="python real_judge.py",
        )


def test_smoke_mode_allows_mock_and_hash():
    spec = DTERunSpec(problem="p", goal="g", embedding_provider="hash")
    enforce_strict_policy(
        spec,
        policy=policy_for_mode("smoke"),
        cache_path=None,
        judge_command="python examples/mock_judge_adapter.py",
    )


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
        SearchNode(node_id="a", claim="route A", confidence=0.7),
        SearchNode(node_id="b", claim="route B", confidence=0.6),
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
    assert status["forced_synthesis"]["control_path"] == str(control_path)
    assert status["control_path"] == str(control_path)
    assert "Forced Synthesis" in report
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
    nodes = [SearchNode(node_id="a", claim="route A")]
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
