import pytest

from dte_backend.models import DTERunSpec
from dte_backend.strict_runner import StrictRunError, enforce_strict_policy, policy_for_mode


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
