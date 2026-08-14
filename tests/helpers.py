"""Test-only builders for valid completed research transitions on the `new` line."""

from __future__ import annotations

from dte_backend.episode_models import ExecutorNodeCandidate
from dte_backend.models import SearchNode


def completed_node(**kwargs) -> SearchNode:
    """Build an active test node with explicit completed-transition data.

    The production model intentionally keeps these persistence fields optional so
    legacy artifacts remain readable. Tests that exercise an active `new` node
    should use this helper instead of relying on legacy empty transition state.
    """

    node_id = str(kwargs.get("node_id", "test-node"))
    claim = str(kwargs.get("claim", node_id))
    kwargs.setdefault("retrospective_method", f"test method for {node_id}: {claim}")
    kwargs.setdefault("epistemic_change_kind", "new_understanding")
    kwargs.setdefault("epistemic_change", f"test epistemic change for {node_id}: {claim}")
    return SearchNode(**kwargs)


def completed_candidate(**kwargs) -> ExecutorNodeCandidate:
    """Build an Executor candidate that satisfies the `new` transition contract."""

    node_id = str(kwargs.get("node_id", "test-child"))
    claim = str(kwargs.get("claim", node_id))
    kwargs.setdefault("retrospective_method", f"test method for {node_id}: {claim}")
    kwargs.setdefault("epistemic_change_kind", "new_understanding")
    kwargs.setdefault("epistemic_change", f"test epistemic change for {node_id}: {claim}")
    return ExecutorNodeCandidate(**kwargs)
