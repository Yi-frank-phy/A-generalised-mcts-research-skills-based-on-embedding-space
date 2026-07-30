import pytest

from dte_backend.episode_adapter import build_executor_episode_request
from dte_backend.episode_commit import EpisodeGraph
from dte_backend.episode_models import (
    REASONING_BOUNDARY_REQUIREMENT_PREFIX,
    compute_role_context_manifest_hash,
    reasoning_boundary_requirement,
)
from dte_backend.models import SearchNode


def executor_request(*, transport_hints=None, coverage_requirements=None):
    parent = SearchNode(node_id="parent", claim="develop the assigned branch")
    graph = EpisodeGraph(nodes=[parent])
    return build_executor_episode_request(
        graph,
        parent,
        run_id="reasoning-firewall",
        iteration=1,
        max_returned_children=1,
        objective="produce one bounded child",
        transport_hints=transport_hints,
        coverage_requirements=coverage_requirements,
        isolation_mode="shared_context_single_agent",
    )


def test_new_episode_grants_bind_reasoning_firewall_before_manifest_hash():
    request = executor_request(transport_hints={"adapter_hint": "preserved"})

    assert request.transport_hints == {"adapter_hint": "preserved"}
    assert reasoning_boundary_requirement() in request.coverage_requirements
    assert (
        request.role_execution_contract.context_manifest_hash
        == compute_role_context_manifest_hash(request)
    )


def test_reasoning_firewall_rejects_cross_episode_override():
    with pytest.raises(ValueError, match="backend-reserved"):
        executor_request(
            coverage_requirements=[
                REASONING_BOUNDARY_REQUIREMENT_PREFIX
                + '{"continuity_scope":"whole_run"}'
            ]
        )


def test_reasoning_firewall_is_covered_by_context_manifest():
    request = executor_request()
    original_hash = request.role_execution_contract.context_manifest_hash
    boundary_index = request.coverage_requirements.index(
        reasoning_boundary_requirement()
    )

    request.coverage_requirements[boundary_index] = (
        REASONING_BOUNDARY_REQUIREMENT_PREFIX
        + '{"continuity_scope":"whole_run"}'
    )

    assert compute_role_context_manifest_hash(request) != original_hash
