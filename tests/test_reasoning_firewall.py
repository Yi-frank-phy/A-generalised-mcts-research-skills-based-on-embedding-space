import pytest

from dte_backend.episode_adapter import build_executor_episode_request
from dte_backend.episode_commit import EpisodeGraph
from dte_backend.episode_models import (
    REASONING_BOUNDARY_TRANSPORT_HINT_KEY,
    compute_role_context_manifest_hash,
    reasoning_boundary_transport_hint,
)
from dte_backend.models import SearchNode


def executor_request(*, transport_hints=None):
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
        isolation_mode="shared_context_single_agent",
    )


def test_new_episode_grants_bind_reasoning_firewall_before_manifest_hash():
    request = executor_request(transport_hints={"adapter_hint": "preserved"})

    assert request.transport_hints is not None
    assert request.transport_hints["adapter_hint"] == "preserved"
    assert (
        request.transport_hints[REASONING_BOUNDARY_TRANSPORT_HINT_KEY]
        == reasoning_boundary_transport_hint()
    )
    assert (
        request.role_execution_contract.context_manifest_hash
        == compute_role_context_manifest_hash(request)
    )


def test_reasoning_firewall_rejects_cross_episode_private_reasoning_override():
    with pytest.raises(ValueError, match="backend-reserved"):
        executor_request(
            transport_hints={
                REASONING_BOUNDARY_TRANSPORT_HINT_KEY: {
                    "schema_version": "dte-reasoning-boundary.v1",
                    "continuity_scope": "whole_run",
                    "cross_episode_private_reasoning_allowed": True,
                    "provider_retained_reasoning_attested": True,
                    "provider_compaction_attested": True,
                }
            }
        )


def test_reasoning_firewall_is_covered_by_context_manifest():
    request = executor_request()
    original_hash = request.role_execution_contract.context_manifest_hash
    assert request.transport_hints is not None

    request.transport_hints[REASONING_BOUNDARY_TRANSPORT_HINT_KEY][
        "cross_episode_private_reasoning_allowed"
    ] = True

    assert compute_role_context_manifest_hash(request) != original_hash
