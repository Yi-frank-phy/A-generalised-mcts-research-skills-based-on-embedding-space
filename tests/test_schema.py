import pytest
from pydantic import ValidationError

from dte_backend.models import DTERunSpec, SearchNode, SynthesisControlRequest


def test_run_spec_validates():
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget={"max_iterations": 2, "total_child_budget": 3, "max_research_iterations": 1},
    )
    assert spec.mode == "mandatory_frontier"


def test_search_node_validates():
    node = SearchNode(node_id="n", claim="claim")
    assert node.status == "frontier"


def test_synthesis_control_request_validates_scope():
    request = SynthesisControlRequest(
        action="force_synthesis_after_current_task",
        requested_by="main_agent",
        reason="reviewed checkpoint",
        scope="node_ids",
        node_ids=["n1"],
    )
    assert request.node_ids == ["n1"]


def test_synthesis_control_request_rejects_missing_node_ids():
    with pytest.raises(ValidationError, match="requires at least one node_id"):
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="reviewed checkpoint",
            scope="node_ids",
        )


def test_synthesis_control_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="user",
            reason="reviewed checkpoint",
            scope="all",
            score=1.0,
        )
