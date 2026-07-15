import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode, SynthesisControlRequest


def test_legacy_run_spec_maps_total_child_budget_to_canonical_field():
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget={"max_iterations": 2, "total_child_budget": 3, "max_research_iterations": 1},
    )
    assert spec.mode == "mandatory_frontier"
    assert spec.budget.allocation_mass_per_iteration == 3


def test_budget_rejects_conflicting_legacy_and_canonical_values():
    with pytest.raises(ValidationError, match="total_child_budget"):
        BudgetSpec(allocation_mass_per_iteration=4, total_child_budget=3)


def test_budget_accepts_matching_legacy_and_canonical_values():
    budget = BudgetSpec(allocation_mass_per_iteration=3, total_child_budget=3)
    assert budget.allocation_mass_per_iteration == 3


def test_run_spec_serializes_only_canonical_budget_fields():
    spec = DTERunSpec(problem="p", goal="g", budget={"total_child_budget": 3})
    serialized = spec.model_dump()["budget"]
    assert serialized["allocation_mass_per_iteration"] == 3
    assert serialized["max_children_per_iteration"] == 5
    assert serialized["max_relation_pairs_per_episode"] == 3
    assert serialized["max_relation_enrichment_pairs"] == 3
    assert "total_child_budget" not in serialized


def test_generated_budget_schema_uses_canonical_fields():
    schema = BudgetSpec.model_json_schema()["properties"]
    assert "allocation_mass_per_iteration" in schema
    assert "max_children_per_iteration" in schema
    assert schema["max_relation_pairs_per_episode"]["default"] == 3
    assert schema["max_relation_enrichment_pairs"]["default"] == 3
    assert schema["max_relation_enrichment_pairs"]["minimum"] == 0
    assert "total_child_budget" not in schema


def test_checked_in_run_schema_and_example_use_canonical_budget_fields():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas" / "run_spec.schema.json").read_text(encoding="utf-8"))
    example = json.loads((root / "examples" / "run_spec.json").read_text(encoding="utf-8"))
    schema_budget = schema["properties"]["budget"]
    example_budget = example["budget"]

    assert "allocation_mass_per_iteration" in schema_budget["properties"]
    assert "max_children_per_iteration" in schema_budget["properties"]
    assert "max_relation_pairs_per_episode" in schema_budget["properties"]
    assert "max_relation_enrichment_pairs" in schema_budget["properties"]
    assert "total_child_budget" not in schema_budget["properties"]
    assert example_budget["allocation_mass_per_iteration"] == 3
    assert example_budget["max_children_per_iteration"] == 5
    assert example_budget["max_relation_pairs_per_episode"] == 3
    assert example_budget["max_relation_enrichment_pairs"] == 3
    assert "total_child_budget" not in example_budget
    assert schema["properties"]["operator_policy"]["properties"]["main_agent_may_request_synthesis"]["default"] is True
    assert example["operator_policy"]["main_agent_may_request_synthesis"] is True


def test_search_node_validates():
    node = SearchNode(node_id="n", claim="claim")
    assert node.status == "frontier"


def test_operator_policy_allows_main_agent_synthesis_by_default():
    spec = DTERunSpec(problem="p", goal="g")
    assert spec.operator_policy.main_agent_may_request_synthesis is True


def test_synthesis_control_request_accepts_user():
    request = SynthesisControlRequest(
        action="force_synthesis_after_current_task",
        requested_by="user",
        reason="reviewed checkpoint",
        scope="node_ids",
        node_ids=["n1"],
    )
    assert request.node_ids == ["n1"]
    assert request.requested_by == "user"


def test_synthesis_control_request_accepts_main_agent_for_policy_authorization():
    request = SynthesisControlRequest(
        action="force_synthesis_after_current_task",
        requested_by="main_agent",
        reason="operator proxy requested synthesis",
    )
    assert request.requested_by == "main_agent"


def test_synthesis_control_request_rejects_unknown_requester():
    with pytest.raises(ValidationError, match="requested_by"):
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="executor",
            reason="unauthorized actor",
        )


def test_synthesis_control_request_rejects_missing_node_ids():
    with pytest.raises(ValidationError, match="requires at least one node_id"):
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="user",
            reason="reviewed checkpoint",
            scope="node_ids",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("score", 1.0),
        ("embedding", [0.1]),
        ("local_embedding", [0.1]),
        ("uncertainty", 0.1),
        ("ucb", 1.0),
        ("ucb_score", 1.0),
        ("allocation", 1),
        ("expansion_budget", 1),
        ("graph_status", "closed"),
        ("status", "closed"),
        ("synthesis_result", "bypass"),
    ],
)
def test_synthesis_control_request_rejects_controller_owned_fields(field, value):
    with pytest.raises(ValidationError):
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="reviewed checkpoint",
            scope="all",
            **{field: value},
        )


def test_checked_in_synthesis_control_schema_and_example_match_operator_contract():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas" / "synthesis_control_request.schema.json").read_text(encoding="utf-8"))
    example = json.loads((root / "examples" / "synthesis_control_request.json").read_text(encoding="utf-8"))

    assert schema == SynthesisControlRequest.model_json_schema()
    assert schema["properties"]["requested_by"]["enum"] == ["user", "main_agent"]
    assert example["requested_by"] == "main_agent"
