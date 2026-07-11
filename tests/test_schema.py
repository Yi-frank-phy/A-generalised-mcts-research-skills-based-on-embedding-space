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
    assert "total_child_budget" not in serialized


def test_generated_budget_schema_uses_canonical_fields():
    schema = BudgetSpec.model_json_schema()["properties"]
    assert "allocation_mass_per_iteration" in schema
    assert "max_children_per_iteration" in schema
    assert "total_child_budget" not in schema


def test_checked_in_run_schema_and_example_use_canonical_budget_fields():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas" / "run_spec.schema.json").read_text(encoding="utf-8"))
    example = json.loads((root / "examples" / "run_spec.json").read_text(encoding="utf-8"))
    schema_budget = schema["properties"]["budget"]
    example_budget = example["budget"]

    assert "allocation_mass_per_iteration" in schema_budget["properties"]
    assert "max_children_per_iteration" in schema_budget["properties"]
    assert "total_child_budget" not in schema_budget["properties"]
    assert example_budget["allocation_mass_per_iteration"] == 3
    assert example_budget["max_children_per_iteration"] == 5
    assert "total_child_budget" not in example_budget


def test_search_node_validates():
    node = SearchNode(node_id="n", claim="claim")
    assert node.status == "frontier"


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


def test_synthesis_control_request_rejects_legacy_main_agent_request():
    with pytest.raises(ValidationError, match="requested_by"):
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="main_agent",
            reason="model root tried to stop the run",
        )


def test_synthesis_control_request_rejects_missing_node_ids():
    with pytest.raises(ValidationError, match="requires at least one node_id"):
        SynthesisControlRequest(
            action="force_synthesis_after_current_task",
            requested_by="user",
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


def test_checked_in_synthesis_control_schema_and_example_are_user_only():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas" / "synthesis_control_request.schema.json").read_text(encoding="utf-8"))
    example = json.loads((root / "examples" / "synthesis_control_request.json").read_text(encoding="utf-8"))

    assert schema == SynthesisControlRequest.model_json_schema()
    assert schema["properties"]["requested_by"]["const"] == "user"
    assert example["requested_by"] == "user"
