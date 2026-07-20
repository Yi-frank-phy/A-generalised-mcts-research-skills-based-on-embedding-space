import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dte_backend.models import (
    BudgetSpec,
    DTERunSpec,
    ExpansionRequest,
    SearchNode,
    SynthesisControlRequest,
)


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence", float("nan")),
        ("density", float("nan")),
        ("uncertainty", float("inf")),
        ("ucb_score", float("-inf")),
        ("local_embedding", [0.0, float("nan")]),
    ],
)
def test_machine_models_reject_nonfinite_floats(field, value):
    with pytest.raises(ValidationError, match="finite number"):
        SearchNode(node_id="n", claim="finite contract", **{field: value})


def test_run_spec_serializes_only_canonical_budget_fields():
    spec = DTERunSpec(problem="p", goal="g", budget={"total_child_budget": 3})
    serialized = spec.model_dump()["budget"]
    assert serialized["allocation_mass_per_iteration"] == 3
    assert serialized["max_children_per_iteration"] == 5
    assert serialized["max_relation_pairs_per_episode"] == 3
    assert serialized["max_relation_enrichment_pairs"] == 3
    assert serialized["max_committed_search_nodes"] == 20
    assert serialized["max_iterations"] == 10
    assert serialized["entropy_plateau_confirmations"] == 2
    assert serialized["continuation_policy"] == "bounded_node_yield_v1"
    assert "total_child_budget" not in serialized


def test_generated_budget_schema_uses_canonical_fields():
    schema = BudgetSpec.model_json_schema()["properties"]
    assert "allocation_mass_per_iteration" in schema
    assert "max_children_per_iteration" in schema
    assert schema["max_relation_pairs_per_episode"]["default"] == 3
    assert schema["max_relation_enrichment_pairs"]["default"] == 3
    assert schema["max_relation_enrichment_pairs"]["minimum"] == 0
    assert schema["max_committed_search_nodes"]["default"] == 20
    assert schema["max_committed_search_nodes"]["maximum"] == 100
    assert schema["max_iterations"]["default"] == 10
    assert schema["entropy_plateau_confirmations"]["default"] == 2
    assert "total_child_budget" not in schema


def test_checked_in_run_schema_and_example_use_canonical_budget_fields():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas" / "run_spec.schema.json").read_text(encoding="utf-8"))
    example = json.loads((root / "examples" / "run_spec.json").read_text(encoding="utf-8"))
    schema_budget = schema["$defs"]["BudgetSpec"]
    example_budget = example["budget"]

    assert "allocation_mass_per_iteration" in schema_budget["properties"]
    assert "max_children_per_iteration" in schema_budget["properties"]
    assert "max_relation_pairs_per_episode" in schema_budget["properties"]
    assert "max_relation_enrichment_pairs" in schema_budget["properties"]
    assert "max_committed_search_nodes" in schema_budget["properties"]
    assert "entropy_plateau_confirmations" in schema_budget["properties"]
    assert "continuation_policy" in schema_budget["properties"]
    assert "total_child_budget" not in schema_budget["properties"]
    assert example_budget["allocation_mass_per_iteration"] == 3
    assert example_budget["max_children_per_iteration"] == 5
    assert example_budget["max_relation_pairs_per_episode"] == 3
    assert example_budget["max_relation_enrichment_pairs"] == 3
    assert example_budget["max_committed_search_nodes"] == 20
    assert example_budget["entropy_plateau_confirmations"] == 2
    assert example_budget["continuation_policy"] == "bounded_node_yield_v1"
    assert "total_child_budget" not in example_budget
    assert schema["$defs"]["OperatorPolicy"]["properties"]["main_agent_may_request_synthesis"]["default"] is True
    assert example["operator_policy"]["main_agent_may_request_synthesis"] is True


def test_search_node_validates():
    node = SearchNode(node_id="n", claim="claim")
    assert node.status == "frontier"


@pytest.mark.parametrize(
    ("schema_name", "model_type"),
    [
        ("search_node.schema.json", SearchNode),
        ("expansion_request.schema.json", ExpansionRequest),
        ("run_spec.schema.json", DTERunSpec),
    ],
)
def test_checked_in_machine_schema_matches_canonical_pydantic_serialization(
    schema_name, model_type
):
    root = Path(__file__).resolve().parents[1]
    checked_schema = json.loads(
        (root / "schemas" / schema_name).read_text(encoding="utf-8")
    )

    assert checked_schema == model_type.model_json_schema(mode="serialization")


def test_machine_contracts_round_trip_canonical_json():
    node = SearchNode(
        node_id="n",
        node_type="evidence",
        claim="claim",
        rationale="rationale",
        assumptions=["assumption"],
        evidence=["evidence"],
        risks=["risk"],
        parent_ids=["parent"],
        confidence=0.8,
        local_embedding=[0.1, 0.2],
        judge_reasoning="reasoning",
        judge_risks=["judge risk"],
        judge_uncertainty_evidence=["gap"],
        judge_result_provenance={"episode_id": "episode"},
        score=0.7,
        density=0.2,
        uncertainty=0.3,
        ucb_score=0.9,
        expansion_budget=2,
        status="closed",
    )
    spec = DTERunSpec(
        problem="problem",
        goal="goal",
        constraints=["constraint"],
        embedding_provider="gemini-embedding-2",
        embedding_dimension=3072,
    )
    request = ExpansionRequest(parent=node, child_count=2, iteration=1, spec=spec)

    for value in (node, spec, request):
        canonical_json = value.model_dump_json()
        assert type(value).model_validate_json(canonical_json) == value


def test_expansion_request_serializes_only_canonical_child_count():
    request = ExpansionRequest(
        parent=SearchNode(node_id="parent", claim="claim"),
        count=2,
        iteration=1,
    )

    serialized = request.model_dump(mode="json")
    checked_schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas" / "expansion_request.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert serialized["child_count"] == 2
    assert "count" not in serialized
    assert "child_count" in checked_schema["properties"]
    assert "count" not in checked_schema["properties"]


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
