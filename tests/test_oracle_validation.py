import pytest

from dte_backend.models import SearchNode
from dte_backend.oracle_validation import validate_relation_output














def test_relation_validation_accepts_conflict():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    result = validate_relation_output(
        nodes,
        {
            "relation": "conflict",
            "source_node_ids": ["a", "b"],
            "rationale": "assumptions clash",
            "discriminator_question": "Which assumption is necessary?",
        },
    )
    assert result.relation == "conflict"
    assert result.discriminator_question


def test_relation_validation_rejects_unknown_node():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    with pytest.raises(ValueError, match="known"):
        validate_relation_output(nodes, {"relation": "equivalent", "source_node_ids": ["a", "z"], "rationale": "x"})


def test_relation_validation_rejects_string_source_ids():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    with pytest.raises(ValueError, match="must be a list"):
        validate_relation_output(
            nodes,
            {"relation": "equivalent", "source_node_ids": "ab", "rationale": "x"},
        )


def test_relation_validation_rejects_duplicate_source_node_id():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    with pytest.raises(ValueError, match="duplicate node IDs"):
        validate_relation_output(
            nodes,
            {"relation": "equivalent", "source_node_ids": ["a", "a"], "rationale": "x"},
        )


@pytest.mark.parametrize("field", ["canonical_node_id", "merged_node", "ready_for_synthesis"])
def test_relation_validation_rejects_every_undeclared_field(field):
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    output = {
        "relation": "independent",
        "source_node_ids": ["a", "b"],
        "rationale": "separate",
        field: "forged",
    }
    with pytest.raises(ValueError, match="forbidden"):
        validate_relation_output(nodes, output)


def test_relation_validation_rejects_duplicate_input_identity():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="a", claim="again")]
    with pytest.raises(ValueError, match="input contains duplicate"):
        validate_relation_output(
            nodes,
            {"relation": "equivalent", "source_node_ids": ["a", "a"], "rationale": "x"},
        )
