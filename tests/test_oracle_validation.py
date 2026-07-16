import pytest

from dte_backend.models import SearchNode
from dte_backend.oracle_validation import validate_judge_output, validate_relation_output


def test_judge_validation_accepts_observable_scores():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    result = validate_judge_output(
        nodes,
        {
            "results": [
                {"node_id": "a", "score": 0.7, "reasoning": "ok", "risks": []},
                {"node_id": "b", "score": 0.4, "reasoning": "weak", "risks": ["risk"]},
            ]
        },
    )
    assert [r.node_id for r in result] == ["a", "b"]


def test_judge_validation_rejects_controller_fields():
    nodes = [SearchNode(node_id="a", claim="A")]
    with pytest.raises(ValueError, match="forbidden"):
        validate_judge_output(nodes, {"results": [{"node_id": "a", "score": 0.7, "reasoning": "x", "ucb_score": 9}]})


def test_judge_validation_rejects_duplicate_node_id_with_complete_coverage():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    with pytest.raises(ValueError, match="duplicate node_id: a"):
        validate_judge_output(
            nodes,
            {
                "results": [
                    {"node_id": "a", "score": 0.1, "reasoning": "first"},
                    {"node_id": "a", "score": 0.9, "reasoning": "second"},
                    {"node_id": "b", "score": 0.5, "reasoning": "complete coverage"},
                ]
            },
        )


def test_judge_validation_rejects_duplicate_input_identity():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="a", claim="again")]
    with pytest.raises(ValueError, match="input contains duplicate"):
        validate_judge_output(
            nodes,
            {"results": [{"node_id": "a", "score": 0.5, "reasoning": "ambiguous"}]},
        )


def test_judge_validation_requires_reasoning():
    nodes = [SearchNode(node_id="a", claim="A")]
    with pytest.raises(ValueError, match="reasoning"):
        validate_judge_output(nodes, {"results": [{"node_id": "a", "score": 0.5}]})


@pytest.mark.parametrize("field", ["density", "status", "graph_revision", "synthesis_result"])
def test_judge_validation_rejects_every_undeclared_field(field):
    nodes = [SearchNode(node_id="a", claim="A")]
    item = {"node_id": "a", "score": 0.7, "reasoning": "x", field: "forged"}
    with pytest.raises(ValueError, match="forbidden"):
        validate_judge_output(nodes, {"results": [item]})


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
