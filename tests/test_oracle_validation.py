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
