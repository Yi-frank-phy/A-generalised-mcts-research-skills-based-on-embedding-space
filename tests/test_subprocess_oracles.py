import sys

from dte_backend.models import SearchNode
from dte_backend.subprocess_oracles import run_subprocess_judge, run_subprocess_relation


def test_mock_judge_subprocess():
    nodes = [SearchNode(node_id="a", claim="A"), SearchNode(node_id="b", claim="B")]
    results = run_subprocess_judge([sys.executable, "examples/mock_judge_adapter.py"], nodes)
    assert len(results) == 2


def test_mock_relation_subprocess():
    nodes = [SearchNode(node_id="a", claim="same"), SearchNode(node_id="b", claim=" same ")]
    result = run_subprocess_relation([sys.executable, "examples/mock_relation_adapter.py"], nodes)
    assert result.relation == "equivalent"
