import os
import sys

from dte_backend.models import SearchNode
from dte_backend.subprocess_oracles import run_subprocess_relation




def test_mock_relation_subprocess(monkeypatch):
    monkeypatch.setenv("DTE_ALLOW_MOCK_ADAPTER", "1")
    nodes = [SearchNode(node_id="a", claim="same"), SearchNode(node_id="b", claim=" same ")]
    result = run_subprocess_relation([sys.executable, "examples/mock_relation_adapter.py"], nodes)
    assert result.relation == "equivalent"


