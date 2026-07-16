import json
import subprocess
import sys

import pytest

from dte_backend.adapter import build_subprocess_adapter, validate_adapter_output
from dte_backend.models import BudgetSpec, DTERunSpec, ExpansionRequest, SearchNode
from dte_backend.runner import run_frontier_search


def test_validate_adapter_output_rejects_metric_prefill():
    parent = SearchNode(node_id="p", claim="parent")
    raw = json.dumps(
        {
            "nodes": [
                {
                    "node_id": "child",
                    "claim": "child",
                    "parent_ids": ["p"],
                    "score": 0.9,
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="controller-owned field: score"):
        validate_adapter_output(parent, child_count=1, raw_output=raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("local_embedding", None),
        ("judge_reasoning", None),
        ("score", None),
        ("uncertainty", None),
        ("ucb_score", None),
        ("expansion_budget", 0),
    ],
)
def test_validate_adapter_output_rejects_controller_owned_field_presence(field, value):
    parent = SearchNode(node_id="p", claim="parent")
    raw = {"nodes": [{"node_id": "child", "claim": "child", "parent_ids": ["p"], field: value}]}

    with pytest.raises(ValueError, match=field):
        validate_adapter_output(parent, child_count=1, raw_output=raw)


def test_validate_adapter_output_still_accepts_legal_output():
    parent = SearchNode(node_id="p", claim="parent")
    raw = {"nodes": [{"node_id": "child", "claim": "child", "parent_ids": ["p"]}]}

    children = validate_adapter_output(parent, child_count=1, raw_output=raw)

    assert [child.node_id for child in children] == ["child"]


@pytest.mark.parametrize(
    "nodes,reason",
    [
        (
            [
                {"node_id": "child", "claim": "one", "parent_ids": ["p"]},
                {"node_id": "child", "claim": "two", "parent_ids": ["p"]},
            ],
            "duplicate node_id",
        ),
        ([{"node_id": "p", "claim": "collision", "parent_ids": ["p"]}], "conflicts"),
        (
            [{"node_id": "child", "claim": "duplicate parents", "parent_ids": ["p", "p"]}],
            "duplicate parent",
        ),
        (
            [{"node_id": "child", "claim": "self", "parent_ids": ["p", "child"]}],
            "parent itself",
        ),
    ],
)
def test_validate_adapter_output_rejects_identity_and_ancestry_violations(nodes, reason):
    parent = SearchNode(node_id="p", claim="parent")
    with pytest.raises(ValueError, match=reason):
        validate_adapter_output(parent, child_count=2, raw_output={"nodes": nodes})


def test_mock_subprocess_adapter_boundary():
    parent = SearchNode(node_id="p", claim="parent")
    adapter = build_subprocess_adapter([sys.executable, "examples/mock_executor_adapter.py"])
    request = ExpansionRequest(parent=parent, child_count=2, iteration=1)
    children = adapter(request)
    assert len(children) == 2
    assert all(child.status == "frontier" for child in children)
    assert all("p" in child.parent_ids for child in children)
    assert all(child.score is None for child in children)


def test_runner_uses_external_executor_adapter():
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=1),
    )
    nodes = [SearchNode(node_id="p", claim="parent", confidence=0.6)]
    adapter = build_subprocess_adapter([sys.executable, "examples/mock_executor_adapter.py"])
    result = run_frontier_search(spec, nodes, executor_adapter=adapter)
    assert any(node.node_id == "p" and node.status == "closed" for node in result.nodes)
    assert any(node.node_id.startswith("adapter-") for node in result.nodes)


def test_cli_executor_command(tmp_path):
    out_dir = tmp_path / "run"
    command = [
        sys.executable,
        "-m",
        "dte_backend",
        "run",
        "--spec",
        "examples/run_spec.json",
        "--nodes",
        "examples/frontier_nodes.json",
        "--out-dir",
        str(out_dir),
        "--executor-command",
        f"{sys.executable} examples/mock_executor_adapter.py",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.returncode == 0
    assert (out_dir / "nodes.json").exists()
    assert (out_dir / "cache_stats.json").exists()
