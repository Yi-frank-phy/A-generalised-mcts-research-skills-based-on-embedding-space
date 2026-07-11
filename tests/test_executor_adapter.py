import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from dte_backend.executor_adapter import (
    ExpansionRequest,
    SubprocessExecutorAdapter,
    validate_executor_children,
)
from dte_backend.expansion import expand_frontier
from dte_backend.models import BudgetSpec, DTERunSpec, SearchNode
from dte_backend.runner import run_frontier_search


class RecordingAdapter:
    def __init__(self) -> None:
        self.requests: list[ExpansionRequest] = []

    def expand(self, request: ExpansionRequest) -> list[SearchNode]:
        self.requests.append(request)
        return [
            SearchNode(
                node_id=f"{request.parent.node_id}-adapter-child",
                claim=f"adapter child for {request.parent.claim}",
                rationale="structured executor output",
                parent_ids=[request.parent.node_id],
                confidence=0.44,
            )
        ]


class BadObjectAdapter:
    def expand(self, request: ExpansionRequest) -> list[SearchNode]:
        return [
            SearchNode(
                node_id=f"{request.parent.node_id}-bad-child",
                claim="bad child",
                parent_ids=[request.parent.node_id],
                score=0.9,
            )
        ]


def test_expand_frontier_uses_executor_adapter_and_closes_parent():
    parent = SearchNode(node_id="p", claim="parent")
    adapter = RecordingAdapter()

    nodes = expand_frontier([parent], {"p": 1}, iteration=2, executor_adapter=adapter)

    assert parent.status == "closed"
    assert len(adapter.requests) == 1
    assert adapter.requests[0].parent.node_id == "p"
    assert nodes[-1].parent_ids == ["p"]
    assert nodes[-1].status == "frontier"


def test_expand_frontier_validates_object_adapter_output_before_consuming():
    parent = SearchNode(node_id="p", claim="parent")

    with pytest.raises(ValueError, match="controller-owned field: score"):
        expand_frontier([parent], {"p": 1}, iteration=2, executor_adapter=BadObjectAdapter())


def test_run_frontier_search_keeps_adapter_inside_mandatory_loop():
    spec = DTERunSpec(
        problem="p",
        goal="g",
        budget=BudgetSpec(max_iterations=1, allocation_mass_per_iteration=1),
    )
    adapter = RecordingAdapter()
    result = run_frontier_search(spec, [SearchNode(node_id="p", claim="parent")], executor_adapter=adapter)

    assert len(adapter.requests) == 1
    assert result.traces[0].allocations
    assert result.nodes[0].score is not None
    assert result.nodes[-1].score is None
    assert "DTE Prototype Report" in result.report


def test_executor_child_cannot_return_synthesis_node():
    parent = SearchNode(node_id="p", claim="parent")
    child = SearchNode(node_id="s", node_type="synthesis", claim="bad", parent_ids=["p"], status="synthesis")

    with pytest.raises(ValueError, match="synthesis"):
        validate_executor_children(parent, 1, [child])


def test_executor_child_count_cannot_exceed_budget():
    parent = SearchNode(node_id="p", claim="parent")
    children = [
        SearchNode(node_id="c1", claim="child 1", parent_ids=["p"]),
        SearchNode(node_id="c2", claim="child 2", parent_ids=["p"]),
    ]

    with pytest.raises(ValueError, match="budget 1"):
        validate_executor_children(parent, 1, children)


def test_executor_child_must_include_parent_id():
    parent = SearchNode(node_id="p", claim="parent")
    child = SearchNode(node_id="c", claim="bad", parent_ids=["other"])

    with pytest.raises(ValueError, match="parent id"):
        validate_executor_children(parent, 1, [child])


def test_executor_child_must_return_to_frontier():
    parent = SearchNode(node_id="p", claim="parent")
    child = SearchNode(node_id="c", claim="bad", parent_ids=["p"], status="closed")

    with pytest.raises(ValueError, match="frontier"):
        validate_executor_children(parent, 1, [child])


def test_executor_child_cannot_prefill_judge_metrics():
    parent = SearchNode(node_id="p", claim="parent")
    child = SearchNode(node_id="c", claim="bad", parent_ids=["p"], score=0.9)

    with pytest.raises(ValueError, match="metrics"):
        validate_executor_children(parent, 1, [child])


def test_executor_child_cannot_prefill_expansion_budget():
    parent = SearchNode(node_id="p", claim="parent")
    child = SearchNode(node_id="c", claim="bad", parent_ids=["p"], expansion_budget=1)

    with pytest.raises(ValueError, match="expansion budgets"):
        validate_executor_children(parent, 1, [child])


def test_search_node_rejects_extra_free_form_fields():
    with pytest.raises(ValidationError):
        SearchNode.model_validate(
            {
                "node_id": "c",
                "node_type": "candidate",
                "claim": "child",
                "final_answer": "executor tries to bypass DTE synthesis",
            }
        )


def test_expansion_request_rejects_extra_free_form_fields():
    parent = SearchNode(node_id="p", claim="parent")

    with pytest.raises(ValidationError):
        ExpansionRequest.model_validate(
            {
                "parent": parent.model_dump(),
                "count": 1,
                "iteration": 1,
                "final_answer": "request tries to smuggle a conclusion",
            }
        )


def test_subprocess_executor_adapter_reads_structured_json(tmp_path):
    script = tmp_path / "adapter.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "request = json.loads(sys.stdin.read())",
                "parent_id = request['parent']['node_id']",
                "print(json.dumps({'nodes': [{",
                "    'node_id': parent_id + '-child',",
                "    'node_type': 'candidate',",
                "    'claim': 'child',",
                "    'rationale': 'from subprocess',",
                "    'parent_ids': [parent_id],",
                "    'confidence': 0.4",
                "}]}))",
            ]
        ),
        encoding="utf-8",
    )
    parent = SearchNode(node_id="p", claim="parent")
    request = ExpansionRequest(parent=parent, count=1, iteration=1)
    adapter = SubprocessExecutorAdapter([sys.executable, str(script)])

    children = adapter.expand(request)

    assert children[0].node_id == "p-child"
    assert children[0].parent_ids == ["p"]


def test_subprocess_executor_adapter_rejects_nonzero_exit(tmp_path):
    script = tmp_path / "adapter.py"
    script.write_text("import sys\nsys.stderr.write('failed')\nsys.exit(3)\n", encoding="utf-8")
    parent = SearchNode(node_id="p", claim="parent")
    request = ExpansionRequest(parent=parent, count=1, iteration=1)
    adapter = SubprocessExecutorAdapter([sys.executable, str(script)])

    with pytest.raises(RuntimeError, match="failed"):
        adapter.expand(request)


def test_subprocess_executor_adapter_requires_node_list(tmp_path):
    script = tmp_path / "adapter.py"
    script.write_text("print('{\"not_nodes\": []}')\n", encoding="utf-8")
    parent = SearchNode(node_id="p", claim="parent")
    request = ExpansionRequest(parent=parent, count=1, iteration=1)
    adapter = SubprocessExecutorAdapter([sys.executable, str(script)])

    with pytest.raises(ValueError, match="nodes list"):
        adapter.expand(request)


def test_validate_executor_cli_runs_example_adapter():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dte_backend",
            "validate-executor",
            "--request",
            "examples/expansion_request.json",
            "--executor-command",
            f"{sys.executable} examples/echo_executor_adapter.py",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "seed-direct-executor-1" in completed.stdout
