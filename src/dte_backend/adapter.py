"""Executor adapter boundary for external Codex/Kimi/OpenClaw episodes.

An adapter is allowed to perform local research/coding/proof work, but it must
return structured SearchNode children. It cannot judge, allocate, or synthesize.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from .models import DTERunSpec, ExpansionRequest, SearchNode

ExecutorAdapter = Callable[[ExpansionRequest], list[SearchNode]]

_FORBIDDEN_PREFILLED_METRICS = {"score", "uncertainty", "ucb_score"}


def _extract_raw_nodes(data: Any) -> list[dict[str, Any]]:
    """Accept either a JSON list or {"nodes": [...]} from adapter stdout."""

    if isinstance(data, list):
        raw_nodes = data
    elif isinstance(data, dict) and isinstance(data.get("nodes"), list):
        raw_nodes = data["nodes"]
    else:
        raise ValueError("executor adapter must return a JSON list or {'nodes': [...]} object")

    if not all(isinstance(item, dict) for item in raw_nodes):
        raise ValueError("executor adapter returned non-object node entries")
    return raw_nodes


def validate_adapter_output(parent: SearchNode, child_count: int, raw_output: str | Any) -> list[SearchNode]:
    """Validate raw adapter output against DTE's executor boundary.

    The adapter is deliberately not trusted. It must not pre-fill DTE metrics,
    create synthesis nodes, exceed the assigned child budget, or omit the parent
    id. This is what prevents external subagents from bypassing the controller.
    """

    data = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
    raw_nodes = _extract_raw_nodes(data)
    if len(raw_nodes) > child_count:
        raise ValueError(f"executor adapter returned {len(raw_nodes)} nodes for budget {child_count}")

    nodes: list[SearchNode] = []
    for raw in raw_nodes:
        for metric in _FORBIDDEN_PREFILLED_METRICS:
            if raw.get(metric) is not None:
                raise ValueError(f"executor adapter may not pre-fill DTE metric: {metric}")
        if int(raw.get("expansion_budget", 0)) != 0:
            raise ValueError("executor adapter may not pre-fill expansion_budget")

        node = SearchNode.model_validate(raw)
        if node.node_type == "synthesis" or node.status == "synthesis":
            raise ValueError("executor adapter may not produce synthesis nodes")
        if node.status != "frontier":
            raise ValueError("executor adapter children must have status='frontier'")
        if parent.node_id not in node.parent_ids:
            raise ValueError("executor adapter child must include the expanded parent id")
        nodes.append(node)

    return nodes


def run_subprocess_executor(command: Sequence[str], request: ExpansionRequest, timeout: float = 120.0) -> list[SearchNode]:
    """Run an external executor command with ExpansionRequest on stdin.

    Args:
        command: argv-style command, for example ["python", "adapter.py"].
        request: validated DTE expansion request.
        timeout: subprocess timeout in seconds.

    Returns:
        Validated SearchNode children.
    """

    completed = subprocess.run(
        list(command),
        input=request.model_dump_json(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "executor adapter failed "
            f"with code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return validate_adapter_output(request.parent, request.child_count, completed.stdout)


def build_subprocess_adapter(command: Sequence[str], timeout: float = 120.0) -> ExecutorAdapter:
    """Create an ExecutorAdapter function around a subprocess command."""

    def adapter(request: ExpansionRequest) -> list[SearchNode]:
        return run_subprocess_executor(command, request, timeout=timeout)

    return adapter
