"""Executor adapter boundary for DTE expansion.

Adapters run external research/coding/proof episodes, but they do not judge,
allocate, or synthesize. Their only authority is to return structured child
SearchNode objects that re-enter the mandatory DTE loop.
"""

from __future__ import annotations

import json
import subprocess
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .models import DTERunSpec, SearchNode


class ExpansionRequest(BaseModel):
    """Structured input passed to one executor episode."""

    model_config = ConfigDict(extra="forbid")

    parent: SearchNode
    count: int = Field(ge=0)
    iteration: int = Field(ge=1)
    spec: DTERunSpec | None = None

    def to_json_dict(self) -> dict:
        """Return the stable JSON payload sent to external adapters."""

        return self.model_dump()


class ExecutorAdapter(Protocol):
    """Adapter protocol used by the DTE expansion role."""

    def expand(self, request: ExpansionRequest) -> list[SearchNode]:
        """Return child SearchNodes for the given parent and budget."""


def validate_executor_children(parent: SearchNode, count: int, children: list[SearchNode]) -> list[SearchNode]:
    """Validate that executor output cannot bypass Judge/Evolution/Synthesis."""

    if len(children) > count:
        raise ValueError(f"Executor returned {len(children)} children for budget {count}")

    for child in children:
        if child.node_type == "synthesis" or child.status == "synthesis":
            raise ValueError("Executor adapters cannot return synthesis nodes")
        if parent.node_id not in child.parent_ids:
            raise ValueError(f"Executor child {child.node_id} must include parent id {parent.node_id}")
        if child.score is not None or child.uncertainty is not None or child.ucb_score is not None:
            raise ValueError("Executor adapters cannot pre-fill Judge/Evolution metrics")
        if child.expansion_budget != 0:
            raise ValueError("Executor adapters cannot pre-fill expansion budgets")
        if child.status != "frontier":
            raise ValueError("Executor children must return to the frontier")

    return children


class SubprocessExecutorAdapter:
    """Run an external adapter command with JSON stdin/stdout.

    The command receives an ExpansionRequest JSON object on stdin and must write
    either a JSON list of SearchNode objects or {"nodes": [...]} to stdout.
    """

    def __init__(self, command: list[str], timeout_seconds: float = 120.0) -> None:
        self.command = list(command)
        self.timeout_seconds = timeout_seconds

    def expand(self, request: ExpansionRequest) -> list[SearchNode]:
        payload = json.dumps(request.to_json_dict(), ensure_ascii=False)
        completed = subprocess.run(
            self.command,
            input=payload,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"executor exited with {completed.returncode}")

        data = json.loads(completed.stdout)
        raw_nodes = data["nodes"] if isinstance(data, dict) and "nodes" in data else data
        if not isinstance(raw_nodes, list):
            raise ValueError("Executor stdout must be a JSON list or an object with a nodes list")

        children = [SearchNode.model_validate(item) for item in raw_nodes]
        return validate_executor_children(request.parent, request.count, children)
