"""Subprocess runners for DTE oracle tasks."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence

from .models import SearchNode
from .oracle_validation import validate_relation_output
from .oracles import RelationOracleResult, make_relation_task

RelationAdapter = Callable[[list[SearchNode]], RelationOracleResult]






def run_subprocess_relation(command: Sequence[str], nodes: list[SearchNode], timeout: float = 360.0) -> RelationOracleResult:
    task = make_relation_task(nodes)
    payload = {"task": task.__dict__, "nodes": [node.model_dump() for node in nodes]}
    completed = subprocess.run(
        list(command),
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"relation oracle failed: {completed.stderr.strip()}")
    return validate_relation_output(nodes, completed.stdout)


def build_subprocess_relation_adapter(command: Sequence[str], timeout: float = 360.0) -> RelationAdapter:
    def adapter(nodes: list[SearchNode]) -> RelationOracleResult:
        return run_subprocess_relation(command, nodes, timeout=timeout)

    return adapter
