"""Subprocess runners for DTE oracle tasks."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence

from .models import SearchNode
from .oracle_validation import validate_judge_output, validate_relation_output
from .oracles import JudgeOracleResult, RelationOracleResult, make_judge_task, make_relation_task

JudgeAdapter = Callable[[list[SearchNode]], list[JudgeOracleResult]]
RelationAdapter = Callable[[list[SearchNode]], RelationOracleResult]


def run_subprocess_judge(command: Sequence[str], nodes: list[SearchNode], timeout: float = 180.0) -> list[JudgeOracleResult]:
    task = make_judge_task(nodes)
    payload = {"task": task.__dict__, "nodes": [node.model_dump() for node in nodes]}
    completed = subprocess.run(list(command), input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"judge oracle failed: {completed.stderr.strip()}")
    return validate_judge_output(nodes, completed.stdout)


def build_subprocess_judge_adapter(command: Sequence[str], timeout: float = 180.0) -> JudgeAdapter:
    def adapter(nodes: list[SearchNode]) -> list[JudgeOracleResult]:
        return run_subprocess_judge(command, nodes, timeout=timeout)

    return adapter


def run_subprocess_relation(command: Sequence[str], nodes: list[SearchNode], timeout: float = 180.0) -> RelationOracleResult:
    task = make_relation_task(nodes)
    payload = {"task": task.__dict__, "nodes": [node.model_dump() for node in nodes]}
    completed = subprocess.run(list(command), input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"relation oracle failed: {completed.stderr.strip()}")
    return validate_relation_output(nodes, completed.stdout)


def build_subprocess_relation_adapter(command: Sequence[str], timeout: float = 180.0) -> RelationAdapter:
    def adapter(nodes: list[SearchNode]) -> RelationOracleResult:
        return run_subprocess_relation(command, nodes, timeout=timeout)

    return adapter
