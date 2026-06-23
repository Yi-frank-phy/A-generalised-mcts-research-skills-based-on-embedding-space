"""Validators for observable DTE oracle outputs.

Judge and relation oracles may be performed by subagents. The backend validates
what they return before it can affect the DTE graph.
"""

from __future__ import annotations

import json
from typing import Any

from .models import SearchNode
from .oracles import JudgeOracleResult, RelationOracleResult


def parse_json_output(raw_output: str | Any) -> Any:
    if isinstance(raw_output, str):
        return json.loads(raw_output)
    return raw_output


def validate_judge_output(nodes: list[SearchNode], raw_output: str | Any) -> list[JudgeOracleResult]:
    """Validate Judge output: scores and reasoning only, no controller fields."""

    data = parse_json_output(raw_output)
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise ValueError("judge oracle must return a list or {'results': [...]} object")

    allowed_ids = {node.node_id for node in nodes}
    seen: set[str] = set()
    parsed: list[JudgeOracleResult] = []
    forbidden = {"embedding", "uncertainty", "ucb_score", "expansion_budget", "node_type", "claim"}

    for item in results:
        if not isinstance(item, dict):
            raise ValueError("judge oracle result entries must be objects")
        overlap = forbidden.intersection(item)
        if overlap:
            raise ValueError(f"judge oracle returned forbidden fields: {sorted(overlap)}")
        node_id = str(item.get("node_id", ""))
        if node_id not in allowed_ids:
            raise ValueError(f"judge oracle returned unknown node_id: {node_id}")
        score = float(item.get("score"))
        if not 0.0 <= score <= 1.0:
            raise ValueError("judge score must be in [0, 1]")
        risks = item.get("risks", [])
        if not isinstance(risks, list):
            raise ValueError("judge risks must be a list")
        parsed.append(
            JudgeOracleResult(
                node_id=node_id,
                score=score,
                reasoning=str(item.get("reasoning", "")),
                risks=[str(r) for r in risks],
            )
        )
        seen.add(node_id)

    missing = allowed_ids - seen
    if missing:
        raise ValueError(f"judge oracle omitted node ids: {sorted(missing)}")
    return parsed


def validate_relation_output(nodes: list[SearchNode], raw_output: str | Any) -> RelationOracleResult:
    """Validate relation output for merge/discriminator decisions."""

    data = parse_json_output(raw_output)
    if not isinstance(data, dict):
        raise ValueError("relation oracle must return an object")
    relation = data.get("relation")
    if relation not in {"equivalent", "complementary", "conflict", "independent"}:
        raise ValueError("invalid relation oracle relation")
    source_ids = [str(x) for x in data.get("source_node_ids", [])]
    allowed_ids = {node.node_id for node in nodes}
    if len(source_ids) < 2 or not set(source_ids).issubset(allowed_ids):
        raise ValueError("relation oracle source_node_ids must contain at least two known nodes")
    return RelationOracleResult(
        relation=relation,
        source_node_ids=source_ids,
        rationale=str(data.get("rationale", "")),
        discriminator_question=None if data.get("discriminator_question") is None else str(data.get("discriminator_question")),
    )
