"""Validators for observable DTE oracle outputs.

Judge and relation oracles may be performed by subagents. The backend validates
what they return before it can affect the DTE graph.
"""

from __future__ import annotations

import json
from typing import Any

from .models import SearchNode
from .episode_models import EpisodeRequest, EpisodeResult, compute_output_hash
from .relation_models import RelationEpisodeOutput
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

    node_ids = [node.node_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("judge oracle input contains duplicate node IDs")
    allowed_ids = set(node_ids)
    seen: set[str] = set()
    parsed: list[JudgeOracleResult] = []
    allowed_fields = {"node_id", "score", "reasoning", "risks"}

    for item in results:
        if not isinstance(item, dict):
            raise ValueError("judge oracle result entries must be objects")
        unexpected = set(item) - allowed_fields
        if unexpected:
            raise ValueError(f"judge oracle returned forbidden fields: {sorted(unexpected)}")
        node_id = str(item.get("node_id", ""))
        if node_id not in allowed_ids:
            raise ValueError(f"judge oracle returned unknown node_id: {node_id}")
        if node_id in seen:
            raise ValueError(f"judge oracle returned duplicate node_id: {node_id}")
        reasoning = item.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("judge oracle reasoning must be a non-empty string")
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
                reasoning=reasoning,
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
    allowed_fields = {
        "relation",
        "source_node_ids",
        "rationale",
        "discriminator_question",
    }
    unexpected = set(data) - allowed_fields
    if unexpected:
        raise ValueError(f"relation oracle returned forbidden fields: {sorted(unexpected)}")
    relation = data.get("relation")
    if relation not in {"equivalent", "complementary", "conflict", "independent"}:
        raise ValueError("invalid relation oracle relation")
    raw_source_ids = data.get("source_node_ids", [])
    if not isinstance(raw_source_ids, list):
        raise ValueError("relation oracle source_node_ids must be a list")
    source_ids = [str(x) for x in raw_source_ids]
    node_ids = [node.node_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("relation oracle input contains duplicate node IDs")
    allowed_ids = set(node_ids)
    if len(source_ids) < 2:
        raise ValueError("relation oracle source_node_ids must contain at least two known nodes")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("relation oracle source_node_ids must not contain duplicate node IDs")
    if not set(source_ids).issubset(allowed_ids):
        raise ValueError("relation oracle source_node_ids must contain only known nodes")
    return RelationOracleResult(
        relation=relation,
        source_node_ids=source_ids,
        rationale=str(data.get("rationale", "")),
        discriminator_question=None if data.get("discriminator_question") is None else str(data.get("discriminator_question")),
    )


def validate_relation_episode_output(
    request: EpisodeRequest | dict[str, Any],
    raw_output: EpisodeResult | dict[str, Any],
) -> EpisodeResult:
    """Validate the App-native Relation envelope without mutating graph state."""

    request_payload = (
        request.model_dump(mode="json") if isinstance(request, EpisodeRequest) else request
    )
    result_payload = (
        raw_output.model_dump(mode="json")
        if isinstance(raw_output, EpisodeResult)
        else raw_output
    )
    parsed_request = EpisodeRequest.model_validate(request_payload)
    result = EpisodeResult.model_validate(result_payload)
    if parsed_request.role != "relation" or parsed_request.relation_payload is None:
        raise ValueError("Relation episode guard requires role='relation' request")
    if result.role != "relation" or not isinstance(result.structured_output, RelationEpisodeOutput):
        raise ValueError("Relation episode guard requires RelationEpisodeOutput")
    for field_name in ("episode_id", "attempt_id", "run_id", "input_graph_revision", "selected_node_revisions"):
        if getattr(result, field_name) != getattr(parsed_request, field_name):
            raise ValueError(f"Relation episode {field_name} mismatch")
    if result.status != "completed":
        raise ValueError("Relation episode result must be completed")
    if result.schema_version != parsed_request.output_schema_version:
        raise ValueError("Relation episode schema version mismatch")
    if result.output_hash != compute_output_hash(result.structured_output, result.schema_version):
        raise ValueError("Relation episode output hash mismatch")

    granted = {pair.candidate_id: pair for pair in parsed_request.relation_payload.candidate_pairs}
    observations = result.structured_output.observations
    ids = [observation.candidate_id for observation in observations]
    if len(ids) != len(set(ids)) or set(ids) != set(granted):
        raise ValueError("Relation episode observations must exactly match granted candidates")
    unordered_pairs = [(item.left_node_id, item.right_node_id) for item in observations]
    if len(unordered_pairs) != len(set(unordered_pairs)):
        raise ValueError("Relation episode contains duplicate unordered pairs")
    for observation in observations:
        pair = granted[observation.candidate_id]
        if (observation.left_node_id, observation.right_node_id) != (pair.left.node_id, pair.right.node_id):
            raise ValueError("Relation episode observation pair mismatch")
        allowed_refs = {
            evidence.evidence_ref
            for node in (pair.left, pair.right)
            for evidence in node.evidence
        }
        if not set(observation.evidence_refs).issubset(allowed_refs):
            raise ValueError("Relation episode contains an invalid evidence reference")
    return result
