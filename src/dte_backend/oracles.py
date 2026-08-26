"""Subagent oracle task contracts for DTE.

Complementary merge, conflict merge, and discriminator generation are not
hard-coded backend intelligence. They are callable oracle tasks that a strong
main agent or subagent may perform, then return as structured data. The Python
backend validates and consumes the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .models import SearchNode


OracleKind = Literal["equivalent_merge", "complementary_merge", "conflict_merge", "discriminator"]


@dataclass(frozen=True)
class OracleTask:
    """A structured task for a model/subagent oracle."""

    kind: OracleKind
    node_ids: list[str]
    instruction: str
    required_output: dict[str, object]




@dataclass(frozen=True)
class RelationOracleResult:
    """Observable relation output for merge/discriminator tasks."""

    relation: Literal["equivalent", "complementary", "conflict", "independent"]
    source_node_ids: list[str]
    rationale: str
    discriminator_question: str | None = None




def make_relation_task(nodes: list[SearchNode]) -> OracleTask:
    return OracleTask(
        kind="complementary_merge",
        node_ids=[n.node_id for n in nodes],
        instruction=(
            "Classify whether these nodes are equivalent, complementary, conflicting, "
            "or independent. If conflicting, propose a discriminator question."
        ),
        required_output={
            "relation": "equivalent|complementary|conflict|independent",
            "source_node_ids": ["..."],
            "rationale": "...",
            "discriminator_question": "optional",
        },
    )
