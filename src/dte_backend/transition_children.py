"""Deterministic completed-transition children for offline validation."""

from __future__ import annotations
import uuid
from .models import SearchNode


def deterministic_transition_children(parent: SearchNode, count: int, iteration: int) -> list[SearchNode]:
    methods = [
        ("assumption audit", "sharper_unknown"),
        ("constructive derivation", "new_understanding"),
        ("representation change", "new_understanding"),
        ("boundary analysis", "sharper_unknown"),
    ]
    result: list[SearchNode] = []
    for index in range(max(0, count)):
        method, kind = methods[index % len(methods)]
        result.append(SearchNode(
            node_id=str(uuid.uuid4()),
            claim=f"{parent.claim} — {method}",
            rationale=f"Offline continuation using {method}.",
            assumptions=list(parent.assumptions),
            evidence=list(parent.evidence),
            risks=list(parent.risks),
            parent_ids=[parent.node_id],
            confidence=parent.confidence,
            retrospective_method=method,
            epistemic_change_kind=kind,
            epistemic_change=f"Applied {method} to {parent.node_id} at iteration {iteration}.",
        ))
    return result
