"""Batch Judge prototype for DTE.

The real system should call a strong model here. This prototype keeps a
schema-compatible deterministic judge so Codex/tests can exercise the full DTE
loop without network access.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import SearchNode
from .text_features import tokenize


@dataclass(frozen=True)
class JudgeResult:
    node_id: str
    score: float
    reasoning: str


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def heuristic_score(node: SearchNode) -> JudgeResult:
    """Score one node with a transparent deterministic heuristic.

    The heuristic is deliberately simple: confidence and evidence help, obvious
    risk/unsupportedness hurts. It is not a replacement for model judging; it is
    a stable backend contract for the prototype.
    """

    text_len = len(tokenize(" ".join([node.claim, node.rationale])))
    evidence_bonus = min(0.20, 0.06 * len(node.evidence))
    assumption_penalty = min(0.12, 0.03 * max(0, len(node.assumptions) - 2))
    risk_penalty = min(0.24, 0.06 * len(node.risks))
    detail_bonus = min(0.10, text_len / 300.0)

    score = _clamp01(node.confidence + evidence_bonus + detail_bonus - assumption_penalty - risk_penalty)
    reasoning = (
        f"heuristic judge: confidence={node.confidence:.2f}, "
        f"evidence_bonus={evidence_bonus:.2f}, detail_bonus={detail_bonus:.2f}, "
        f"assumption_penalty={assumption_penalty:.2f}, risk_penalty={risk_penalty:.2f}"
    )
    return JudgeResult(node_id=node.node_id, score=score, reasoning=reasoning)


def batch_judge(nodes: list[SearchNode]) -> list[JudgeResult]:
    """Judge all frontier nodes in one logical batch."""

    return [heuristic_score(node) for node in nodes if node.status == "frontier"]
