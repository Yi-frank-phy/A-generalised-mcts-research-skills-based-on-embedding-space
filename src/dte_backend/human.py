"""Human-in-the-loop trigger protocol for the main Codex agent.

The main agent asks the user in chat. The backend only creates a structured
question when the search state has reached a low-temperature ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import SearchNode


@dataclass(frozen=True)
class HumanQuestion:
    question_type: str
    question: str
    options: list[str]
    context: str


def maybe_create_human_question(
    frontier: list[SearchNode],
    entropy_plateau: bool,
    min_score_gap: float = 0.05,
) -> HumanQuestion | None:
    """Create a compact branch-selection question when DTE is stuck."""

    if not entropy_plateau or len(frontier) < 2:
        return None
    ranked = sorted(frontier, key=lambda n: (n.ucb_score if n.ucb_score is not None else n.confidence), reverse=True)
    top, second = ranked[0], ranked[1]
    top_score = top.ucb_score if top.ucb_score is not None else top.confidence
    second_score = second.ucb_score if second.ucb_score is not None else second.confidence
    if abs(top_score - second_score) > min_score_gap:
        return None
    return HumanQuestion(
        question_type="branch_choice",
        question="DTE reached an entropy plateau with two near-tied frontier branches. Should the next run prioritize one branch or generate a discriminator task?",
        options=[
            f"Prioritize {top.node_id}: {top.claim}",
            f"Prioritize {second.node_id}: {second.claim}",
            "Generate a discriminator task between these branches",
        ],
        context=f"top_ucb={top_score:.3f}; second_ucb={second_score:.3f}",
    )
