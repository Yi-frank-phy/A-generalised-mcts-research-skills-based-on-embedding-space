from collections.abc import Callable, Sequence

import numpy as np

from .allocation import (
    boltzmann_probabilities,
    select_next_index,
    temperature_for_target_entropy,
)
from .controller import score_transition_frontier
from .transition import MethodEpistemicTransition, embed_method_epistemic_transitions


def score_transition_directions(
    transitions: Sequence[MethodEpistemicTransition],
    propulsion_values: np.ndarray,
    embed_fn: Callable[[str], Sequence[float]],
    bandwidth: float | None = None,
) -> dict[str, object]:
    """Score already-completed transition directions in canonical pair space."""
    if not transitions:
        raise ValueError("transitions must be non-empty")

    embeddings = embed_method_epistemic_transitions(transitions, embed_fn)
    result = score_transition_frontier(
        embeddings,
        propulsion_values=propulsion_values,
        bandwidth=bandwidth,
    )
    return {
        **result,
        "transitions": tuple(transitions),
    }


def select_transition_direction(
    transitions: Sequence[MethodEpistemicTransition],
    propulsion_values: np.ndarray,
    embed_fn: Callable[[str], Sequence[float]],
    selection_quantile: float,
    bandwidth: float | None = None,
) -> dict[str, object]:
    """Select one continuation direction after geometry/UCB recomputation.

    The scheduler uses the current provisional breadth closure H_B = H_geom.
    It does not generate or execute a prospective intervention; that belongs to
    the downstream execution layer after a completed direction has been chosen.
    """
    scored = score_transition_directions(
        transitions,
        propulsion_values=propulsion_values,
        embed_fn=embed_fn,
        bandwidth=bandwidth,
    )
    ucb_scores = np.asarray(scored["ucb_scores"], dtype=float)
    target_entropy = float(scored["target_entropy"])
    temperature = temperature_for_target_entropy(ucb_scores, target_entropy)
    probabilities = boltzmann_probabilities(ucb_scores, temperature)
    selected_index = select_next_index(probabilities, selection_quantile)

    return {
        **scored,
        "temperature": float(temperature),
        "probabilities": probabilities,
        "selected_index": selected_index,
        "selected_transition": transitions[selected_index],
    }
