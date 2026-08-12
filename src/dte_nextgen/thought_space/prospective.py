"""Compatibility helpers for proposing pre-execution interventions.

A ``ProspectiveThought`` is an intervention proposal only. It is not the
canonical DTE search microstate and must not be used as the geometry consumed
by the transition UCB, entropy, or frontier-displacement controller.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

CANONICALIZATION_VERSION = "prospective-thought-v1"
PROSPECTIVE_THOUGHT_ROLE = "intervention_proposal_only"


def _normalize_field(text: str) -> str:
    return " ".join(str(text).split())


@dataclass(frozen=True)
class ProspectiveThought:
    observation: str
    possible_structure: str
    discriminating_test: str

    def canonical_text(self) -> str:
        return (
            f"OBSERVATION:\n{_normalize_field(self.observation)}\n\n"
            f"POSSIBLE_STRUCTURE:\n{_normalize_field(self.possible_structure)}\n\n"
            f"DISCRIMINATING_TEST:\n{_normalize_field(self.discriminating_test)}"
        )


PROSPECTIVE_THOUGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string", "minLength": 1},
        "possible_structure": {"type": "string", "minLength": 1},
        "discriminating_test": {"type": "string", "minLength": 1},
    },
    "required": ["observation", "possible_structure", "discriminating_test"],
    "additionalProperties": False,
}


def prospective_thought_batch_schema(candidate_count: int) -> dict:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    return {
        "type": "array",
        "items": PROSPECTIVE_THOUGHT_SCHEMA,
        "minItems": candidate_count,
        "maxItems": candidate_count,
    }


def build_notice_instruction(candidate_count: int) -> str:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    return (
        f"Generate exactly {candidate_count} distinct prospective structural thoughts from the current research state. "
        "Each thought must contain only observation, possible_structure, and discriminating_test. "
        "The observation must contain only facts or relations explicit in the supplied research state. "
        "Do not invent new named operators, states, graph features, physical assumptions, parameters, measurements, or protocol components that are not present in the research state. "
        "The possible_structure may propose a new hypothesis or relation, but it must be framed as a possibility rather than an observed fact. "
        "The discriminating_test must test that hypothesis using only objects already present in the research state or generic operations such as algebraic rewriting, comparison, or counterexample construction. "
        "Do not rank candidates. Do not score candidates. Do not predict the execution outcome. "
        "State what was noticed, one possible structural relation worth testing, and a concrete discriminating test."
    )


def embed_prospective_thoughts(
    thoughts: Sequence[ProspectiveThought],
    embed_fn: Callable[[str], Sequence[float]],
) -> np.ndarray:
    """Compatibility embedder for proposal experiments, not transition state."""
    if not thoughts:
        return np.empty((0, 0), dtype=float)

    vectors: list[np.ndarray] = []
    expected_dimension: int | None = None
    for thought in thoughts:
        vector = np.asarray(embed_fn(thought.canonical_text()), dtype=float)
        if vector.ndim != 1 or len(vector) == 0:
            raise ValueError("embedding must be a non-empty 1D vector")
        if not np.isfinite(vector).all():
            raise ValueError("embedding must contain only finite values")
        if expected_dimension is None:
            expected_dimension = len(vector)
        elif len(vector) != expected_dimension:
            raise ValueError("embedding dimension must be consistent across thoughts")
        vectors.append(vector)
    return np.vstack(vectors)
