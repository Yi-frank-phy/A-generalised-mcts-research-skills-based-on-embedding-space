from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

METHOD_EPISTEMIC_TRANSITION_VERSION = "method-epistemic-transition-v1"
_ALLOWED_CHANGE_KINDS = {
    "new_understanding",
    "sharper_unknown",
    "no_material_change",
}


def _normalize_field(text: str) -> str:
    return " ".join(str(text).split())


@dataclass(frozen=True)
class MethodEpistemicTransition:
    """Completed research transition embedded as method -> epistemic change.

    ``context_q`` is retained for proposal/context use only. It is deliberately
    excluded from ``canonical_text`` so topical source context cannot become
    part of the method--epistemic-change geometry.
    """

    retrospective_method: str
    epistemic_change_kind: str
    epistemic_change: str
    context_q: str | None = None

    def __post_init__(self) -> None:
        if not _normalize_field(self.retrospective_method):
            raise ValueError("retrospective_method must be non-empty")
        if self.epistemic_change_kind not in _ALLOWED_CHANGE_KINDS:
            raise ValueError(
                "epistemic_change_kind must be one of: "
                "new_understanding, sharper_unknown, no_material_change"
            )
        if not _normalize_field(self.epistemic_change):
            raise ValueError("epistemic_change must be non-empty")

    def canonical_text(self) -> str:
        return (
            "METHOD_EPISTEMIC_TRANSITION_V1\n"
            f"RETROSPECTIVE_METHOD:\n{_normalize_field(self.retrospective_method)}\n\n"
            f"EPISTEMIC_CHANGE_KIND:\n{self.epistemic_change_kind}\n\n"
            f"EPISTEMIC_CHANGE:\n{_normalize_field(self.epistemic_change)}"
        )


def embed_method_epistemic_transitions(
    transitions: Sequence[MethodEpistemicTransition],
    embed_fn: Callable[[str], Sequence[float]],
) -> np.ndarray:
    """Embed only canonical method--epistemic-change text, preserving order."""
    if not transitions:
        return np.empty((0, 0), dtype=float)

    vectors: list[np.ndarray] = []
    expected_dimension: int | None = None
    for transition in transitions:
        vector = np.asarray(embed_fn(transition.canonical_text()), dtype=float)
        if vector.ndim != 1 or len(vector) == 0:
            raise ValueError("embedding must be a non-empty 1D vector")
        if not np.isfinite(vector).all():
            raise ValueError("embedding must contain only finite values")
        if expected_dimension is None:
            expected_dimension = len(vector)
        elif len(vector) != expected_dimension:
            raise ValueError("embedding dimension must be consistent across transitions")
        vectors.append(vector)

    return np.vstack(vectors)
