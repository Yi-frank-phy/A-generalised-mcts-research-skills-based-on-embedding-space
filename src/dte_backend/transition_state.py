"""Canonical completed research-transition state for the new release line."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models import SearchNode

_ALLOWED_CHANGE_KINDS = {
    "new_understanding",
    "sharper_unknown",
    "no_material_change",
}


def _clean(value: str | None) -> str:
    return " ".join(str(value or "").split())


def require_completed_transition(node: SearchNode) -> tuple[str, str, str]:
    method = _clean(getattr(node, "retrospective_method", None))
    kind = _clean(getattr(node, "epistemic_change_kind", None))
    change = _clean(getattr(node, "epistemic_change", None))
    if not method or kind not in _ALLOWED_CHANGE_KINDS or not change:
        raise ValueError(
            f"active node {node.node_id!r} lacks a valid completed transition"
        )
    return method, kind, change


def canonical_transition_text(node: SearchNode) -> str:
    method, kind, change = require_completed_transition(node)
    return (
        "METHOD_EPISTEMIC_TRANSITION_V1\n"
        f"RETROSPECTIVE_METHOD:\n{method}\n\n"
        f"EPISTEMIC_CHANGE_KIND:\n{kind}\n\n"
        f"EPISTEMIC_CHANGE:\n{change}"
    )


def embed_transition_nodes(nodes: Sequence[SearchNode], provider: object) -> np.ndarray:
    if not nodes:
        return np.empty((0, 0), dtype=float)
    texts = [canonical_transition_text(node) for node in nodes]
    raw = provider.embed_texts(texts)  # type: ignore[attr-defined]
    vectors = np.asarray(raw, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(nodes) or vectors.shape[1] == 0:
        raise ValueError("embedding provider returned an invalid transition batch")
    if not np.isfinite(vectors).all():
        raise ValueError("transition embeddings must be finite")
    return vectors
