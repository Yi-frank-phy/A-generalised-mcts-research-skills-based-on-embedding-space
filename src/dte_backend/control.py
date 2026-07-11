"""Strict-run user control file support.

The user-authored control file is intentionally narrow: it can request synthesis
after the current safe task, but it cannot judge nodes, allocate budget, or
mutate graph state.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import ForcedSynthesisRecord, SearchNode, SynthesisControlRequest


def load_synthesis_control(path: str | Path | None, nodes: list[SearchNode]) -> SynthesisControlRequest | None:
    """Return a validated user synthesis request if the control file exists."""

    if path is None:
        return None
    control_path = Path(path)
    if not control_path.exists():
        return None

    request = SynthesisControlRequest.model_validate(json.loads(control_path.read_text(encoding="utf-8")))
    if request.scope == "node_ids":
        known_ids = {node.node_id for node in nodes}
        missing = [node_id for node_id in request.node_ids if node_id not in known_ids]
        if missing:
            raise ValueError(f"synthesis control references unknown node ids: {', '.join(missing)}")
    return request


def record_forced_synthesis(
    request: SynthesisControlRequest,
    nodes: list[SearchNode],
    control_path: str | Path | None = None,
) -> ForcedSynthesisRecord:
    """Convert a validated request into immutable run metadata."""

    selected = {node.node_id for node in nodes} if request.scope == "all" else set(request.node_ids)
    frontier_ids = {node.node_id for node in nodes if node.status == "frontier"}
    left_unexplored = sorted(frontier_ids - selected)
    return ForcedSynthesisRecord(
        stop_reason="user_interrupted_for_synthesis",
        requested_by=request.requested_by,
        reason=request.reason,
        scope=request.scope,
        node_ids=list(request.node_ids),
        left_unexplored_node_ids=left_unexplored,
        control_path=None if control_path is None else str(control_path),
    )
