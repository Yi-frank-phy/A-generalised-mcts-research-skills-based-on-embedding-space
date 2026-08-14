"""Cold-start completed transitions for the `new` release line."""

from __future__ import annotations
from .models import DTERunSpec, SearchNode


def seed_frontier(spec: DTERunSpec) -> tuple[list[SearchNode], dict[str, object]]:
    """Create four completed initialization transitions, not prospective states."""

    constraints = list(spec.constraints[:3])
    rows = [
        (
            "seed-direct",
            "Direct constructive route",
            "direct construction from the stated constraints",
            "sharper_unknown",
            "Whether the direct construction satisfies all stated constraints.",
        ),
        (
            "seed-counter",
            "Boundary-first route",
            "boundary-case and counterexample decomposition",
            "sharper_unknown",
            "Whether a boundary or low-dimensional case invalidates the current route.",
        ),
        (
            "seed-formalism",
            "Alternative-formalism route",
            "alternative-formalism decomposition",
            "sharper_unknown",
            "Whether a change of representation reveals an equivalence or hidden structure.",
        ),
        (
            "seed-relations",
            "Relation-discriminator route",
            "relation and discriminator decomposition",
            "new_understanding",
            "Separated equivalent, complementary, and conflicting continuations as distinct cases.",
        ),
    ]
    nodes = [
        SearchNode(
            node_id=node_id,
            claim=f"{label} for: {spec.problem}",
            rationale=f"Cold-start decomposition toward goal: {spec.goal}",
            assumptions=constraints,
            confidence=0.5,
            retrospective_method=method,
            epistemic_change_kind=kind,
            epistemic_change=change,
        )
        for node_id, label, method, kind, change in rows
    ]
    return nodes, {
        "seed_coordinate": "completed_method_epistemic_transition",
        "seed_count": len(nodes),
    }
