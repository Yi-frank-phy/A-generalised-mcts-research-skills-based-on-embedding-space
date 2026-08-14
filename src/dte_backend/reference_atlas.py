"""Problem-independent quadrature anchors for the new research-space atlas.

These nodes are geometry/reference cells only. They are never live frontier
members, realized evidence, or Judge observations.
"""

from __future__ import annotations
from .models import SearchNode

_METHODS = (
    "direct constructive derivation",
    "assumption stress test",
    "counterexample search",
    "boundary and limiting-case analysis",
    "symmetry and invariant extraction",
    "representation change",
    "duality or equivalence mapping",
    "dimensional reduction",
    "decomposition into subproblems",
    "no-go obstruction proof",
    "perturbative approximation",
    "exact small-case calculation",
    "consistency cross-check",
    "alternative formalism translation",
    "relation and merge discrimination",
    "backward reconstruction of the key move",
)

_CHANGES = (
    ("new_understanding", "Identified an explicit mechanism that advances the problem."),
    ("new_understanding", "Exposed a structural equivalence between previously separate routes."),
    ("new_understanding", "Derived a concrete construction or proof skeleton."),
    ("sharper_unknown", "Isolated the weakest unresolved assumption that controls progress."),
    ("sharper_unknown", "Narrowed the failure boundary to a specific case or condition."),
    ("sharper_unknown", "Separated two plausible continuations that require different evidence."),
    ("no_material_change", "Tested the route but found no material epistemic change."),
    ("no_material_change", "Reformulated the route without changing the unresolved structure."),
)


def packaged_reference_nodes() -> tuple[SearchNode, ...]:
    """Return the fixed 128-cell method→epistemic reference atlas."""

    nodes: list[SearchNode] = []
    for method_index, method in enumerate(_METHODS):
        for change_index, (kind, change) in enumerate(_CHANGES):
            nodes.append(
                SearchNode(
                    node_id=f"reference-{method_index:02d}-{change_index:02d}",
                    claim="reference geometry cell",
                    status="archived",
                    retrospective_method=method,
                    epistemic_change_kind=kind,
                    epistemic_change=change,
                )
            )
    return tuple(nodes)


def combined_reference_nodes(initial_nodes: list[SearchNode]) -> tuple[SearchNode, ...]:
    """Freeze generic method-space anchors plus run-specific initial transitions."""

    return (*packaged_reference_nodes(), *(node.model_copy(deep=True) for node in initial_nodes))
