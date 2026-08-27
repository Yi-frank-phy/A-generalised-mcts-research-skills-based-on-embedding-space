"""Problem-independent quadrature anchors for the new research-space atlas.

These nodes are geometry/reference cells only. They are never live frontier
members, realized evidence, or Judge observations.
"""

from __future__ import annotations
from .models import SearchNode
from .transition_state import canonical_transition_text

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
    """Freeze generic anchors plus unique run-specific initial transitions.

    Exact duplicate canonical transitions are one quadrature location, not extra
    volume cells.  Keeping duplicate cells creates a degenerate zero-radius
    ground state whose entropy floor can exceed the singleton-frontier target
    entropy and make the Boltzmann match unsatisfiable.
    """

    packaged = list(packaged_reference_nodes())
    seen = {canonical_transition_text(node) for node in packaged}
    unique_initial: list[SearchNode] = []
    for node in initial_nodes:
        key = canonical_transition_text(node)
        if key in seen:
            continue
        seen.add(key)
        unique_initial.append(node.model_copy(deep=True))
    return (*packaged, *unique_initial)
