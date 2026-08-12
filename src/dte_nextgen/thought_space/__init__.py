"""Experimental next-generation DTE geometry primitives.

Prospective thoughts remain compatibility/proposal helpers. The authoritative
next-generation search coordinate is a completed method--epistemic transition.
"""

from .controller import frontier_standard_deviations, score_transition_frontier
from .entropy import adaptive_bandwidth, configurational_entropy, normalized_kernel_density
from .metric import FrozenThoughtMetric, FrozenTransitionMetric, MetricIdentity
from .prospective import (
    CANONICALIZATION_VERSION,
    PROSPECTIVE_THOUGHT_ROLE,
    PROSPECTIVE_THOUGHT_SCHEMA,
    ProspectiveThought,
    build_notice_instruction,
    embed_prospective_thoughts,
    prospective_thought_batch_schema,
)
from .return_metric import null_adjusted_geometric_return, rbf_mmd2
from .transition import (
    METHOD_EPISTEMIC_TRANSITION_VERSION,
    MethodEpistemicTransition,
    embed_method_epistemic_transitions,
)

__all__ = [
    "CANONICALIZATION_VERSION",
    "PROSPECTIVE_THOUGHT_ROLE",
    "PROSPECTIVE_THOUGHT_SCHEMA",
    "ProspectiveThought",
    "build_notice_instruction",
    "prospective_thought_batch_schema",
    "embed_prospective_thoughts",
    "METHOD_EPISTEMIC_TRANSITION_VERSION",
    "MethodEpistemicTransition",
    "embed_method_epistemic_transitions",
    "adaptive_bandwidth",
    "normalized_kernel_density",
    "configurational_entropy",
    "frontier_standard_deviations",
    "score_transition_frontier",
    "MetricIdentity",
    "FrozenThoughtMetric",
    "FrozenTransitionMetric",
    "rbf_mmd2",
    "null_adjusted_geometric_return",
]
