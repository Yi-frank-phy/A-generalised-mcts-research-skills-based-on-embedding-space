"""Experimental prospective thought-space primitives for next-generation DTE."""

from .entropy import adaptive_bandwidth, configurational_entropy, normalized_kernel_density
from .metric import FrozenThoughtMetric, MetricIdentity
from .prospective import (
    CANONICALIZATION_VERSION,
    PROSPECTIVE_THOUGHT_SCHEMA,
    ProspectiveThought,
    build_notice_instruction,
    embed_prospective_thoughts,
    prospective_thought_batch_schema,
)
from .return_metric import null_adjusted_geometric_return, rbf_mmd2

__all__ = [
    "CANONICALIZATION_VERSION",
    "PROSPECTIVE_THOUGHT_SCHEMA",
    "ProspectiveThought",
    "build_notice_instruction",
    "prospective_thought_batch_schema",
    "embed_prospective_thoughts",
    "adaptive_bandwidth",
    "normalized_kernel_density",
    "configurational_entropy",
    "MetricIdentity",
    "FrozenThoughtMetric",
    "rbf_mmd2",
    "null_adjusted_geometric_return",
]
