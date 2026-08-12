from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .prospective import CANONICALIZATION_VERSION, ProspectiveThought, embed_prospective_thoughts
from .transition import (
    METHOD_EPISTEMIC_TRANSITION_VERSION,
    MethodEpistemicTransition,
    embed_method_epistemic_transitions,
)

_SUPPORTED_CANONICALIZATION_VERSIONS = {
    CANONICALIZATION_VERSION,
    METHOD_EPISTEMIC_TRANSITION_VERSION,
}


@dataclass(frozen=True)
class MetricIdentity:
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    canonicalization_version: str
    normalization_policy: str
    kernel_family: str
    bandwidth: float

    def __post_init__(self) -> None:
        if self.embedding_dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if self.canonicalization_version not in _SUPPORTED_CANONICALIZATION_VERSIONS:
            raise ValueError("unsupported canonicalization version")
        if self.normalization_policy not in {"none", "l2"}:
            raise ValueError("normalization policy must be one of: none, l2")
        if self.kernel_family != "rbf":
            raise ValueError("kernel family must be rbf")
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")


def _validate_frozen_cloud(cloud: np.ndarray, identity: MetricIdentity) -> np.ndarray:
    if cloud.size == 0:
        return cloud
    if cloud.shape[1] != identity.embedding_dimension:
        raise ValueError(
            "embedding dimension does not match frozen metric identity: "
            f"expected {identity.embedding_dimension}, got {cloud.shape[1]}"
        )
    if identity.normalization_policy == "l2":
        norms = np.linalg.norm(cloud, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("embedding normalization does not match frozen metric identity")
    return cloud


@dataclass(frozen=True)
class FrozenThoughtMetric:
    """Compatibility metric for the superseded prospective-thought experiment."""

    identity: MetricIdentity
    embed_fn: Callable[[str], Sequence[float]] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.identity.canonicalization_version != CANONICALIZATION_VERSION:
            raise ValueError("metric identity does not match prospective thought serializer")

    def embed_cloud(self, thoughts: Sequence[ProspectiveThought]) -> np.ndarray:
        cloud = embed_prospective_thoughts(thoughts, self.embed_fn)
        return _validate_frozen_cloud(cloud, self.identity)


@dataclass(frozen=True)
class FrozenTransitionMetric:
    """Frozen metric for authoritative method--epistemic transition geometry."""

    identity: MetricIdentity
    embed_fn: Callable[[str], Sequence[float]] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.identity.canonicalization_version != METHOD_EPISTEMIC_TRANSITION_VERSION:
            raise ValueError("metric identity does not match transition serializer")

    def embed_cloud(
        self,
        transitions: Sequence[MethodEpistemicTransition],
    ) -> np.ndarray:
        cloud = embed_method_epistemic_transitions(transitions, self.embed_fn)
        return _validate_frozen_cloud(cloud, self.identity)
