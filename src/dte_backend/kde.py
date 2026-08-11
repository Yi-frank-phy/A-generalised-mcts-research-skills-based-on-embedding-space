"""KDE utilities for DTE frontier geometry.

The old DTE backend treated embedding-space density as the physical state of the
frontier. This module keeps that idea explicit and computes the pairwise distance
matrix once per batch, then reuses it for soft density, bounded entropy, and
absolute local uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class KDEState:
    """Density observables for one frontier batch."""

    log_density: list[float]
    uncertainty: list[float]
    spatial_entropy: float
    bandwidth2: float


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def pairwise_squared_distance(embeddings: list[list[float]], normalize: bool = True) -> np.ndarray:
    """Compute pairwise squared Euclidean distance for embedding vectors."""

    if not embeddings:
        return np.zeros((0, 0), dtype=float)
    x = np.asarray(embeddings, dtype=float)
    if x.ndim != 2:
        raise ValueError("embeddings must be a 2D array-like object")
    if normalize:
        x = _normalize(x)
    diff = x[:, None, :] - x[None, :, :]
    return np.sum(diff * diff, axis=2)


def estimate_bandwidth2(dist2: np.ndarray) -> float:
    """Return h^2 for the legacy adaptive scale h = median(pair distance)/sqrt(2)."""

    if dist2.size == 0:
        return 1.0
    nonzero = dist2[dist2 > 1e-12]
    if nonzero.size == 0:
        return 1.0
    return max(float(np.median(nonzero)) / 2.0, 1e-8)


def _logsumexp(values: np.ndarray, axis: int = 1) -> np.ndarray:
    max_v = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_v, axis=axis) + np.log(np.sum(np.exp(values - max_v), axis=axis))


def compute_kde_state(embeddings: list[list[float]]) -> KDEState:
    """Compute soft density, absolute local uncertainty, and bounded entropy."""

    n = len(embeddings)
    if n == 0:
        return KDEState(log_density=[], uncertainty=[], spatial_entropy=0.0, bandwidth2=1.0)
    if n == 1:
        return KDEState(log_density=[0.0], uncertainty=[1.0], spatial_entropy=0.0, bandwidth2=1.0)

    dist2 = pairwise_squared_distance(embeddings, normalize=True)
    bandwidth2 = estimate_bandwidth2(dist2)
    log_kernel = -dist2 / (2.0 * bandwidth2)
    log_density_np = _logsumexp(log_kernel, axis=1) - math.log(n)
    spatial_entropy = float(-np.mean(log_density_np))

    density = np.exp(log_density_np)
    effective_local_count = np.maximum(float(n) * density, 1.0)
    uncertainty = 1.0 / np.sqrt(effective_local_count)

    return KDEState(
        log_density=[float(v) for v in log_density_np],
        uncertainty=[float(v) for v in uncertainty],
        spatial_entropy=spatial_entropy,
        bandwidth2=bandwidth2,
    )
