"""Entropy and temperature controller for DTE frontier search.

This module restores the old DTE idea that search should be controlled by the
state of the frontier, not by a fixed iteration counter alone. The current
calculation is a lightweight KDE-style entropy proxy over normalized embeddings;
real runs should use a high-quality embedding provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EntropyState:
    """Search-phase observables for one DTE iteration."""

    spatial_entropy: float
    entropy_delta: float | None
    effective_temperature: float
    normalized_temperature: float
    should_synthesize: bool
    stop_reason: str | None = None


def _logsumexp(values: np.ndarray, axis: int = 1) -> np.ndarray:
    max_v = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_v, axis=axis) + np.log(np.sum(np.exp(values - max_v), axis=axis))


def spatial_entropy_from_embeddings(embeddings: list[list[float]]) -> float:
    """Estimate dimensionless spatial entropy from frontier embeddings.

    For N<2 there is no meaningful spatial spread, so entropy is 0. For N>=2,
    vectors are L2-normalized, pairwise squared distances are used in a Gaussian
    kernel, and entropy is `-mean(log density)`. This is a proxy suitable for
    control; it is not a calibrated thermodynamic entropy.
    """

    if len(embeddings) < 2:
        return 0.0
    x = np.asarray(embeddings, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    diff = x[:, None, :] - x[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    nonzero = d2[d2 > 1e-12]
    if nonzero.size == 0:
        return 0.0
    bandwidth2 = float(np.median(nonzero))
    bandwidth2 = max(bandwidth2, 1e-8)
    log_kernel = -d2 / (2.0 * bandwidth2)
    log_density = _logsumexp(log_kernel, axis=1) - math.log(x.shape[0])
    return float(-np.mean(log_density))


def evaluate_entropy_state(
    spatial_entropy: float,
    previous_entropy: float | None,
    iteration: int,
    min_iterations: int,
    entropy_change_threshold: float,
    t_max: float = 1.0,
) -> EntropyState:
    """Convert entropy history into normalized temperature and stop signal."""

    if previous_entropy is None:
        return EntropyState(
            spatial_entropy=spatial_entropy,
            entropy_delta=None,
            effective_temperature=float(t_max),
            normalized_temperature=1.0,
            should_synthesize=False,
            stop_reason=None,
        )

    delta = abs(spatial_entropy - previous_entropy) / max(abs(previous_entropy), 1.0)
    # High entropy change means high search temperature; plateau means low temp.
    normalized_temperature = min(1.0, delta / max(entropy_change_threshold, 1e-12))
    effective_temperature = float(t_max * normalized_temperature)
    should_stop = iteration >= min_iterations and delta < entropy_change_threshold
    return EntropyState(
        spatial_entropy=spatial_entropy,
        entropy_delta=float(delta),
        effective_temperature=effective_temperature,
        normalized_temperature=float(normalized_temperature),
        should_synthesize=should_stop,
        stop_reason="entropy_plateau" if should_stop else None,
    )
