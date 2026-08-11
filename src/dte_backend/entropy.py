"""Diversity-state and entropy-matched allocation temperature for DTE search.

The geometry layer supplies a bounded soft-discrete entropy. The allocation
controller chooses the Boltzmann temperature whose action entropy matches that
current geometry for the current UCB spectrum. Cross-iteration entropy delta and
plateau telemetry remain separate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from .kde import compute_kde_state


@dataclass(frozen=True)
class EntropyState:
    """Search-phase observables for one DTE iteration.

    ``normalized_temperature`` is retained as a persisted compatibility field.
    Its value is the normalized geometry-entropy coordinate ``H/log(N)``; it is
    not the physical/effective Boltzmann temperature used for allocation.
    """

    spatial_entropy: float
    entropy_delta: float | None
    effective_temperature: float
    normalized_temperature: float
    plateau_signal: bool
    consecutive_plateau_count: int

    @property
    def should_synthesize(self) -> bool:
        """Deprecated compatibility alias; callers must treat this as a signal."""

        return self.plateau_signal

    @property
    def stop_reason(self) -> str | None:
        """Deprecated compatibility label, not a synthesis decision."""

        return "entropy_plateau" if self.plateau_signal else None


def spatial_entropy_from_embeddings(embeddings: list[list[float]]) -> float:
    """Return the bounded current-frontier soft-discrete entropy."""

    return compute_kde_state(embeddings).spatial_entropy


def _normalized_entropy_coordinate(
    spatial_entropy: float,
    frontier_size: int,
) -> float:
    """Map bounded geometry entropy to its legacy persisted [0, 1] coordinate."""

    if frontier_size < 0:
        raise ValueError("frontier_size must be nonnegative")
    if not math.isfinite(spatial_entropy):
        raise ValueError("spatial_entropy must be finite")
    if frontier_size <= 1:
        if abs(spatial_entropy) > 1e-12:
            raise ValueError("singleton/empty frontier entropy must be zero")
        return 0.0

    max_entropy = math.log(frontier_size)
    tolerance = 1e-12
    if spatial_entropy < -tolerance or spatial_entropy > max_entropy + tolerance:
        raise ValueError("spatial_entropy outside the current proxy's [0, log N] range")

    return float(min(1.0, max(0.0, spatial_entropy / max_entropy)))


def _boltzmann_probabilities(scores: np.ndarray, temperature: float) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=float)
    if temperature <= 0.0:
        winners = scores == np.max(scores)
        return winners.astype(float) / float(np.sum(winners))
    scaled = scores / float(temperature)
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    return weights / np.sum(weights)


def _boltzmann_entropy(scores: np.ndarray, temperature: float) -> float:
    probabilities = _boltzmann_probabilities(scores, temperature)
    positive = probabilities > 0.0
    if not np.any(positive):
        return 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def temperature_for_target_entropy(
    ucb_scores: Sequence[float],
    target_entropy: float,
    *,
    entropy_tolerance: float = 1e-10,
    max_bisection_steps: int = 120,
) -> float:
    """Invert Boltzmann action entropy for the current UCB spectrum.

    For non-degenerate scores the Boltzmann entropy is monotone increasing in
    temperature. Degenerate spectra cannot realize a lower-than-uniform action
    entropy; in that case any positive temperature is equivalent, so return 1.0
    as a deterministic compatibility representative.
    """

    values = np.asarray(list(ucb_scores), dtype=float)
    if values.ndim != 1:
        raise ValueError("ucb_scores must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("ucb_scores must be finite")
    if not math.isfinite(target_entropy):
        raise ValueError("target_entropy must be finite")
    n = int(values.size)
    if n == 0:
        if abs(target_entropy) > entropy_tolerance:
            raise ValueError("empty UCB spectrum can only target zero entropy")
        return 0.0
    max_entropy = math.log(n)
    if target_entropy < -entropy_tolerance or target_entropy > max_entropy + entropy_tolerance:
        raise ValueError("target_entropy must lie in [0, log N]")
    if n == 1:
        return 0.0

    target = min(max(float(target_entropy), 0.0), max_entropy)
    span = float(np.max(values) - np.min(values))
    if span <= 1e-15:
        return 1.0

    entropy_at_zero = _boltzmann_entropy(values, 0.0)
    if target <= entropy_at_zero + entropy_tolerance:
        return 0.0

    high = span
    high_entropy = _boltzmann_entropy(values, high)
    while high_entropy < target - entropy_tolerance:
        high *= 2.0
        if not math.isfinite(high) or high > 1e15 * max(span, 1.0):
            return high
        high_entropy = _boltzmann_entropy(values, high)

    low = 0.0
    for _ in range(max_bisection_steps):
        mid = (low + high) / 2.0
        entropy = _boltzmann_entropy(values, mid)
        if abs(entropy - target) <= entropy_tolerance:
            return float(mid)
        if entropy < target:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def evaluate_entropy_state(
    spatial_entropy: float,
    frontier_size: int,
    ucb_scores: Sequence[float],
    previous_entropy: float | None,
    iteration: int,
    min_iterations: int,
    entropy_change_threshold: float,
    previous_plateau_count: int = 0,
    plateau_confirmations: int = 1,
    t_max: float = 1.0,
) -> EntropyState:
    """Solve current allocation temperature and update entropy-history telemetry."""

    del t_max  # retained only for call-site/spec compatibility; no longer defines T.
    if len(ucb_scores) != frontier_size:
        raise ValueError("ucb_scores length must equal frontier_size")

    normalized_entropy = _normalized_entropy_coordinate(
        spatial_entropy,
        frontier_size,
    )
    effective_temperature = temperature_for_target_entropy(
        ucb_scores,
        spatial_entropy,
    )

    if previous_entropy is None:
        return EntropyState(
            spatial_entropy=spatial_entropy,
            entropy_delta=None,
            effective_temperature=effective_temperature,
            normalized_temperature=normalized_entropy,
            plateau_signal=False,
            consecutive_plateau_count=0,
        )

    delta = abs(spatial_entropy - previous_entropy) / max(abs(previous_entropy), 1.0)
    current_is_plateau = delta < entropy_change_threshold
    consecutive_plateau_count = (
        previous_plateau_count + 1 if current_is_plateau else 0
    )
    plateau_signal = (
        iteration >= min_iterations
        and consecutive_plateau_count >= plateau_confirmations
    )
    return EntropyState(
        spatial_entropy=spatial_entropy,
        entropy_delta=float(delta),
        effective_temperature=effective_temperature,
        normalized_temperature=normalized_entropy,
        plateau_signal=plateau_signal,
        consecutive_plateau_count=consecutive_plateau_count,
    )
