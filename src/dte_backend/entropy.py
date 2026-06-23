"""Entropy and temperature controller for DTE frontier search.

Search should be controlled by the state of the frontier, not by a fixed
iteration counter alone. Spatial entropy is computed from the same KDE state
that provides uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kde import compute_kde_state


@dataclass(frozen=True)
class EntropyState:
    """Search-phase observables for one DTE iteration."""

    spatial_entropy: float
    entropy_delta: float | None
    effective_temperature: float
    normalized_temperature: float
    should_synthesize: bool
    stop_reason: str | None = None


def spatial_entropy_from_embeddings(embeddings: list[list[float]]) -> float:
    """Estimate dimensionless frontier spatial entropy from embeddings."""

    return compute_kde_state(embeddings).spatial_entropy


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
