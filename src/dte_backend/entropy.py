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
    """Estimate dimensionless frontier spatial entropy from embeddings."""

    return compute_kde_state(embeddings).spatial_entropy


def evaluate_entropy_state(
    spatial_entropy: float,
    previous_entropy: float | None,
    iteration: int,
    min_iterations: int,
    entropy_change_threshold: float,
    previous_plateau_count: int = 0,
    plateau_confirmations: int = 1,
    t_max: float = 1.0,
) -> EntropyState:
    """Convert entropy history into temperature and a non-authoritative plateau signal."""

    if previous_entropy is None:
        return EntropyState(
            spatial_entropy=spatial_entropy,
            entropy_delta=None,
            effective_temperature=float(t_max),
            normalized_temperature=1.0,
            plateau_signal=False,
            consecutive_plateau_count=0,
        )

    delta = abs(spatial_entropy - previous_entropy) / max(abs(previous_entropy), 1.0)
    normalized_temperature = min(1.0, delta / max(entropy_change_threshold, 1e-12))
    effective_temperature = float(t_max * normalized_temperature)
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
        normalized_temperature=float(normalized_temperature),
        plateau_signal=plateau_signal,
        consecutive_plateau_count=consecutive_plateau_count,
    )
