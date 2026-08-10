"""Diversity-state and temperature controller for DTE frontier search.

The current KDE-derived population observable remains a provisional,
batch-relative proxy. This module only maps that current-state observable to the
system allocation temperature and keeps cross-iteration delta/plateau telemetry
separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    """Return the legacy batch-relative KDE diversity proxy."""

    return compute_kde_state(embeddings).spatial_entropy


def _normalized_temperature_from_current_state(
    spatial_entropy: float,
    frontier_size: int,
) -> float:
    """Map the current population diversity proxy to the canonical [0, 1] coordinate."""

    if frontier_size < 0:
        raise ValueError("frontier_size must be nonnegative")
    if not math.isfinite(spatial_entropy):
        raise ValueError("spatial_entropy must be finite")
    if frontier_size <= 1:
        return 0.0

    max_entropy = math.log(frontier_size)
    tolerance = 1e-12
    if spatial_entropy < -tolerance or spatial_entropy > max_entropy + tolerance:
        raise ValueError("spatial_entropy outside the current proxy's [0, log N] range")

    # Clamp only floating-point noise at the analytic endpoints.
    return float(min(1.0, max(0.0, spatial_entropy / max_entropy)))


def evaluate_entropy_state(
    spatial_entropy: float,
    frontier_size: int,
    previous_entropy: float | None,
    iteration: int,
    min_iterations: int,
    entropy_change_threshold: float,
    previous_plateau_count: int = 0,
    plateau_confirmations: int = 1,
    t_max: float = 1.0,
) -> EntropyState:
    """Map current diversity to temperature and history only to plateau telemetry."""

    normalized_temperature = _normalized_temperature_from_current_state(
        spatial_entropy,
        frontier_size,
    )
    effective_temperature = float(t_max * normalized_temperature)

    if previous_entropy is None:
        return EntropyState(
            spatial_entropy=spatial_entropy,
            entropy_delta=None,
            effective_temperature=effective_temperature,
            normalized_temperature=normalized_temperature,
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
        normalized_temperature=normalized_temperature,
        plateau_signal=plateau_signal,
        consecutive_plateau_count=consecutive_plateau_count,
    )
