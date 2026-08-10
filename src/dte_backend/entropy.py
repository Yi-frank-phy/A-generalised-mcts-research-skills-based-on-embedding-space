"""Compatibility controller for the legacy DTE KDE proxy.

The retained `spatial_entropy` fields refer to the historical batch-relative
kernel surprisal proxy. They must not be interpreted as calibrated research
entropy or compared across incompatible metric identities.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kde import KDEMetricIdentity, LEGACY_KDE_METRIC_IDENTITY, compute_kde_state


@dataclass(frozen=True)
class MetricObservation:
    """One metric value together with the exact identity that produced it."""

    value: float
    identity: KDEMetricIdentity = LEGACY_KDE_METRIC_IDENTITY


def relative_metric_delta(
    current: MetricObservation,
    previous: MetricObservation | None,
) -> float | None:
    """Return a relative delta only for observations from one metric identity."""

    if previous is None or current.identity != previous.identity:
        return None
    return abs(current.value - previous.value) / max(abs(previous.value), 1.0)


@dataclass(frozen=True)
class EntropyState:
    """Compatibility observables derived from the legacy KDE proxy."""

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
    """Return the legacy batch-relative kernel surprisal compatibility field."""

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
    """Retain the historical proxy controller as a non-authoritative signal."""

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
    normalized_temperature = min(
        1.0, delta / max(entropy_change_threshold, 1e-12)
    )
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
