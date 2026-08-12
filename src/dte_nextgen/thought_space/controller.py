import numpy as np

from .entropy import adaptive_bandwidth, configurational_entropy, normalized_kernel_density


def _validate_frontier(embeddings: np.ndarray) -> np.ndarray:
    points = np.asarray(embeddings, dtype=float)
    if points.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(points) == 0:
        raise ValueError("embeddings must be non-empty")
    if not np.isfinite(points).all():
        raise ValueError("embeddings must contain only finite values")
    return points


def frontier_standard_deviations(
    embeddings: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Original DTE geometric uncertainty over the current live frontier.

    The same self-including KDE density used by configurational entropy gives
    ``SD_i = 1 / sqrt(N * rho_i)``.  No history count or return statistic enters
    this calculation.
    """
    points = _validate_frontier(embeddings)
    density = normalized_kernel_density(points, bandwidth)
    return 1.0 / np.sqrt(len(points) * density)


def score_transition_frontier(
    embeddings: np.ndarray,
    propulsion_values: np.ndarray,
    bandwidth: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Score completed transition directions using live geometry only.

    ``propulsion_values`` are externally supplied estimates of expected
    null-adjusted whole-frontier displacement.  This function deliberately does
    not infer V from historical rewards: it only combines the supplied V with
    geometric SD through ``U = V + SD`` and reports the shared geometry entropy.
    """
    points = _validate_frontier(embeddings)
    values = np.asarray(propulsion_values, dtype=float)
    if values.ndim != 1 or len(values) != len(points):
        raise ValueError("propulsion_values must contain one value per live transition")
    if not np.isfinite(values).all():
        raise ValueError("propulsion_values must contain only finite values")

    resolved_bandwidth = (
        adaptive_bandwidth(points) if bandwidth is None else float(bandwidth)
    )
    if resolved_bandwidth <= 0.0 or not np.isfinite(resolved_bandwidth):
        raise ValueError("bandwidth must be finite and positive")

    density = normalized_kernel_density(points, resolved_bandwidth)
    standard_deviations = 1.0 / np.sqrt(len(points) * density)
    ucb_scores = values + standard_deviations
    target_entropy = configurational_entropy(points, resolved_bandwidth)

    return {
        "embeddings": points,
        "densities": density,
        "values": values,
        "standard_deviations": standard_deviations,
        "ucb_scores": ucb_scores,
        "target_entropy": float(target_entropy),
        "bandwidth": float(resolved_bandwidth),
    }
