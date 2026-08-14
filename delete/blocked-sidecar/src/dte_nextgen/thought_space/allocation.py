import numpy as np


def boltzmann_probabilities(ucb_scores: np.ndarray, temperature: float) -> np.ndarray:
    """Return stable probabilities proportional to exp(U_i / T)."""
    scores = np.asarray(ucb_scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("ucb_scores must be a non-empty 1D array")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")

    scaled = scores / temperature
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    return weights / np.sum(weights)


def boltzmann_entropy(ucb_scores: np.ndarray, temperature: float) -> float:
    probabilities = boltzmann_probabilities(ucb_scores, temperature)
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def temperature_for_target_entropy(
    ucb_scores: np.ndarray,
    target_entropy: float,
    min_temperature: float = 1e-6,
    initial_max_temperature: float = 1.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float:
    """Invert Boltzmann entropy for the current provisional H_B = H_geom closure."""
    scores = np.asarray(ucb_scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("ucb_scores must be a non-empty 1D array")
    if min_temperature <= 0.0 or initial_max_temperature <= 0.0:
        raise ValueError("temperature bounds must be positive")

    max_entropy = float(np.log(len(scores)))
    if target_entropy < 0.0 or target_entropy > max_entropy:
        raise ValueError("target_entropy must lie in [0, log(N)]")

    if np.allclose(scores, scores[0]):
        return float(initial_max_temperature)

    low = float(min_temperature)
    h_low = boltzmann_entropy(scores, low)
    if target_entropy <= h_low + tolerance:
        return low

    high = float(initial_max_temperature)
    h_high = boltzmann_entropy(scores, high)
    while h_high < target_entropy and high < 1e12:
        high *= 2.0
        h_high = boltzmann_entropy(scores, high)

    if h_high < target_entropy:
        raise ValueError("could not bracket target entropy")

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        h_mid = boltzmann_entropy(scores, mid)
        if abs(h_mid - target_entropy) <= tolerance:
            return float(mid)
        if h_mid < target_entropy:
            low = mid
        else:
            high = mid

    return float(0.5 * (low + high))


def select_next_index(probabilities: np.ndarray, quantile: float) -> int:
    """Select exactly one categorical action from an explicit uniform quantile."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or len(probs) == 0:
        raise ValueError("probabilities must be a non-empty 1D array")
    if not np.isfinite(probs).all() or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isfinite(quantile) or not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must lie in [0, 1)")

    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")

    cumulative = np.cumsum(probs / total)
    index = int(np.searchsorted(cumulative, quantile, side="right"))
    return min(index, len(probs) - 1)
