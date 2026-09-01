"""Entropy-matched radial distribution for the new controller."""

from __future__ import annotations
import numpy as np
from .space_measure import intrinsic_cell_volumes, volume_reward_statistics


def _vec(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or len(out) == 0 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite non-empty 1D array")
    return out


def _phase_space(radii: np.ndarray, volumes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = _vec(radii, "radii")
    v = _vec(volumes, "cell_volumes")
    if len(r) != len(v) or np.any(r < 0.0) or np.any(v <= 0.0):
        raise ValueError("radii/volumes must align, with radii >=0 and volumes >0")
    return r, v


def boltzmann_distribution(radii: np.ndarray, temperature: float, volumes: np.ndarray) -> np.ndarray:
    r, v = _phase_space(radii, volumes)
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logw = np.log(v) - r / temp
    logw -= np.max(logw)
    weights = np.exp(logw)
    return weights / np.sum(weights)


def entropy_of_density(probabilities: np.ndarray, volumes: np.ndarray) -> float:
    p = _vec(probabilities, "probabilities")
    v = _vec(volumes, "cell_volumes")
    if len(p) != len(v) or np.any(p < 0.0) or np.any(v <= 0.0) or np.sum(p) <= 0.0:
        raise ValueError("invalid probability/volume arrays")
    p = p / np.sum(p)
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log(p[mask] / v[mask])))


def temperature_for_entropy(radii: np.ndarray, volumes: np.ndarray, target_entropy: float) -> float:
    r, v = _phase_space(radii, volumes)
    target = float(target_entropy)
    if not np.isfinite(target) or np.allclose(r, r[0]):
        raise ValueError("target entropy and radii must define a nontrivial phase space")
    def h(temp: float) -> float:
        return entropy_of_density(boltzmann_distribution(r, temp, v), v)
    tolerance = 1e-10
    low = 1e-8
    cold = h(low)
    if target < cold - tolerance:
        raise ValueError("target_entropy is below the cold-limit entropy")
    if abs(target - cold) <= tolerance:
        return low
    if target > np.log(np.sum(v)) + tolerance:
        raise ValueError("target_entropy exceeds the local phase-space maximum")
    high = 1.0
    while h(high) < target - tolerance and high < 1e16:
        high *= 2.0
    if h(high) < target - tolerance:
        raise ValueError("could not bracket target_entropy")
    for _ in range(256):
        mid = 0.5 * (low + high)
        current = h(mid)
        if abs(current - target) <= tolerance:
            return float(mid)
        if current < target:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


def node_reward_sd_from_occupancy(
    radii: np.ndarray,
    reference_density: np.ndarray,
    occupancy_fraction: float,
    *,
    reward_values: np.ndarray | None = None,
) -> dict[str, float]:
    rho = float(occupancy_fraction)
    if not np.isfinite(rho) or not 0.0 < rho <= 1.0:
        raise ValueError("occupancy_fraction must lie in (0,1]")
    volumes = intrinsic_cell_volumes(reference_density)
    target = float(-np.log(rho))
    temperature = temperature_for_entropy(radii, volumes, target)
    probabilities = boltzmann_distribution(radii, temperature, volumes)

    if reward_values is None:
        stats = volume_reward_statistics(radii, volumes, probabilities)
        reward_mean = float(stats["mean"])
        reward_sd = float(stats["sd"])
    else:
        reward = _vec(reward_values, "reward_values")
        if len(reward) != len(radii) or np.any(reward < 0.0):
            raise ValueError("reward_values must be non-negative and align with radii")
        reward_mean = float(np.sum(probabilities * reward))
        reward_sd = float(
            np.sqrt(np.sum(probabilities * (reward - reward_mean) ** 2))
        )

    return {
        "temperature": temperature,
        "target_entropy": target,
        "volume_reward_mean": reward_mean,
        "volume_reward_sd": reward_sd,
        "geometric_half_peak_sd": min(float(temperature * np.log(2.0)), float(np.max(radii))),
    }
