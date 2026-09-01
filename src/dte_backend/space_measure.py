"""Frozen-atlas proper-volume measure for the new controller."""

from __future__ import annotations
import numpy as np


def _vec(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or len(out) == 0 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite non-empty 1D array")
    return out


def intrinsic_cell_volumes(reference_density: np.ndarray) -> np.ndarray:
    density = _vec(reference_density, "reference_density")
    if np.any(density <= 0.0):
        raise ValueError("reference_density must be positive")
    inverse = 1.0 / density
    return len(density) * inverse / np.sum(inverse)


def _phase_space(radii: np.ndarray, volumes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = _vec(radii, "radii")
    v = _vec(volumes, "cell_volumes")
    if len(r) != len(v) or np.any(r < 0.0) or np.any(v <= 0.0):
        raise ValueError("radii/volumes must align, with radii >= 0 and volumes > 0")
    return r, v


def _volume_profile(radii: np.ndarray, volumes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, v = _phase_space(radii, volumes)
    order = np.argsort(r, kind="stable")
    unique, inverse = np.unique(r[order], return_inverse=True)
    shell = np.zeros(len(unique), dtype=float)
    np.add.at(shell, inverse, v[order])
    positive = unique > 0.0
    if not np.any(positive):
        raise ValueError("radii must contain a positive accessible radius")
    return (
        np.concatenate(([0.0], unique[positive])),
        np.concatenate(([0.0], np.cumsum(shell[positive]))),
    )


def intrinsic_proper_volume_at_radii(
    radii: np.ndarray,
    cell_volumes: np.ndarray,
    query_radii: np.ndarray,
) -> np.ndarray:
    """Evaluate one frozen-source cumulative proper-volume profile in batch."""
    query = np.asarray(query_radii, dtype=float)
    if not np.isfinite(query).all() or np.any(query < 0.0):
        raise ValueError("query_radii must be finite and non-negative")
    profile_r, profile_v = _volume_profile(radii, cell_volumes)
    return np.asarray(
        np.interp(
            query,
            profile_r,
            profile_v,
            left=0.0,
            right=float(profile_v[-1]),
        ),
        dtype=float,
    )


def intrinsic_proper_volume_at_radius(
    radii: np.ndarray,
    cell_volumes: np.ndarray,
    query_radius: float,
) -> float:
    query = float(query_radius)
    return float(
        intrinsic_proper_volume_at_radii(
            radii,
            cell_volumes,
            np.asarray([query], dtype=float),
        )[0]
    )


def volume_reward_statistics(radii: np.ndarray, cell_volumes: np.ndarray, probabilities: np.ndarray) -> dict[str, np.ndarray | float]:
    r, v = _phase_space(radii, cell_volumes)
    p = _vec(probabilities, "probabilities")
    if len(p) != len(r) or np.any(p < 0.0) or np.sum(p) <= 0.0:
        raise ValueError("probabilities must be non-negative and align with radii")
    p = p / np.sum(p)
    reward = intrinsic_proper_volume_at_radii(r, v, r)
    mean = float(np.sum(p * reward))
    sd = float(np.sqrt(np.sum(p * (reward - mean) ** 2)))
    return {"reward_values": reward, "mean": mean, "sd": sd}
