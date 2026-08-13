"""Live-frontier occupancy on frozen-atlas proper-volume geometry."""

from __future__ import annotations

import numpy as np

from .geometry import nearest_reference_indices
from .volume_measure import intrinsic_cell_volumes, intrinsic_proper_volume_at_radius


def proper_volume_displacements(
    *,
    live_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    geodesic_distances: np.ndarray,
    reference_density: np.ndarray,
) -> np.ndarray:
    live = np.asarray(live_embeddings, dtype=float)
    reference = np.asarray(reference_embeddings, dtype=float)
    geodesic = np.asarray(geodesic_distances, dtype=float)
    density = np.asarray(reference_density, dtype=float)
    if live.ndim != 2 or len(live) == 0 or not np.isfinite(live).all():
        raise ValueError("live_embeddings must be a finite non-empty 2D array")
    if reference.ndim != 2 or len(reference) == 0 or not np.isfinite(reference).all():
        raise ValueError("reference_embeddings must be a finite non-empty 2D array")
    if live.shape[1] != reference.shape[1]:
        raise ValueError("live and reference embeddings must have the same dimension")
    if geodesic.shape != (len(reference), len(reference)):
        raise ValueError("geodesic_distances must be square over the reference atlas")
    if density.shape != (len(reference),) or np.any(density <= 0.0):
        raise ValueError("reference_density must be positive with one value per atlas cell")

    anchors = nearest_reference_indices(live, reference)
    volumes = intrinsic_cell_volumes(density)
    result = np.zeros((len(live), len(live)), dtype=float)
    for i, raw_source in enumerate(anchors):
        source = int(raw_source)
        radii = geodesic[source]
        for j, raw_target in enumerate(anchors):
            move_radius = float(geodesic[source, int(raw_target)])
            result[i, j] = intrinsic_proper_volume_at_radius(radii, volumes, move_radius)
    np.fill_diagonal(result, 0.0)
    return result


def occupancy_from_proper_volume(
    proper_volume_distance: np.ndarray,
    volume_bandwidth: float = 1.0,
) -> np.ndarray:
    distance = np.asarray(proper_volume_distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1] or len(distance) == 0:
        raise ValueError("proper_volume_distance must be a non-empty square matrix")
    if not np.isfinite(distance).all() or np.any(distance < 0.0):
        raise ValueError("proper_volume_distance must be finite and non-negative")
    scale = float(volume_bandwidth)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("volume_bandwidth must be finite and positive")
    rho = np.mean(np.exp(-distance / scale), axis=1)
    return np.clip(rho, np.finfo(float).tiny, 1.0)


def estimate_live_occupancy(
    *,
    live_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    geodesic_distances: np.ndarray,
    reference_density: np.ndarray,
    volume_bandwidth: float = 1.0,
) -> dict[str, np.ndarray | float]:
    displacement = proper_volume_displacements(
        live_embeddings=live_embeddings,
        reference_embeddings=reference_embeddings,
        geodesic_distances=geodesic_distances,
        reference_density=reference_density,
    )
    occupancy = occupancy_from_proper_volume(displacement, volume_bandwidth)
    return {
        "proper_volume_displacements": displacement,
        "occupancy_fractions": occupancy,
        "volume_bandwidth": float(volume_bandwidth),
    }
