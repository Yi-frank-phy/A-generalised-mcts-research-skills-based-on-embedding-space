"""Proper-volume UCB scorer on a frozen metric-measure atlas."""

from __future__ import annotations

import numpy as np

from .geometry import all_pairs_geodesic_distances, reference_radii_for_queries
from .occupancy import estimate_live_occupancy
from .volume_boltzmann import node_reward_sd_from_occupancy


def _cloud(values: np.ndarray, name: str) -> np.ndarray:
    cloud = np.asarray(values, dtype=float)
    if cloud.ndim != 2 or len(cloud) == 0 or not np.isfinite(cloud).all():
        raise ValueError(f"{name} must be a finite non-empty 2D array")
    return cloud


def score_metric_measure_frontier(
    *,
    propulsion_values: np.ndarray,
    node_radii: np.ndarray,
    reference_density: np.ndarray,
    occupancy_fractions: np.ndarray,
) -> dict[str, object]:
    values = np.asarray(propulsion_values, dtype=float)
    radii = np.asarray(node_radii, dtype=float)
    density = np.asarray(reference_density, dtype=float)
    occupancy = np.asarray(occupancy_fractions, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0.0) or not np.isfinite(values).all():
        raise ValueError("propulsion_values must be finite, non-negative, and non-empty")
    if radii.ndim != 2 or radii.shape[0] != len(values):
        raise ValueError("node_radii must contain one row per live transition")
    if density.shape != (radii.shape[1],) or np.any(density <= 0.0):
        raise ValueError("reference_density must be positive with one value per atlas cell")
    if occupancy.shape != (len(values),) or np.any(occupancy <= 0.0) or np.any(occupancy > 1.0):
        raise ValueError("occupancy_fractions must contain one value in (0,1] per live transition")

    results = [
        node_reward_sd_from_occupancy(radii[i], density, float(occupancy[i]))
        for i in range(len(values))
    ]
    sd = np.asarray([float(item["volume_reward_sd"]) for item in results])
    entropies = np.asarray([float(item["target_entropy"]) for item in results])
    return {
        "values": values,
        "standard_deviations": sd,
        "sd_source": "proper_volume_boltzmann_reward",
        "ucb_scores": values + sd,
        "target_entropies": entropies,
        "allocation_target_entropy": float(np.mean(entropies)),
        "temperatures": np.asarray([float(item["temperature"]) for item in results]),
        "geometric_half_peak_sd": np.asarray(
            [float(item["geometric_half_peak_sd"]) for item in results]
        ),
        "boltzmann_reward_means": np.asarray(
            [float(item["volume_reward_mean"]) for item in results]
        ),
        "occupancy_fractions": occupancy,
    }


def score_proper_volume_embeddings(
    *,
    live_embeddings: np.ndarray,
    propulsion_values: np.ndarray,
    reference_embeddings: np.ndarray,
    graph_k: int,
    volume_bandwidth: float = 1.0,
    reference_density: np.ndarray | None = None,
) -> dict[str, object]:
    live = _cloud(live_embeddings, "live_embeddings")
    reference = _cloud(reference_embeddings, "reference_embeddings")
    if live.shape[1] != reference.shape[1]:
        raise ValueError("live and reference embeddings must have the same dimension")
    if len(reference) < len(live):
        raise ValueError("reference atlas must contain at least as many cells as the live frontier")
    values = np.asarray(propulsion_values, dtype=float)
    if values.shape != (len(live),):
        raise ValueError("propulsion_values must contain one value per live transition")

    geodesic = all_pairs_geodesic_distances(reference, k=int(graph_k))
    if reference_density is None:
        density = np.ones(len(reference), dtype=float)
        density_source = "uniform_frozen_reference_measure"
    else:
        density = np.asarray(reference_density, dtype=float)
        if density.shape != (len(reference),) or np.any(density <= 0.0):
            raise ValueError("reference_density must be positive with one value per atlas cell")
        density_source = "supplied_experimental_correction"

    node_radii = reference_radii_for_queries(live, reference, geodesic)
    occupancy = estimate_live_occupancy(
        live_embeddings=live,
        reference_embeddings=reference,
        geodesic_distances=geodesic,
        reference_density=density,
        volume_bandwidth=volume_bandwidth,
    )
    scored = score_metric_measure_frontier(
        propulsion_values=values,
        node_radii=node_radii,
        reference_density=density,
        occupancy_fractions=occupancy["occupancy_fractions"],
    )
    return {
        **scored,
        "live_embeddings": live,
        "reference_embeddings": reference,
        "reference_density": density,
        "reference_density_source": density_source,
        "geodesic_distances": geodesic,
        "node_radii": node_radii,
        "proper_volume_displacements": occupancy["proper_volume_displacements"],
        "volume_bandwidth": float(volume_bandwidth),
    }
