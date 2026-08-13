import numpy as np

from dte_backend.space_distribution import node_reward_sd_from_occupancy
from dte_backend.space_geometry import all_pairs_geodesic_distances, query_geodesic_distance, reference_radii_for_queries
from dte_backend.space_measure import intrinsic_cell_volumes, intrinsic_proper_volume_at_radius, volume_reward_statistics


def _circle(angles: list[float]) -> np.ndarray:
    return np.asarray([[np.cos(a), np.sin(a)] for a in angles], dtype=float)


def test_proper_volume_is_not_node_normalized() -> None:
    radii = np.asarray([0.0, 1.0, 2.0, 4.0])
    volumes = np.asarray([2.0, 1.0, 3.0, 4.0])
    assert np.isclose(intrinsic_proper_volume_at_radius(radii, volumes, 0.0), 0.0)
    assert np.isclose(intrinsic_proper_volume_at_radius(radii, volumes, 1.5), 2.5)
    assert np.isclose(intrinsic_proper_volume_at_radius(radii, volumes, 4.0), 8.0)


def test_reward_sd_is_sd_of_same_proper_volume_variable() -> None:
    radii = np.asarray([0.0, 1.0, 2.0, 4.0])
    volumes = np.asarray([2.0, 1.0, 3.0, 4.0])
    probability = np.asarray([0.1, 0.2, 0.3, 0.4])
    stats = volume_reward_statistics(radii, volumes, probability)
    assert np.allclose(stats["reward_values"], [0.0, 1.0, 4.0, 8.0])
    assert np.isclose(stats["mean"], 4.6)
    assert np.isclose(stats["sd"], np.sqrt(9.44))


def test_uniform_frozen_atlas_has_one_common_volume_unit() -> None:
    assert np.allclose(intrinsic_cell_volumes(np.ones(5)), np.ones(5))


def test_sparse_angular_graph_recovers_chain_geodesic() -> None:
    reference = _circle([0.0, 0.3, 0.6, 0.9, 1.2])
    geodesic = all_pairs_geodesic_distances(reference, k=1)
    assert np.isclose(geodesic[0, 4], 1.2, atol=1e-10)
    assert np.allclose(reference_radii_for_queries(reference[[0]], reference, geodesic)[0], geodesic[0])
    assert np.isclose(query_geodesic_distance(reference[0], reference[2], reference, geodesic), 0.6, atol=1e-10)


def test_lower_occupancy_produces_larger_reward_uncertainty_on_same_atlas() -> None:
    radii = np.asarray([0.0, 0.3, 0.6, 0.9, 1.2])
    density = np.ones(5)
    crowded = node_reward_sd_from_occupancy(radii, density, 0.75)
    isolated = node_reward_sd_from_occupancy(radii, density, 0.40)
    assert float(crowded["volume_reward_sd"]) < float(isolated["volume_reward_sd"])
