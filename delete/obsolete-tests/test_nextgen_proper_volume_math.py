import numpy as np

from dte_nextgen.thought_space.geometry import (
    all_pairs_geodesic_distances,
    query_geodesic_distance,
    reference_radii_for_queries,
)
from dte_nextgen.thought_space.occupancy import estimate_live_occupancy
from dte_nextgen.thought_space.volume_measure import (
    intrinsic_cell_volumes,
    intrinsic_proper_volume_at_radius,
    volume_reward_statistics,
)


def _unit_circle(angles: list[float]) -> np.ndarray:
    return np.asarray([[np.cos(a), np.sin(a)] for a in angles], dtype=float)


def test_cumulative_proper_volume_is_absolute_not_node_normalized() -> None:
    radii = np.asarray([0.0, 1.0, 2.0, 4.0])
    volumes = np.asarray([2.0, 1.0, 3.0, 4.0])

    assert np.isclose(intrinsic_proper_volume_at_radius(radii, volumes, 0.0), 0.0)
    assert np.isclose(intrinsic_proper_volume_at_radius(radii, volumes, 1.5), 2.5)
    assert np.isclose(intrinsic_proper_volume_at_radius(radii, volumes, 4.0), 8.0)
    assert np.isclose(
        intrinsic_proper_volume_at_radius(radii, 10.0 * volumes, 1.5),
        25.0,
    )


def test_reward_sd_is_standard_deviation_of_the_same_proper_volume_variable() -> None:
    radii = np.asarray([0.0, 1.0, 2.0, 4.0])
    volumes = np.asarray([2.0, 1.0, 3.0, 4.0])
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.4])

    stats = volume_reward_statistics(radii, volumes, probabilities)

    assert np.allclose(stats["reward_values"], np.asarray([0.0, 1.0, 4.0, 8.0]))
    assert np.isclose(stats["mean"], 4.6)
    assert np.isclose(stats["sd"], np.sqrt(9.44))


def test_equal_reference_measure_uses_one_common_atlas_volume_unit() -> None:
    volumes = intrinsic_cell_volumes(np.ones(5))

    assert np.allclose(volumes, np.ones(5))
    assert np.isclose(np.sum(volumes), 5.0)


def test_sparse_angular_graph_recovers_chain_geodesic_and_query_distance() -> None:
    reference = _unit_circle([0.0, 0.3, 0.6, 0.9, 1.2])
    geodesic = all_pairs_geodesic_distances(reference, k=1)

    assert np.isclose(geodesic[0, 4], 1.2, atol=1e-10)
    radii = reference_radii_for_queries(reference[[0]], reference, geodesic)
    assert np.allclose(radii[0], geodesic[0])
    assert np.isclose(
        query_geodesic_distance(reference[0], reference[2], reference, geodesic),
        0.6,
        atol=1e-10,
    )


def test_live_duplicate_directions_have_higher_occupancy_than_isolated_direction() -> None:
    reference = _unit_circle([0.0, 0.3, 0.6, 0.9, 1.2])
    live = _unit_circle([0.0, 0.0, 1.2])
    geodesic = all_pairs_geodesic_distances(reference, k=1)

    result = estimate_live_occupancy(
        live_embeddings=live,
        reference_embeddings=reference,
        geodesic_distances=geodesic,
        reference_density=np.ones(len(reference)),
        volume_bandwidth=1.0,
    )

    rho = result["occupancy_fractions"]
    assert np.isclose(result["proper_volume_displacements"][0, 1], 0.0)
    assert rho[0] > rho[2]
    assert rho[1] > rho[2]
