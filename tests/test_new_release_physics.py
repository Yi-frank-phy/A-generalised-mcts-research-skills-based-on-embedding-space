import numpy as np

from dte_backend.controller_atlas import FrozenReferenceAtlas
from dte_backend.controller_value import (
    proper_volume_distance_matrix,
    proper_volume_values_for_queries,
)
from dte_backend.space_distribution import node_reward_sd_from_occupancy
from dte_backend.space_geometry import (
    all_pairs_geodesic_distances,
    nearest_reference_indices,
    query_geodesic_distance,
    query_geodesic_distance_matrix,
    query_reference_weights,
    reference_radii_for_queries,
)
from dte_backend.space_measure import (
    intrinsic_cell_volumes,
    intrinsic_proper_volume_at_radius,
    volume_reward_statistics,
)


def _circle(angles: list[float]) -> np.ndarray:
    return np.asarray([[np.cos(a), np.sin(a)] for a in angles], dtype=float)


def _atlas(reference: np.ndarray, geodesic: np.ndarray) -> FrozenReferenceAtlas:
    return FrozenReferenceAtlas(
        node_ids=tuple(f"r{i}" for i in range(len(reference))),
        embeddings=reference,
        geodesic_distances=geodesic,
        reference_density=np.ones(len(reference)),
        graph_k=1,
        identity="toy",
    )


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
    assert np.allclose(
        reference_radii_for_queries(
            reference[[0]], reference, geodesic, neighbor_count=2
        )[0],
        geodesic[0],
    )
    assert np.isclose(
        query_geodesic_distance(
            reference[0], reference[2], reference, geodesic, neighbor_count=2
        ),
        0.6,
        atol=1e-10,
    )


def test_continuous_query_extension_is_exact_on_reference_atlas() -> None:
    reference = _circle([0.0, 0.3, 0.6, 0.9, 1.2])
    geodesic = all_pairs_geodesic_distances(reference, k=1)
    profiles = reference_radii_for_queries(
        reference, reference, geodesic, neighbor_count=2
    )
    pairwise = query_geodesic_distance_matrix(
        reference, reference, reference, geodesic, neighbor_count=2
    )
    assert np.allclose(profiles, geodesic, atol=1e-12)
    assert np.allclose(pairwise, geodesic, atol=1e-12)


def test_modified_shepard_support_stays_local() -> None:
    reference = _circle([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    query = _circle([0.13])
    weights = query_reference_weights(query, reference, neighbor_count=2)[0]
    assert np.isclose(np.sum(weights), 1.0)
    assert np.count_nonzero(weights > 1e-10) == 2
    assert np.all(weights[3:] == 0.0)


def test_continuous_query_extension_resolves_motion_inside_old_nearest_cell() -> None:
    reference = _circle([0.0, 0.3, 0.6, 0.9, 1.2])
    geodesic = all_pairs_geodesic_distances(reference, k=1)
    queries = _circle([0.05, 0.10])
    assert np.array_equal(nearest_reference_indices(queries, reference), [0, 0])
    distance = query_geodesic_distance(
        queries[0], queries[1], reference, geodesic, neighbor_count=2
    )
    assert 0.0 < distance < 0.3


def test_continuous_query_extension_removes_old_voronoi_boundary_jump() -> None:
    reference = _circle([0.0, 0.3, 0.6, 0.9, 1.2])
    geodesic = all_pairs_geodesic_distances(reference, k=1)
    epsilon = 1e-4
    queries = _circle([0.15 - epsilon, 0.15 + epsilon])
    old_anchors = nearest_reference_indices(queries, reference)
    assert old_anchors[0] != old_anchors[1]
    distance = query_geodesic_distance(
        queries[0], queries[1], reference, geodesic, neighbor_count=2
    )
    profiles = reference_radii_for_queries(
        queries, reference, geodesic, neighbor_count=2
    )
    assert distance < 0.01
    assert np.max(np.abs(profiles[0] - profiles[1])) < 0.01


def test_proper_volume_field_extension_is_exact_on_reference_sources() -> None:
    reference = _circle([0.0, 0.3, 0.6, 0.9, 1.2])
    geodesic = all_pairs_geodesic_distances(reference, k=1)
    atlas = _atlas(reference, geodesic)
    expected = intrinsic_proper_volume_at_radius(
        geodesic[0],
        np.ones(len(reference)),
        geodesic[0, 2],
    )
    observed = proper_volume_distance_matrix(reference[[0]], reference[[2]], atlas)[0, 0]
    assert np.isclose(observed, expected, atol=1e-12)


def test_proper_volume_field_is_continuous_when_source_leaves_atlas_vertex() -> None:
    reference = _circle([0.0, 0.3, 0.6, 0.9, 1.2])
    geodesic = all_pairs_geodesic_distances(reference, k=1)
    atlas = _atlas(reference, geodesic)
    sources = _circle([0.0, 1e-6])
    target = reference[[2]]
    observed = proper_volume_distance_matrix(sources, target, atlas)[:, 0]
    assert np.isclose(observed[0], 2.0, atol=1e-12)
    assert abs(observed[1] - observed[0]) < 1e-6


def test_boltzmann_reward_values_use_same_continuous_proper_volume_field() -> None:
    reference = _circle([0.0, 0.3, 0.6, 0.9, 1.2])
    geodesic = all_pairs_geodesic_distances(reference, k=1)
    atlas = _atlas(reference, geodesic)
    sources = _circle([0.0, 1e-6])
    radii = reference_radii_for_queries(
        sources, reference, geodesic, neighbor_count=2
    )
    rewards = proper_volume_values_for_queries(sources, radii, atlas)
    assert np.allclose(rewards[0], [0.0, 1.0, 2.0, 3.0, 4.0], atol=1e-12)
    assert np.max(np.abs(rewards[1] - rewards[0])) < 1e-5


def test_lower_occupancy_produces_larger_reward_uncertainty_on_same_atlas() -> None:
    radii = np.asarray([0.0, 0.3, 0.6, 0.9, 1.2])
    density = np.ones(5)
    crowded = node_reward_sd_from_occupancy(radii, density, 0.75)
    isolated = node_reward_sd_from_occupancy(radii, density, 0.40)
    assert float(crowded["volume_reward_sd"]) < float(isolated["volume_reward_sd"])
