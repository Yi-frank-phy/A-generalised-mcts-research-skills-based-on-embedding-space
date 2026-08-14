import numpy as np

from dte_nextgen.thought_space.controller import (
    frontier_standard_deviations,
    score_transition_frontier,
)
from dte_nextgen.thought_space.entropy import (
    configurational_entropy,
    normalized_kernel_density,
)


def test_five_nearby_live_transitions_have_lower_sd_than_isolated_direction() -> None:
    points = np.asarray([[0.0], [0.001], [0.002], [0.003], [0.004], [10.0]])
    bandwidth = 0.1

    density = normalized_kernel_density(points, bandwidth)
    sd = frontier_standard_deviations(points, bandwidth)

    assert np.all(sd[:5] < sd[5])
    assert np.allclose(sd, 1.0 / np.sqrt(len(points) * density))


def test_identical_k_cluster_recovers_inverse_sqrt_k() -> None:
    points = np.asarray([[0.0], [0.0], [0.0], [100.0]])
    sd = frontier_standard_deviations(points, bandwidth=0.1)

    assert np.allclose(sd[:3], 1.0 / np.sqrt(3.0), atol=1e-6)
    assert np.isclose(sd[3], 1.0, atol=1e-6)


def test_controller_sd_depends_only_on_live_geometry() -> None:
    points = np.asarray([[0.0], [0.01], [10.0]])

    low_values = score_transition_frontier(
        points,
        propulsion_values=np.asarray([0.0, 0.0, 0.0]),
        bandwidth=0.1,
    )
    changed_values = score_transition_frontier(
        points,
        propulsion_values=np.asarray([1.0, 0.4, 0.9]),
        bandwidth=0.1,
    )

    assert np.allclose(
        low_values["standard_deviations"],
        changed_values["standard_deviations"],
    )
    assert np.allclose(
        changed_values["ucb_scores"],
        changed_values["values"] + changed_values["standard_deviations"],
    )


def test_controller_entropy_and_sd_use_the_same_density_field() -> None:
    points = np.asarray([[0.0], [0.01], [2.0]])
    bandwidth = 0.2

    result = score_transition_frontier(
        points,
        propulsion_values=np.asarray([0.2, 0.4, 0.3]),
        bandwidth=bandwidth,
    )
    density = normalized_kernel_density(points, bandwidth)

    assert np.allclose(result["densities"], density)
    assert np.allclose(
        result["standard_deviations"],
        1.0 / np.sqrt(len(points) * density),
    )
    assert np.isclose(
        result["target_entropy"],
        configurational_entropy(points, bandwidth),
    )
