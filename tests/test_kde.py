import math

import numpy as np

from dte_backend.kde import compute_kde_state, estimate_bandwidth2, pairwise_squared_distance


def test_kde_state_shapes():
    state = compute_kde_state([[1, 0], [0, 1], [-1, 0]])
    assert len(state.log_density) == 3
    assert len(state.uncertainty) == 3
    assert state.bandwidth2 > 0


def test_pairwise_distance_square_shape():
    dist = pairwise_squared_distance([[1, 0], [0, 1]])
    assert dist.shape == (2, 2)


def test_bandwidth2_is_half_the_median_nonzero_squared_pair_distance():
    dist2 = pairwise_squared_distance([[1, 0], [0, 1], [-1, 0]])
    nonzero = dist2[dist2 > 1e-12]
    expected = float(np.median(nonzero)) / 2.0

    assert math.isclose(estimate_bandwidth2(dist2), expected, rel_tol=0.0, abs_tol=1e-12)


def test_kde_uncertainty_preserves_absolute_local_evidence_scale():
    state = compute_kde_state([[1, 0], [1, 0], [0, 1]])
    n = 3
    expected = [1.0 / math.sqrt(n * math.exp(log_rho)) for log_rho in state.log_density]

    assert np.allclose(state.uncertainty, expected, rtol=0.0, atol=1e-12)
    assert 0.0 < state.uncertainty[0] < 1.0
    assert 0.0 < state.uncertainty[2] <= 1.0
    assert state.uncertainty[2] > state.uncertainty[0]
