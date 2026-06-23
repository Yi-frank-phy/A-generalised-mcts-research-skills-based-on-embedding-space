from dte_backend.kde import compute_kde_state, pairwise_squared_distance


def test_kde_state_shapes():
    state = compute_kde_state([[1, 0], [0, 1], [-1, 0]])
    assert len(state.log_density) == 3
    assert len(state.uncertainty) == 3
    assert state.bandwidth2 > 0


def test_pairwise_distance_square_shape():
    dist = pairwise_squared_distance([[1, 0], [0, 1]])
    assert dist.shape == (2, 2)
