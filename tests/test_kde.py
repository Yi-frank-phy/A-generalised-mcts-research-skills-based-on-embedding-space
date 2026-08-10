import math

import numpy as np
import pytest

from dte_backend.kde import compute_kde_state, pairwise_squared_distance


def test_kde_state_shapes():
    state = compute_kde_state([[1, 0], [0, 1], [-1, 0]])
    assert len(state.log_density) == 3
    assert len(state.uncertainty) == 3
    assert state.bandwidth2 > 0


def test_pairwise_distance_square_shape():
    dist = pairwise_squared_distance([[1, 0], [0, 1]])
    assert dist.shape == (2, 2)


def test_two_distinct_points_have_batch_relative_closed_form():
    expected = -math.log((1.0 + math.exp(-0.5)) / 2.0)
    near = compute_kde_state([[1.0, 0.0], [0.99, 0.1]])
    far = compute_kde_state([[1.0, 0.0], [-1.0, 0.0]])
    assert near.spatial_entropy == pytest.approx(expected)
    assert far.spatial_entropy == pytest.approx(expected)


def test_equidistant_support_matches_closed_form():
    for n in (3, 4, 8):
        gram = np.full((n, n), -1.0 / (n - 1))
        np.fill_diagonal(gram, 1.0)
        vals, vecs = np.linalg.eigh(gram)
        positive = vals > 1e-10
        coords = vecs[:, positive] @ np.diag(np.sqrt(vals[positive]))
        state = compute_kde_state(coords.tolist())
        expected = -math.log((1.0 + (n - 1) * math.exp(-0.5)) / n)
        assert state.spatial_entropy == pytest.approx(expected)
        assert expected < 0.5


def test_legacy_kde_metric_identity_is_explicit_and_versioned():
    from dte_backend.kde import LEGACY_KDE_METRIC_IDENTITY

    identity = LEGACY_KDE_METRIC_IDENTITY
    assert identity.name == "batch_relative_kernel_surprisal"
    assert identity.version == 1
    assert identity.embedding_normalization == "l2_per_vector"
    assert identity.kernel == "gaussian"
    assert (
        identity.bandwidth_rule
        == "median_nonzero_pairwise_squared_distance_per_batch"
    )
    assert identity.self_kernel_included is True


def test_batch_relative_kernel_surprisal_alias_matches_legacy_field():
    state = compute_kde_state([[1, 0], [0, 1], [-1, 0]])
    assert state.batch_relative_kernel_surprisal == state.spatial_entropy
