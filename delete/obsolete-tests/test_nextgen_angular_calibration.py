import numpy as np
import pytest

from dte_nextgen.thought_space.angular import (
    FrozenEmpiricalAngularCalibration,
    l2_normalize_rows,
    off_diagonal_cosines,
    pairwise_cosine_matrix,
)


def test_l2_normalization_is_invariant_to_positive_row_rescaling():
    base = np.array([[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]])
    scaled = np.array([[30.0, 40.0, 0.0], [0.0, 0.2, 0.0]])

    assert np.allclose(l2_normalize_rows(base), l2_normalize_rows(scaled))


def test_zero_norm_embedding_row_is_rejected():
    with pytest.raises(ValueError, match="zero-norm"):
        l2_normalize_rows(np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_pairwise_cosine_recovers_exact_simple_3d_angles():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    cosine = pairwise_cosine_matrix(vectors)

    assert np.allclose(np.diag(cosine), 1.0)
    assert np.isclose(cosine[0, 1], 0.0)
    assert np.isclose(cosine[0, 2], -1.0)


def test_off_diagonal_cosines_use_each_unordered_pair_once():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    assert np.allclose(off_diagonal_cosines(vectors), [0.0, -1.0, 0.0])


def test_empirical_3d_uniform_reference_approximately_recovers_raw_cosine():
    background_cosines = np.linspace(-1.0, 1.0, 1001)
    calibration = FrozenEmpiricalAngularCalibration.from_background_cosines(
        background_cosines
    )
    probe = np.array([-0.8, -0.2, 0.0, 0.4, 0.9])

    calibrated = calibration.calibrate_cosines(probe)

    assert np.all(np.diff(calibrated) >= 0.0)
    assert np.all(calibrated >= -1.0)
    assert np.all(calibrated <= 1.0)
    assert np.allclose(calibrated, probe, atol=3e-3)


def test_empirical_calibration_expands_high_dimensional_like_cosine_concentration():
    narrow_background = np.linspace(-0.03, 0.03, 1001)
    calibration = FrozenEmpiricalAngularCalibration.from_background_cosines(
        narrow_background
    )

    calibrated = calibration.calibrate_cosines(np.array([-0.02, 0.0, 0.02]))

    assert calibrated[0] < -0.5
    assert abs(calibrated[1]) < 0.01
    assert calibrated[2] > 0.5


def test_background_embeddings_are_frozen_as_sorted_pair_cosines():
    background = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    calibration = FrozenEmpiricalAngularCalibration.from_background_embeddings(
        background
    )

    expected = np.sort(off_diagonal_cosines(background))
    assert np.allclose(calibration.background_cosines, expected)


def test_nonfinite_or_out_of_range_cosines_are_rejected():
    calibration = FrozenEmpiricalAngularCalibration.from_background_cosines(
        np.linspace(-0.5, 0.5, 11)
    )

    with pytest.raises(ValueError, match="finite"):
        calibration.calibrate_cosines(np.array([np.nan]))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        calibration.calibrate_cosines(np.array([1.01]))
