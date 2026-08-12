import numpy as np
import pytest

from dte_nextgen.thought_space.effective_sphere import (
    fit_effective_dimension,
    sphere_fourth_moment,
    summarize_effective_sphere_null,
)


def _sphere_cosines(dimension: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = (dimension - 1.0) / 2.0
    return 2.0 * rng.beta(a, a, size=count) - 1.0


def test_second_moment_fit_recovers_known_sphere_dimension():
    cosines = _sphere_cosines(dimension=64, count=200_000, seed=11)

    fitted = fit_effective_dimension(cosines)

    assert fitted == pytest.approx(64.0, rel=0.03)


def test_fitted_dimension_predicts_fourth_moment_without_refitting():
    cosines = _sphere_cosines(dimension=96, count=250_000, seed=23)
    fitted = fit_effective_dimension(cosines)
    observed_fourth = float(np.mean(cosines**4))
    predicted_fourth = sphere_fourth_moment(fitted)

    assert observed_fourth / predicted_fourth == pytest.approx(1.0, rel=0.04)


def test_summary_exposes_shift_that_an_isotropic_sphere_cannot_explain():
    cosines = _sphere_cosines(dimension=128, count=100_000, seed=31) + 0.12
    cosines = np.clip(cosines, -1.0, 1.0)

    summary = summarize_effective_sphere_null(cosines)

    assert summary["mean_cosine"] > 0.10
    assert summary["mean_z_score"] > 5.0


def test_invalid_cosines_are_rejected():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        fit_effective_dimension(np.array([0.0, 1.2]))
