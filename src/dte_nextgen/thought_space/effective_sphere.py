from __future__ import annotations

import math

import numpy as np


def _validated_cosines(cosines: np.ndarray) -> np.ndarray:
    values = np.asarray(cosines, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("cosines must be a non-empty 1D array")
    if not np.isfinite(values).all():
        raise ValueError("cosines must be finite")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("cosines must lie in [-1, 1]")
    return values


def fit_effective_dimension(cosines: np.ndarray) -> float:
    """Fit isotropic-sphere dimension from the exact second-moment identity.

    For independent uniform directions on S^(d-1), E[cos(theta)^2] = 1 / d.
    No centering or additional scale parameter is fitted here.
    """
    values = _validated_cosines(cosines)
    second_moment = float(np.mean(values**2))
    if second_moment <= 0.0:
        raise ValueError("cosine second moment must be positive")
    return 1.0 / second_moment


def sphere_fourth_moment(dimension: float) -> float:
    """Return the exact isotropic-sphere fourth moment E[cos(theta)^4]."""
    d = float(dimension)
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError("dimension must be a positive finite number")
    return 3.0 / (d * (d + 2.0))


def summarize_effective_sphere_null(cosines: np.ndarray) -> dict[str, float]:
    """Return parameter-light diagnostics for an isotropic effective-sphere null."""
    values = _validated_cosines(cosines)
    count = len(values)
    mean_cosine = float(np.mean(values))
    second_moment = float(np.mean(values**2))
    fourth_moment = float(np.mean(values**4))
    fitted_dimension = fit_effective_dimension(values)
    predicted_fourth = sphere_fourth_moment(fitted_dimension)

    null_sd = math.sqrt(1.0 / fitted_dimension)
    mean_standard_error = null_sd / math.sqrt(count)
    mean_z_score = (
        abs(mean_cosine) / mean_standard_error
        if mean_standard_error > 0.0
        else math.inf
    )

    return {
        "count": float(count),
        "mean_cosine": mean_cosine,
        "second_moment": second_moment,
        "fourth_moment": fourth_moment,
        "fitted_dimension": fitted_dimension,
        "predicted_fourth_moment": predicted_fourth,
        "fourth_moment_ratio": fourth_moment / predicted_fourth,
        "mean_z_score": mean_z_score,
    }
