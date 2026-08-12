from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_finite_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    points = np.asarray(embeddings, dtype=float)
    if points.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(points) == 0:
        raise ValueError("embeddings must be non-empty")
    if not np.isfinite(points).all():
        raise ValueError("embeddings must be finite")
    return points


def l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """Return rowwise L2-normalized embeddings without fitting any frontier scale."""
    points = _as_finite_2d_embeddings(embeddings)
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("embeddings must not contain a zero-norm row")
    return points / norms[:, None]


def pairwise_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine matrix after rowwise L2 normalization."""
    unit = l2_normalize_rows(embeddings)
    cosine = unit @ unit.T
    return np.clip(cosine, -1.0, 1.0)


def off_diagonal_cosines(embeddings: np.ndarray) -> np.ndarray:
    """Return each unordered pair cosine exactly once, in upper-triangle order."""
    points = _as_finite_2d_embeddings(embeddings)
    if len(points) < 2:
        raise ValueError("at least two embeddings are required")
    cosine = pairwise_cosine_matrix(points)
    upper = np.triu_indices(len(points), k=1)
    return cosine[upper]


@dataclass(frozen=True)
class FrozenEmpiricalAngularCalibration:
    """Frozen empirical random-pair cosine reference for angular calibration.

    A new cosine c is mapped through the frozen empirical mid-rank CDF F_0 as
    C_0(c) = 2 F_0(c) - 1. The reference is deliberately external to the live
    frontier so scoring cannot redefine its own angular resolution each round.
    """

    background_cosines: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.background_cosines, dtype=float, copy=True)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("background_cosines must be a non-empty 1D array")
        if not np.isfinite(values).all():
            raise ValueError("background_cosines must be finite")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("background_cosines must lie in [-1, 1]")
        values.sort()
        values.setflags(write=False)
        object.__setattr__(self, "background_cosines", values)

    @classmethod
    def from_background_cosines(
        cls, background_cosines: np.ndarray
    ) -> "FrozenEmpiricalAngularCalibration":
        return cls(np.asarray(background_cosines, dtype=float))

    @classmethod
    def from_background_embeddings(
        cls, embeddings: np.ndarray
    ) -> "FrozenEmpiricalAngularCalibration":
        return cls(off_diagonal_cosines(embeddings))

    def calibrate_cosines(self, cosines: np.ndarray) -> np.ndarray:
        """Map finite cosine values to their frozen-background rank coordinate."""
        values = np.asarray(cosines, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("cosines must be finite")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("cosines must lie in [-1, 1]")

        background = self.background_cosines
        left = np.searchsorted(background, values, side="left")
        right = np.searchsorted(background, values, side="right")
        cdf = (left + right) / (2.0 * len(background))
        calibrated = 2.0 * cdf - 1.0

        # Preserve the exact cosine endpoints even when the empirical reference
        # itself contains no sample at -1 or +1.
        calibrated = np.where(values <= -1.0, -1.0, calibrated)
        calibrated = np.where(values >= 1.0, 1.0, calibrated)
        return np.asarray(calibrated, dtype=float)
