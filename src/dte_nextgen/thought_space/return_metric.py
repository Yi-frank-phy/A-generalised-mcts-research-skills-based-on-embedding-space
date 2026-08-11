import numpy as np


def _as_cloud(points: np.ndarray) -> np.ndarray:
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2:
        raise ValueError("embedding cloud must be a 2D array")
    if len(cloud) == 0:
        raise ValueError("embedding cloud must be non-empty")
    return cloud


def _rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    diff = x[:, None, :] - y[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    return np.exp(-dist_sq / (2.0 * bandwidth * bandwidth))


def rbf_mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Biased empirical MMD^2 under one frozen RBF metric."""
    x_cloud = _as_cloud(x)
    y_cloud = _as_cloud(y)
    if x_cloud.shape[1] != y_cloud.shape[1]:
        raise ValueError("embedding clouds must have the same dimension")

    k_xx = _rbf_kernel(x_cloud, x_cloud, bandwidth)
    k_yy = _rbf_kernel(y_cloud, y_cloud, bandwidth)
    k_xy = _rbf_kernel(x_cloud, y_cloud, bandwidth)
    value = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(max(0.0, value))


def null_adjusted_geometric_return(
    before: np.ndarray,
    after: np.ndarray,
    null_a: np.ndarray,
    null_b: np.ndarray,
    bandwidth: float,
) -> float:
    """Experimental movement proxy above same-state sampling drift, scaled to [0,1]."""
    observed = rbf_mmd2(before, after, bandwidth)
    null_drift = rbf_mmd2(null_a, null_b, bandwidth)
    return float(max(0.0, observed - null_drift) / 2.0)
