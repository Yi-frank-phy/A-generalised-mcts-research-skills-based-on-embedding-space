import numpy as np


def adaptive_bandwidth(embeddings: np.ndarray) -> float:
    """Current-frontier relative scale: median pair distance / sqrt(2)."""
    points = np.asarray(embeddings, dtype=float)
    if points.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(points) == 0:
        raise ValueError("embeddings must be non-empty")
    if len(points) == 1:
        return 1.0

    diff = points[:, None, :] - points[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    upper = np.triu_indices(len(points), k=1)
    distances = np.sqrt(dist_sq[upper])
    median_distance = float(np.median(distances))
    if median_distance < 1e-10:
        return 1e-3
    return median_distance / np.sqrt(2.0)


def normalized_kernel_density(embeddings: np.ndarray, bandwidth: float) -> np.ndarray:
    """Self-including RBF soft occupancy rho_i = (1/N) sum_j K(z_i,z_j)."""
    points = np.asarray(embeddings, dtype=float)
    if points.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(points) == 0:
        raise ValueError("embeddings must be non-empty")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    diff = points[:, None, :] - points[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    kernels = np.exp(-dist_sq / (2.0 * bandwidth * bandwidth))
    return np.mean(kernels, axis=1)


def configurational_entropy(embeddings: np.ndarray, bandwidth: float) -> float:
    """Bounded soft-discrete entropy H = -(1/N) sum_i log rho_i."""
    density = normalized_kernel_density(embeddings, bandwidth)
    return float(-np.mean(np.log(density)))
