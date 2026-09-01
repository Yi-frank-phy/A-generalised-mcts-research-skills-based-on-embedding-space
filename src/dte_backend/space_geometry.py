"""Frozen research-space angular graph geometry for the new controller."""

from __future__ import annotations
import heapq
import numpy as np


def _normalized(points: np.ndarray, name: str) -> np.ndarray:
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or len(cloud) == 0 or cloud.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(cloud).all():
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(cloud, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} rows must have non-zero norm")
    return cloud / norms[:, None]


def pairwise_angular_distances(embeddings: np.ndarray) -> np.ndarray:
    cloud = _normalized(embeddings, "embeddings")
    distance = np.arccos(np.clip(cloud @ cloud.T, -1.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    return distance


def symmetric_knn_graph(embeddings: np.ndarray, k: int) -> np.ndarray:
    distance = pairwise_angular_distances(embeddings)
    n = len(distance)
    k = int(k)
    if k < 1 or k >= n:
        raise ValueError("k must satisfy 1 <= k < number of embeddings")
    graph = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(graph, 0.0)
    for i in range(n):
        neighbours = [int(j) for j in np.argsort(distance[i], kind="stable") if j != i][:k]
        for j in neighbours:
            edge = float(distance[i, j])
            graph[i, j] = min(graph[i, j], edge)
            graph[j, i] = min(graph[j, i], edge)
    return graph


def _dijkstra(graph: np.ndarray, source: int) -> np.ndarray:
    distance = np.full(len(graph), np.inf, dtype=float)
    distance[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, int(source))]
    while queue:
        current, u = heapq.heappop(queue)
        if current > distance[u]:
            continue
        for raw_v in np.flatnonzero(np.isfinite(graph[u])):
            v = int(raw_v)
            if v == u:
                continue
            candidate = current + float(graph[u, v])
            if candidate < distance[v] - 1e-15:
                distance[v] = candidate
                heapq.heappush(queue, (candidate, v))
    return distance


def all_pairs_geodesic_distances(embeddings: np.ndarray, k: int) -> np.ndarray:
    graph = symmetric_knn_graph(embeddings, k)
    geodesic = np.vstack([_dijkstra(graph, i) for i in range(len(graph))])
    if not np.isfinite(geodesic).all():
        raise ValueError("kNN graph is disconnected; increase k or use one connected atlas")
    return geodesic


def nearest_reference_indices(query_embeddings: np.ndarray, reference_embeddings: np.ndarray) -> np.ndarray:
    """Legacy zero-order query quantization helper.

    The authoritative proper-volume geometry no longer uses this function for
    live/query distances. It remains available for compatibility and explicit
    nearest-neighbour diagnostics.
    """
    query = _normalized(query_embeddings, "query_embeddings")
    reference = _normalized(reference_embeddings, "reference_embeddings")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference embeddings must have the same dimension")
    return np.argmin(np.arccos(np.clip(query @ reference.T, -1.0, 1.0)), axis=1).astype(int)


def _validate_geodesic(values: np.ndarray, count: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (count, count):
        raise ValueError("geodesic_distances must be square over the reference atlas")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0) or not np.allclose(matrix, matrix.T):
        raise ValueError("geodesic_distances must be finite, non-negative, and symmetric")
    return matrix


def query_reference_weights(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    *,
    neighbor_count: int | None = None,
    power: float = 2.0,
) -> np.ndarray:
    """Continuous local partition-of-unity weights over the frozen atlas.

    Off atlas this uses a compact-support modified Shepard construction. The
    support radius is just outside the requested local-neighbour radius, so a
    landmark entering or leaving support does so with vanishing weight. This
    avoids both hard kNN membership jumps and the near-uniform global Shepard
    weights caused by high-dimensional angular concentration.

    At an exact reference location the corresponding reference row is recovered
    exactly. ``neighbor_count=None`` retains all references for compatibility;
    the authoritative controller supplies a local count tied to frozen graph k.
    """
    query = _normalized(query_embeddings, "query_embeddings")
    reference = _normalized(reference_embeddings, "reference_embeddings")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference embeddings must have the same dimension")
    exponent = float(power)
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("power must be finite and positive")

    n_reference = len(reference)
    if neighbor_count is None:
        support_count = n_reference
    else:
        support_count = int(neighbor_count)
        if support_count < 1:
            raise ValueError("neighbor_count must be positive")
        support_count = min(support_count, n_reference)

    cosine = np.clip(query @ reference.T, -1.0, 1.0)
    angular = np.arccos(cosine)
    # Detect actual coincident normalized references by chord distance rather
    # than trusting arccos(1-eps), which can be around 1e-8.
    chord = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    coincident = chord <= 1e-12

    weights = np.zeros_like(angular)
    eps = np.finfo(float).eps
    for row in range(len(query)):
        exact = np.flatnonzero(coincident[row])
        if len(exact):
            weights[row, exact] = 1.0 / float(len(exact))
            continue

        local_radius = float(
            np.partition(angular[row], support_count - 1)[support_count - 1]
        )
        support_radius = max(
            local_radius * (1.0 + 1e-6),
            local_radius + 1e-12,
            eps,
        )
        inside = angular[row] < support_radius
        distances = np.maximum(angular[row, inside], eps)
        taper = np.maximum(support_radius - angular[row, inside], 0.0) / support_radius
        raw = np.power(taper / distances, exponent)
        mass = float(np.sum(raw))
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("could not construct continuous query interpolation weights")
        weights[row, inside] = raw / mass
    return weights


def reference_radii_for_queries(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    geodesic_distances: np.ndarray,
    *,
    neighbor_count: int | None = None,
) -> np.ndarray:
    """Continuously extend atlas geodesic distance profiles to arbitrary queries."""
    reference = _normalized(reference_embeddings, "reference_embeddings")
    geodesic = _validate_geodesic(geodesic_distances, len(reference))
    weights = query_reference_weights(
        query_embeddings,
        reference_embeddings,
        neighbor_count=neighbor_count,
    )
    return np.asarray(weights @ geodesic, dtype=float)


def query_geodesic_distance_matrix(
    query_a: np.ndarray,
    query_b: np.ndarray,
    reference_embeddings: np.ndarray,
    geodesic_distances: np.ndarray,
    *,
    neighbor_count: int | None = None,
) -> np.ndarray:
    """Pairwise continuous extension of the frozen graph metric.

    The atlas metric is embedded isometrically into L-infinity by its distance
    profiles. Interpolated query profiles are compared in that same norm. This
    is a continuous pseudometric off atlas and exactly equals the frozen graph
    geodesic for reference vertices.
    """
    profile_a = reference_radii_for_queries(
        query_a,
        reference_embeddings,
        geodesic_distances,
        neighbor_count=neighbor_count,
    )
    profile_b = reference_radii_for_queries(
        query_b,
        reference_embeddings,
        geodesic_distances,
        neighbor_count=neighbor_count,
    )
    return np.max(
        np.abs(profile_a[:, None, :] - profile_b[None, :, :]),
        axis=2,
    )


def query_geodesic_distance(
    query_a: np.ndarray,
    query_b: np.ndarray,
    reference_embeddings: np.ndarray,
    geodesic_distances: np.ndarray,
    *,
    neighbor_count: int | None = None,
) -> float:
    pair = query_geodesic_distance_matrix(
        np.asarray(query_a, dtype=float)[None, :],
        np.asarray(query_b, dtype=float)[None, :],
        reference_embeddings,
        geodesic_distances,
        neighbor_count=neighbor_count,
    )
    return float(pair[0, 0])
