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


def reference_radii_for_queries(query_embeddings: np.ndarray, reference_embeddings: np.ndarray, geodesic_distances: np.ndarray) -> np.ndarray:
    reference = _normalized(reference_embeddings, "reference_embeddings")
    geodesic = _validate_geodesic(geodesic_distances, len(reference))
    return geodesic[nearest_reference_indices(query_embeddings, reference_embeddings)].copy()


def query_geodesic_distance(query_a: np.ndarray, query_b: np.ndarray, reference_embeddings: np.ndarray, geodesic_distances: np.ndarray) -> float:
    reference = _normalized(reference_embeddings, "reference_embeddings")
    geodesic = _validate_geodesic(geodesic_distances, len(reference))
    anchors = nearest_reference_indices(
        np.vstack([np.asarray(query_a, dtype=float), np.asarray(query_b, dtype=float)]),
        reference_embeddings,
    )
    return float(geodesic[int(anchors[0]), int(anchors[1])])
