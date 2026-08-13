"""Sparse angular geometry for the next-generation proper-volume controller."""

from __future__ import annotations

import heapq

import numpy as np


def _normalized_cloud(points: np.ndarray, name: str) -> np.ndarray:
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
    """Return pairwise angular distances between L2-normalized embeddings."""
    cloud = _normalized_cloud(embeddings, "embeddings")
    cosine = np.clip(cloud @ cloud.T, -1.0, 1.0)
    distance = np.arccos(cosine)
    np.fill_diagonal(distance, 0.0)
    return distance


def symmetric_knn_graph(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Build a symmetric kNN-union graph with angular edge lengths."""
    distance = pairwise_angular_distances(embeddings)
    n = len(distance)
    neighbour_count = int(k)
    if neighbour_count < 1 or neighbour_count >= n:
        raise ValueError("k must satisfy 1 <= k < number of embeddings")

    graph = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(graph, 0.0)
    for i in range(n):
        order = np.argsort(distance[i], kind="stable")
        neighbours = [int(j) for j in order if j != i][:neighbour_count]
        for j in neighbours:
            edge = float(distance[i, j])
            graph[i, j] = min(graph[i, j], edge)
            graph[j, i] = min(graph[j, i], edge)
    return graph


def _dijkstra(graph: np.ndarray, source: int) -> np.ndarray:
    n = len(graph)
    distance = np.full(n, np.inf, dtype=float)
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
    """Return all-pairs shortest paths on a connected sparse angular graph."""
    graph = symmetric_knn_graph(embeddings, k)
    geodesic = np.vstack([_dijkstra(graph, i) for i in range(len(graph))])
    if not np.isfinite(geodesic).all():
        raise ValueError(
            "kNN graph is disconnected; increase k or restrict the reference atlas "
            "to one connected component"
        )
    return geodesic


def nearest_reference_indices(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
) -> np.ndarray:
    """Map each query embedding to its nearest frozen reference location."""
    query = _normalized_cloud(query_embeddings, "query_embeddings")
    reference = _normalized_cloud(reference_embeddings, "reference_embeddings")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference embeddings must have the same dimension")
    cosine = np.clip(query @ reference.T, -1.0, 1.0)
    angular = np.arccos(cosine)
    return np.argmin(angular, axis=1).astype(int)


def _validate_geodesic_matrix(
    geodesic_distances: np.ndarray,
    reference_count: int,
) -> np.ndarray:
    geodesic = np.asarray(geodesic_distances, dtype=float)
    if geodesic.shape != (reference_count, reference_count):
        raise ValueError("geodesic_distances must be square over the reference atlas")
    if not np.isfinite(geodesic).all() or np.any(geodesic < 0.0):
        raise ValueError("geodesic_distances must be finite and non-negative")
    if not np.allclose(geodesic, geodesic.T):
        raise ValueError("geodesic_distances must be symmetric")
    return geodesic


def reference_radii_for_queries(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    geodesic_distances: np.ndarray,
) -> np.ndarray:
    """Return atlas-cell geodesic radii for each anchored query."""
    reference = _normalized_cloud(reference_embeddings, "reference_embeddings")
    geodesic = _validate_geodesic_matrix(geodesic_distances, len(reference))
    anchors = nearest_reference_indices(query_embeddings, reference_embeddings)
    return geodesic[anchors].copy()


def query_geodesic_distance(
    query_a: np.ndarray,
    query_b: np.ndarray,
    reference_embeddings: np.ndarray,
    geodesic_distances: np.ndarray,
) -> float:
    """Return geodesic distance after mapping both queries to the frozen atlas."""
    reference = _normalized_cloud(reference_embeddings, "reference_embeddings")
    geodesic = _validate_geodesic_matrix(geodesic_distances, len(reference))
    queries = np.vstack([np.asarray(query_a, dtype=float), np.asarray(query_b, dtype=float)])
    anchors = nearest_reference_indices(queries, reference_embeddings)
    return float(geodesic[int(anchors[0]), int(anchors[1])])
