"""Run-local realized-return evidence for the next-generation controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Record:
    embedding: np.ndarray
    mean_return: float
    count: int


class TransitionHistory:
    def __init__(self) -> None:
        self._records: dict[str, _Record] = {}

    def __len__(self) -> int:
        return len(self._records)

    def record(self, node_id: str, embedding: np.ndarray, observed_return: float) -> None:
        vector = np.asarray(embedding, dtype=float)
        value = float(observed_return)
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if vector.ndim != 1 or len(vector) == 0 or not np.isfinite(vector).all():
            raise ValueError("embedding must be a finite non-empty 1D vector")
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("observed_return must be finite and non-negative")

        current = self._records.get(node_id)
        if current is None:
            self._records[node_id] = _Record(vector.copy(), value, 1)
            return
        if current.embedding.shape != vector.shape or not np.allclose(current.embedding, vector):
            raise ValueError("existing node embedding cannot change")
        count = current.count + 1
        current.mean_return += (value - current.mean_return) / count
        current.count = count

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._records:
            return np.empty((0, 0)), np.array([], dtype=float), np.array([], dtype=int)
        records = list(self._records.values())
        return (
            np.vstack([record.embedding for record in records]),
            np.asarray([record.mean_return for record in records], dtype=float),
            np.asarray([record.count for record in records], dtype=int),
        )
