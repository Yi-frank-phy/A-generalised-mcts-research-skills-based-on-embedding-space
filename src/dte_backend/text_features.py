"""Deterministic local text features for the DTE prototype.

This module is intentionally cheap and offline. It is *not* a semantic embedding
replacement. It exists so the prototype can exercise the DTE entropy/novelty
loop without calling an external model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Return coarse tokens for English/Chinese mixed research notes."""

    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def node_text_parts(claim: str, rationale: str = "", assumptions: Iterable[str] = (), evidence: Iterable[str] = (), risks: Iterable[str] = ()) -> str:
    """Create the canonical text representation used for local features."""

    return "\n".join(
        [claim or "", rationale or "", "\n".join(assumptions), "\n".join(evidence), "\n".join(risks)]
    ).strip()


def hashed_embedding(text: str, dim: int = 64) -> list[float]:
    """Create a stable hashed bag-of-words embedding.

    This avoids Python's randomized hash and keeps tests reproducible. The vector
    is L2-normalized. Empty text returns a zero vector.
    """

    if dim <= 0:
        raise ValueError("dim must be positive")

    vec = np.zeros(dim, dtype=float)
    tokens = tokenize(text)
    if not tokens:
        return vec.tolist()

    counts = Counter(tokens)
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign * (1.0 + math.log(count))

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


def cosine_distance_matrix(embeddings: list[list[float]]) -> np.ndarray:
    """Return pairwise cosine distance matrix for normalized embeddings."""

    if not embeddings:
        return np.empty((0, 0), dtype=float)
    arr = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    arr = arr / safe
    sim = np.clip(arr @ arr.T, -1.0, 1.0)
    return 1.0 - sim
