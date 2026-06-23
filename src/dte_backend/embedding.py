"""Embedding providers for DTE search geometry.

The formal DTE entropy/temperature controller depends on a continuous geometry.
For real research runs, use a strong external provider such as Gemini Embedding 2
or another Qwen3-Embedding-8B-class endpoint. The hash provider exists only for
offline tests and fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol
from urllib import request as urlrequest

from .text_features import hashed_embedding


class EmbeddingProvider(Protocol):
    """Common embedding provider interface."""

    name: str
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into fixed-size float vectors."""


@dataclass
class HashEmbeddingProvider:
    """Deterministic local fallback; not a semantic-quality embedding."""

    dim: int = 64
    name: str = "hash"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [hashed_embedding(text, dim=self.dim) for text in texts]


@dataclass
class GeminiEmbedding2Provider:
    """Gemini Embedding 2 provider using the Google Generative Language REST API.

    This provider is intentionally dependency-light. It is not exercised in CI.
    It expects `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment.

    DTE should call this provider with short node summaries, not full executor
    transcripts. Persistent caching should be enabled before high-volume use.
    """

    dim: int = 3072
    model: str = "gemini-embedding-2"
    api_key: str | None = None
    name: str = "gemini-embedding-2"

    def _key(self) -> str:
        key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GeminiEmbedding2Provider requires GEMINI_API_KEY or GOOGLE_API_KEY")
        return key

    def _embed_one(self, text: str) -> list[float]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:embedContent?key={self._key()}"
        )
        payload = {
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dim,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlrequest.urlopen(req, timeout=60) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        values = parsed.get("embedding", {}).get("values")
        if not isinstance(values, list):
            raise RuntimeError(f"Gemini embedding response missing embedding.values: {parsed}")
        return [float(v) for v in values]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Free AI Studio embedding does not expose a batch tier in the same way;
        # keep batching at the DTE cache/request layer, not inside this REST loop.
        return [self._embed_one(text) for text in texts]


def get_embedding_provider(name: str = "hash", dim: int = 3072) -> EmbeddingProvider:
    """Create an embedding provider by name."""

    if name == "hash":
        return HashEmbeddingProvider(dim=dim)
    if name == "gemini-embedding-2":
        return GeminiEmbedding2Provider(dim=dim)
    raise ValueError("embedding provider must be 'hash' or 'gemini-embedding-2'")
