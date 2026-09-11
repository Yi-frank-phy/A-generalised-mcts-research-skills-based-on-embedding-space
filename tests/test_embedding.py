import json
from urllib import error as urlerror

import pytest

from dte_backend.embedding import GeminiEmbedding2Provider, HashEmbeddingProvider, get_embedding_provider


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_hash_embedding_provider_batches():
    provider = HashEmbeddingProvider(dim=16)
    vectors = provider.embed_texts(["alpha", "beta"])
    assert len(vectors) == 2
    assert all(len(v) == 16 for v in vectors)


def test_get_gemini_embedding_provider_metadata():
    provider = get_embedding_provider("gemini-embedding-2", dim=1536)
    assert isinstance(provider, GeminiEmbedding2Provider)
    assert provider.dim == 1536


def test_gemini_embedding_provider_defaults_to_max_geometry():
    assert GeminiEmbedding2Provider().dim == 3072
    assert get_embedding_provider("gemini-embedding-2").dim == 3072


def test_gemini_embedding_provider_uses_batch_endpoint(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        payload = json.loads(request.data.decode("utf-8"))
        return _Response(
            {"embeddings": [{"values": [float(index), 1.0]} for index, _ in enumerate(payload["requests"])]}
        )

    monkeypatch.setattr("dte_backend.embedding.urlrequest.urlopen", fake_urlopen)
    provider = GeminiEmbedding2Provider(dim=2, api_key="test-key")

    assert provider.embed_texts(["alpha", "beta"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert len(requests) == 1
    assert ":batchEmbedContents?" in requests[0].full_url
    payload = json.loads(requests[0].data.decode("utf-8"))
    assert [item["model"] for item in payload["requests"]] == [
        "models/gemini-embedding-2",
        "models/gemini-embedding-2",
    ]
    assert all(item["outputDimensionality"] == 2 for item in payload["requests"])


def test_gemini_embedding_provider_chunks_large_batches(monkeypatch):
    batch_sizes = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        batch_sizes.append(len(payload["requests"]))
        return _Response({"embeddings": [{"values": [1.0]} for _ in payload["requests"]]})

    monkeypatch.setattr("dte_backend.embedding.urlrequest.urlopen", fake_urlopen)
    provider = GeminiEmbedding2Provider(dim=1, api_key="test-key")

    assert len(provider.embed_texts([str(index) for index in range(101)])) == 101
    assert batch_sizes == [100, 1]


def test_gemini_embedding_provider_waits_for_rate_limit_window(monkeypatch):
    attempts = 0
    sleeps = []

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urlerror.HTTPError(request.full_url, 429, "rate limited", {}, None)
        return _Response({"embeddings": [{"values": [1.0]}]})

    monkeypatch.setattr("dte_backend.embedding.urlrequest.urlopen", fake_urlopen)
    monkeypatch.setattr("dte_backend.embedding.time.sleep", sleeps.append)
    provider = GeminiEmbedding2Provider(dim=1, api_key="test-key")

    assert provider.embed_texts(["alpha"]) == [[1.0]]
    assert attempts == 2
    assert sleeps == [60.0]


def test_unknown_embedding_provider_rejected():
    with pytest.raises(ValueError):
        get_embedding_provider("tiny-random", dim=64)
