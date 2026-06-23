import pytest

from dte_backend.embedding import GeminiEmbedding2Provider, HashEmbeddingProvider, get_embedding_provider


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


def test_unknown_embedding_provider_rejected():
    with pytest.raises(ValueError):
        get_embedding_provider("tiny-random", dim=64)
