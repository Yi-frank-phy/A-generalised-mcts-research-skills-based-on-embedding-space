from copy import deepcopy
import math

import pytest

from dte_backend.cache import DTECache, EmbeddingCacheNamespace, embedding_cache_key
from dte_backend.file_cache import FileDTECache
from dte_backend.models import SearchNode
from dte_backend.novelty import ensure_embeddings, estimate_frontier_kde_state


class StaticEmbeddingProvider:
    name = "static-test"

    def __init__(self, dim, vectors):
        self.dim = dim
        self.vectors = vectors
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return deepcopy(self.vectors)


def namespace(provider, dimension):
    return EmbeddingCacheNamespace(
        provider=provider.name,
        model_snapshot=provider.name,
        dimension=dimension,
        contract_version="embedding-v1",
    )


def test_embedding_batch_validates_existing_cache_and_provider_before_install():
    provider = StaticEmbeddingProvider(3, [[7.0, 8.0, 9.0]])
    cache = DTECache()
    existing = SearchNode(node_id="existing", claim="existing", local_embedding=[1, 2, 3])
    cached = SearchNode(node_id="cached", claim="cached")
    generated = SearchNode(node_id="generated", claim="generated")
    cache.set_embedding(cached, [4.0, 5.0, 6.0], namespace=namespace(provider, 3))

    ensure_embeddings(
        [existing, cached, generated],
        cache=cache,
        provider=provider,
        expected_dimension=3,
    )

    assert existing.local_embedding == [1.0, 2.0, 3.0]
    assert cached.local_embedding == [4.0, 5.0, 6.0]
    assert generated.local_embedding == [7.0, 8.0, 9.0]
    assert len(provider.calls) == 1
    assert len(provider.calls[0]) == 1
    assert cache.get_embedding(existing, namespace=namespace(provider, 3)) == existing.local_embedding
    assert cache.get_embedding(generated, namespace=namespace(provider, 3)) == generated.local_embedding


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([], "wrong number of vectors"),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "wrong number of vectors"),
        ([[1.0, 2.0]], "dimension mismatch"),
        ([[1.0, math.nan, 3.0]], "not finite"),
        ([[1.0, math.inf, 3.0]], "not finite"),
    ],
)
def test_invalid_provider_batch_leaves_nodes_and_cache_unchanged(vectors, message):
    provider = StaticEmbeddingProvider(3, vectors)
    cache = DTECache()
    existing = SearchNode(node_id="existing", claim="existing", local_embedding=[1.0, 2.0, 3.0])
    missing = SearchNode(node_id="missing", claim="missing")
    node_before = [deepcopy(existing.local_embedding), deepcopy(missing.local_embedding)]
    cache_before = deepcopy(cache.__dict__)

    with pytest.raises(ValueError, match=message):
        ensure_embeddings(
            [existing, missing],
            cache=cache,
            provider=provider,
            expected_dimension=3,
        )

    assert [existing.local_embedding, missing.local_embedding] == node_before
    assert cache.__dict__ == cache_before


def test_provider_metadata_must_match_expected_dimension_before_any_lookup_or_call():
    provider = StaticEmbeddingProvider(2, [[1.0, 2.0]])
    cache = DTECache()
    node = SearchNode(node_id="n", claim="claim")
    cache_before = deepcopy(cache.__dict__)

    with pytest.raises(ValueError, match="provider dimension"):
        ensure_embeddings(
            [node],
            cache=cache,
            provider=provider,
            expected_dimension=3,
        )

    assert provider.calls == []
    assert node.local_embedding is None
    assert cache.__dict__ == cache_before


@pytest.mark.parametrize(
    ("invalid_vector", "message"),
    [
        ([1.0, 2.0], "dimension mismatch"),
        ([1.0, math.nan, 3.0], "not finite"),
    ],
)
def test_invalid_existing_vector_prevents_cache_or_provider_mutation(invalid_vector, message):
    provider = StaticEmbeddingProvider(3, [[4.0, 5.0, 6.0]])
    cache = DTECache()
    invalid = SearchNode(node_id="invalid", claim="invalid")
    invalid.local_embedding = invalid_vector
    missing = SearchNode(node_id="missing", claim="missing")
    cache_before = deepcopy(cache.__dict__)

    with pytest.raises(ValueError, match=f"existing node.*{message}"):
        ensure_embeddings(
            [invalid, missing],
            cache=cache,
            provider=provider,
            expected_dimension=3,
        )

    if message == "not finite":
        assert invalid.local_embedding is not None
        assert math.isnan(invalid.local_embedding[1])
    else:
        assert invalid.local_embedding == invalid_vector
    assert missing.local_embedding is None
    assert provider.calls == []
    assert cache.__dict__ == cache_before


@pytest.mark.parametrize(
    ("invalid_vector", "message"),
    [
        ([1.0, 2.0], "dimension mismatch"),
        ([1.0, math.inf, 3.0], "not finite"),
    ],
)
def test_invalid_cached_vector_rolls_back_cache_stats_and_all_node_changes(invalid_vector, message):
    provider = StaticEmbeddingProvider(3, [[7.0, 8.0, 9.0]])
    cache = DTECache()
    cached = SearchNode(node_id="cached", claim="cached")
    other = SearchNode(node_id="other", claim="other")
    key = embedding_cache_key(cached, namespace=namespace(provider, 3))
    cache.embeddings[key] = invalid_vector
    cache_before = deepcopy(cache.__dict__)

    with pytest.raises(ValueError, match=f"cached node.*{message}"):
        ensure_embeddings(
            [cached, other],
            cache=cache,
            provider=provider,
            expected_dimension=3,
        )

    assert cached.local_embedding is None
    assert other.local_embedding is None
    assert provider.calls == []
    assert cache.__dict__ == cache_before


class FailingFileCache(FileDTECache):
    def __init__(self, path):
        super().__init__(path)
        self.set_calls = 0

    def set_embedding(self, node, embedding, namespace):
        self.set_calls += 1
        super().set_embedding(node, embedding, namespace=namespace)
        if self.set_calls == 2:
            raise OSError("second cache write failed")


def test_cache_install_failure_rolls_back_file_cache_and_nodes(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache = FailingFileCache(cache_path)
    provider = StaticEmbeddingProvider(3, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    nodes = [
        SearchNode(node_id="a", claim="A"),
        SearchNode(node_id="b", claim="B"),
    ]
    cache_before = deepcopy(cache.__dict__)

    with pytest.raises(OSError, match="second cache write failed"):
        ensure_embeddings(
            nodes,
            cache=cache,
            provider=provider,
            expected_dimension=3,
        )

    assert [node.local_embedding for node in nodes] == [None, None]
    assert cache.__dict__ == cache_before
    assert not cache_path.exists()


def test_estimate_frontier_propagates_expected_dimension_atomically():
    provider = StaticEmbeddingProvider(3, [[1.0, 2.0]])
    nodes = [SearchNode(node_id="n", claim="claim")]

    with pytest.raises(ValueError, match="dimension mismatch"):
        estimate_frontier_kde_state(
            nodes,
            provider=provider,
            expected_dimension=3,
        )

    assert nodes[0].local_embedding is None
