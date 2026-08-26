from dte_backend.cache import EmbeddingCacheNamespace
from dte_backend.file_cache import FileDTECache
from dte_backend.models import SearchNode


def test_file_cache_persists_embedding(tmp_path):
    path = tmp_path / "dte_cache.json"
    node = SearchNode(node_id="n", claim="claim")
    cache = FileDTECache(path)
    cache.set_embedding(node, [0.1, 0.2])
    cache2 = FileDTECache(path)
    assert cache2.get_embedding(node) == [0.1, 0.2]




def test_file_cache_does_not_share_embedding_across_namespaces(tmp_path):
    path = tmp_path / "dte_cache.json"
    node = SearchNode(node_id="n", claim="claim")
    first = EmbeddingCacheNamespace("hash", "hash-v1", 64, "embedding-v1")
    second = EmbeddingCacheNamespace("hash", "hash-v1", 128, "embedding-v1")
    cache = FileDTECache(path)
    cache.set_embedding(node, [0.1, 0.2], namespace=first)

    assert cache.get_embedding(node, namespace=second) is None


